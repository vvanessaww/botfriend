"""Whisper-based transcription with optional speaker diarization."""

from __future__ import annotations

import math
import re
from collections import deque
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from botfriend.models import TranscriptSegment
from botfriend.pipeline.audio import AudioChunk

# If Whisper thinks there's a >40% chance nobody is speaking, skip it.
NO_SPEECH_THRESHOLD = 0.4

# Segments with avg_logprob below this are low-confidence garbage.
MIN_LOG_PROB = -1.0

# Common Whisper hallucinations on background noise.
_HALLUCINATION_PATTERNS = re.compile(
    r"^("
    r"(i'?m not sure|thank you|thanks for watching|"
    r"you|bye|okay|oh|hmm|uh|um|ah|so|"
    r"subscribe|like and subscribe|see you next time|"
    r"the end|\.+|,+)"
    r"[.!?, ]*"
    r")+$",
    re.IGNORECASE,
)


class Transcriber:
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._diarizer: Optional[object] = None
        # Track recent outputs to catch repetition hallucinations
        self._recent_texts: deque[str] = deque(maxlen=5)

    def enable_diarization(self, hf_token: str) -> None:
        from pyannote.audio import Pipeline

        self._diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )

    def transcribe(self, chunk: AudioChunk) -> list[TranscriptSegment]:
        segments_iter, info = self._model.transcribe(
            chunk.data,
            beam_size=1,
            language="en",
            vad_filter=True,
            vad_parameters=dict(
                min_speech_duration_ms=250,
                min_silence_duration_ms=400,
                speech_pad_ms=100,
            ),
            condition_on_previous_text=False,
            temperature=0.0,
        )

        results: list[TranscriptSegment] = []
        for seg in segments_iter:
            # High no-speech probability — skip
            if seg.no_speech_prob > NO_SPEECH_THRESHOLD:
                continue

            # Very low confidence — skip
            if seg.avg_logprob < MIN_LOG_PROB:
                continue

            text = seg.text.strip()
            if not text:
                continue

            # Filter known hallucination phrases
            if _HALLUCINATION_PATTERNS.match(text):
                continue

            # Filter exact repetitions (Whisper loops)
            normalized = text.lower().strip("., !?")
            if normalized in self._recent_texts:
                continue

            confidence = math.exp(seg.avg_logprob)
            speaker = self._identify_speaker(chunk, seg.start, seg.end)
            results.append(
                TranscriptSegment(
                    text=text,
                    speaker=speaker,
                    start_time=seg.start,
                    end_time=seg.end,
                    confidence=confidence,
                )
            )
            self._recent_texts.append(normalized)

        return results

    def _identify_speaker(
        self, chunk: AudioChunk, start: float, end: float
    ) -> str:
        if self._diarizer is None:
            return "Speaker"

        import torch

        waveform = torch.from_numpy(chunk.data).unsqueeze(0)
        result = self._diarizer(
            {"waveform": waveform, "sample_rate": chunk.sample_rate}
        )

        # pyannote >= 3.1 returns DiarizeOutput; get the annotation from it
        annotation = getattr(result, "speaker_diarization", result)

        mid_point = (start + end) / 2
        for turn, _, speaker_label in annotation.itertracks(yield_label=True):
            if turn.start <= mid_point <= turn.end:
                return speaker_label
        return "Speaker"
