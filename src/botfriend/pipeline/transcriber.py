"""Whisper-based transcription with optional speaker diarization."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from botfriend.models import TranscriptSegment
from botfriend.pipeline.audio import AudioChunk

NO_SPEECH_THRESHOLD = 0.6


class Transcriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._diarizer: Optional[object] = None

    def enable_diarization(self, hf_token: str) -> None:
        from pyannote.audio import Pipeline

        self._diarizer = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    def transcribe(self, chunk: AudioChunk) -> list[TranscriptSegment]:
        segments_iter, info = self._model.transcribe(
            chunk.data,
            beam_size=5,
            language="en",
            vad_filter=True,
        )

        results: list[TranscriptSegment] = []
        for seg in segments_iter:
            if seg.no_speech_prob > NO_SPEECH_THRESHOLD:
                continue
            text = seg.text.strip()
            if not text:
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
        return results

    def _identify_speaker(
        self, chunk: AudioChunk, start: float, end: float
    ) -> str:
        if self._diarizer is None:
            return "Speaker"

        import torch

        waveform = torch.from_numpy(chunk.data).unsqueeze(0)
        diarization = self._diarizer(
            {"waveform": waveform, "sample_rate": chunk.sample_rate}
        )

        mid_point = (start + end) / 2
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if turn.start <= mid_point <= turn.end:
                return speaker_label
        return "Speaker"
