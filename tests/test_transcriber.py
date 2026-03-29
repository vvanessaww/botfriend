import pytest
import math
import numpy as np
from unittest.mock import MagicMock, patch
from botfriend.pipeline.audio import AudioChunk
from botfriend.pipeline.transcriber import Transcriber
from botfriend.models import TranscriptSegment


class TestTranscriber:
    def _make_chunk(self, duration: float = 3.0) -> AudioChunk:
        samples = int(16000 * duration)
        return AudioChunk(data=np.zeros(samples, dtype=np.float32), sample_rate=16000)

    @patch("botfriend.pipeline.transcriber.WhisperModel")
    def test_init_loads_model(self, mock_whisper_cls):
        transcriber = Transcriber(model_size="base")
        mock_whisper_cls.assert_called_once_with("base", device="cpu", compute_type="int8")

    @patch("botfriend.pipeline.transcriber.WhisperModel")
    def test_transcribe_returns_segments(self, mock_whisper_cls):
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = " we should make the button blue"
        mock_segment.avg_logprob = -0.3
        mock_segment.no_speech_prob = 0.1

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98

        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        transcriber = Transcriber(model_size="base")
        chunk = self._make_chunk()
        segments = transcriber.transcribe(chunk)

        assert len(segments) == 1
        assert segments[0].text == "we should make the button blue"
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 2.5

    @patch("botfriend.pipeline.transcriber.WhisperModel")
    def test_transcribe_filters_no_speech(self, mock_whisper_cls):
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 3.0
        mock_segment.text = " "
        mock_segment.avg_logprob = -1.5
        mock_segment.no_speech_prob = 0.9

        mock_info = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        transcriber = Transcriber(model_size="base")
        segments = transcriber.transcribe(self._make_chunk())

        assert len(segments) == 0

    @patch("botfriend.pipeline.transcriber.WhisperModel")
    def test_transcribe_sets_confidence_from_logprob(self, mock_whisper_cls):
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " hello"
        mock_segment.avg_logprob = -0.2
        mock_segment.no_speech_prob = 0.05

        mock_info = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        transcriber = Transcriber(model_size="base")
        segments = transcriber.transcribe(self._make_chunk())

        assert len(segments) == 1
        expected_confidence = math.exp(-0.2)
        assert abs(segments[0].confidence - expected_confidence) < 0.01

    @patch("botfriend.pipeline.transcriber.WhisperModel")
    def test_transcribe_assigns_default_speaker(self, mock_whisper_cls):
        mock_model = MagicMock()
        mock_whisper_cls.return_value = mock_model

        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " hello"
        mock_segment.avg_logprob = -0.3
        mock_segment.no_speech_prob = 0.05

        mock_info = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        transcriber = Transcriber(model_size="base")
        segments = transcriber.transcribe(self._make_chunk())

        assert segments[0].speaker == "Speaker"
