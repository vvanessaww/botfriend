import pytest
import asyncio
import numpy as np
from unittest.mock import patch, MagicMock
from botfriend.pipeline.audio import AudioCapture, AudioChunk


class TestAudioChunk:
    def test_create_chunk(self):
        data = np.zeros(48000, dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=16000)
        assert chunk.sample_rate == 16000
        assert len(chunk.data) == 48000

    def test_duration_seconds(self):
        data = np.zeros(48000, dtype=np.float32)
        chunk = AudioChunk(data=data, sample_rate=16000)
        assert chunk.duration_seconds == 3.0


class TestAudioCapture:
    def test_create_with_defaults(self):
        capture = AudioCapture()
        assert capture.sample_rate == 16000
        assert capture.chunk_duration == 2.0
        assert capture.overlap_duration == 0.5

    def test_create_with_custom_params(self):
        capture = AudioCapture(sample_rate=44100, chunk_duration=5.0, overlap_duration=2.0)
        assert capture.sample_rate == 44100
        assert capture.chunk_duration == 5.0
        assert capture.overlap_duration == 2.0

    def test_chunk_size_calculation(self):
        capture = AudioCapture(sample_rate=16000, chunk_duration=3.0)
        assert capture.chunk_size == 48000

    def test_overlap_size_calculation(self):
        capture = AudioCapture(sample_rate=16000, overlap_duration=1.0)
        assert capture.overlap_size == 16000

    @pytest.mark.asyncio
    async def test_produces_chunks_from_mock_audio(self):
        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0, overlap_duration=0.0)

        chunks_received = []

        async def collect_chunks():
            count = 0
            async for chunk in capture.stream():
                chunks_received.append(chunk)
                count += 1
                if count >= 2:
                    capture.stop()

        fake_audio = np.random.randn(16000).astype(np.float32)

        with patch("sounddevice.InputStream") as mock_stream:
            mock_instance = MagicMock()
            mock_stream.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)

            async def feed_audio():
                await asyncio.sleep(0.05)
                for _ in range(3):
                    capture._on_audio_data(fake_audio.reshape(-1, 1))
                    await asyncio.sleep(0.01)

            await asyncio.gather(
                collect_chunks(),
                feed_audio(),
            )

        assert len(chunks_received) >= 2
        for chunk in chunks_received:
            assert isinstance(chunk, AudioChunk)
            assert chunk.sample_rate == 16000
