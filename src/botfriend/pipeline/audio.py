"""Microphone audio capture with chunked async streaming."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioChunk:
    data: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return len(self.data) / self.sample_rate


class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 3.0,
        overlap_duration: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self._buffer = np.array([], dtype=np.float32)
        self._queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._running = False

    @property
    def chunk_size(self) -> int:
        return int(self.sample_rate * self.chunk_duration)

    @property
    def overlap_size(self) -> int:
        return int(self.sample_rate * self.overlap_duration)

    def _on_audio_data(self, indata: np.ndarray) -> None:
        mono = indata[:, 0] if indata.ndim > 1 else indata
        self._buffer = np.concatenate([self._buffer, mono])

        while len(self._buffer) >= self.chunk_size:
            chunk_data = self._buffer[: self.chunk_size].copy()
            self._buffer = self._buffer[self.chunk_size - self.overlap_size :]
            try:
                self._queue.put_nowait(
                    AudioChunk(data=chunk_data, sample_rate=self.sample_rate)
                )
            except asyncio.QueueFull:
                pass

    def stop(self) -> None:
        self._running = False

    async def stream(self) -> AsyncIterator[AudioChunk]:
        self._running = True
        loop = asyncio.get_event_loop()

        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            if status:
                pass
            loop.call_soon_threadsafe(self._on_audio_data, indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while self._running:
                try:
                    chunk = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                    yield chunk
                except asyncio.TimeoutError:
                    continue
