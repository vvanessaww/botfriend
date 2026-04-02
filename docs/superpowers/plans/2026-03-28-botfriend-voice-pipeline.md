# Botfriend Voice Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a terminal-based tool that listens to live conversations via microphone, extracts actionable code changes using Claude Code CLI, and executes them against a local codebase with user confirmation.

**Architecture:** Four-stage async pipeline (audio capture -> transcription -> action extraction -> execution) rendered in a three-pane Textual TUI. Each stage communicates via async queues. Two separate Claude Code CLI usages: one stateless for extraction, one persistent for coding.

**Tech Stack:** Python 3.11+, textual, faster-whisper, pyannote-audio, sounddevice, numpy

---

## File Structure

```
botfriend/
├── src/
│   └── botfriend/
│       ├── __init__.py              # Package init, version
│       ├── main.py                  # CLI entry point (argparse, startup)
│       ├── app.py                   # Textual App subclass, layout, keybindings
│       ├── models.py                # Dataclasses: TranscriptSegment, ActionItem
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── audio.py             # Mic capture via sounddevice, async chunking
│       │   ├── transcriber.py       # Whisper + pyannote transcription
│       │   └── extractor.py         # Action extraction via Claude CLI
│       ├── queue/
│       │   ├── __init__.py
│       │   └── action_queue.py      # ActionQueue state machine
│       ├── executor/
│       │   ├── __init__.py
│       │   └── claude_runner.py     # Claude Code CLI wrapper for code execution
│       └── widgets/
│           ├── __init__.py
│           ├── transcript_panel.py  # RichLog-based transcript display
│           ├── queue_panel.py       # Selectable action item list
│           └── execution_log.py     # RichLog-based execution output
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_action_queue.py
│   ├── test_audio.py
│   ├── test_transcriber.py
│   ├── test_extractor.py
│   ├── test_claude_runner.py
│   └── test_app.py
├── pyproject.toml
└── README.md
```

---

### Task 1: Project Scaffolding & Package Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/botfriend/__init__.py`
- Create: `src/botfriend/pipeline/__init__.py`
- Create: `src/botfriend/queue/__init__.py`
- Create: `src/botfriend/executor/__init__.py`
- Create: `src/botfriend/widgets/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "botfriend"
version = "0.1.0"
description = "Real-time voice-to-code pipeline using Claude Code"
requires-python = ">=3.11"
dependencies = [
    "textual>=0.85.0",
    "faster-whisper>=1.0.0",
    "pyannote.audio>=3.1",
    "sounddevice>=0.4.6",
    "numpy>=1.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.0",
]

[project.scripts]
botfriend = "botfriend.main:main"
```

- [ ] **Step 2: Create package init files**

`src/botfriend/__init__.py`:
```python
"""Botfriend: Real-time voice-to-code pipeline."""

__version__ = "0.1.0"
```

`src/botfriend/pipeline/__init__.py`:
```python
```

`src/botfriend/queue/__init__.py`:
```python
```

`src/botfriend/executor/__init__.py`:
```python
```

`src/botfriend/widgets/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

- [ ] **Step 3: Install the package in dev mode**

Run: `cd /Users/vanessawang/botfriend && python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

Expected: Package installs successfully with all dependencies.

Note: `pyannote.audio` and `faster-whisper` have large transitive dependencies (torch, etc.). This step may take several minutes.

- [ ] **Step 4: Verify installation**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -c "import botfriend; print(botfriend.__version__)"`

Expected: `0.1.0`

- [ ] **Step 5: Commit**

```bash
git init
git add pyproject.toml src/ tests/__init__.py
git commit -m "chore: scaffold botfriend package with dependencies"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/botfriend/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for data models**

`tests/test_models.py`:
```python
import pytest
from dataclasses import FrozenInstanceError
from botfriend.models import (
    TranscriptSegment,
    ActionItem,
    ActionStatus,
)


class TestTranscriptSegment:
    def test_create_segment(self):
        seg = TranscriptSegment(
            text="we should make the button blue",
            speaker="Speaker 1",
            start_time=0.0,
            end_time=2.5,
            confidence=0.95,
        )
        assert seg.text == "we should make the button blue"
        assert seg.speaker == "Speaker 1"
        assert seg.start_time == 0.0
        assert seg.end_time == 2.5
        assert seg.confidence == 0.95

    def test_segment_is_immutable(self):
        seg = TranscriptSegment(
            text="hello",
            speaker="Speaker 1",
            start_time=0.0,
            end_time=1.0,
            confidence=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            seg.text = "modified"


class TestActionItem:
    def test_create_action_item(self):
        item = ActionItem(
            description="Change CTA button color to blue",
            speakers=("Speaker 1", "Speaker 2"),
            confidence="high",
        )
        assert item.description == "Change CTA button color to blue"
        assert item.speakers == ("Speaker 1", "Speaker 2")
        assert item.confidence == "high"
        assert item.status == ActionStatus.PENDING

    def test_action_item_with_status(self):
        item = ActionItem(
            description="Add modal",
            speakers=("Speaker 1",),
            confidence="medium",
            status=ActionStatus.EXECUTING,
        )
        assert item.status == ActionStatus.EXECUTING

    def test_action_item_replace_status(self):
        item = ActionItem(
            description="Add modal",
            speakers=("Speaker 1",),
            confidence="high",
        )
        updated = item.with_status(ActionStatus.CONFIRMED)
        assert updated.status == ActionStatus.CONFIRMED
        assert item.status == ActionStatus.PENDING  # original unchanged

    def test_action_item_replace_description(self):
        item = ActionItem(
            description="Make it blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        updated = item.with_description("Make it teal")
        assert updated.description == "Make it teal"
        assert item.description == "Make it blue"  # original unchanged


class TestActionStatus:
    def test_status_values(self):
        assert ActionStatus.PENDING.value == "pending"
        assert ActionStatus.CONFIRMED.value == "confirmed"
        assert ActionStatus.EXECUTING.value == "executing"
        assert ActionStatus.DONE.value == "done"
        assert ActionStatus.FAILED.value == "failed"
        assert ActionStatus.REJECTED.value == "rejected"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_models.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'botfriend.models'`

- [ ] **Step 3: Implement data models**

`src/botfriend/models.py`:
```python
"""Immutable data models for the botfriend pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class ActionStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    speaker: str
    start_time: float
    end_time: float
    confidence: float


@dataclass(frozen=True)
class ActionItem:
    description: str
    speakers: tuple[str, ...]
    confidence: str
    status: ActionStatus = ActionStatus.PENDING

    def with_status(self, status: ActionStatus) -> ActionItem:
        return replace(self, status=status)

    def with_description(self, description: str) -> ActionItem:
        return replace(self, description=description)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_models.py -v`

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/botfriend/models.py tests/test_models.py
git commit -m "feat: add immutable data models for transcript segments and action items"
```

---

### Task 3: Action Queue State Machine

**Files:**
- Create: `src/botfriend/queue/action_queue.py`
- Create: `tests/test_action_queue.py`

- [ ] **Step 1: Write failing tests for ActionQueue**

`tests/test_action_queue.py`:
```python
import pytest
from botfriend.models import ActionItem, ActionStatus
from botfriend.queue.action_queue import ActionQueue


class TestActionQueue:
    def test_add_item(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        updated_queue = queue.add(item)
        assert len(updated_queue) == 1
        assert updated_queue.items[0].description == "Make button blue"
        assert len(queue) == 0  # original unchanged

    def test_confirm_item(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        queue = queue.add(item)
        queue = queue.confirm(0)
        assert queue.items[0].status == ActionStatus.CONFIRMED

    def test_reject_item(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        queue = queue.add(item)
        queue = queue.reject(0)
        assert queue.items[0].status == ActionStatus.REJECTED

    def test_confirm_out_of_bounds_raises(self):
        queue = ActionQueue()
        with pytest.raises(IndexError):
            queue.confirm(0)

    def test_confirm_non_pending_raises(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
            status=ActionStatus.EXECUTING,
        )
        queue = queue.add(item)
        with pytest.raises(ValueError, match="Can only confirm pending items"):
            queue.confirm(0)

    def test_next_confirmed_returns_first_confirmed(self):
        queue = ActionQueue()
        item1 = ActionItem(description="A", speakers=("S1",), confidence="high")
        item2 = ActionItem(description="B", speakers=("S1",), confidence="high")
        queue = queue.add(item1).add(item2)
        queue = queue.confirm(0)
        result = queue.next_confirmed()
        assert result is not None
        assert result[1].description == "A"

    def test_next_confirmed_returns_none_when_empty(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item)
        assert queue.next_confirmed() is None

    def test_mark_executing(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item).confirm(0)
        queue = queue.mark_executing(0)
        assert queue.items[0].status == ActionStatus.EXECUTING

    def test_mark_done(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item).confirm(0).mark_executing(0)
        queue = queue.mark_done(0)
        assert queue.items[0].status == ActionStatus.DONE

    def test_mark_failed(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item).confirm(0).mark_executing(0)
        queue = queue.mark_failed(0)
        assert queue.items[0].status == ActionStatus.FAILED

    def test_edit_item(self):
        queue = ActionQueue()
        item = ActionItem(description="Make it blue", speakers=("S1",), confidence="high")
        queue = queue.add(item)
        queue = queue.edit(0, "Make it teal")
        assert queue.items[0].description == "Make it teal"
        assert queue.items[0].status == ActionStatus.PENDING

    def test_pending_items(self):
        queue = ActionQueue()
        item1 = ActionItem(description="A", speakers=("S1",), confidence="high")
        item2 = ActionItem(description="B", speakers=("S1",), confidence="high")
        queue = queue.add(item1).add(item2).confirm(0)
        pending = queue.pending_items()
        assert len(pending) == 1
        assert pending[0][1].description == "B"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_action_queue.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ActionQueue**

`src/botfriend/queue/action_queue.py`:
```python
"""Immutable action queue with state machine transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from botfriend.models import ActionItem, ActionStatus


@dataclass(frozen=True)
class ActionQueue:
    items: tuple[ActionItem, ...] = ()

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: ActionItem) -> ActionQueue:
        return ActionQueue(items=(*self.items, item))

    def _update_at(self, index: int, item: ActionItem) -> ActionQueue:
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Index {index} out of range for queue of size {len(self.items)}")
        new_items = list(self.items)
        new_items[index] = item
        return ActionQueue(items=tuple(new_items))

    def confirm(self, index: int) -> ActionQueue:
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Index {index} out of range")
        item = self.items[index]
        if item.status != ActionStatus.PENDING:
            raise ValueError(f"Can only confirm pending items, got {item.status.value}")
        return self._update_at(index, item.with_status(ActionStatus.CONFIRMED))

    def reject(self, index: int) -> ActionQueue:
        if index < 0 or index >= len(self.items):
            raise IndexError(f"Index {index} out of range")
        item = self.items[index]
        if item.status != ActionStatus.PENDING:
            raise ValueError(f"Can only reject pending items, got {item.status.value}")
        return self._update_at(index, item.with_status(ActionStatus.REJECTED))

    def mark_executing(self, index: int) -> ActionQueue:
        item = self.items[index]
        return self._update_at(index, item.with_status(ActionStatus.EXECUTING))

    def mark_done(self, index: int) -> ActionQueue:
        item = self.items[index]
        return self._update_at(index, item.with_status(ActionStatus.DONE))

    def mark_failed(self, index: int) -> ActionQueue:
        item = self.items[index]
        return self._update_at(index, item.with_status(ActionStatus.FAILED))

    def edit(self, index: int, new_description: str) -> ActionQueue:
        item = self.items[index]
        return self._update_at(index, item.with_description(new_description))

    def next_confirmed(self) -> tuple[int, ActionItem] | None:
        for i, item in enumerate(self.items):
            if item.status == ActionStatus.CONFIRMED:
                return (i, item)
        return None

    def pending_items(self) -> list[tuple[int, ActionItem]]:
        return [
            (i, item)
            for i, item in enumerate(self.items)
            if item.status == ActionStatus.PENDING
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_action_queue.py -v`

Expected: All 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/botfriend/queue/action_queue.py tests/test_action_queue.py
git commit -m "feat: add immutable ActionQueue with state machine transitions"
```

---

### Task 4: Audio Capture Pipeline

**Files:**
- Create: `src/botfriend/pipeline/audio.py`
- Create: `tests/test_audio.py`

- [ ] **Step 1: Write failing tests for AudioCapture**

`tests/test_audio.py`:
```python
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
        assert capture.chunk_duration == 3.0
        assert capture.overlap_duration == 1.0

    def test_create_with_custom_params(self):
        capture = AudioCapture(sample_rate=44100, chunk_duration=5.0, overlap_duration=2.0)
        assert capture.sample_rate == 44100
        assert capture.chunk_duration == 5.0
        assert capture.overlap_duration == 2.0

    def test_chunk_size_calculation(self):
        capture = AudioCapture(sample_rate=16000, chunk_duration=3.0)
        assert capture.chunk_size == 48000  # 16000 * 3

    def test_overlap_size_calculation(self):
        capture = AudioCapture(sample_rate=16000, overlap_duration=1.0)
        assert capture.overlap_size == 16000  # 16000 * 1

    @pytest.mark.asyncio
    async def test_produces_chunks_from_mock_audio(self):
        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0, overlap_duration=0.0)

        # Simulate audio callback pushing data
        chunks_received = []

        async def collect_chunks():
            count = 0
            async for chunk in capture.stream():
                chunks_received.append(chunk)
                count += 1
                if count >= 2:
                    capture.stop()

        # Simulate feeding audio data
        fake_audio = np.random.randn(16000).astype(np.float32)

        with patch("sounddevice.InputStream") as mock_stream:
            mock_instance = MagicMock()
            mock_stream.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_stream.return_value.__exit__ = MagicMock(return_value=False)

            # Feed audio through the internal buffer
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_audio.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement AudioCapture**

`src/botfriend/pipeline/audio.py`:
```python
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
                pass  # drop chunk if consumer is too slow

    def stop(self) -> None:
        self._running = False

    async def stream(self) -> AsyncIterator[AudioChunk]:
        self._running = True
        loop = asyncio.get_event_loop()

        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            if status:
                pass  # silently handle status (overflows etc)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_audio.py -v`

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/botfriend/pipeline/audio.py tests/test_audio.py
git commit -m "feat: add async audio capture with chunked streaming and overlap"
```

---

### Task 5: Transcription Pipeline (Whisper + Pyannote)

**Files:**
- Create: `src/botfriend/pipeline/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write failing tests for Transcriber**

`tests/test_transcriber.py`:
```python
import pytest
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
        mock_segment.no_speech_prob = 0.9  # high no-speech prob

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
        # confidence derived from avg_logprob: exp(-0.2) ≈ 0.818
        assert 0.8 < segments[0].confidence < 0.85

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

        # No diarization pipeline -> default speaker
        transcriber = Transcriber(model_size="base")
        segments = transcriber.transcribe(self._make_chunk())

        assert segments[0].speaker == "Speaker"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_transcriber.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement Transcriber**

`src/botfriend/pipeline/transcriber.py`:
```python
"""Whisper-based transcription with optional speaker diarization."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from botfriend.models import TranscriptSegment
from botfriend.pipeline.audio import AudioChunk

NO_SPEECH_THRESHOLD = 0.6
EMPTY_TEXT_THRESHOLD = 0.5


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_transcriber.py -v`

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/botfriend/pipeline/transcriber.py tests/test_transcriber.py
git commit -m "feat: add Whisper transcription with optional pyannote diarization"
```

---

### Task 6: Claude Code CLI Runner (Executor)

**Files:**
- Create: `src/botfriend/executor/claude_runner.py`
- Create: `tests/test_claude_runner.py`

- [ ] **Step 1: Write failing tests for ClaudeRunner**

`tests/test_claude_runner.py`:
```python
import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from botfriend.executor.claude_runner import ClaudeRunner, ExecutionResult


class TestExecutionResult:
    def test_create_success(self):
        result = ExecutionResult(text="Done", session_id="abc123", success=True)
        assert result.text == "Done"
        assert result.session_id == "abc123"
        assert result.success is True

    def test_create_failure(self):
        result = ExecutionResult(text="Error occurred", session_id=None, success=False)
        assert result.success is False


class TestClaudeRunner:
    def test_create_runner(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        assert runner.project_dir == "/tmp/myproject"
        assert runner.session_id is None

    @pytest.mark.asyncio
    async def test_execute_action_success(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        mock_result = json.dumps({
            "type": "result",
            "result": "Changed button color to blue",
            "session_id": "sess_abc123",
        })

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_result.encode(), b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await runner.execute("Change CTA button color to blue")

            assert result.success is True
            assert result.text == "Changed button color to blue"
            assert result.session_id == "sess_abc123"
            assert runner.session_id == "sess_abc123"

    @pytest.mark.asyncio
    async def test_execute_preserves_session(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        runner = runner.with_session_id("existing_session")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                json.dumps({"result": "ok", "session_id": "existing_session"}).encode(),
                b"",
            )
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            await runner.execute("Do something")

            call_args = mock_exec.call_args[0]
            assert "--resume" in call_args
            assert "existing_session" in call_args

    @pytest.mark.asyncio
    async def test_execute_action_failure(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"Something went wrong")
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            result = await runner.execute("Do something impossible")

            assert result.success is False
            assert "Something went wrong" in result.text

    @pytest.mark.asyncio
    async def test_extract_actions_returns_json(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        mock_result = json.dumps({
            "type": "result",
            "result": json.dumps([
                {"description": "Make button blue", "speakers": ["Speaker 1"], "confidence": "high"}
            ]),
            "session_id": "sess_extract",
        })

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_result.encode(), b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            actions = await runner.extract_actions("transcript text here", [])

            assert len(actions) == 1
            assert actions[0]["description"] == "Make button blue"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_claude_runner.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ClaudeRunner**

`src/botfriend/executor/claude_runner.py`:
```python
"""Claude Code CLI wrapper for action extraction and code execution."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any


EXTRACTION_PROMPT_TEMPLATE = """You are analyzing a live conversation between people discussing a software product. Extract any actionable code changes mentioned.

Here is the recent conversation transcript:

{transcript}

Previously extracted action items (do NOT re-extract these):
{previous_items}

Return ONLY a JSON array of NEW action items. Each item must have:
- "description": a clear, specific description of what code change to make
- "speakers": list of speaker labels who contributed to this idea
- "confidence": "high", "medium", or "low"

If there are no new actionable items, return an empty array: []
Return ONLY the JSON array, no other text."""


@dataclass(frozen=True)
class ExecutionResult:
    text: str
    session_id: str | None
    success: bool


class ClaudeRunner:
    def __init__(self, project_dir: str) -> None:
        self.project_dir = project_dir
        self.session_id: str | None = None

    def with_session_id(self, session_id: str) -> ClaudeRunner:
        runner = ClaudeRunner(project_dir=self.project_dir)
        runner.session_id = session_id
        return runner

    async def execute(self, action_description: str) -> ExecutionResult:
        cmd = [
            "claude",
            "-p", action_description,
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]
        if self.session_id:
            cmd.extend(["--resume", self.session_id])

        env = {**os.environ}
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_dir,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip() or stdout.decode().strip() or "Unknown error"
            return ExecutionResult(
                text=f"Error (exit {proc.returncode}): {err}",
                session_id=self.session_id,
                success=False,
            )

        raw = stdout.decode()
        try:
            data = json.loads(raw)
            text = data.get("result", raw)
            new_session_id = data.get("session_id", self.session_id)
            self.session_id = new_session_id
            return ExecutionResult(text=text, session_id=new_session_id, success=True)
        except json.JSONDecodeError:
            return ExecutionResult(text=raw, session_id=self.session_id, success=True)

    async def extract_actions(
        self,
        transcript: str,
        previous_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        previous_str = json.dumps(previous_items, indent=2) if previous_items else "None"
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            transcript=transcript,
            previous_items=previous_str,
        )

        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
        ]

        env = {**os.environ}
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return []

        raw = stdout.decode()
        try:
            data = json.loads(raw)
            result_text = data.get("result", raw)
            actions = json.loads(result_text)
            if isinstance(actions, list):
                return actions
            return []
        except (json.JSONDecodeError, TypeError):
            return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_claude_runner.py -v`

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/botfriend/executor/claude_runner.py tests/test_claude_runner.py
git commit -m "feat: add Claude Code CLI runner for extraction and execution"
```

---

### Task 7: Action Extractor Pipeline Stage

**Files:**
- Create: `src/botfriend/pipeline/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write failing tests for ActionExtractor**

`tests/test_extractor.py`:
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from botfriend.pipeline.extractor import ActionExtractor
from botfriend.models import TranscriptSegment, ActionItem


class TestActionExtractor:
    def _make_segment(self, text: str, speaker: str = "Speaker 1") -> TranscriptSegment:
        return TranscriptSegment(
            text=text,
            speaker=speaker,
            start_time=0.0,
            end_time=2.0,
            confidence=0.9,
        )

    def test_create_extractor(self):
        mock_runner = MagicMock()
        extractor = ActionExtractor(runner=mock_runner, interval_seconds=10.0)
        assert extractor.interval_seconds == 10.0

    def test_add_segment_to_buffer(self):
        mock_runner = MagicMock()
        extractor = ActionExtractor(runner=mock_runner)
        seg = self._make_segment("make the button blue")
        extractor.add_segment(seg)
        assert len(extractor.transcript_buffer) == 1

    def test_format_transcript(self):
        mock_runner = MagicMock()
        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("make the button blue", "Speaker 1"))
        extractor.add_segment(self._make_segment("yeah and bigger", "Speaker 2"))
        formatted = extractor.format_transcript()
        assert "[Speaker 1]" in formatted
        assert "make the button blue" in formatted
        assert "[Speaker 2]" in formatted

    @pytest.mark.asyncio
    async def test_extract_returns_action_items(self):
        mock_runner = AsyncMock()
        mock_runner.extract_actions.return_value = [
            {"description": "Make button blue", "speakers": ["Speaker 1"], "confidence": "high"},
        ]

        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("make the button blue"))

        items = await extractor.extract()
        assert len(items) == 1
        assert items[0].description == "Make button blue"
        assert items[0].speakers == ("Speaker 1",)
        assert items[0].confidence == "high"

    @pytest.mark.asyncio
    async def test_extract_filters_low_confidence(self):
        mock_runner = AsyncMock()
        mock_runner.extract_actions.return_value = [
            {"description": "Maybe change color", "speakers": ["Speaker 1"], "confidence": "low"},
        ]

        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("maybe change the color?"))

        items = await extractor.extract()
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_extract_tracks_previous_items(self):
        mock_runner = AsyncMock()
        mock_runner.extract_actions.return_value = [
            {"description": "A", "speakers": ["S1"], "confidence": "high"},
        ]

        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("do A"))
        await extractor.extract()

        # Second extraction should pass previous items
        mock_runner.extract_actions.return_value = []
        extractor.add_segment(self._make_segment("do B"))
        await extractor.extract()

        call_args = mock_runner.extract_actions.call_args_list[1]
        previous = call_args[0][1]  # second positional arg
        assert len(previous) == 1
        assert previous[0]["description"] == "A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_extractor.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement ActionExtractor**

`src/botfriend/pipeline/extractor.py`:
```python
"""Extracts actionable items from accumulated transcript using Claude CLI."""

from __future__ import annotations

from typing import Any

from botfriend.models import ActionItem, TranscriptSegment
from botfriend.executor.claude_runner import ClaudeRunner

ALLOWED_CONFIDENCES = {"high", "medium"}


class ActionExtractor:
    def __init__(
        self,
        runner: ClaudeRunner,
        interval_seconds: float = 10.0,
    ) -> None:
        self.runner = runner
        self.interval_seconds = interval_seconds
        self.transcript_buffer: list[TranscriptSegment] = []
        self._previous_items: list[dict[str, Any]] = []

    def add_segment(self, segment: TranscriptSegment) -> None:
        self.transcript_buffer.append(segment)

    def format_transcript(self) -> str:
        lines = []
        for seg in self.transcript_buffer:
            lines.append(f"[{seg.speaker}] {seg.text}")
        return "\n".join(lines)

    async def extract(self) -> list[ActionItem]:
        if not self.transcript_buffer:
            return []

        transcript_text = self.format_transcript()
        raw_actions = await self.runner.extract_actions(
            transcript_text,
            self._previous_items,
        )

        items: list[ActionItem] = []
        for raw in raw_actions:
            confidence = raw.get("confidence", "low")
            if confidence not in ALLOWED_CONFIDENCES:
                continue
            speakers = raw.get("speakers", [])
            item = ActionItem(
                description=raw["description"],
                speakers=tuple(speakers),
                confidence=confidence,
            )
            items.append(item)
            self._previous_items.append({
                "description": raw["description"],
                "speakers": speakers,
                "confidence": confidence,
            })

        return items
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_extractor.py -v`

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/botfriend/pipeline/extractor.py tests/test_extractor.py
git commit -m "feat: add action extractor with deduplication and confidence filtering"
```

---

### Task 8: TUI Widgets

**Files:**
- Create: `src/botfriend/widgets/transcript_panel.py`
- Create: `src/botfriend/widgets/queue_panel.py`
- Create: `src/botfriend/widgets/execution_log.py`

- [ ] **Step 1: Implement TranscriptPanel**

`src/botfriend/widgets/transcript_panel.py`:
```python
"""Scrollable live transcript panel."""

from textual.widgets import RichLog
from botfriend.models import TranscriptSegment


class TranscriptPanel(RichLog):
    DEFAULT_CSS = """
    TranscriptPanel {
        border: solid $primary;
        border-title-color: $text;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__(highlight=True, markup=True, auto_scroll=True, wrap=True)
        self.border_title = "Live Transcript"

    def add_segment(self, segment: TranscriptSegment) -> None:
        color = "cyan" if "1" in segment.speaker else "green"
        self.write(f"[bold {color}][{segment.speaker}][/] {segment.text}")
```

- [ ] **Step 2: Implement QueuePanel**

`src/botfriend/widgets/queue_panel.py`:
```python
"""Action item queue panel with keyboard navigation."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from botfriend.models import ActionItem, ActionStatus
from botfriend.queue.action_queue import ActionQueue

STATUS_ICONS = {
    ActionStatus.PENDING: "",
    ActionStatus.CONFIRMED: "",
    ActionStatus.EXECUTING: "",
    ActionStatus.DONE: "",
    ActionStatus.FAILED: "",
    ActionStatus.REJECTED: "",
}


class QueueItemWidget(Static):
    DEFAULT_CSS = """
    QueueItemWidget {
        padding: 0 1;
        height: auto;
    }
    QueueItemWidget.selected {
        background: $accent;
    }
    """

    def __init__(self, item: ActionItem, index: int, selected: bool = False) -> None:
        icon = STATUS_ICONS.get(item.status, "")
        speakers_str = ", ".join(item.speakers) if item.speakers else ""
        text = f"{icon} {item.description}\n  ({speakers_str}) [{item.confidence}]"
        super().__init__(text, classes="selected" if selected else "")
        self.item_index = index


class QueuePanel(Widget):
    DEFAULT_CSS = """
    QueuePanel {
        border: solid $secondary;
        border-title-color: $text;
        height: 1fr;
        overflow-y: auto;
    }
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "confirm_item", "Confirm"),
        Binding("x", "reject_item", "Reject"),
        Binding("e", "edit_item", "Edit"),
    ]

    selected_index: reactive[int] = reactive(0)

    class ItemConfirmed(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    class ItemRejected(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    class ItemEditRequested(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self.border_title = "Action Queue"
        self._queue = ActionQueue()

    def compose(self) -> ComposeResult:
        yield VerticalScroll()

    def update_queue(self, queue: ActionQueue) -> None:
        self._queue = queue
        self._render_items()

    def _render_items(self) -> None:
        container = self.query_one(VerticalScroll)
        container.remove_children()
        for i, item in enumerate(self._queue.items):
            widget = QueueItemWidget(item, i, selected=(i == self.selected_index))
            container.mount(widget)

    def watch_selected_index(self, value: int) -> None:
        self._render_items()

    def action_move_up(self) -> None:
        if self.selected_index > 0:
            self.selected_index -= 1

    def action_move_down(self) -> None:
        if self.selected_index < len(self._queue) - 1:
            self.selected_index += 1

    def action_confirm_item(self) -> None:
        if self.selected_index < len(self._queue):
            item = self._queue.items[self.selected_index]
            if item.status == ActionStatus.PENDING:
                self.post_message(self.ItemConfirmed(self.selected_index))

    def action_reject_item(self) -> None:
        if self.selected_index < len(self._queue):
            item = self._queue.items[self.selected_index]
            if item.status == ActionStatus.PENDING:
                self.post_message(self.ItemRejected(self.selected_index))

    def action_edit_item(self) -> None:
        if self.selected_index < len(self._queue):
            item = self._queue.items[self.selected_index]
            if item.status == ActionStatus.PENDING:
                self.post_message(self.ItemEditRequested(self.selected_index))
```

- [ ] **Step 3: Implement ExecutionLog**

`src/botfriend/widgets/execution_log.py`:
```python
"""Scrollable execution log panel."""

from textual.widgets import RichLog


class ExecutionLog(RichLog):
    DEFAULT_CSS = """
    ExecutionLog {
        border: solid $warning;
        border-title-color: $text;
        height: auto;
        min-height: 5;
        max-height: 12;
    }
    """

    def __init__(self) -> None:
        super().__init__(highlight=True, markup=True, auto_scroll=True, wrap=True)
        self.border_title = "Execution Log"

    def log_started(self, description: str) -> None:
        self.write(f"[bold yellow]Running:[/] {description}")

    def log_done(self, description: str) -> None:
        self.write(f"[bold green]Done:[/] {description}")

    def log_failed(self, description: str, error: str) -> None:
        self.write(f"[bold red]Failed:[/] {description}")
        self.write(f"  [red]{error}[/]")
```

- [ ] **Step 4: Commit**

```bash
git add src/botfriend/widgets/
git commit -m "feat: add TUI widgets for transcript, queue, and execution log"
```

---

### Task 9: Main Textual App

**Files:**
- Create: `src/botfriend/app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write basic app test**

`tests/test_app.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from textual.testing import AppTest
from botfriend.app import BotfriendApp


class TestBotfriendApp:
    @pytest.mark.asyncio
    async def test_app_starts_and_shows_panels(self):
        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            # Verify all three panels exist
            assert app.query_one("TranscriptPanel") is not None
            assert app.query_one("QueuePanel") is not None
            assert app.query_one("ExecutionLog") is not None

    @pytest.mark.asyncio
    async def test_app_quit_keybinding(self):
        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("q")
            # App should initiate quit
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_app.py -v`

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement BotfriendApp**

`src/botfriend/app.py`:
```python
"""Main Textual TUI application for Botfriend."""

from __future__ import annotations

import asyncio
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical

from botfriend.executor.claude_runner import ClaudeRunner
from botfriend.models import ActionItem, TranscriptSegment
from botfriend.pipeline.audio import AudioCapture
from botfriend.pipeline.extractor import ActionExtractor
from botfriend.pipeline.transcriber import Transcriber
from botfriend.queue.action_queue import ActionQueue
from botfriend.widgets.execution_log import ExecutionLog
from botfriend.widgets.queue_panel import QueuePanel
from botfriend.widgets.transcript_panel import TranscriptPanel


class BotfriendApp(App):
    CSS = """
    #top-row {
        height: 1fr;
    }
    #transcript {
        width: 2fr;
    }
    #queue {
        width: 1fr;
    }
    #execution {
        height: auto;
        min-height: 5;
        max-height: 12;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause/Resume"),
    ]

    def __init__(
        self,
        project_dir: str,
        model_size: str = "base",
        hf_token: Optional[str] = None,
        enable_audio: bool = True,
    ) -> None:
        super().__init__()
        self.project_dir = project_dir
        self.model_size = model_size
        self.hf_token = hf_token
        self.enable_audio = enable_audio
        self._queue = ActionQueue()
        self._paused = False
        self._executing = False

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="top-row"):
                yield TranscriptPanel().data_bind()  # noqa
                yield QueuePanel()
            yield ExecutionLog()

    def on_mount(self) -> None:
        # Fix widget IDs for CSS targeting
        self.query_one(TranscriptPanel).id = "transcript"
        self.query_one(QueuePanel).id = "queue"
        self.query_one(ExecutionLog).id = "execution"

        # Focus the queue panel for keyboard navigation
        self.query_one(QueuePanel).focus()

        if self.enable_audio:
            self._start_pipeline()

    def _start_pipeline(self) -> None:
        self._audio = AudioCapture()
        self._transcriber = Transcriber(model_size=self.model_size)
        self._coding_runner = ClaudeRunner(project_dir=self.project_dir)
        self._extraction_runner = ClaudeRunner(project_dir=self.project_dir)
        self._extractor = ActionExtractor(runner=self._extraction_runner)

        if self.hf_token:
            self._transcriber.enable_diarization(self.hf_token)

        self.run_worker(self._audio_loop(), exclusive=False)
        self.run_worker(self._extraction_loop(), exclusive=False)

    async def _audio_loop(self) -> None:
        transcript_panel = self.query_one(TranscriptPanel)
        async for chunk in self._audio.stream():
            if self._paused:
                continue
            segments = self._transcriber.transcribe(chunk)
            for seg in segments:
                self.call_from_thread(transcript_panel.add_segment, seg)
                self._extractor.add_segment(seg)

    async def _extraction_loop(self) -> None:
        while True:
            await asyncio.sleep(self._extractor.interval_seconds)
            if self._paused:
                continue
            items = await self._extractor.extract()
            for item in items:
                self._queue = self._queue.add(item)
                queue_panel = self.query_one(QueuePanel)
                self.call_from_thread(queue_panel.update_queue, self._queue)

    async def _execute_next(self) -> None:
        if self._executing:
            return
        result = self._queue.next_confirmed()
        if result is None:
            return

        index, item = result
        self._executing = True
        self._queue = self._queue.mark_executing(index)
        self.query_one(QueuePanel).update_queue(self._queue)

        exec_log = self.query_one(ExecutionLog)
        exec_log.log_started(item.description)

        execution_result = await self._coding_runner.execute(item.description)

        if execution_result.success:
            self._queue = self._queue.mark_done(index)
            exec_log.log_done(item.description)
        else:
            self._queue = self._queue.mark_failed(index)
            exec_log.log_failed(item.description, execution_result.text)

        self.query_one(QueuePanel).update_queue(self._queue)
        self._executing = False

        # Check if there's another confirmed item
        await self._execute_next()

    def on_queue_panel_item_confirmed(self, event: QueuePanel.ItemConfirmed) -> None:
        self._queue = self._queue.confirm(event.index)
        self.query_one(QueuePanel).update_queue(self._queue)
        self.run_worker(self._execute_next(), exclusive=False)

    def on_queue_panel_item_rejected(self, event: QueuePanel.ItemRejected) -> None:
        self._queue = self._queue.reject(event.index)
        self.query_one(QueuePanel).update_queue(self._queue)

    def on_queue_panel_item_edit_requested(self, event: QueuePanel.ItemEditRequested) -> None:
        item = self._queue.items[event.index]
        self.push_screen(
            EditScreen(item.description, event.index),
        )

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        status = "PAUSED" if self._paused else "LISTENING"
        self.notify(f"Mic: {status}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/test_app.py -v`

Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/botfriend/app.py tests/test_app.py
git commit -m "feat: add main Textual TUI app with pipeline orchestration"
```

---

### Task 10: Edit Screen for Action Items

**Files:**
- Modify: `src/botfriend/app.py` (add import and handler)
- Create: `src/botfriend/widgets/edit_screen.py`

- [ ] **Step 1: Implement EditScreen**

`src/botfriend/widgets/edit_screen.py`:
```python
"""Inline edit screen for modifying action item descriptions."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Static


class EditScreen(ModalScreen):
    DEFAULT_CSS = """
    EditScreen {
        align: center middle;
    }
    #edit-dialog {
        width: 60;
        height: auto;
        border: thick $primary;
        padding: 1 2;
        background: $surface;
    }
    #edit-label {
        margin-bottom: 1;
    }
    #edit-input {
        margin-bottom: 1;
    }
    #edit-hint {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    class EditCompleted(Message):
        def __init__(self, index: int, new_description: str) -> None:
            self.index = index
            self.new_description = new_description
            super().__init__()

    def __init__(self, description: str, index: int) -> None:
        super().__init__()
        self._description = description
        self._index = index

    def compose(self) -> ComposeResult:
        with Static(id="edit-dialog"):
            yield Static("Edit action item:", id="edit-label")
            yield Input(value=self._description, id="edit-input")
            yield Static("Enter to save, Escape to cancel", id="edit-hint")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(None)
        self.app.post_message(
            self.EditCompleted(self._index, event.value)
        )

    def action_cancel(self) -> None:
        self.dismiss(None)
```

- [ ] **Step 2: Add edit handler to BotfriendApp**

Add to `src/botfriend/app.py` at the top imports:
```python
from botfriend.widgets.edit_screen import EditScreen
```

Add method to the `BotfriendApp` class:
```python
    def on_edit_screen_edit_completed(self, event: EditScreen.EditCompleted) -> None:
        self._queue = self._queue.edit(event.index, event.new_description)
        self.query_one(QueuePanel).update_queue(self._queue)
```

- [ ] **Step 3: Commit**

```bash
git add src/botfriend/widgets/edit_screen.py src/botfriend/app.py
git commit -m "feat: add inline edit screen for action item descriptions"
```

---

### Task 11: CLI Entry Point

**Files:**
- Create: `src/botfriend/main.py`

- [ ] **Step 1: Implement CLI entry point**

`src/botfriend/main.py`:
```python
"""CLI entry point for botfriend."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Botfriend: Real-time voice-to-code pipeline",
    )
    parser.add_argument(
        "project_dir",
        help="Path to the project directory for code execution",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )

    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    if not os.path.isdir(project_dir):
        print(f"Error: Not a directory: {project_dir}", file=sys.stderr)
        sys.exit(1)

    hf_token = None
    if not args.no_diarization:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print(
                "Speaker diarization requires a HuggingFace token.\n"
                "Set HF_TOKEN env var or run with --no-diarization.\n"
                "Get a free token at https://huggingface.co/settings/tokens",
                file=sys.stderr,
            )
            sys.exit(1)

    from botfriend.app import BotfriendApp

    app = BotfriendApp(
        project_dir=project_dir,
        model_size=args.model,
        hf_token=hf_token,
    )
    app.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the CLI entrypoint works**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m botfriend.main --help`

Expected:
```
usage: main.py [-h] [--model {tiny,base,small,medium,large-v3}] [--no-diarization] project_dir

Botfriend: Real-time voice-to-code pipeline

positional arguments:
  project_dir           Path to the project directory for code execution

options:
  -h, --help            show this help message and exit
  --model {tiny,base,small,medium,large-v3}
                        Whisper model size (default: base)
  --no-diarization      Disable speaker diarization
```

- [ ] **Step 3: Commit**

```bash
git add src/botfriend/main.py
git commit -m "feat: add CLI entry point with argparse"
```

---

### Task 12: Integration Test & Final Wiring

**Files:**
- Modify: `tests/test_app.py` (add integration test)

- [ ] **Step 1: Write integration test with mocked pipeline**

Add to `tests/test_app.py`:
```python
class TestBotfriendIntegration:
    @pytest.mark.asyncio
    async def test_full_flow_with_mock_data(self):
        """Test the full flow: add item to queue, confirm, verify state."""
        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            from botfriend.models import ActionItem

            # Simulate action items arriving
            item = ActionItem(
                description="Change button color to blue",
                speakers=("Speaker 1",),
                confidence="high",
            )
            app._queue = app._queue.add(item)
            app.query_one("QueuePanel").update_queue(app._queue)

            # Focus queue and confirm
            app.query_one("QueuePanel").focus()
            await pilot.press("enter")

            # Verify item was confirmed
            assert app._queue.items[0].status.value == "confirmed"

    @pytest.mark.asyncio
    async def test_reject_item_flow(self):
        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            from botfriend.models import ActionItem

            item = ActionItem(
                description="Remove header",
                speakers=("Speaker 2",),
                confidence="medium",
            )
            app._queue = app._queue.add(item)
            app.query_one("QueuePanel").update_queue(app._queue)

            app.query_one("QueuePanel").focus()
            await pilot.press("x")

            assert app._queue.items[0].status.value == "rejected"

    @pytest.mark.asyncio
    async def test_pause_toggle(self):
        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app._paused is False
            await pilot.press("p")
            assert app._paused is True
            await pilot.press("p")
            assert app._paused is False
```

- [ ] **Step 2: Run all tests**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/ -v --tb=short`

Expected: All tests pass.

- [ ] **Step 3: Check test coverage**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/ --cov=botfriend --cov-report=term-missing`

Expected: Coverage >= 80% on models, action_queue, audio, transcriber, extractor, claude_runner.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add integration tests for full TUI flow"
```

---

### Task 13: README & Final Polish

**Files:**
- Create: `README.md`
- Create: `.gitignore`

- [ ] **Step 1: Create .gitignore**

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
```

- [ ] **Step 2: Create README**

`README.md`:
```markdown
# Botfriend

Real-time voice-to-code pipeline. Listens to live conversations, extracts actionable code changes, and executes them via Claude Code CLI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Speaker diarization requires a free HuggingFace token:
1. Create account at https://huggingface.co
2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create token at https://huggingface.co/settings/tokens
4. Set: `export HF_TOKEN=hf_your_token_here`

## Usage

```bash
botfriend /path/to/your/project
```

Options:
- `--model {tiny,base,small,medium,large-v3}` — Whisper model size (default: base)
- `--no-diarization` — Disable speaker identification

## Keyboard Controls

| Key | Action |
|-----|--------|
| Up/Down | Navigate action queue |
| Enter | Confirm selected action |
| x | Reject/dismiss action |
| e | Edit action before confirming |
| p | Pause/resume listening |
| q | Quit |
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore README.md
git commit -m "docs: add README and gitignore"
```

- [ ] **Step 4: Run full test suite one final time**

Run: `cd /Users/vanessawang/botfriend && source .venv/bin/activate && python -m pytest tests/ -v --cov=botfriend`

Expected: All tests pass, coverage >= 80%.
