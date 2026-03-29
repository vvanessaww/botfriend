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
from botfriend.widgets.edit_screen import EditScreen
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
                yield TranscriptPanel()
                yield QueuePanel()
            yield ExecutionLog()

    def on_mount(self) -> None:
        self.query_one(TranscriptPanel).id = "transcript"
        self.query_one(QueuePanel).id = "queue"
        self.query_one(ExecutionLog).id = "execution"
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

    def on_edit_screen_edit_completed(self, event) -> None:
        self._queue = self._queue.edit(event.index, event.new_description)
        self.query_one(QueuePanel).update_queue(self._queue)

    def action_toggle_pause(self) -> None:
        self._paused = not self._paused
        status = "PAUSED" if self._paused else "LISTENING"
        self.notify(f"Mic: {status}")
