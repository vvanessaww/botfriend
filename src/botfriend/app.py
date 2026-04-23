"""Main Textual TUI application for Botfriend."""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
from botfriend.widgets.waveform import Waveform


class BotfriendApp(App):
    CSS = """
    #left-col {
        width: 2fr;
    }
    #right-col {
        width: 1fr;
    }
    #waveform {
        height: 3;
    }
    #transcript {
        height: 1fr;
    }
    #queue {
        height: 1fr;
    }
    #execution {
        height: auto;
        min-height: 8;
        max-height: 20;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("p", "toggle_pause", "Pause/Resume"),
    ]

    def __init__(
        self,
        project_dir: str,
        transcriber: Optional[Transcriber] = None,
        hf_token: Optional[str] = None,
        enable_audio: bool = True,
    ) -> None:
        super().__init__()
        self.project_dir = project_dir
        self._transcriber = transcriber
        self.hf_token = hf_token
        self.enable_audio = enable_audio
        self._queue = ActionQueue()
        self._paused = False
        self._executing = False
        self._listening = False

    def compose(self) -> ComposeResult:
        log_path = Path(self.project_dir) / "botfriend.log"
        with Horizontal():
            with Vertical(id="left-col"):
                yield Waveform()
                yield TranscriptPanel()
            with Vertical(id="right-col"):
                yield QueuePanel()
                yield ExecutionLog(log_path=log_path)

    def on_mount(self) -> None:
        self.query_one(Waveform).id = "waveform"
        self.query_one(TranscriptPanel).id = "transcript"
        self.query_one(QueuePanel).id = "queue"
        self.query_one(ExecutionLog).id = "execution"
        self.query_one(QueuePanel).focus()

        exec_log = self.query_one(ExecutionLog)

        if self.enable_audio:
            self._start_pipeline()
            exec_log.write("[bold green]Mic LISTENING[/] -- press [bold]p[/] to pause")
            self._waveform_timer = self.set_interval(1 / 15, self._refresh_waveform)
        else:
            exec_log.write("[dim]Audio disabled (test mode)[/]")

    def _refresh_waveform(self) -> None:
        try:
            self.query_one(Waveform).refresh()
        except Exception:
            pass

    def _start_pipeline(self) -> None:
        waveform = self.query_one(Waveform)
        self._audio = AudioCapture(on_level=waveform.push_level)
        if self._transcriber is None:
            self._transcriber = Transcriber()
        self._coding_runner = ClaudeRunner(project_dir=self.project_dir)
        self._extraction_runner = ClaudeRunner(project_dir=self.project_dir)
        self._extractor = ActionExtractor(runner=self._extraction_runner)
        self._listening = True

        self.run_worker(self._audio_loop(), exclusive=False)
        self.run_worker(self._extraction_loop(), exclusive=False)

    async def _audio_loop(self) -> None:
        transcript_panel = self.query_one(TranscriptPanel)
        exec_log = self.query_one(ExecutionLog)
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        async for chunk in self._audio.stream():
            if self._paused:
                continue
            try:
                transcript_panel.show_processing()
                segments = await loop.run_in_executor(
                    executor,
                    functools.partial(self._transcriber.transcribe, chunk),
                )
                transcript_panel.hide_processing()
                for seg in segments:
                    transcript_panel.add_segment(seg)
                    self._extractor.add_segment(seg)
            except Exception as e:
                transcript_panel.hide_processing()
                exec_log.write(f"[bold red]Transcription error:[/] {e}")

    async def _extraction_loop(self) -> None:
        exec_log = self.query_one(ExecutionLog)
        while True:
            await asyncio.sleep(self._extractor.interval_seconds)
            if self._paused:
                continue
            if not self._extractor.transcript_buffer:
                continue
            try:
                exec_log.write("[dim]Extracting action items...[/]")
                items = await self._extractor.extract()
                if items:
                    exec_log.write(f"[bold cyan]Found {len(items)} action item(s)[/]")
                    for item in items:
                        self._queue = self._queue.add(item)
                    self.query_one(QueuePanel).update_queue(self._queue)
                else:
                    exec_log.write("[dim]No new action items found[/]")
            except Exception as e:
                exec_log.write(f"[bold red]Extraction error:[/] {e}")

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
            summary = execution_result.text
            if len(summary) > 300:
                summary = summary[:300] + "..."
            exec_log.write(f"  [dim]{summary}[/]")
            try:
                diff = await self._coding_runner.get_diff()
                if diff:
                    exec_log.write(f"  [bold magenta]Changes:[/]")
                    for line in diff.split("\n")[:10]:
                        exec_log.write(f"    {line}")
            except Exception:
                pass
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
        exec_log = self.query_one(ExecutionLog)
        if self._paused:
            if hasattr(self, "_audio"):
                self._audio.stop()
            self._listening = False
            exec_log.write("[bold yellow]Mic PAUSED[/] -- press [bold]p[/] to unpause")
        else:
            self._listening = True
            if self.enable_audio:
                self.run_worker(self._restart_audio(), exclusive=False)
            exec_log.write("[bold green]Mic LISTENING[/] -- press [bold]p[/] to pause")

    async def _restart_audio(self) -> None:
        """Restart the audio stream after unpausing."""
        waveform = self.query_one(Waveform)
        self._audio = AudioCapture(on_level=waveform.push_level)
        transcript_panel = self.query_one(TranscriptPanel)
        exec_log = self.query_one(ExecutionLog)
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        async for chunk in self._audio.stream():
            if self._paused:
                break
            try:
                transcript_panel.show_processing()
                segments = await loop.run_in_executor(
                    executor,
                    functools.partial(self._transcriber.transcribe, chunk),
                )
                transcript_panel.hide_processing()
                for seg in segments:
                    transcript_panel.add_segment(seg)
                    self._extractor.add_segment(seg)
            except Exception as e:
                transcript_panel.hide_processing()
                exec_log.write(f"[bold red]Transcription error:[/] {e}")
