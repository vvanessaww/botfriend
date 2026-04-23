"""Scrollable live transcript panel with processing indicator."""

from __future__ import annotations

from textual.widgets import RichLog
from botfriend.models import TranscriptSegment

_DOT_FRAMES = [".  ", ".. ", "...", " ..", "  ."]


class TranscriptPanel(RichLog):
    DEFAULT_CSS = """
    TranscriptPanel {
        border: solid $primary;
        border-title-color: $text;
        border-subtitle-color: $text-muted;
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__(highlight=True, markup=True, auto_scroll=True, wrap=True)
        self.border_title = "Live Transcript"
        self._dot_frame: int = 0
        self._dot_timer = None
        self._processing = False

    def add_segment(self, segment: TranscriptSegment) -> None:
        self.hide_processing()
        color = "cyan" if "1" in segment.speaker else "green"
        self.write(f"[bold {color}][{segment.speaker}][/] {segment.text}")

    def show_processing(self) -> None:
        if self._processing:
            return
        self._processing = True
        self._dot_frame = 0
        self._update_subtitle()
        self._dot_timer = self.set_interval(0.3, self._tick_dots)

    def hide_processing(self) -> None:
        if not self._processing:
            return
        self._processing = False
        if self._dot_timer is not None:
            self._dot_timer.stop()
            self._dot_timer = None
        self.border_subtitle = ""

    def _tick_dots(self) -> None:
        if not self._processing:
            return
        self._dot_frame = (self._dot_frame + 1) % len(_DOT_FRAMES)
        self._update_subtitle()

    def _update_subtitle(self) -> None:
        dots = _DOT_FRAMES[self._dot_frame]
        self.border_subtitle = f"transcribing {dots}"
