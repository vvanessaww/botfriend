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
