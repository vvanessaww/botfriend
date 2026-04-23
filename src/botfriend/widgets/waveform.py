"""Real-time audio waveform visualizer."""

from __future__ import annotations

from collections import deque

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


# Block characters for rendering amplitude levels (low to high)
_BARS = " ░▒▓█"


class Waveform(Widget):
    """Displays a rolling waveform of recent audio levels.

    Updated externally by calling `push_level(rms_value)` from the
    audio callback.  The widget re-renders on every timer tick so
    visual updates are decoupled from transcription speed.
    """

    DEFAULT_CSS = """
    Waveform {
        height: 3;
        border: solid $accent;
        border-title-color: $text;
        padding: 0 1;
    }
    Waveform .waveform-bar {
        height: 1;
    }
    """

    # How many columns of history to show
    WIDTH = 80

    def __init__(self) -> None:
        super().__init__()
        self.border_title = "Audio"
        self._levels: deque[float] = deque([0.0] * self.WIDTH, maxlen=self.WIDTH)
        self._peak: float = 0.0
        self._clipping = False

    def push_level(self, rms: float) -> None:
        """Push a new RMS level (0.0 - 1.0) into the rolling buffer."""
        self._levels.append(min(rms, 1.0))
        self._peak = max(self._peak * 0.95, rms)  # fast attack, slow decay

    def render(self) -> str:
        levels = list(self._levels)
        if not levels:
            return ""

        # Scale levels to bar characters
        bar_chars = []
        for level in levels:
            idx = int(level * (len(_BARS) - 1))
            idx = min(idx, len(_BARS) - 1)
            bar_chars.append(_BARS[idx])

        waveform_line = "".join(bar_chars)

        # Status line with peak meter
        peak_bar_width = 20
        filled = int(self._peak * peak_bar_width)
        filled = min(filled, peak_bar_width)
        peak_bar = "█" * filled + "░" * (peak_bar_width - filled)

        if self._peak > 0.02:
            status = f"Peak [{peak_bar}] ●  LISTENING"
        else:
            status = f"Peak [{peak_bar}] ○  quiet"

        return f"{waveform_line}\n{status}"
