"""Scrollable execution log panel with file logging."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.widgets import RichLog

logger = logging.getLogger(__name__)


def _strip_markup(text: str) -> str:
    """Remove Rich markup tags for plain text logging."""
    return re.sub(r"\[/?[^\]]*\]", "", text)


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

    def __init__(self, log_path: Optional[Path] = None) -> None:
        super().__init__(highlight=True, markup=True, auto_scroll=True, wrap=True)
        self.border_title = "Execution Log"
        self._log_path = log_path

    def _write_to_file(self, text: str) -> None:
        if self._log_path is None:
            return
        try:
            plain = _strip_markup(text)
            timestamp = datetime.now().strftime("%H:%M:%S")
            with open(self._log_path, "a") as f:
                f.write(f"[{timestamp}] {plain}\n")
        except OSError as e:
            logger.warning("Failed to write to log file %s: %s", self._log_path, e)

    def write(self, *args, **kwargs) -> None:
        if args:
            self._write_to_file(str(args[0]))
        return super().write(*args, **kwargs)

    def log_started(self, description: str) -> None:
        self.write(f"[bold yellow]Running:[/] {description}")

    def log_done(self, description: str) -> None:
        self.write(f"[bold green]Done:[/] {description}")

    def log_failed(self, description: str, error: str) -> None:
        self.write(f"[bold red]Failed:[/] {description}")
        self.write(f"  [red]{error}[/]")
