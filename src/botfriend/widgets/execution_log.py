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
