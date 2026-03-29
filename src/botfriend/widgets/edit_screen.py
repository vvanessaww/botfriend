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
