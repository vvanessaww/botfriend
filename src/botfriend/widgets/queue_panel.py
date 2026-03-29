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

    can_focus = True
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
