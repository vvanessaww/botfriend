"""Immutable action queue with state machine transitions."""

from __future__ import annotations

from dataclasses import dataclass
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
