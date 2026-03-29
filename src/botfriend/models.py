"""Immutable data models for the botfriend pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class ActionStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    speaker: str
    start_time: float
    end_time: float
    confidence: float


@dataclass(frozen=True)
class ActionItem:
    description: str
    speakers: tuple[str, ...]
    confidence: str
    status: ActionStatus = ActionStatus.PENDING

    def with_status(self, status: ActionStatus) -> ActionItem:
        return replace(self, status=status)

    def with_description(self, description: str) -> ActionItem:
        return replace(self, description=description)
