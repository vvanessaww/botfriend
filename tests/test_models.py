import pytest
from dataclasses import FrozenInstanceError
from botfriend.models import (
    TranscriptSegment,
    ActionItem,
    ActionStatus,
)


class TestTranscriptSegment:
    def test_create_segment(self):
        seg = TranscriptSegment(
            text="we should make the button blue",
            speaker="Speaker 1",
            start_time=0.0,
            end_time=2.5,
            confidence=0.95,
        )
        assert seg.text == "we should make the button blue"
        assert seg.speaker == "Speaker 1"
        assert seg.start_time == 0.0
        assert seg.end_time == 2.5
        assert seg.confidence == 0.95

    def test_segment_is_immutable(self):
        seg = TranscriptSegment(
            text="hello",
            speaker="Speaker 1",
            start_time=0.0,
            end_time=1.0,
            confidence=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            seg.text = "modified"


class TestActionItem:
    def test_create_action_item(self):
        item = ActionItem(
            description="Change CTA button color to blue",
            speakers=("Speaker 1", "Speaker 2"),
            confidence="high",
        )
        assert item.description == "Change CTA button color to blue"
        assert item.speakers == ("Speaker 1", "Speaker 2")
        assert item.confidence == "high"
        assert item.status == ActionStatus.PENDING

    def test_action_item_with_status(self):
        item = ActionItem(
            description="Add modal",
            speakers=("Speaker 1",),
            confidence="medium",
            status=ActionStatus.EXECUTING,
        )
        assert item.status == ActionStatus.EXECUTING

    def test_action_item_replace_status(self):
        item = ActionItem(
            description="Add modal",
            speakers=("Speaker 1",),
            confidence="high",
        )
        updated = item.with_status(ActionStatus.CONFIRMED)
        assert updated.status == ActionStatus.CONFIRMED
        assert item.status == ActionStatus.PENDING  # original unchanged

    def test_action_item_replace_description(self):
        item = ActionItem(
            description="Make it blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        updated = item.with_description("Make it teal")
        assert updated.description == "Make it teal"
        assert item.description == "Make it blue"  # original unchanged


class TestActionStatus:
    def test_status_values(self):
        assert ActionStatus.PENDING.value == "pending"
        assert ActionStatus.CONFIRMED.value == "confirmed"
        assert ActionStatus.EXECUTING.value == "executing"
        assert ActionStatus.DONE.value == "done"
        assert ActionStatus.FAILED.value == "failed"
        assert ActionStatus.REJECTED.value == "rejected"
