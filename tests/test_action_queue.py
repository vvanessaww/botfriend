import pytest
from botfriend.models import ActionItem, ActionStatus
from botfriend.queue.action_queue import ActionQueue


class TestActionQueue:
    def test_add_item(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        updated_queue = queue.add(item)
        assert len(updated_queue) == 1
        assert updated_queue.items[0].description == "Make button blue"
        assert len(queue) == 0  # original unchanged

    def test_confirm_item(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        queue = queue.add(item)
        queue = queue.confirm(0)
        assert queue.items[0].status == ActionStatus.CONFIRMED

    def test_reject_item(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
        )
        queue = queue.add(item)
        queue = queue.reject(0)
        assert queue.items[0].status == ActionStatus.REJECTED

    def test_confirm_out_of_bounds_raises(self):
        queue = ActionQueue()
        with pytest.raises(IndexError):
            queue.confirm(0)

    def test_confirm_non_pending_raises(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
            status=ActionStatus.EXECUTING,
        )
        queue = queue.add(item)
        with pytest.raises(ValueError, match="Can only confirm pending items"):
            queue.confirm(0)

    def test_next_confirmed_returns_first_confirmed(self):
        queue = ActionQueue()
        item1 = ActionItem(description="A", speakers=("S1",), confidence="high")
        item2 = ActionItem(description="B", speakers=("S1",), confidence="high")
        queue = queue.add(item1).add(item2)
        queue = queue.confirm(0)
        result = queue.next_confirmed()
        assert result is not None
        assert result[1].description == "A"

    def test_next_confirmed_returns_none_when_empty(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item)
        assert queue.next_confirmed() is None

    def test_mark_executing(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item).confirm(0)
        queue = queue.mark_executing(0)
        assert queue.items[0].status == ActionStatus.EXECUTING

    def test_mark_done(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item).confirm(0).mark_executing(0)
        queue = queue.mark_done(0)
        assert queue.items[0].status == ActionStatus.DONE

    def test_mark_failed(self):
        queue = ActionQueue()
        item = ActionItem(description="A", speakers=("S1",), confidence="high")
        queue = queue.add(item).confirm(0).mark_executing(0)
        queue = queue.mark_failed(0)
        assert queue.items[0].status == ActionStatus.FAILED

    def test_edit_item(self):
        queue = ActionQueue()
        item = ActionItem(description="Make it blue", speakers=("S1",), confidence="high")
        queue = queue.add(item)
        queue = queue.edit(0, "Make it teal")
        assert queue.items[0].description == "Make it teal"
        assert queue.items[0].status == ActionStatus.PENDING

    def test_pending_items(self):
        queue = ActionQueue()
        item1 = ActionItem(description="A", speakers=("S1",), confidence="high")
        item2 = ActionItem(description="B", speakers=("S1",), confidence="high")
        queue = queue.add(item1).add(item2).confirm(0)
        pending = queue.pending_items()
        assert len(pending) == 1
        assert pending[0][1].description == "B"

    def test_reject_out_of_bounds_raises(self):
        queue = ActionQueue()
        with pytest.raises(IndexError):
            queue.reject(0)

    def test_reject_non_pending_raises(self):
        queue = ActionQueue()
        item = ActionItem(
            description="Make button blue",
            speakers=("Speaker 1",),
            confidence="high",
            status=ActionStatus.EXECUTING,
        )
        queue = queue.add(item)
        with pytest.raises(ValueError, match="Can only reject pending items"):
            queue.reject(0)

    def test_mark_executing_out_of_bounds_raises(self):
        queue = ActionQueue()
        with pytest.raises(IndexError):
            queue.mark_executing(0)

    def test_mark_done_out_of_bounds_raises(self):
        queue = ActionQueue()
        with pytest.raises(IndexError):
            queue.mark_done(0)

    def test_mark_failed_out_of_bounds_raises(self):
        queue = ActionQueue()
        with pytest.raises(IndexError):
            queue.mark_failed(0)

    def test_edit_out_of_bounds_raises(self):
        queue = ActionQueue()
        with pytest.raises(IndexError):
            queue.edit(0, "new description")
