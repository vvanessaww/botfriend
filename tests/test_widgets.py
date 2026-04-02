"""Tests for TUI widget components."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from botfriend.models import ActionItem, ActionStatus
from botfriend.queue.action_queue import ActionQueue


class TestTranscriptPanel:
    def test_add_segment_renders_speaker_text(self):
        from botfriend.models import TranscriptSegment
        from botfriend.widgets.transcript_panel import TranscriptPanel

        panel = TranscriptPanel()
        segment = TranscriptSegment(
            text="make the button blue",
            speaker="Speaker 1",
            start_time=0.0,
            end_time=1.5,
            confidence=0.9,
        )
        # add_segment should not raise
        panel.add_segment(segment)

    def test_speaker_color_assignment(self):
        from botfriend.widgets.transcript_panel import TranscriptPanel

        panel = TranscriptPanel()
        assert panel.border_title == "Live Transcript"


class TestQueuePanel:
    def test_empty_queue_renders(self):
        from botfriend.widgets.queue_panel import QueuePanel

        panel = QueuePanel()
        assert panel.border_title == "Action Queue"
        assert len(panel._queue) == 0

    @pytest.mark.asyncio
    async def test_update_queue_stores_items(self):
        from botfriend.app import BotfriendApp
        from botfriend.widgets.queue_panel import QueuePanel

        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)):
            panel = app.query_one(QueuePanel)
            item = ActionItem(
                description="Add footer",
                speakers=("Speaker 1",),
                confidence="high",
            )
            queue = ActionQueue().add(item)
            panel.update_queue(queue)
            assert len(panel._queue) == 1
            assert panel._queue.items[0].description == "Add footer"

    @pytest.mark.asyncio
    async def test_selected_index_starts_at_zero(self):
        from botfriend.app import BotfriendApp
        from botfriend.widgets.queue_panel import QueuePanel

        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)):
            panel = app.query_one(QueuePanel)
            assert panel.selected_index == 0

    @pytest.mark.asyncio
    async def test_move_up_does_not_go_negative(self):
        from botfriend.app import BotfriendApp
        from botfriend.widgets.queue_panel import QueuePanel

        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)):
            panel = app.query_one(QueuePanel)
            panel.action_move_up()
            assert panel.selected_index == 0


class TestQueueItemWidget:
    def test_renders_with_status_icon(self):
        from botfriend.widgets.queue_panel import QueueItemWidget, STATUS_ICONS

        item = ActionItem(
            description="Fix the nav",
            speakers=("Speaker 2",),
            confidence="medium",
        )
        widget = QueueItemWidget(item, index=0, selected=False)
        assert widget.item_index == 0

    def test_selected_class_applied(self):
        from botfriend.widgets.queue_panel import QueueItemWidget

        item = ActionItem(
            description="Update colors",
            speakers=(),
            confidence="high",
        )
        widget = QueueItemWidget(item, index=1, selected=True)
        assert "selected" in widget.classes

    def test_unselected_class_not_applied(self):
        from botfriend.widgets.queue_panel import QueueItemWidget

        item = ActionItem(
            description="Update colors",
            speakers=(),
            confidence="high",
        )
        widget = QueueItemWidget(item, index=0, selected=False)
        assert "selected" not in widget.classes


class TestExecutionLog:
    def test_strip_markup(self):
        from botfriend.widgets.execution_log import _strip_markup

        assert _strip_markup("[bold red]Error:[/] something broke") == "Error: something broke"
        assert _strip_markup("no markup here") == "no markup here"
        assert _strip_markup("[dim]faded[/dim]") == "faded"

    def test_write_to_file(self, tmp_path):
        from botfriend.widgets.execution_log import ExecutionLog

        log_path = tmp_path / "test.log"
        log = ExecutionLog(log_path=log_path)
        log._write_to_file("[bold]test message[/]")

        content = log_path.read_text()
        assert "test message" in content
        assert "[bold]" not in content  # markup stripped

    def test_write_to_file_skips_when_no_path(self):
        from botfriend.widgets.execution_log import ExecutionLog

        log = ExecutionLog(log_path=None)
        # should not raise
        log._write_to_file("test")

    def test_write_to_file_logs_warning_on_error(self, tmp_path):
        from botfriend.widgets.execution_log import ExecutionLog

        # Point to a directory (not a file) so open() fails
        log = ExecutionLog(log_path=tmp_path)
        with patch("botfriend.widgets.execution_log.logger") as mock_logger:
            log._write_to_file("test")
            mock_logger.warning.assert_called_once()


class TestEditScreen:
    def test_stores_description_and_index(self):
        from botfriend.widgets.edit_screen import EditScreen

        screen = EditScreen(description="change the color", index=3)
        assert screen._description == "change the color"
        assert screen._index == 3
