import pytest
from unittest.mock import MagicMock, patch


class TestBotfriendApp:
    @pytest.mark.asyncio
    async def test_app_starts_and_shows_panels(self):
        from botfriend.app import BotfriendApp
        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            from botfriend.widgets.transcript_panel import TranscriptPanel
            from botfriend.widgets.queue_panel import QueuePanel
            from botfriend.widgets.execution_log import ExecutionLog
            assert app.query_one(TranscriptPanel) is not None
            assert app.query_one(QueuePanel) is not None
            assert app.query_one(ExecutionLog) is not None

    @pytest.mark.asyncio
    async def test_app_quit_keybinding(self):
        from botfriend.app import BotfriendApp
        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("q")


class TestBotfriendIntegration:
    @pytest.mark.asyncio
    async def test_full_flow_with_mock_data(self):
        """Test the full flow: add item to queue, confirm, verify state."""
        import asyncio
        from unittest.mock import AsyncMock
        from botfriend.app import BotfriendApp
        from botfriend.models import ActionItem
        from botfriend.widgets.queue_panel import QueuePanel

        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        # Prevent _execute_next from running (no _coding_runner in test mode)
        app._execute_next = AsyncMock(return_value=None)
        async with app.run_test(size=(120, 40)) as pilot:
            item = ActionItem(
                description="Change button color to blue",
                speakers=("Speaker 1",),
                confidence="high",
            )
            app._queue = app._queue.add(item)
            app.query_one(QueuePanel).update_queue(app._queue)

            app.query_one(QueuePanel).focus()
            await pilot.press("enter")
            # Allow message processing
            try:
                await pilot.pause()
            except AttributeError:
                await asyncio.sleep(0.1)

            assert app._queue.items[0].status.value == "confirmed"

    @pytest.mark.asyncio
    async def test_reject_item_flow(self):
        import asyncio
        from botfriend.app import BotfriendApp
        from botfriend.models import ActionItem
        from botfriend.widgets.queue_panel import QueuePanel

        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            item = ActionItem(
                description="Remove header",
                speakers=("Speaker 2",),
                confidence="medium",
            )
            app._queue = app._queue.add(item)
            app.query_one(QueuePanel).update_queue(app._queue)

            app.query_one(QueuePanel).focus()
            await pilot.press("x")
            try:
                await pilot.pause()
            except AttributeError:
                await asyncio.sleep(0.1)

            assert app._queue.items[0].status.value == "rejected"

    @pytest.mark.asyncio
    async def test_pause_toggle(self):
        import asyncio
        from botfriend.app import BotfriendApp

        app = BotfriendApp(project_dir="/tmp/test", enable_audio=False)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app._paused is False
            await pilot.press("p")
            try:
                await pilot.pause()
            except AttributeError:
                await asyncio.sleep(0.1)
            assert app._paused is True
            await pilot.press("p")
            try:
                await pilot.pause()
            except AttributeError:
                await asyncio.sleep(0.1)
            assert app._paused is False
