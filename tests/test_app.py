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
