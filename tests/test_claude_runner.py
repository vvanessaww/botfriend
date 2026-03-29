import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from botfriend.executor.claude_runner import ClaudeRunner, ExecutionResult


class TestExecutionResult:
    def test_create_success(self):
        result = ExecutionResult(text="Done", session_id="abc123", success=True)
        assert result.text == "Done"
        assert result.session_id == "abc123"
        assert result.success is True

    def test_create_failure(self):
        result = ExecutionResult(text="Error occurred", session_id=None, success=False)
        assert result.success is False


class TestClaudeRunner:
    def test_create_runner(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        assert runner.project_dir == "/tmp/myproject"
        assert runner.session_id is None

    @pytest.mark.asyncio
    async def test_execute_action_success(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        mock_result = json.dumps({
            "type": "result",
            "result": "Changed button color to blue",
            "session_id": "sess_abc123",
        })

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_result.encode(), b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await runner.execute("Change CTA button color to blue")

            assert result.success is True
            assert result.text == "Changed button color to blue"
            assert result.session_id == "sess_abc123"
            assert runner.session_id == "sess_abc123"

    @pytest.mark.asyncio
    async def test_execute_preserves_session(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        runner = runner.with_session_id("existing_session")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (
                json.dumps({"result": "ok", "session_id": "existing_session"}).encode(),
                b"",
            )
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            await runner.execute("Do something")

            call_args = mock_exec.call_args[0]
            assert "--resume" in call_args
            assert "existing_session" in call_args

    @pytest.mark.asyncio
    async def test_execute_action_failure(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (b"", b"Something went wrong")
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            result = await runner.execute("Do something impossible")

            assert result.success is False
            assert "Something went wrong" in result.text

    @pytest.mark.asyncio
    async def test_extract_actions_returns_json(self):
        runner = ClaudeRunner(project_dir="/tmp/myproject")
        mock_result = json.dumps({
            "type": "result",
            "result": json.dumps([
                {"description": "Make button blue", "speakers": ["Speaker 1"], "confidence": "high"}
            ]),
            "session_id": "sess_extract",
        })

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_result.encode(), b"")
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            actions = await runner.extract_actions("transcript text here", [])

            assert len(actions) == 1
            assert actions[0]["description"] == "Make button blue"
