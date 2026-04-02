"""Tests for CLI entry point."""

import logging
import sys
from unittest.mock import patch, MagicMock

import pytest

from botfriend.main import main


class TestMainCLI:
    def test_missing_project_dir_exits(self):
        with patch("sys.argv", ["botfriend"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2  # argparse exits with 2 on missing args

    def test_invalid_project_dir_exits(self, tmp_path):
        fake_path = str(tmp_path / "nonexistent")
        with patch("sys.argv", ["botfriend", fake_path]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 1

    def test_no_hf_token_without_flag_exits(self, tmp_path):
        with patch("sys.argv", ["botfriend", str(tmp_path)]):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_no_diarization_skips_token_check(self, tmp_path):
        """With --no-diarization, should proceed past HF token check."""
        with patch("sys.argv", ["botfriend", str(tmp_path), "--no-diarization"]):
            with patch.dict("os.environ", {}, clear=True):
                mock_transcriber = MagicMock()
                mock_app_instance = MagicMock()
                with patch("botfriend.pipeline.transcriber.Transcriber", return_value=mock_transcriber):
                    with patch("botfriend.app.BotfriendApp", return_value=mock_app_instance):
                        main()
                        mock_app_instance.run.assert_called_once()

    def test_debug_flag_sets_logging_level(self, tmp_path):
        # Reset root logger so basicConfig can reconfigure it
        root = logging.getLogger()
        original_level = root.level
        original_handlers = root.handlers[:]
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        try:
            with patch("sys.argv", ["botfriend", str(tmp_path), "--no-diarization", "--debug"]):
                with patch.dict("os.environ", {}, clear=True):
                    with patch("botfriend.pipeline.transcriber.Transcriber", return_value=MagicMock()):
                        with patch("botfriend.app.BotfriendApp", return_value=MagicMock()):
                            main()
                            assert logging.getLogger().level == logging.DEBUG
        finally:
            root.handlers = original_handlers
            root.setLevel(original_level)

    def test_model_flag_passed_to_transcriber(self, tmp_path):
        with patch("sys.argv", ["botfriend", str(tmp_path), "--no-diarization", "--model", "tiny"]):
            with patch.dict("os.environ", {}, clear=True):
                with patch("botfriend.pipeline.transcriber.Transcriber") as mock_cls:
                    mock_cls.return_value = MagicMock()
                    with patch("botfriend.app.BotfriendApp", return_value=MagicMock()):
                        main()
                        mock_cls.assert_called_once_with(model_size="tiny")
