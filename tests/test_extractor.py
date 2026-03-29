import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from botfriend.pipeline.extractor import ActionExtractor
from botfriend.models import TranscriptSegment, ActionItem


class TestActionExtractor:
    def _make_segment(self, text: str, speaker: str = "Speaker 1") -> TranscriptSegment:
        return TranscriptSegment(
            text=text,
            speaker=speaker,
            start_time=0.0,
            end_time=2.0,
            confidence=0.9,
        )

    def test_create_extractor(self):
        mock_runner = MagicMock()
        extractor = ActionExtractor(runner=mock_runner, interval_seconds=10.0)
        assert extractor.interval_seconds == 10.0

    def test_add_segment_to_buffer(self):
        mock_runner = MagicMock()
        extractor = ActionExtractor(runner=mock_runner)
        seg = self._make_segment("make the button blue")
        extractor.add_segment(seg)
        assert len(extractor.transcript_buffer) == 1

    def test_format_transcript(self):
        mock_runner = MagicMock()
        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("make the button blue", "Speaker 1"))
        extractor.add_segment(self._make_segment("yeah and bigger", "Speaker 2"))
        formatted = extractor.format_transcript()
        assert "[Speaker 1]" in formatted
        assert "make the button blue" in formatted
        assert "[Speaker 2]" in formatted

    @pytest.mark.asyncio
    async def test_extract_returns_action_items(self):
        mock_runner = AsyncMock()
        mock_runner.extract_actions.return_value = [
            {"description": "Make button blue", "speakers": ["Speaker 1"], "confidence": "high"},
        ]

        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("make the button blue"))

        items = await extractor.extract()
        assert len(items) == 1
        assert items[0].description == "Make button blue"
        assert items[0].speakers == ("Speaker 1",)
        assert items[0].confidence == "high"

    @pytest.mark.asyncio
    async def test_extract_filters_low_confidence(self):
        mock_runner = AsyncMock()
        mock_runner.extract_actions.return_value = [
            {"description": "Maybe change color", "speakers": ["Speaker 1"], "confidence": "low"},
        ]

        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("maybe change the color?"))

        items = await extractor.extract()
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_extract_tracks_previous_items(self):
        mock_runner = AsyncMock()
        mock_runner.extract_actions.return_value = [
            {"description": "A", "speakers": ["S1"], "confidence": "high"},
        ]

        extractor = ActionExtractor(runner=mock_runner)
        extractor.add_segment(self._make_segment("do A"))
        await extractor.extract()

        mock_runner.extract_actions.return_value = []
        extractor.add_segment(self._make_segment("do B"))
        await extractor.extract()

        call_args = mock_runner.extract_actions.call_args_list[1]
        previous = call_args[0][1]
        assert len(previous) == 1
        assert previous[0]["description"] == "A"
