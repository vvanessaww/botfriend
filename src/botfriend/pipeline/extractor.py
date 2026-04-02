"""Extracts actionable items from accumulated transcript using Claude CLI."""

from __future__ import annotations

from typing import Any

from botfriend.models import ActionItem, TranscriptSegment
from botfriend.executor.claude_runner import ClaudeRunner

ALLOWED_CONFIDENCES = {"high", "medium"}


class ActionExtractor:
    def __init__(
        self,
        runner: ClaudeRunner,
        interval_seconds: float = 5.0,
    ) -> None:
        self.runner = runner
        self.interval_seconds = interval_seconds
        self.transcript_buffer: list[TranscriptSegment] = []
        self._previous_items: list[dict[str, Any]] = []

    def add_segment(self, segment: TranscriptSegment) -> None:
        self.transcript_buffer.append(segment)

    def format_transcript(self) -> str:
        lines = []
        for seg in self.transcript_buffer:
            lines.append(f"[{seg.speaker}] {seg.text}")
        return "\n".join(lines)

    async def extract(self) -> list[ActionItem]:
        if not self.transcript_buffer:
            return []

        transcript_text = self.format_transcript()
        raw_actions = await self.runner.extract_actions(
            transcript_text,
            self._previous_items,
        )

        items: list[ActionItem] = []
        for raw in raw_actions:
            confidence = raw.get("confidence", "low")
            if confidence not in ALLOWED_CONFIDENCES:
                continue
            speakers = raw.get("speakers", [])
            item = ActionItem(
                description=raw["description"],
                speakers=tuple(speakers),
                confidence=confidence,
            )
            items.append(item)
            self._previous_items.append({
                "description": raw["description"],
                "speakers": speakers,
                "confidence": confidence,
            })

        return items
