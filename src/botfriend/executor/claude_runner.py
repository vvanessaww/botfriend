"""Claude Code CLI wrapper for action extraction and code execution."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any


EXTRACTION_PROMPT_TEMPLATE = """You are analyzing a live conversation between people discussing a software product. Extract any actionable code changes mentioned.

Here is the recent conversation transcript:

{transcript}

Previously extracted action items (do NOT re-extract these):
{previous_items}

Return ONLY a JSON array of NEW action items. Each item must have:
- "description": a clear, specific description of what code change to make
- "speakers": list of speaker labels who contributed to this idea
- "confidence": "high", "medium", or "low"

If there are no new actionable items, return an empty array: []
Return ONLY the JSON array, no other text."""


@dataclass(frozen=True)
class ExecutionResult:
    text: str
    session_id: str | None
    success: bool


class ClaudeRunner:
    def __init__(self, project_dir: str) -> None:
        self.project_dir = project_dir
        self.session_id: str | None = None

    def with_session_id(self, session_id: str) -> ClaudeRunner:
        runner = ClaudeRunner(project_dir=self.project_dir)
        runner.session_id = session_id
        return runner

    async def execute(self, action_description: str) -> ExecutionResult:
        cmd = [
            "claude",
            "-p", action_description,
            "--output-format", "json",
            "--dangerously-skip-permissions",
            "--verbose",
        ]
        if self.session_id:
            cmd.extend(["--resume", self.session_id])

        env = {**os.environ}
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.project_dir,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip() or stdout.decode().strip() or "Unknown error"
            return ExecutionResult(
                text=f"Error (exit {proc.returncode}): {err}",
                session_id=self.session_id,
                success=False,
            )

        raw = stdout.decode()
        try:
            data = json.loads(raw)
            text = data.get("result", raw)
            new_session_id = data.get("session_id", self.session_id)
            self.session_id = new_session_id
            return ExecutionResult(text=text, session_id=new_session_id, success=True)
        except json.JSONDecodeError:
            return ExecutionResult(text=raw, session_id=self.session_id, success=True)

    async def extract_actions(
        self,
        transcript: str,
        previous_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        previous_str = json.dumps(previous_items, indent=2) if previous_items else "None"
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            transcript=transcript,
            previous_items=previous_str,
        )

        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
        ]

        env = {**os.environ}
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return []

        raw = stdout.decode()
        try:
            data = json.loads(raw)
            result_text = data.get("result", raw)
            actions = json.loads(result_text)
            if isinstance(actions, list):
                return actions
            return []
        except (json.JSONDecodeError, TypeError):
            return []
