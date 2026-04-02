# Botfriend: Real-Time Voice-to-Code Pipeline

**Date:** 2026-03-28
**Status:** Draft

## Overview

Botfriend is a terminal-based tool that listens to live conversations between people ideating about a software product and extracts actionable code changes in real time. It bridges the gap between user research / design conversations and implementation by turning spoken ideas into queued action items that execute against a live codebase via Claude Code CLI.

### Core Value Proposition

Shortens the loop between user testing / UXR conversations and engineering implementation. Changes happen in real time, collaboratively, as ideas are spoken.

### Key Use Cases

- **Pair ideation:** Two people (e.g., designer + developer) talking through UX changes while the tool implements them live
- **User research sessions:** Multiple participants bouncing ideas; the tool captures and queues actionable items from the discussion
- **Collaborative design reviews:** Stakeholders discussing product changes with immediate execution capability

## Architecture

### Four-Stage Async Pipeline

```
Mic → [Audio Capture] → [Transcription] → [Action Extraction] → [Queue] → [Execution]
         (sounddevice)   (faster-whisper     (Claude Code CLI      (TUI)    (Claude Code
                          + pyannote)         as extractor)                   CLI as coder)
```

Each stage runs as an independent async task. A slow execution never blocks transcription or extraction.

### Stage 1: Audio Capture

- `sounddevice` captures mic audio at 16kHz mono (Whisper's preferred format)
- Audio is buffered in ~3-second chunks with ~1-second overlap to prevent cutting words mid-sentence
- Chunks are pushed to an async queue for the transcription stage

### Stage 2: Transcription + Speaker Diarization

- `faster-whisper` (CTranslate2 backend) transcribes each audio chunk
- Runs on CPU on M-series Macs via the Neural Engine — no GPU required
- `pyannote-audio` labels speakers as `Speaker 1`, `Speaker 2`, etc.
  - Speaker labels are consistent within a session but not across sessions
- Output: timestamped, speaker-labeled text segments pushed to the next queue
- Low-confidence transcriptions are still processed but flagged for the extractor to weigh lower

### Stage 3: Action Extraction

- Transcript accumulates in a rolling buffer
- Every ~10 seconds (or on detected speech pauses), the buffer is sent to Claude Code CLI with an extraction prompt
- Uses a **separate, stateless** Claude Code invocation (not the coding session)
- Extraction prompt:

```
You are analyzing a live conversation between people discussing
a software product. Extract any actionable code changes mentioned.
Return a JSON array of action items, each with:
- description: what to implement
- speakers: who suggested it
- confidence: high/medium/low
```

- Only `high` and `medium` confidence items enter the queue
- The extraction prompt includes previously extracted items to avoid duplicates
- If the conversation refines an idea ("actually make it teal, not blue"), a new item supersedes the old one

### Stage 4: Queue + Execution

- Action items land in the TUI queue with `pending` status
- User confirms/rejects via keyboard
- Confirmed items are sent to Claude Code CLI for code execution
- Execution is **sequential** — one Claude Code CLI call at a time since it operates on the same codebase
- The coding session ID is preserved across executions so Claude maintains context of prior changes

## TUI Layout

Three-pane layout using `textual`:

```
┌─────────────────────────────────────┬──────────────────────────┐
│                                     │                          │
│         LIVE TRANSCRIPT             │      ACTION QUEUE        │
│                                     │                          │
│  [Speaker 1] we should make the     │  ▶ Change CTA button     │
│  button more prominent              │    color to blue          │
│                                     │    (Speaker 1 + 2)       │
│  [Speaker 2] yeah like blue, and    │                          │
│  bigger too                         │    Add onboarding modal   │
│                                     │    after signup           │
│  [Speaker 1] also we need an        │    (Speaker 1)           │
│  onboarding flow after signup       │                          │
│                                     │                          │
├─────────────────────────────────────┴──────────────────────────┤
│  EXECUTION LOG                                                 │
│  ✓ Updated header nav links (session abc123)                   │
│  ⟳ Running: "Add footer component with social links"           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate action queue |
| `Enter` | Confirm selected action (sends to execution) |
| `x` | Reject/dismiss selected action |
| `e` | Edit action item text before confirming (inline editor) |
| `p` | Pause/resume listening |
| `q` | Quit |

### Action Item States

`pending` → `confirmed` → `executing` → `done` / `failed`

## Project Structure

```
botfriend/
├── src/
│   └── botfriend/
│       ├── __init__.py
│       ├── main.py              # Entry point, CLI args
│       ├── app.py               # Textual TUI app
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── audio.py         # Mic capture, chunking
│       │   ├── transcriber.py   # Whisper + pyannote
│       │   └── extractor.py     # Action item extraction via Claude CLI
│       ├── queue/
│       │   ├── __init__.py
│       │   └── action_queue.py  # Action item state machine
│       ├── executor/
│       │   ├── __init__.py
│       │   └── claude_runner.py # Claude Code CLI execution
│       └── models.py            # Data classes (ActionItem, Transcript, etc.)
├── tests/
├── pyproject.toml
└── README.md
```

## Startup

```bash
botfriend /path/to/project
```

- Points at a project directory for code execution
- Downloads Whisper model on first run if not cached (~1.5GB for `medium` model)
- Pyannote model requires a free HuggingFace token — on first run, prompts user to provide it if not set via `HF_TOKEN` env var
- Opens the TUI and starts listening immediately

## Dependencies

- **Python 3.11+**
- `textual` — TUI framework
- `faster-whisper` — local Whisper inference
- `pyannote-audio` — speaker diarization
- `sounddevice` — mic capture
- `numpy` — audio buffer handling

## Edge Cases & Design Decisions

### Audio Quality
Whisper is robust to background noise. Low-confidence transcriptions are flagged but still processed. No action is taken on unintelligible audio — it simply doesn't extract items.

### Conversation vs. Action Items
Not everything said is actionable. The extraction prompt is tuned to only pull out statements implying a code change. The confidence threshold filters ambiguous cases.

### Conflicting Ideas
If Speaker 1 says "make it blue" and Speaker 2 says "actually red," both are extracted as separate queue items. The user resolves conflicts by confirming the one they want (or editing to a compromise). The queue is the resolution layer.

### Long-Running Executions
If a Claude Code execution takes 2+ minutes, new action items keep accumulating in the queue. The transcript and extraction pipeline are never blocked. A visual indicator shows when execution is busy.

### Session Persistence
The Claude Code coding session ID is preserved across executions so it maintains context of prior changes. Restarting botfriend starts a fresh session.

### Extraction Rate Limiting
The extractor fires on a ~10-second interval or on detected speech pauses — not on every transcript chunk. This avoids redundant Claude Code CLI calls.

## Future Roadmap

- **Multi-project support:** Switch target project mid-conversation based on context or explicit command
- **Web UI frontend:** Optional browser-based frontend for richer transcript visualization while keeping the same pipeline engine
- **Session recording:** Save full transcripts with action item history for review
- **Team mode:** Multiple listeners on different machines contributing to the same action queue
