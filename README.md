# Botfriend

Talk about code changes out loud. Botfriend listens, figures out what's actionable, and runs it against your codebase through Claude Code — all in real time.

It started as a way to speed up pair ideation sessions. Two people sit down, talk through what they want to change, and the tool handles everything between "we should make the button blue" and the actual code diff. No context switching, no writing tickets, no "I'll get to that later."

## How it works

Botfriend runs a four-stage async pipeline. Each stage operates independently, so a slow code execution never blocks transcription or extraction.

```
Mic ──→ Audio Capture ──→ Transcription ──→ Action Extraction ──→ Queue ──→ Execution
         sounddevice      faster-whisper      Claude (Haiku)       TUI      Claude Code
                          + pyannote                                         CLI
```

1. **Audio capture** — Records from your mic at 16kHz. Buffers audio in ~2-second chunks with overlap so words don't get cut mid-sentence.

2. **Transcription** — `faster-whisper` transcribes each chunk locally on your machine (no API calls). `pyannote-audio` optionally labels who's talking (Speaker 1, Speaker 2, etc.).

3. **Action extraction** — Every few seconds, the accumulated transcript gets sent to Claude Haiku with a prompt that pulls out anything that sounds like a code change. It tracks what it's already extracted to avoid duplicates. If someone refines an idea ("actually make it teal, not blue"), the new item supersedes the old one. Only high/medium confidence items make it to the queue.

4. **Queue + execution** — Extracted items land in a TUI queue. You review each one — confirm, reject, or edit the description before it runs. Confirmed items execute sequentially through Claude Code CLI, which preserves its session ID across runs so it has context of prior changes.

The TUI has three panes:

```
┌───────────────────────────────────┬────────────────────────────┐
│                                   │                            │
│        LIVE TRANSCRIPT            │       ACTION QUEUE         │
│                                   │                            │
│  [Speaker 1] we should make the   │  ▶ Change CTA button      │
│  button more prominent            │    color to blue           │
│                                   │    (Speaker 1 + 2)         │
│  [Speaker 2] yeah like blue, and  │                            │
│  bigger too                       │    Add onboarding modal    │
│                                   │    after signup            │
│  [Speaker 1] also we need an      │    (Speaker 1)             │
│  onboarding flow after signup     │                            │
│                                   │                            │
├───────────────────────────────────┴────────────────────────────┤
│  EXECUTION LOG                                                 │
│  ✓ Updated header nav links (session abc123)                   │
│  ⟳ Running: "Add footer component with social links"           │
└────────────────────────────────────────────────────────────────┘
```

## When to use it

- **Pair ideation** — Designer + developer talking through UX changes while the tool implements them live
- **User research sessions** — Multiple people bouncing ideas around; botfriend captures and queues the actionable ones
- **Design reviews** — Stakeholders discussing changes with immediate execution so you can see results in the same meeting

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/vvanessaww/botfriend.git
cd botfriend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Speaker diarization (identifying who's talking) is optional but useful for multi-person sessions. It requires a free HuggingFace token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. `export HF_TOKEN=hf_your_token_here`

On first run, Whisper downloads its model weights (~150MB for `base`, ~1.5GB for `medium`). This only happens once.

## Usage

Point it at the project you want to modify:

```bash
botfriend /path/to/your/project
```

That's it. It opens the TUI and starts listening immediately.

Options:
- `--model {tiny,base,small,medium,large-v3}` — Whisper model size (default: `base`). Larger models are more accurate but slower.
- `--no-diarization` — Skip speaker identification if you don't need it or don't have an HF token.

### Keyboard controls

| Key | Action |
|-----|--------|
| `↑` / `↓` | Navigate the action queue |
| `Enter` | Confirm an action (sends it to Claude Code for execution) |
| `x` | Reject/dismiss an action |
| `e` | Edit the action description before confirming |
| `p` | Pause/resume the mic |
| `q` | Quit |

## Architecture

The pipeline is fully async. Each stage communicates through queues, so they never block each other. A long-running Claude Code execution doesn't stall transcription — new items keep accumulating in the queue while it works.

Key design decisions:

- **Immutable data throughout.** Action items are frozen dataclasses with explicit state transitions (`pending → confirmed → executing → done/failed`). The queue itself is immutable — every operation returns a new queue. No mutation, no surprises.

- **Two separate Claude instances.** One stateless Haiku call handles extraction (cheap, fast, no memory needed). A separate Claude Code session handles execution and preserves context across runs so it knows what it already changed.

- **Human in the loop.** Nothing executes without confirmation. The queue is the resolution layer — if two people suggest conflicting ideas, you pick the one you want (or edit it into a compromise).

- **Extraction deduplication.** The extractor passes previously extracted items back to Claude so it doesn't re-extract the same idea. If the conversation refines something, the new version supersedes the old one.

```
botfriend/
├── src/botfriend/
│   ├── main.py              # CLI entry point
│   ├── app.py               # Textual TUI application
│   ├── models.py            # Frozen dataclasses (ActionItem, TranscriptSegment)
│   ├── pipeline/
│   │   ├── audio.py         # Mic capture, chunking, overlap
│   │   ├── transcriber.py   # Whisper + pyannote diarization
│   │   └── extractor.py     # Action extraction via Claude Haiku
│   ├── queue/
│   │   └── action_queue.py  # Immutable queue with state machine
│   ├── executor/
│   │   └── claude_runner.py # Claude Code CLI wrapper, session persistence
│   └── widgets/             # TUI components (transcript, queue, log, editor)
└── tests/                   # Unit + integration tests
```

## Dependencies

All processing runs locally except for the Claude API calls (extraction + execution).

| Dependency | What it does |
|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Local speech-to-text (CTranslate2 backend, runs on CPU) |
| [pyannote-audio](https://github.com/pyannote/pyannote-audio) | Speaker diarization |
| [textual](https://github.com/Textualize/textual) | TUI framework |
| [sounddevice](https://python-sounddevice.readthedocs.io/) | Mic capture |
| numpy | Audio buffer handling |

## Tests

```bash
pytest
pytest --cov=botfriend  # with coverage
```

## License

MIT
