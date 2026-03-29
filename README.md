# Botfriend

Real-time voice-to-code pipeline. Listens to live conversations, extracts actionable code changes, and executes them via Claude Code CLI.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Speaker diarization requires a free HuggingFace token:
1. Create account at https://huggingface.co
2. Accept model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create token at https://huggingface.co/settings/tokens
4. Set: `export HF_TOKEN=hf_your_token_here`

## Usage

```bash
botfriend /path/to/your/project
```

Options:
- `--model {tiny,base,small,medium,large-v3}` — Whisper model size (default: base)
- `--no-diarization` — Disable speaker identification

## Keyboard Controls

| Key | Action |
|-----|--------|
| Up/Down | Navigate action queue |
| Enter | Confirm selected action |
| x | Reject/dismiss action |
| e | Edit action before confirming |
| p | Pause/resume listening |
| q | Quit |
