"""CLI entry point for botfriend."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Botfriend: Real-time voice-to-code pipeline",
    )
    parser.add_argument(
        "project_dir",
        help="Path to the project directory for code execution",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )

    args = parser.parse_args()

    project_dir = os.path.abspath(args.project_dir)
    if not os.path.isdir(project_dir):
        print(f"Error: Not a directory: {project_dir}", file=sys.stderr)
        sys.exit(1)

    hf_token = None
    if not args.no_diarization:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print(
                "Speaker diarization requires a HuggingFace token.\n"
                "Set HF_TOKEN env var or run with --no-diarization.\n"
                "Get a free token at https://huggingface.co/settings/tokens",
                file=sys.stderr,
            )
            sys.exit(1)

    from botfriend.app import BotfriendApp

    app = BotfriendApp(
        project_dir=project_dir,
        model_size=args.model,
        hf_token=hf_token,
    )
    app.run()


if __name__ == "__main__":
    main()
