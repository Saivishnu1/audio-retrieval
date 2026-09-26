"""Command-line interface for diarized transcription of audio clips."""

from __future__ import annotations

import argparse
import logging
import sys

from .exceptions import TranscriptionError
from .transcriber import DIARIZE_MODEL, transcribe_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-clips",
        description="Generate diarized (speaker-labeled) transcripts for every "
        "audio clip in a directory, one transcript file per clip.",
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing audio clips to transcribe (e.g. 'clips')",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="transcripts",
        help="Directory to write transcript files into (default: transcripts)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DIARIZE_MODEL,
        help=f"Diarization-capable transcription model (default: {DIARIZE_MODEL})",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="output_format",
        choices=["txt", "json"],
        default="txt",
        help="Transcript output format (default: txt)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # picks up OPENAI_API_KEY from a local .env, if present
    except ImportError:  # pragma: no cover
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        results = transcribe_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=args.model,
            output_format=args.output_format,
        )
    except TranscriptionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Transcribed {len(results)} file(s) into '{args.output_dir}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
