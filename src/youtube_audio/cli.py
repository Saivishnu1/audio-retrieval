"""Command-line interface for extracting audio from YouTube videos."""

from __future__ import annotations

import argparse
import logging
import sys

from .exceptions import AudioExtractionError
from .extractor import DEFAULT_AUDIO_FORMAT, DEFAULT_AUDIO_QUALITY, extract_audio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-audio",
        description="Extract audio from a YouTube video URL.",
    )
    parser.add_argument("url", help="YouTube video URL to extract audio from")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="downloads",
        help="Directory to save the extracted audio file (default: downloads)",
    )
    parser.add_argument(
        "-f",
        "--format",
        dest="audio_format",
        default=DEFAULT_AUDIO_FORMAT,
        help=f"Target audio format, e.g. mp3, wav, m4a (default: {DEFAULT_AUDIO_FORMAT})",
    )
    parser.add_argument(
        "-q",
        "--quality",
        dest="audio_quality",
        default=DEFAULT_AUDIO_QUALITY,
        help=f"Target audio bitrate in kbps (default: {DEFAULT_AUDIO_QUALITY})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument(
        "-t",
        "--time-range",
        dest="time_range",
        default=None,
        help="Extract only this time range, e.g. '1:30-4:00', '90-240', "
        "or '1:30-' (from 1:30 to the end of the video)",
    )
    range_group.add_argument(
        "-c",
        "--chapter",
        dest="chapter",
        default=None,
        help="Extract only chapter(s) whose title matches this regex, "
        "e.g. 'Intro' or 'Chapter 3'",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    try:
        result = extract_audio(
            url=args.url,
            output_dir=args.output_dir,
            audio_format=args.audio_format,
            audio_quality=args.audio_quality,
            time_range=args.time_range,
            chapter=args.chapter,
        )
    except AudioExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved audio: {result.file_path}")
    print(f"Title: {result.title}")
    if result.duration_seconds is not None:
        print(f"Duration: {result.duration_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
