"""Diarized transcription of audio files via OpenAI's gpt-4o-transcribe-diarize.

Sends an audio file to OpenAI's `/v1/audio/transcriptions` endpoint with
`response_format="diarized_json"`, which returns speaker-labeled segments
(who spoke, what they said, and when). Requires an `OPENAI_API_KEY`
environment variable.

Reference: https://developers.openai.com/api/docs/guides/speech-to-text
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .exceptions import TranscriptionError

logger = logging.getLogger(__name__)

DIARIZE_MODEL = "gpt-4o-transcribe-diarize"
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # OpenAI's hard limit for this endpoint
SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}


@dataclass
class DiarizedSegment:
    """A single speaker-labeled segment of a transcript."""

    speaker: str
    text: str
    start: float
    end: float


@dataclass
class TranscriptResult:
    """The full diarized transcript for one audio file."""

    source_file: Path
    segments: list[DiarizedSegment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """The transcript as plain text, without speaker labels."""
        return " ".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def to_labeled_text(self) -> str:
        """Render the transcript as "[start-end] SPEAKER: text" lines."""
        lines = []
        for seg in self.segments:
            start_str = _format_timestamp(seg.start)
            end_str = _format_timestamp(seg.end)
            lines.append(f"[{start_str}-{end_str}] {seg.speaker}: {seg.text.strip()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "source_file": str(self.source_file),
            "segments": [
                {
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                }
                for seg in self.segments
            ],
        }


def _format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS (or M:SS if under an hour)."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def transcribe_file(
    audio_path: str | Path,
    api_key: Optional[str] = None,
    model: str = DIARIZE_MODEL,
) -> TranscriptResult:
    """Transcribe a single audio file with speaker diarization.

    Args:
        audio_path: Path to the audio file (mp3, wav, m4a, webm, etc.).
        api_key: OpenAI API key. Falls back to the `OPENAI_API_KEY`
            environment variable if not provided.
        model: The diarization-capable transcription model to use.

    Returns:
        A TranscriptResult with speaker-labeled segments.

    Raises:
        TranscriptionError: If the file is missing, too large, an
            unsupported format, or the API call fails.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise TranscriptionError(
            "openai package is not installed. Install it with `pip install openai`."
        ) from exc

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise TranscriptionError(
            f"Unsupported audio format '{audio_path.suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    size = audio_path.stat().st_size
    if size > MAX_FILE_SIZE_BYTES:
        raise TranscriptionError(
            f"'{audio_path.name}' is {size / 1024 / 1024:.1f}MB, which exceeds "
            f"OpenAI's {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB limit for this "
            "endpoint. Split the file into smaller chunks first."
        )

    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
            )
    except Exception as exc:  # openai raises its own exception hierarchy
        raise TranscriptionError(
            f"Failed to transcribe '{audio_path.name}': {exc}"
        ) from exc

    segments = [
        DiarizedSegment(
            speaker=getattr(seg, "speaker", "UNKNOWN"),
            text=getattr(seg, "text", ""),
            start=getattr(seg, "start", 0.0),
            end=getattr(seg, "end", 0.0),
        )
        for seg in getattr(response, "segments", []) or []
    ]

    return TranscriptResult(source_file=audio_path, segments=segments)


def transcribe_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    api_key: Optional[str] = None,
    model: str = DIARIZE_MODEL,
    output_format: str = "txt",
) -> list[TranscriptResult]:
    """Transcribe every supported audio file in a directory.

    Writes one transcript file per input audio file into `output_dir`,
    named after the source file (e.g. `clip.mp3` -> `clip.txt`).

    Args:
        input_dir: Directory containing audio clips to transcribe.
        output_dir: Directory to write one transcript file per clip into.
        api_key: OpenAI API key. Falls back to `OPENAI_API_KEY` env var.
        model: The diarization-capable transcription model to use.
        output_format: "txt" (speaker-labeled lines) or "json" (full
            structured segments).

    Returns:
        A list of TranscriptResult, one per successfully transcribed file.
    """
    if output_format not in ("txt", "json"):
        raise TranscriptionError(f"Unsupported output_format: {output_format}")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not audio_files:
        logger.warning("No supported audio files found in %s", input_dir)

    results = []
    for audio_path in audio_files:
        logger.info("Transcribing %s ...", audio_path.name)
        try:
            result = transcribe_file(audio_path, api_key=api_key, model=model)
        except TranscriptionError as exc:
            logger.error("Skipping %s: %s", audio_path.name, exc)
            continue

        out_path = output_dir / f"{audio_path.stem}.{output_format}"
        if output_format == "json":
            out_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            out_path.write_text(result.to_labeled_text(), encoding="utf-8")

        logger.info("Wrote transcript: %s", out_path)
        results.append(result)

    return results
