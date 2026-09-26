"""Diarized transcription of audio files using OpenAI's transcription API."""

from .exceptions import TranscriptionError
from .transcriber import (
    DiarizedSegment,
    TranscriptResult,
    transcribe_file,
    transcribe_directory,
)

__all__ = [
    "transcribe_file",
    "transcribe_directory",
    "TranscriptResult",
    "DiarizedSegment",
    "TranscriptionError",
]
