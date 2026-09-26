"""Extract audio tracks from YouTube videos."""

from .extractor import AudioResult, extract_audio
from .exceptions import AudioExtractionError

__all__ = ["extract_audio", "AudioResult", "AudioExtractionError"]
