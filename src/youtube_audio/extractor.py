"""Core logic for extracting audio from YouTube videos.

Uses yt-dlp to download the best available audio stream from a YouTube
URL and (optionally) convert it to a target audio format via ffmpeg.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .exceptions import AudioExtractionError

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_AUDIO_QUALITY = "192"  # kbps, only relevant for lossy formats like mp3

_TIME_RANGE_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?|\d{1,2}:\d{2}(?::\d{2})?)"
    r"\s*-\s*"
    r"(\d+(?:\.\d+)?|\d{1,2}:\d{2}(?::\d{2})?|inf|end)?\s*$"
)


@dataclass
class AudioResult:
    """Metadata about a successfully extracted audio file."""

    file_path: Path
    title: str
    duration_seconds: Optional[float]
    source_url: str
    audio_format: str


def _parse_time_range(time_range: str) -> tuple[str, str]:
    """Validate and split a "START-END" time range string.

    Accepts seconds (e.g. "90") or "H:MM:SS"/"M:SS" (e.g. "1:30"). "-" or
    "end"/"inf" as END means "to the end of the video".
    """
    match = _TIME_RANGE_RE.match(time_range)
    if not match:
        raise AudioExtractionError(
            f"Invalid time_range '{time_range}'. Expected format like "
            "'1:30-4:00', '90-240', or '1:30-' (to end of video)."
        )
    start, end = match.group(1), match.group(2)
    if not end or end == "end":
        end = "inf"
    return start, end


def extract_audio(
    url: str,
    output_dir: str | Path = "downloads",
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    audio_quality: str = DEFAULT_AUDIO_QUALITY,
    filename_template: str = "%(title)s.%(ext)s",
    time_range: Optional[str] = None,
    chapter: Optional[str] = None,
) -> AudioResult:
    """Download a YouTube video and extract its audio track.

    Args:
        url: The YouTube video URL (or any URL yt-dlp supports).
        output_dir: Directory to save the extracted audio file into.
        audio_format: Target audio format/codec, e.g. "mp3", "wav", "m4a".
        audio_quality: Target bitrate in kbps for lossy formats (e.g. "192").
        filename_template: yt-dlp output filename template.
        time_range: Optional clip range as "START-END", e.g. "1:30-4:00" or
            "90-240" (seconds). Use "-" as END for "to the end of the video".
            Mutually exclusive with `chapter`.
        chapter: Optional chapter title regex to extract, matched against
            the video's chapter markers (yt-dlp's `--download-sections`
            chapter mode), e.g. "Intro" or "Chapter 3". All chapters whose
            title matches the regex are extracted. Mutually exclusive with
            `time_range`.

    Returns:
        An AudioResult describing the downloaded file.

    Raises:
        AudioExtractionError: If the download or conversion fails, or if
            both/neither of `time_range`/`chapter` semantics are misused.
    """
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise AudioExtractionError(
            "yt-dlp is not installed. Install it with `pip install yt-dlp`."
        ) from exc

    if time_range and chapter:
        raise AudioExtractionError("Specify only one of `time_range` or `chapter`, not both.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / filename_template),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": audio_quality,
            }
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    if time_range:
        start, end = _parse_time_range(time_range)
        section = f"*{start}-{end}"
    elif chapter:
        # No "*" prefix here — "*" denotes a time-range section; a bare
        # regex denotes a chapter-title match.
        section = chapter
    else:
        section = None

    if section:
        # yt_dlp's Python API expects `download_ranges` to be a callable
        # (built via yt_dlp.utils.download_range_func), NOT the raw
        # "--download-sections" CLI string. Passing the string directly
        # under the wrong key ("download_sections") is silently ignored
        # and the full video gets downloaded.
        from yt_dlp.utils import download_range_func

        chapter_regexes, time_ranges = [], []
        # download_range_func's `ranges` are (start, end) tuples for "*"
        # prefixed sections; anything else is treated as a chapter regex.
        if section.startswith("*"):
            start_str, _, end_str = section[1:].partition("-")
            time_ranges.append(
                (_to_seconds(start_str), None if end_str == "inf" else _to_seconds(end_str))
            )
        else:
            chapter_regexes.append(section)

        ydl_opts["download_ranges"] = download_range_func(chapter_regexes, time_ranges)
        ydl_opts["force_keyframes_at_cuts"] = True

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            final_path = Path(ydl.prepare_filename(info))
            # After post-processing, extension changes to the target format.
            final_path = final_path.with_suffix(f".{audio_format}")
    except yt_dlp.utils.DownloadError as exc:
        raise AudioExtractionError(f"Failed to extract audio from '{url}': {exc}") from exc

    if not final_path.exists():
        raise AudioExtractionError(
            f"Expected output file not found after extraction: {final_path}"
        )

    duration = info.get("duration")
    if time_range:
        start, end = _parse_time_range(time_range)
        start_s = _to_seconds(start)
        end_s = duration if end == "inf" else _to_seconds(end)
        if end_s is not None:
            duration = end_s - start_s
    elif chapter and info.get("requested_downloads"):
        # yt-dlp trims via ffmpeg per matched chapter section(s); fall back
        # to summing matched chapters' durations if available in info.
        matched = [
            c["end_time"] - c["start_time"]
            for c in info.get("chapters") or []
            if re.search(chapter, c.get("title", ""))
        ]
        if matched:
            duration = sum(matched)

    return AudioResult(
        file_path=final_path,
        title=info.get("title", "unknown"),
        duration_seconds=duration,
        source_url=url,
        audio_format=audio_format,
    )


def _to_seconds(value: str) -> float:
    """Convert a "H:MM:SS"/"M:SS" or plain-seconds string to float seconds."""
    if ":" not in value:
        return float(value)
    parts = [float(p) for p in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds
