"""Tests for youtube_audio.extractor.

These tests mock yt_dlp so they run offline without hitting YouTube.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from youtube_audio.exceptions import AudioExtractionError
from youtube_audio.extractor import extract_audio


class FakeDownloadError(Exception):
    """Stand-in for yt_dlp.utils.DownloadError."""


def _make_fake_yt_dlp(info: dict, prepared_filename: str, raise_error: bool = False):
    fake_module = MagicMock()
    fake_module.utils.DownloadError = FakeDownloadError

    fake_ydl_instance = MagicMock()
    if raise_error:
        fake_ydl_instance.extract_info.side_effect = FakeDownloadError("boom")
    else:
        fake_ydl_instance.extract_info.return_value = info
    fake_ydl_instance.prepare_filename.return_value = prepared_filename

    fake_ydl_cls = MagicMock()
    fake_ydl_cls.return_value.__enter__.return_value = fake_ydl_instance
    fake_module.YoutubeDL = fake_ydl_cls

    return fake_module


def test_extract_audio_success(tmp_path: Path):
    output_dir = tmp_path / "downloads"
    expected_file = output_dir / "My Video.mp3"

    info = {"title": "My Video", "duration": 123.4}
    fake_yt_dlp = _make_fake_yt_dlp(
        info=info,
        prepared_filename=str(output_dir / "My Video.webm"),
    )

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp}):
        # Ensure the "downloaded" file exists so extract_audio finds it.
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_file.touch()

        result = extract_audio(
            url="https://www.youtube.com/watch?v=abc123",
            output_dir=output_dir,
            audio_format="mp3",
        )

    assert result.file_path == expected_file
    assert result.title == "My Video"
    assert result.duration_seconds == 123.4
    assert result.audio_format == "mp3"


def test_extract_audio_download_error(tmp_path: Path):
    fake_yt_dlp = _make_fake_yt_dlp(info={}, prepared_filename="", raise_error=True)

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp}):
        with pytest.raises(AudioExtractionError):
            extract_audio(
                url="https://www.youtube.com/watch?v=bad",
                output_dir=tmp_path / "downloads",
            )


def test_extract_audio_missing_output_file(tmp_path: Path):
    output_dir = tmp_path / "downloads"
    info = {"title": "Ghost Video", "duration": 10.0}
    fake_yt_dlp = _make_fake_yt_dlp(
        info=info,
        prepared_filename=str(output_dir / "Ghost Video.webm"),
    )

    with patch.dict("sys.modules", {"yt_dlp": fake_yt_dlp}):
        # Do NOT create the expected output file.
        with pytest.raises(AudioExtractionError):
            extract_audio(
                url="https://www.youtube.com/watch?v=ghost",
                output_dir=output_dir,
                audio_format="mp3",
            )
