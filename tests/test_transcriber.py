"""Tests for transcription.transcriber.

These tests mock the openai client so they run offline without hitting
OpenAI's API.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from transcription.exceptions import TranscriptionError
from transcription.transcriber import (
    MAX_FILE_SIZE_BYTES,
    transcribe_directory,
    transcribe_file,
)


def _fake_segment(speaker: str, text: str, start: float, end: float):
    return SimpleNamespace(speaker=speaker, text=text, start=start, end=end)


def _make_fake_openai_module(segments):
    fake_response = SimpleNamespace(segments=segments)
    fake_client_instance = MagicMock()
    fake_client_instance.audio.transcriptions.create.return_value = fake_response

    fake_module = MagicMock()
    fake_module.OpenAI.return_value = fake_client_instance
    return fake_module, fake_client_instance


def test_transcribe_file_success(tmp_path: Path):
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"fake audio data")

    segments = [
        _fake_segment("Speaker A", "Hello there.", 0.0, 2.5),
        _fake_segment("Speaker B", "Hi, how are you?", 2.5, 5.0),
    ]
    fake_openai, fake_client = _make_fake_openai_module(segments)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        result = transcribe_file(audio_path, api_key="fake-key")

    assert result.source_file == audio_path
    assert len(result.segments) == 2
    assert result.segments[0].speaker == "Speaker A"
    assert result.full_text == "Hello there. Hi, how are you?"
    labeled = result.to_labeled_text()
    assert "Speaker A: Hello there." in labeled
    assert "Speaker B: Hi, how are you?" in labeled
    fake_client.audio.transcriptions.create.assert_called_once()
    _, kwargs = fake_client.audio.transcriptions.create.call_args
    assert kwargs["model"] == "gpt-4o-transcribe-diarize"
    assert kwargs["response_format"] == "diarized_json"


def test_transcribe_file_missing(tmp_path: Path):
    with pytest.raises(TranscriptionError):
        transcribe_file(tmp_path / "does_not_exist.mp3")


def test_transcribe_file_unsupported_format(tmp_path: Path):
    bad_path = tmp_path / "clip.txt"
    bad_path.write_text("not audio")
    with pytest.raises(TranscriptionError):
        transcribe_file(bad_path)


def test_transcribe_file_too_large(tmp_path: Path):
    audio_path = tmp_path / "big.mp3"
    audio_path.write_bytes(b"\0" * (MAX_FILE_SIZE_BYTES + 1))
    with pytest.raises(TranscriptionError, match="exceeds"):
        transcribe_file(audio_path)


def test_transcribe_directory_writes_one_file_per_clip(tmp_path: Path):
    input_dir = tmp_path / "clips"
    input_dir.mkdir()
    output_dir = tmp_path / "transcripts"

    (input_dir / "a.mp3").write_bytes(b"fake")
    (input_dir / "b.wav").write_bytes(b"fake")
    (input_dir / "ignore.txt").write_text("not audio")

    segments = [_fake_segment("Speaker A", "Some text.", 0.0, 1.0)]
    fake_openai, _ = _make_fake_openai_module(segments)

    with patch.dict("sys.modules", {"openai": fake_openai}):
        results = transcribe_directory(input_dir, output_dir, api_key="fake-key")

    assert len(results) == 2
    assert (output_dir / "a.txt").exists()
    assert (output_dir / "b.txt").exists()
    assert not (output_dir / "ignore.txt").exists()
    assert "Speaker A: Some text." in (output_dir / "a.txt").read_text(encoding="utf-8")
