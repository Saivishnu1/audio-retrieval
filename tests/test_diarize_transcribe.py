"""Tests for transcription.diarize_transcribe.

Mocks the Deepgram SDK so these run offline without hitting the API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "transcription"))
import diarize_transcribe as dt  # noqa: E402


def _paragraph(speaker, sentences, start, end):
    return SimpleNamespace(
        speaker=speaker,
        sentences=[SimpleNamespace(text=s) for s in sentences],
        start=start,
        end=end,
    )


def test_format_timestamp_under_an_hour():
    assert dt.format_timestamp(0) == "0:00"
    assert dt.format_timestamp(65) == "1:05"
    assert dt.format_timestamp(3599) == "59:59"


def test_format_timestamp_over_an_hour():
    assert dt.format_timestamp(3600) == "1:00:00"
    assert dt.format_timestamp(3725) == "1:02:05"


def test_segments_from_paragraphs_joins_sentences_and_labels_speaker():
    paragraphs = [
        _paragraph(0, ["Hello.", "How are you?"], 0.0, 2.5),
        _paragraph(1, ["I'm good."], 2.5, 4.0),
    ]

    segments = dt._segments_from_paragraphs(paragraphs)

    assert segments == [
        {"speaker": "Speaker 0", "text": "Hello. How are you?", "start": 0.0, "end": 2.5},
        {"speaker": "Speaker 1", "text": "I'm good.", "start": 2.5, "end": 4.0},
    ]


def test_segments_from_paragraphs_skips_empty_text():
    paragraphs = [
        _paragraph(0, ["  ", ""], 0.0, 1.0),
        _paragraph(1, ["Real content."], 1.0, 2.0),
    ]

    segments = dt._segments_from_paragraphs(paragraphs)

    assert len(segments) == 1
    assert segments[0]["text"] == "Real content."


def test_segments_from_paragraphs_defaults_missing_speaker_to_zero():
    paragraphs = [_paragraph(None, ["Unlabeled."], 0.0, 1.0)]

    segments = dt._segments_from_paragraphs(paragraphs)

    assert segments[0]["speaker"] == "Speaker 0"


def _fake_deepgram_response(paragraphs):
    alternative = SimpleNamespace(paragraphs=SimpleNamespace(paragraphs=paragraphs))
    channel = SimpleNamespace(alternatives=[alternative])
    return SimpleNamespace(results=SimpleNamespace(channels=[channel]))


def test_transcribe_and_diarize_node_success(tmp_path: Path):
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"fake audio bytes")

    response = _fake_deepgram_response([_paragraph(0, ["Test."], 0.0, 1.0)])
    fake_client = MagicMock()
    fake_client.listen.v1.media.transcribe_file.return_value = response

    fake_deepgram_module = MagicMock()
    fake_deepgram_module.DeepgramClient.return_value = fake_client

    with patch.dict("sys.modules", {"deepgram": fake_deepgram_module}):
        state = dt.transcribe_and_diarize_node(
            {"audio_path": str(audio_path), "api_key": "fake-key"}
        )

    assert "error" not in state
    assert state["labeled_segments"] == [
        {"speaker": "Speaker 0", "text": "Test.", "start": 0.0, "end": 1.0}
    ]


def test_transcribe_and_diarize_node_request_failure(tmp_path: Path):
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"fake audio bytes")

    fake_client = MagicMock()
    fake_client.listen.v1.media.transcribe_file.side_effect = RuntimeError("boom")

    fake_deepgram_module = MagicMock()
    fake_deepgram_module.DeepgramClient.return_value = fake_client

    with patch.dict("sys.modules", {"deepgram": fake_deepgram_module}):
        state = dt.transcribe_and_diarize_node(
            {"audio_path": str(audio_path), "api_key": "fake-key"}
        )

    assert "error" in state
    assert "boom" in state["error"]


def test_transcribe_and_diarize_node_skips_if_already_errored():
    state = dt.transcribe_and_diarize_node({"error": "earlier failure"})
    assert state == {"error": "earlier failure"}


def test_write_node_renders_labeled_lines(tmp_path: Path):
    output_dir = tmp_path / "transcripts"
    state = {
        "audio_path": "clips/example.mp3",
        "output_dir": str(output_dir),
        "labeled_segments": [
            {"speaker": "Speaker 0", "text": "Hi.", "start": 0.0, "end": 1.5},
            {"speaker": "Speaker 1", "text": "Hello back.", "start": 1.5, "end": 3.0},
        ],
    }

    result = dt.write_node(state)

    out_path = output_dir / "example.txt"
    assert result["output_path"] == str(out_path)
    assert out_path.read_text(encoding="utf-8") == (
        "[0:00-0:02] Speaker 0: Hi.\n[0:02-0:03] Speaker 1: Hello back."
    )
