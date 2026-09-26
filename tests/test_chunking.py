"""Tests for ingestion.chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.chunking import (
    LONG_TURN_THRESHOLD_SEC,
    MAX_SPLIT_CHUNK_SEC,
    MIN_SPLIT_CHUNK_SEC,
    SHORT_TURN_MAX_WORDS,
    chunk_transcript,
)


def write_transcript(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "clip.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# Rule 1: base case
# ---------------------------------------------------------------------
def test_base_case_one_line_one_chunk(tmp_path: Path):
    path = write_transcript(
        tmp_path,
        [
            "[0:00-0:10] Speaker 0: This is a short normal turn of speech here.",
        ],
    )
    chunks, summary = chunk_transcript(path, clip_file="clips/x.mp3", clip_offset_sec=100.0)

    assert len(chunks) == 1
    assert summary.from_base == 1
    assert summary.from_split == 0
    c = chunks[0]
    assert c.file == "clips/x.mp3"
    assert c.speaker == "Speaker 0"
    assert c.start_sec == 100.0
    assert c.end_sec == 110.0
    assert c.text == "This is a short normal turn of speech here."
    assert c.source == "base"


# ---------------------------------------------------------------------
# Rule 2: long-turn splitting
# ---------------------------------------------------------------------
def test_long_turn_gets_split_at_sentence_boundaries(tmp_path: Path):
    assert LONG_TURN_THRESHOLD_SEC < 200  # sanity check on the fixture below
    long_text = " ".join([f"This is sentence number {i}." for i in range(1, 21)])
    path = write_transcript(tmp_path, [f"[0:00-3:20] Speaker 0: {long_text}"])  # 200s

    chunks, summary = chunk_transcript(path, clip_file="clips/x.mp3", clip_offset_sec=0.0)

    assert summary.from_split > 1
    assert len(chunks) == summary.from_split
    for c in chunks:
        assert c.source == "split"
        assert c.speaker == "Speaker 0"
        duration = c.end_sec - c.start_sec
        # Allow slack since these are proportional character-based estimates.
        assert duration <= MAX_SPLIT_CHUNK_SEC + 15
    # Chunks should be in chronological order and roughly tile the span.
    assert chunks[0].start_sec == pytest.approx(0.0, abs=1.0)
    assert chunks[-1].end_sec == pytest.approx(200.0, abs=1.0)


def test_split_chunks_have_contextual_embedding_input_but_clean_text(tmp_path: Path):
    long_text = " ".join([f"Sentence {i} has some unique content in it." for i in range(1, 21)])
    path = write_transcript(tmp_path, [f"[0:00-3:00] Speaker 0: {long_text}"])

    chunks, _ = chunk_transcript(path, clip_file="clips/x.mp3")

    assert len(chunks) >= 2
    # Second piece onward should have embedding_input that includes a bit
    # of the previous piece's last sentence, but `text` must stay clean
    # (only this chunk's own words).
    second = chunks[1]
    assert second.text != second.embedding_input
    assert second.text in second.embedding_input
    assert not second.embedding_input.startswith(second.text)


# ---------------------------------------------------------------------
# Rule 3: short-turn merge/drop
# ---------------------------------------------------------------------
def test_short_turn_merges_into_following_same_speaker_chunk(tmp_path: Path):
    path = write_transcript(
        tmp_path,
        [
            "[0:00-0:01] Speaker 0: Yeah.",
            "[0:01-0:15] Speaker 0: And that is the actual point I wanted to make today.",
        ],
    )
    chunks, summary = chunk_transcript(path, clip_file="clips/x.mp3")

    assert len(chunks) == 1
    assert summary.short_turns_merged == 1
    assert summary.short_turns_dropped == 0
    assert chunks[0].text.startswith("Yeah.")
    assert chunks[0].start_sec == 0.0


def test_short_turn_merges_into_preceding_same_speaker_chunk_when_no_next(tmp_path: Path):
    path = write_transcript(
        tmp_path,
        [
            "[0:00-0:15] Speaker 0: Here is a full sentence establishing some content.",
            "[0:15-0:16] Speaker 0: Right.",
        ],
    )
    chunks, summary = chunk_transcript(path, clip_file="clips/x.mp3")

    assert len(chunks) == 1
    assert summary.short_turns_merged == 1
    assert chunks[0].text.endswith("Right.")
    assert chunks[0].end_sec == 0.0 + 16


def test_short_turn_with_no_same_speaker_neighbor_is_dropped(tmp_path: Path):
    path = write_transcript(
        tmp_path,
        [
            "[0:00-0:15] Speaker 0: Here is a full sentence establishing some content.",
            "[0:15-0:16] Speaker 1: Mhmm.",
            "[0:16-0:30] Speaker 0: And here is another full sentence from speaker zero.",
        ],
    )
    chunks, summary = chunk_transcript(path, clip_file="clips/x.mp3")

    # The lone backchannel from Speaker 1 has no adjacent Speaker 1 chunk,
    # so it must be dropped, not merged into a different speaker's chunk.
    assert summary.short_turns_dropped == 1
    assert summary.short_turns_merged == 0
    assert all(c.speaker != "Speaker 1" for c in chunks)
    assert len(chunks) == 2


def test_short_turn_never_merges_across_different_speakers(tmp_path: Path):
    path = write_transcript(
        tmp_path,
        [
            "[0:00-0:15] Speaker 0: Here is a full sentence establishing some content.",
            "[0:15-0:16] Speaker 1: Okay.",
        ],
    )
    chunks, summary = chunk_transcript(path, clip_file="clips/x.mp3")

    assert summary.short_turns_dropped == 1
    assert len(chunks) == 1
    assert chunks[0].speaker == "Speaker 0"
    assert "Okay" not in chunks[0].text


# ---------------------------------------------------------------------
# Rule 4: every chunk maps to exactly one speaker + carries its own timing
# ---------------------------------------------------------------------
def test_every_chunk_has_required_fields(tmp_path: Path):
    path = write_transcript(
        tmp_path,
        [
            "[0:00-0:10] Speaker 0: A normal length turn of speech right here.",
            "[0:10-0:20] Speaker 1: Another normal length turn of speech right here.",
        ],
    )
    chunks, _ = chunk_transcript(path, clip_file="clips/x.mp3", clip_offset_sec=50.0)

    for c in chunks:
        assert c.file == "clips/x.mp3"
        assert c.speaker in ("Speaker 0", "Speaker 1")
        assert c.start_sec >= 50.0
        assert c.end_sec > c.start_sec
        assert c.text
        assert c.embedding_input


# ---------------------------------------------------------------------
# Rule 5: summary stats
# ---------------------------------------------------------------------
def test_summary_reports_duration_stats(tmp_path: Path):
    path = write_transcript(
        tmp_path,
        [
            "[0:00-0:10] Speaker 0: A normal length turn of speech right here.",
            "[0:10-0:30] Speaker 1: Another normal length turn of speech right here now.",
        ],
    )
    _, summary = chunk_transcript(path, clip_file="clips/x.mp3")

    assert summary.total_chunks == 2
    assert summary.min_duration == pytest.approx(10.0)
    assert summary.max_duration == pytest.approx(20.0)
    assert summary.mean_duration == pytest.approx(15.0)
    assert "total_chunks=2" in summary.report()
