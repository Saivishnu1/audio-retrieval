"""Turn diarized transcript segments into retrieval-ready chunks.

Rules: one turn = one chunk by default; turns over
LONG_TURN_THRESHOLD_SEC split at sentence boundaries into
[MIN_SPLIT_CHUNK_SEC, MAX_SPLIT_CHUNK_SEC] pieces (timing estimated
proportionally by character position, since transcripts only carry
paragraph-level timestamps); turns under SHORT_TURN_MAX_WORDS merge into
an adjacent same-speaker chunk or are dropped. Each chunk carries `text`
(own words, for display/keyword search) separately from
`embedding_input` (text plus preceding-turn context, for the embedder only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

LONG_TURN_THRESHOLD_SEC = 75.0  # lines longer than this get split
MIN_SPLIT_CHUNK_SEC = 20.0
MAX_SPLIT_CHUNK_SEC = 60.0
SHORT_TURN_MAX_WORDS = 5
CONTEXT_MAX_CHARS = 300  # cap on how much preceding-turn text is prepended

SEGMENT_RE = re.compile(
    r"^\[(?P<start>\d+(?::\d{2}){0,2})-(?P<end>\d+(?::\d{2}){0,2})\]\s*"
    r"(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)
# Splits on sentence-ending punctuation while keeping it attached to the
# preceding sentence.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class RawSegment:
    start: float
    end: float
    speaker: str
    text: str


@dataclass
class Chunk:
    file: str
    speaker: str
    start_sec: float
    end_sec: float
    text: str
    embedding_input: str
    source: str  # "base" | "split" | "merged"


@dataclass
class ChunkingSummary:
    total_chunks: int = 0
    from_base: int = 0
    from_split: int = 0
    from_merge: int = 0
    short_turns_merged: int = 0
    short_turns_dropped: int = 0
    durations: list[float] = field(default_factory=list)

    @property
    def min_duration(self) -> float:
        return min(self.durations) if self.durations else 0.0

    @property
    def max_duration(self) -> float:
        return max(self.durations) if self.durations else 0.0

    @property
    def mean_duration(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 0.0

    def report(self) -> str:
        return (
            f"Chunking summary:\n"
            f"  total_chunks={self.total_chunks}\n"
            f"  from_base={self.from_base} from_split={self.from_split} "
            f"from_merge={self.from_merge}\n"
            f"  short_turns_merged={self.short_turns_merged} "
            f"short_turns_dropped={self.short_turns_dropped}\n"
            f"  duration_sec: min={self.min_duration:.1f} "
            f"max={self.max_duration:.1f} mean={self.mean_duration:.1f}"
        )


def parse_timestamp(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def parse_transcript_file(path: Path) -> list[RawSegment]:
    segments: list[RawSegment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = SEGMENT_RE.match(line)
        if not match:
            continue
        segments.append(
            RawSegment(
                start=parse_timestamp(match.group("start")),
                end=parse_timestamp(match.group("end")),
                speaker=match.group("speaker").strip(),
                text=match.group("text").strip(),
            )
        )
    return segments


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts or [text]


def _split_long_segment(seg: RawSegment, clip_offset: float) -> list[Chunk]:
    sentences = _split_sentences(seg.text)
    duration = seg.end - seg.start
    total_chars = sum(len(s) for s in sentences) or 1

    pieces: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for sentence in sentences:
        current.append(sentence)
        current_chars += len(sentence)
        est_duration_so_far = (current_chars / total_chars) * duration
        if est_duration_so_far >= MAX_SPLIT_CHUNK_SEC:
            pieces.append(current)
            current = []
            current_chars = 0
    if current:
        pieces.append(current)
    if len(pieces) >= 2:
        last_chars = sum(len(s) for s in pieces[-1])
        est_last_duration = (last_chars / total_chars) * duration
        if est_last_duration < MIN_SPLIT_CHUNK_SEC:
            pieces[-2].extend(pieces[-1])
            pieces.pop()

    chunks: list[Chunk] = []
    chars_consumed = 0
    prev_piece_last_sentence = ""
    for piece_sentences in pieces:
        piece_text = " ".join(piece_sentences)
        piece_chars = sum(len(s) for s in piece_sentences)

        piece_start_frac = chars_consumed / total_chars
        piece_end_frac = (chars_consumed + piece_chars) / total_chars
        piece_start = seg.start + piece_start_frac * duration
        piece_end = seg.start + piece_end_frac * duration
        chars_consumed += piece_chars

        if prev_piece_last_sentence:
            embed_input = f"{prev_piece_last_sentence} {piece_text}"[-CONTEXT_MAX_CHARS:]
        else:
            embed_input = piece_text

        chunks.append(
            Chunk(
                file="",  # filled in by caller
                speaker=seg.speaker,
                start_sec=clip_offset + piece_start,
                end_sec=clip_offset + piece_end,
                text=piece_text,
                embedding_input=embed_input,
                source="split",
            )
        )
        prev_piece_last_sentence = piece_sentences[-1]

    return chunks


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _prior_context_text(prior_text: str) -> str:
    sentences = _split_sentences(prior_text)
    context = " ".join(sentences[-2:])
    return context[-CONTEXT_MAX_CHARS:]


def chunk_transcript(
    transcript_path: str | Path,
    clip_file: str,
    clip_offset_sec: float = 0.0,
) -> tuple[list[Chunk], ChunkingSummary]:
    segments = parse_transcript_file(Path(transcript_path))
    summary = ChunkingSummary()

    pending: list[list[Chunk] | None] = [None] * len(segments)

    for i, seg in enumerate(segments):
        duration = seg.end - seg.start
        word_count = _word_count(seg.text)

        if word_count <= SHORT_TURN_MAX_WORDS and duration < LONG_TURN_THRESHOLD_SEC:
            pending[i] = None  # handled in pass 2 (merge/drop)
            continue

        if duration > LONG_TURN_THRESHOLD_SEC:
            split_chunks = _split_long_segment(seg, clip_offset_sec)
            for c in split_chunks:
                c.file = clip_file
            pending[i] = split_chunks
            summary.from_split += len(split_chunks)
            continue

        # Base case.
        prior_text = segments[i - 1].text if i > 0 else ""
        embed_input = seg.text
        if prior_text:
            ctx = _prior_context_text(prior_text)
            embed_input = f"{ctx} {seg.text}"[-CONTEXT_MAX_CHARS - len(seg.text) :]
        pending[i] = [
            Chunk(
                file=clip_file,
                speaker=seg.speaker,
                start_sec=clip_offset_sec + seg.start,
                end_sec=clip_offset_sec + seg.end,
                text=seg.text,
                embedding_input=embed_input,
                source="base",
            )
        ]
        summary.from_base += 1

    for i, seg in enumerate(segments):
        if pending[i] is not None:
            continue

        merged = False

        if i + 1 < len(segments) and pending[i + 1] and segments[i + 1].speaker == seg.speaker:
            target = pending[i + 1][0]
            target.text = f"{seg.text} {target.text}".strip()
            target.embedding_input = f"{seg.text} {target.embedding_input}".strip()
            target.start_sec = min(target.start_sec, clip_offset_sec + seg.start)
            target.source = "merged"
            merged = True
        elif i - 1 >= 0 and pending[i - 1] and segments[i - 1].speaker == seg.speaker:
            target = pending[i - 1][-1]
            target.text = f"{target.text} {seg.text}".strip()
            target.embedding_input = f"{target.embedding_input} {seg.text}".strip()
            target.end_sec = max(target.end_sec, clip_offset_sec + seg.end)
            target.source = "merged"
            merged = True

        if merged:
            summary.short_turns_merged += 1
        else:
            summary.short_turns_dropped += 1

    chunks: list[Chunk] = []
    for group in pending:
        if group:
            chunks.extend(group)

    summary.from_merge = sum(1 for c in chunks if c.source == "merged")
    summary.total_chunks = len(chunks)
    summary.durations = [c.end_sec - c.start_sec for c in chunks]

    return chunks, summary
