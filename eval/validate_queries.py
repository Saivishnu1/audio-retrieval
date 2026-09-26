"""Validate eval/golden_queries.yaml against the actual transcripts.

Orchestrated with LangGraph, following the same structure as
`transcription/diarize_transcribe.py`: a `GraphState` TypedDict threaded
through nodes, a `build_graph()` that wires them up, and a `main()` that
drives one graph run (this script only ever needs one run, unlike the
per-file loop in diarize_transcribe.py).

Pipeline:

    load  ->  check_spans  ->  check_paraphrase_overlap  ->  check_leakage  ->  report

    - load: reads manifest.yaml and golden_queries.yaml, parses every
      referenced transcript into timestamped segments.
    - check_spans: every query's (file, start_sec, end_sec) must fall
      inside the clip's real range in the manifest AND overlap an actual
      transcript segment (catches typo'd timestamps pointing at nothing).
    - check_paraphrase_overlap: every `paraphrase`-bucket query must share
      less than PARAPHRASE_MAX_OVERLAP of its content words with the
      matched span's transcript text -- otherwise it's just a copy, which
      would make the retrieval eval trivially easy. `exact_keyword`
      queries are checked the opposite way (should largely match).
    - check_leakage: flags (warning only) a query whose distinctive words
      also heavily overlap a DIFFERENT file's transcript, which would
      make its "relevant" label ambiguous.
    - report: prints bucket counts, warnings, and errors; sets the
      process exit code.

Usage:
    python eval/validate_queries.py
    python eval/validate_queries.py --queries eval/golden_queries.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, TypedDict

import yaml

PARAPHRASE_MAX_OVERLAP = 0.30  # fraction of query content words allowed to
                                # also appear in the source span's text
EXACT_KEYWORD_MIN_OVERLAP = 0.60  # fallback threshold when not verbatim
LEAKAGE_MIN_OVERLAP = 0.80  # cross-file word-overlap that triggers a warning
MIN_LEAKAGE_CHECK_WORDS = 3  # skip the leakage check for very short queries

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
    "to", "for", "with", "from", "by", "is", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "not", "no", "so", "that",
    "this", "these", "those", "it", "its", "as", "into", "than", "then",
    "you", "your", "i", "we", "they", "he", "she", "his", "her", "their",
    "what", "why", "how", "when", "where", "who", "which", "can", "could",
    "would", "should", "will", "just", "about", "there", "here", "up",
    "out", "over", "all", "some", "any", "more", "most", "much", "many",
    "i'm", "im", "my", "mine", "one", "only", "me", "myself", "am",
}

SEGMENT_RE = re.compile(
    r"^\[(?P<start>\d+(?::\d{2}){0,2})-(?P<end>\d+(?::\d{2}){0,2})\]\s*"
    r"(?P<speaker>[^:]+):\s*(?P<text>.*)$"
)


# --------------------------------------------------------------------------
# Shared data types
# --------------------------------------------------------------------------
class Segment(TypedDict):
    start: float
    end: float
    speaker: str
    text: str


class GraphState(TypedDict, total=False):
    """State threaded through the LangGraph pipeline."""

    queries_path: str
    manifest_path: str
    manifest: dict
    queries: list[dict]
    # clip file path -> parsed transcript segments
    transcripts: dict[str, list[Segment]]
    # clip file path -> full lowercase transcript text (for leakage check)
    full_text: dict[str, str]
    bucket_counts: dict[str, dict[str, int]]
    errors: list[str]
    warnings: list[str]
    error: Optional[str]  # fatal error (e.g. files not found) -> short-circuits


# --------------------------------------------------------------------------
# Helpers (pure functions, no graph/state dependency)
# --------------------------------------------------------------------------
def parse_timestamp(ts: str) -> float:
    parts = [float(p) for p in ts.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def load_transcript_segments(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        match = SEGMENT_RE.match(line)
        if not match:
            continue
        segments.append(
            {
                "start": parse_timestamp(match.group("start")),
                "end": parse_timestamp(match.group("end")),
                "speaker": match.group("speaker").strip(),
                "text": match.group("text").strip(),
            }
        )
    return segments


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def overlap_fraction(query_words: set[str], source_words: set[str]) -> float:
    if not query_words:
        return 0.0
    return len(query_words & source_words) / len(query_words)


def find_manifest_clip(manifest: dict, clip_path: str) -> Optional[dict]:
    for clip in manifest.get("clips", []):
        if clip["file"] == clip_path:
            return clip
    return None


def segments_overlapping(
    segments: list[Segment], rel_start: float, rel_end: float
) -> list[Segment]:
    return [s for s in segments if s["start"] < rel_end and s["end"] > rel_start]


# --------------------------------------------------------------------------
# Node: load
# --------------------------------------------------------------------------
def load_node(state: GraphState) -> GraphState:
    queries_path = Path(state["queries_path"])
    manifest_path = Path(state["manifest_path"])

    if not manifest_path.exists():
        return {**state, "error": f"manifest not found: {manifest_path}"}
    if not queries_path.exists():
        return {**state, "error": f"queries file not found: {queries_path}"}

    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    with open(queries_path, encoding="utf-8") as f:
        query_data = yaml.safe_load(f)

    queries = (query_data or {}).get("queries", [])
    if not queries:
        return {**state, "error": f"no queries found in {queries_path}"}

    transcripts: dict[str, list[Segment]] = {}
    full_text: dict[str, str] = {}
    for clip in manifest.get("clips", []):
        clip_path = clip["file"]
        transcript_path = Path(clip["transcript"])
        segments = (
            load_transcript_segments(transcript_path)
            if transcript_path.exists()
            else []
        )
        transcripts[clip_path] = segments
        full_text[clip_path] = " ".join(s["text"].lower() for s in segments)

    print(f"Loaded {len(queries)} queries from {queries_path}")
    print(f"Loaded {len(transcripts)} clip transcripts from {manifest_path}\n")

    return {
        **state,
        "manifest": manifest,
        "queries": queries,
        "transcripts": transcripts,
        "full_text": full_text,
        "bucket_counts": {},
        "errors": [],
        "warnings": [],
    }


# --------------------------------------------------------------------------
# Node: check_spans
# --------------------------------------------------------------------------
def check_spans_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    manifest = state["manifest"]
    transcripts = state["transcripts"]
    errors = list(state["errors"])
    bucket_counts = {k: dict(v) for k, v in state["bucket_counts"].items()}

    for q in state["queries"]:
        qid = q.get("id", "<no id>")
        bucket = q.get("bucket")
        query_text = q.get("query", "")
        relevant = q.get("relevant", [])

        if bucket is None:
            errors.append(f"{qid}: missing 'bucket'")
        if not query_text:
            errors.append(f"{qid}: missing 'query' text")
        if not relevant:
            errors.append(f"{qid}: 'relevant' list is empty")
            continue

        for span in relevant:
            clip_path = span.get("file")
            start_sec = span.get("start_sec")
            end_sec = span.get("end_sec")

            file_bucket_counts = bucket_counts.setdefault(clip_path, {})
            file_bucket_counts[bucket] = file_bucket_counts.get(bucket, 0) + 1

            entry = find_manifest_clip(manifest, clip_path)
            if entry is None:
                errors.append(f"{qid}: '{clip_path}' not found in manifest.yaml")
                continue

            if start_sec is None or end_sec is None:
                errors.append(f"{qid}: span for {clip_path} missing start_sec/end_sec")
                continue
            if start_sec >= end_sec:
                errors.append(f"{qid}: span for {clip_path} has start_sec >= end_sec")
                continue
            if not (entry["start_sec"] <= start_sec <= entry["end_sec"]):
                errors.append(
                    f"{qid}: start_sec {start_sec} for {clip_path} is outside the "
                    f"clip's actual range [{entry['start_sec']}, {entry['end_sec']}]"
                )
            if not (entry["start_sec"] <= end_sec <= entry["end_sec"]):
                errors.append(
                    f"{qid}: end_sec {end_sec} for {clip_path} is outside the "
                    f"clip's actual range [{entry['start_sec']}, {entry['end_sec']}]"
                )

            clip_offset = float(entry["start_sec"])
            rel_start = start_sec - clip_offset
            rel_end = end_sec - clip_offset

            segments = transcripts.get(clip_path, [])
            if not segments:
                errors.append(f"{qid}: no transcript segments loaded for {clip_path}")
                continue

            overlapping = segments_overlapping(segments, rel_start, rel_end)
            if not overlapping:
                errors.append(
                    f"{qid}: span [{start_sec}, {end_sec}] (clip-relative "
                    f"[{rel_start:.1f}, {rel_end:.1f}]) for {clip_path} does not "
                    f"overlap any transcript segment"
                )

    return {**state, "errors": errors, "bucket_counts": bucket_counts}


# --------------------------------------------------------------------------
# Node: check_paraphrase_overlap (also checks exact_keyword the other way)
# --------------------------------------------------------------------------
def check_paraphrase_overlap_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    manifest = state["manifest"]
    transcripts = state["transcripts"]
    errors = list(state["errors"])
    warnings = list(state["warnings"])

    for q in state["queries"]:
        qid = q.get("id", "<no id>")
        bucket = q.get("bucket")
        query_text = q.get("query", "")

        for span in q.get("relevant", []):
            clip_path = span.get("file")
            start_sec, end_sec = span.get("start_sec"), span.get("end_sec")
            entry = find_manifest_clip(manifest, clip_path)
            if entry is None or start_sec is None or end_sec is None:
                continue  # already reported by check_spans_node

            clip_offset = float(entry["start_sec"])
            rel_start, rel_end = start_sec - clip_offset, end_sec - clip_offset
            segments = transcripts.get(clip_path, [])
            overlapping = segments_overlapping(segments, rel_start, rel_end)
            if not overlapping:
                continue  # already reported by check_spans_node

            source_text = " ".join(s["text"] for s in overlapping)

            if bucket == "paraphrase":
                q_words = content_words(query_text)
                src_words = content_words(source_text)
                frac = overlap_fraction(q_words, src_words)
                if frac >= PARAPHRASE_MAX_OVERLAP:
                    errors.append(
                        f"{qid}: paraphrase word-overlap {frac:.0%} >= "
                        f"{PARAPHRASE_MAX_OVERLAP:.0%} threshold with source "
                        f"span text -- too close to verbatim. "
                        f"Shared words: {sorted(q_words & src_words)}"
                    )

            if bucket == "exact_keyword":
                normalized_query = re.sub(r"[^a-z0-9 ]", "", query_text.lower())
                normalized_source = re.sub(r"[^a-z0-9 ]", "", source_text.lower())
                if normalized_query not in normalized_source:
                    q_words = content_words(query_text)
                    src_words = content_words(source_text)
                    frac = overlap_fraction(q_words, src_words)
                    if frac < EXACT_KEYWORD_MIN_OVERLAP:
                        warnings.append(
                            f"{qid}: exact_keyword query not found verbatim in "
                            f"source span, and word overlap is only {frac:.0%} "
                            f"-- double check the timestamp"
                        )

    return {**state, "errors": errors, "warnings": warnings}


# --------------------------------------------------------------------------
# Node: check_leakage
# --------------------------------------------------------------------------
def check_leakage_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    manifest = state["manifest"]
    full_text = state["full_text"]
    warnings = list(state["warnings"])

    for q in state["queries"]:
        qid = q.get("id", "<no id>")
        bucket = q.get("bucket")
        query_text = q.get("query", "")
        relevant = q.get("relevant", [])

        if bucket == "multi_span":
            continue  # deliberately spans multiple files

        relevant_files = {s.get("file") for s in relevant}
        q_words = content_words(query_text)
        if len(q_words) < MIN_LEAKAGE_CHECK_WORDS:
            continue

        for clip in manifest.get("clips", []):
            other_file = clip["file"]
            if other_file in relevant_files:
                continue
            other_words = content_words(full_text.get(other_file, ""))
            frac = overlap_fraction(q_words, other_words)
            if frac >= LEAKAGE_MIN_OVERLAP:
                warnings.append(
                    f"{qid}: query also has {frac:.0%} word-overlap with "
                    f"unrelated file {other_file} -- possible ambiguous label"
                )

    return {**state, "warnings": warnings}


# --------------------------------------------------------------------------
# Node: report
# --------------------------------------------------------------------------
def report_node(state: GraphState) -> GraphState:
    if state.get("error"):
        print(f"FATAL: {state['error']}")
        return state

    print("Per-file bucket counts:")
    for clip_path, counts in state["bucket_counts"].items():
        total = sum(counts.values())
        print(f"  {clip_path}: {counts} (total {total})")
    print()

    warnings = state["warnings"]
    errors = state["errors"]

    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
        print()

    if errors:
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print(f"\nFAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
    else:
        print(f"PASSED: 0 errors, {len(warnings)} warning(s)")

    return state


# --------------------------------------------------------------------------
# Build the graph
# --------------------------------------------------------------------------
def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("load", load_node)
    graph.add_node("check_spans", check_spans_node)
    graph.add_node("check_paraphrase_overlap", check_paraphrase_overlap_node)
    graph.add_node("check_leakage", check_leakage_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("load")
    graph.add_edge("load", "check_spans")
    graph.add_edge("check_spans", "check_paraphrase_overlap")
    graph.add_edge("check_paraphrase_overlap", "check_leakage")
    graph.add_edge("check_leakage", "report")
    graph.add_edge("report", END)

    return graph.compile()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries", default="eval/golden_queries.yaml", help="Path to the golden query set"
    )
    parser.add_argument("--manifest", default="manifest.yaml", help="Path to manifest.yaml")
    args = parser.parse_args()

    app = build_graph()
    final_state = app.invoke(
        {"queries_path": args.queries, "manifest_path": args.manifest}
    )

    if final_state.get("error"):
        return 1
    return 1 if final_state.get("errors") else 0


if __name__ == "__main__":
    sys.exit(main())
