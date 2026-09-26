"""Diarized transcription using Deepgram, orchestrated with LangGraph.

Standalone script — copy/paste this whole file into a single Google Colab
cell and it works as-is, or run it locally as `python diarize_transcribe.py`.
No dependency on the rest of this repo's package.

Deepgram's transcription API natively supports speaker diarization in a
single call (`diarize=True`), so no separate diarization model/pass is
needed here (unlike the Whisper + pyannote combo).

Pipeline (a LangGraph StateGraph, one run per audio file):

    transcribe_and_diarize (Deepgram)  ->  write

    - transcribe_and_diarize: Deepgram's `nova-2` (or newer) model with
      diarization enabled -> word-level results tagged with a speaker id,
      which are grouped into speaker-labeled segments.
    - write: renders "[start-end] SPEAKER: text" lines to a .txt file.

Setup
-----
Local:
    pip install deepgram-sdk langgraph python-dotenv
    # .env (or exported env vars) must contain:
    #   DEEPGRAM_API_KEY=...

Google Colab:
    !pip install -q deepgram-sdk langgraph
    # Set DEEPGRAM_API_KEY via Colab's "Secrets" (key icon in the left
    # sidebar), OR just run:
    #   import os
    #   os.environ["DEEPGRAM_API_KEY"] = "..."
    # before running this cell. Then paste this whole file into a cell.

Get a Deepgram API key
-----------------------
https://console.deepgram.com/ -> Create a project -> API Keys -> Create.
New accounts get free trial credit, no card required to start.

Configuration
-------------
Edit INPUT_DIR / OUTPUT_DIR / DEEPGRAM_MODEL below, or set them as env
vars (DIARIZE_INPUT_DIR / DIARIZE_OUTPUT_DIR / DEEPGRAM_MODEL) before
running.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TypedDict

# --------------------------------------------------------------------------
# Configuration — edit these to taste (or override via env vars)
# --------------------------------------------------------------------------
INPUT_DIR = os.environ.get("DIARIZE_INPUT_DIR", "clips")
OUTPUT_DIR = os.environ.get("DIARIZE_OUTPUT_DIR", "transcripts")
DEEPGRAM_MODEL = os.environ.get("DEEPGRAM_MODEL", "nova-2")
SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

# --------------------------------------------------------------------------
# Load local .env if present (harmless no-op on Colab if the file is missing
# or python-dotenv isn't installed there)
# --------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# --------------------------------------------------------------------------
# Shared data types
# --------------------------------------------------------------------------
class LabeledSegment(TypedDict):
    speaker: str
    text: str
    start: float
    end: float


class GraphState(TypedDict, total=False):
    """State threaded through the LangGraph pipeline for one audio file."""

    audio_path: str
    output_dir: str
    api_key: str
    labeled_segments: list[LabeledSegment]
    output_path: str
    error: Optional[str]


def format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


# --------------------------------------------------------------------------
# Node: transcribe_and_diarize (Deepgram, one call does both)
# --------------------------------------------------------------------------
def _segments_from_paragraphs(paragraphs: list) -> list[LabeledSegment]:
    """Build labeled segments from Deepgram's `paragraphs` output.

    Requires `paragraphs=True` in the request. Unlike the raw per-word
    `words` list (which is always lowercase/unpunctuated), each paragraph's
    sentences already have smart-formatted casing and punctuation applied,
    and are pre-grouped by speaker turn.
    """
    segments: list[LabeledSegment] = []
    for paragraph in paragraphs:
        speaker = paragraph.speaker if paragraph.speaker is not None else 0
        text = " ".join(sentence.text for sentence in paragraph.sentences).strip()
        if not text:
            continue
        segments.append(
            {
                "speaker": f"Speaker {speaker}",
                "text": text,
                "start": paragraph.start,
                "end": paragraph.end,
            }
        )
    return segments


def transcribe_and_diarize_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    from deepgram import DeepgramClient

    audio_path = Path(state["audio_path"])
    print(f"Transcribing + diarizing (Deepgram): {audio_path.name}")

    try:
        client = DeepgramClient(api_key=state["api_key"])
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        response = client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=DEEPGRAM_MODEL,
            smart_format=True,
            punctuate=True,
            paragraphs=True,
            diarize=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {**state, "error": f"deepgram request failed: {exc}"}

    try:
        channel = response.results.channels[0]
        paragraphs = channel.alternatives[0].paragraphs.paragraphs
    except (AttributeError, IndexError, KeyError) as exc:
        return {**state, "error": f"unexpected deepgram response shape: {exc}"}

    segments = _segments_from_paragraphs(paragraphs)
    return {**state, "labeled_segments": segments}


# --------------------------------------------------------------------------
# Node: write (render + save the transcript file)
# --------------------------------------------------------------------------
def write_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    audio_path = Path(state["audio_path"])
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"[{format_timestamp(seg['start'])}-{format_timestamp(seg['end'])}] "
        f"{seg['speaker']}: {seg['text']}"
        for seg in state.get("labeled_segments", [])
    ]
    out_path = output_dir / f"{audio_path.stem}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")
    return {**state, "output_path": str(out_path)}


# --------------------------------------------------------------------------
# Build the graph
# --------------------------------------------------------------------------
def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("transcribe_and_diarize", transcribe_and_diarize_node)
    graph.add_node("write", write_node)

    graph.set_entry_point("transcribe_and_diarize")
    graph.add_edge("transcribe_and_diarize", "write")
    graph.add_edge("write", END)

    return graph.compile()


def main() -> None:
    api_key = os.environ.get("DEEPGRAM_API_KEY")

    if not api_key:
        raise SystemExit(
            "DEEPGRAM_API_KEY is not set. Get one at https://console.deepgram.com/, "
            "then set it as an env var, in a local .env file, or via Colab "
            "secrets/os.environ before running this script."
        )

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    audio_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not audio_files:
        raise SystemExit(f"No supported audio files found in '{input_dir}'")

    app = build_graph()

    for audio_path in audio_files:
        initial_state: GraphState = {
            "audio_path": str(audio_path),
            "output_dir": str(output_dir),
            "api_key": api_key,
        }
        final_state = app.invoke(initial_state)
        if final_state.get("error"):
            print(f"ERROR processing {audio_path.name}: {final_state['error']}")

    print(f"\nDone. Transcripts written to '{output_dir}'")


if __name__ == "__main__":
    main()
