"""Idempotent ingestion: manifest.yaml + transcripts -> chunks table.

LangGraph pipeline: load_and_chunk -> upsert_keyword_rows -> embed_and_update.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

import yaml
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .chunking import Chunk as ChunkData
from .chunking import ChunkingSummary, chunk_transcript
from core.embedder import Embedder
from core.models import Chunk, File, Speaker


class FileChunks(TypedDict):
    clip: dict
    chunks: list[ChunkData]
    summary: ChunkingSummary


class GraphState(TypedDict, total=False):
    manifest_path: str
    embedder: Embedder
    embed_batch_size: int
    target: str  # "primary" or "local"
    file_chunks: list[FileChunks]
    chunk_row_ids: dict[str, dict[int, int]]
    total_chunks: int
    error: Optional[str]


def load_manifest(manifest_path: str | Path) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_chunk_node(state: GraphState) -> GraphState:
    manifest_path = Path(state["manifest_path"])
    if not manifest_path.exists():
        return {**state, "error": f"manifest not found: {manifest_path}"}

    manifest = load_manifest(manifest_path)
    file_chunks: list[FileChunks] = []

    for clip in manifest.get("clips", []):
        chunks, summary = chunk_transcript(
            clip["transcript"], clip_file=clip["file"], clip_offset_sec=clip["start_sec"]
        )
        print(f"{clip['file']}: {summary.report()}")
        file_chunks.append({"clip": clip, "chunks": chunks, "summary": summary})

    total = sum(len(fc["chunks"]) for fc in file_chunks)
    return {**state, "file_chunks": file_chunks, "total_chunks": total}


def _upsert_file(session: Session, clip: dict) -> File:
    values = dict(
        file_path=clip["file"],
        title=clip.get("title"),
        podcast=clip.get("podcast"),
        source_url=clip.get("source_url"),
        start_sec=clip["start_sec"],
        end_sec=clip["end_sec"],
        sha256=clip.get("sha256"),
    )
    stmt = (
        pg_insert(File)
        .values(**values)
        .on_conflict_do_update(index_elements=["file_path"], set_=values)
        .returning(File.id)
    )
    file_id = session.execute(stmt).scalar_one()
    return session.get(File, file_id)


def _upsert_speaker(
    session: Session, file_id: int, label: str, real_name: Optional[str]
) -> Speaker:
    stmt = (
        pg_insert(Speaker)
        .values(file_id=file_id, label=label, real_name=real_name)
        .on_conflict_do_update(
            index_elements=["file_id", "label"], set_={"real_name": real_name}
        )
        .returning(Speaker.id)
    )
    speaker_id = session.execute(stmt).scalar_one()
    return session.get(Speaker, speaker_id)


def _upsert_chunk_keyword_row(
    session: Session,
    file_id: int,
    speaker_id: int,
    chunk_index: int,
    chunk: ChunkData,
) -> int:
    values = dict(
        file_id=file_id,
        speaker_id=speaker_id,
        chunk_index=chunk_index,
        start_sec=chunk.start_sec,
        end_sec=chunk.end_sec,
        text=chunk.text,
        embedding_input=chunk.embedding_input,
        source=chunk.source,
    )
    stmt = (
        pg_insert(Chunk)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["file_id", "chunk_index"],
            set_={k: v for k, v in values.items() if k not in ("file_id", "chunk_index")},
        )
        .returning(Chunk.id)
    )
    return session.execute(stmt).scalar_one()


def upsert_keyword_rows_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    from core.db import get_session

    session = get_session()
    chunk_row_ids: dict[str, dict[int, int]] = {}

    try:
        for fc in state["file_chunks"]:
            clip = fc["clip"]
            file_row = _upsert_file(session, clip)

            speaker_ids: dict[str, int] = {}
            for label, real_name in clip.get("speakers", {}).items():
                speaker_row = _upsert_speaker(session, file_row.id, label, real_name)
                speaker_ids[label] = speaker_row.id

            for chunk in fc["chunks"]:
                if chunk.speaker not in speaker_ids:
                    speaker_row = _upsert_speaker(session, file_row.id, chunk.speaker, None)
                    speaker_ids[chunk.speaker] = speaker_row.id

            row_ids: dict[int, int] = {}
            for chunk_index, chunk in enumerate(fc["chunks"]):
                row_id = _upsert_chunk_keyword_row(
                    session,
                    file_id=file_row.id,
                    speaker_id=speaker_ids[chunk.speaker],
                    chunk_index=chunk_index,
                    chunk=chunk,
                )
                row_ids[chunk_index] = row_id
            chunk_row_ids[clip["file"]] = row_ids

        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        return {**state, "error": f"upsert_keyword_rows failed: {exc}"}
    finally:
        session.close()

    return {**state, "chunk_row_ids": chunk_row_ids}


def embed_and_update_node(state: GraphState) -> GraphState:
    if state.get("error"):
        return state

    from core.db import get_session

    embedder: Embedder = state["embedder"]
    batch_size = state.get("embed_batch_size", 64)
    embedding_col, model_col = (
        (Chunk.embedding, "embedding_model")
        if state.get("target") != "local"
        else (Chunk.embedding_local, "embedding_local_model")
    )
    session = get_session()

    try:
        for fc in state["file_chunks"]:
            clip = fc["clip"]
            chunks = fc["chunks"]
            row_ids = state["chunk_row_ids"][clip["file"]]

            for batch_start in range(0, len(chunks), batch_size):
                batch = chunks[batch_start : batch_start + batch_size]
                embeddings = embedder.embed_documents([c.embedding_input for c in batch])
                for offset, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                    chunk_index = batch_start + offset
                    row_id = row_ids[chunk_index]
                    session.execute(
                        update(Chunk)
                        .where(Chunk.id == row_id)
                        .values(**{
                            embedding_col.key: embedding,
                            model_col: embedder.model_name,
                        })
                    )
            print(f"{clip['file']}: embedded {len(chunks)} chunks (target={state.get('target', 'primary')})")

        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        return {**state, "error": f"embed_and_update failed: {exc}"}
    finally:
        session.close()

    return state


def build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(GraphState)
    graph.add_node("load_and_chunk", load_and_chunk_node)
    graph.add_node("upsert_keyword_rows", upsert_keyword_rows_node)
    graph.add_node("embed_and_update", embed_and_update_node)

    graph.set_entry_point("load_and_chunk")
    graph.add_edge("load_and_chunk", "upsert_keyword_rows")
    graph.add_edge("upsert_keyword_rows", "embed_and_update")
    graph.add_edge("embed_and_update", END)

    return graph.compile()


def main() -> None:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest manifest.yaml chunks into Postgres.")
    parser.add_argument("--manifest", default="manifest.yaml")
    parser.add_argument(
        "--embedder",
        choices=["openai", "local"],
        default="openai",
        help="Which Embedder backend to use (default: openai)",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Only run the keyword-side upsert; skip the embedding step "
        "(e.g. to bring up keyword search quickly, embed later).",
    )
    args = parser.parse_args()

    if args.embedder == "openai":
        from core.embedder import OpenAIEmbedder

        embedder: Embedder = OpenAIEmbedder()
        target = "primary"
    else:
        from core.embedder import LocalEmbedder

        embedder = LocalEmbedder()
        target = "local"

    app = build_graph()
    initial_state: GraphState = {
        "manifest_path": args.manifest,
        "embedder": embedder,
        "embed_batch_size": 64,
        "target": target,
    }

    if args.skip_embed:
        state = load_and_chunk_node(initial_state)
        state = upsert_keyword_rows_node(state)
    else:
        state = app.invoke(initial_state)

    if state.get("error"):
        raise SystemExit(f"Error: {state['error']}")

    print(f"\nIngested {state.get('total_chunks', 0)} chunks total.")


if __name__ == "__main__":
    main()
