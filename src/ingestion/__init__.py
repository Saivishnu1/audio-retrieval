"""Offline chunking + ingestion pipeline: manifest.yaml + transcripts -> Postgres."""

from .chunking import Chunk, ChunkingSummary, chunk_transcript
from .ingest import build_graph as build_ingest_graph
from .ingest import main as ingest_main

__all__ = [
    "chunk_transcript",
    "Chunk",
    "ChunkingSummary",
    "build_ingest_graph",
    "ingest_main",
]
