"""Offline chunking + ingestion pipeline: manifest.yaml + transcripts -> Postgres."""

from .chunking import Chunk, ChunkingSummary, chunk_transcript

__all__ = [
    "chunk_transcript",
    "Chunk",
    "ChunkingSummary",
]
