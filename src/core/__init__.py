"""Shared infrastructure: embedding backends, the DB schema, and the connection.

Used by both `ingestion` (writes) and `retrieval` (reads).
"""

from .embedder import Embedder, LocalEmbedder, OpenAIEmbedder
from .models import Base, Chunk, File, Speaker

__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "LocalEmbedder",
    "Base",
    "File",
    "Speaker",
    "Chunk",
]
