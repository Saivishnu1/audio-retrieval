"""SQLAlchemy schema. Source of truth for Alembic autogenerate (see alembic/env.py)."""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIMENSION = 1536  # OpenAIEmbedder
LOCAL_EMBEDDING_DIMENSION = 384  # LocalEmbedder


class Base(DeclarativeBase):
    pass


class File(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    podcast: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    start_sec: Mapped[float | None] = mapped_column(Double)
    end_sec: Mapped[float | None] = mapped_column(Double)
    sha256: Mapped[str | None] = mapped_column(Text)

    speakers: Mapped[list["Speaker"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )


class Speaker(Base):
    __tablename__ = "speakers"
    __table_args__ = (UniqueConstraint("file_id", "label", name="uq_speaker_file_label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(Text, nullable=False)  # e.g. "Speaker 0"
    real_name: Mapped[str | None] = mapped_column(Text)  # e.g. "Dan Snow"

    file: Mapped["File"] = relationship(back_populates="speakers")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="speaker")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("file_id", "chunk_index", name="uq_chunk_file_index"),
        Index("chunks_tsv_idx", "tsv", postgresql_using="gin"),
        Index(
            "chunks_text_trgm_idx", "text", postgresql_using="gin",
            postgresql_ops={"text": "gin_trgm_ops"},
        ),
        Index(
            "chunks_embedding_hnsw_idx", "embedding", postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index(
            "chunks_embedding_local_hnsw_idx", "embedding_local", postgresql_using="hnsw",
            postgresql_ops={"embedding_local": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("files.id", ondelete="CASCADE"))
    speaker_id: Mapped[int] = mapped_column(ForeignKey("speakers.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_sec: Mapped[float] = mapped_column(Double, nullable=False)
    end_sec: Mapped[float] = mapped_column(Double, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_input: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)  # base|split|merged
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSION))
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    embedding_local: Mapped[list[float] | None] = mapped_column(
        Vector(LOCAL_EMBEDDING_DIMENSION)
    )
    embedding_local_model: Mapped[str | None] = mapped_column(String(128))
    tsv: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    file: Mapped["File"] = relationship(back_populates="chunks")
    speaker: Mapped["Speaker"] = relationship(back_populates="chunks")
