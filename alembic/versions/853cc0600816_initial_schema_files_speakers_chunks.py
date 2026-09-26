"""initial schema: files, speakers, chunks

Revision ID: 853cc0600816
Revises:
Create Date: 2026-09-26 13:09:34.281741

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '853cc0600816'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        'files',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('podcast', sa.Text(), nullable=True),
        sa.Column('source_url', sa.Text(), nullable=True),
        sa.Column('start_sec', sa.Double(), nullable=True),
        sa.Column('end_sec', sa.Double(), nullable=True),
        sa.Column('sha256', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_path'),
    )
    op.create_table(
        'speakers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('file_id', sa.Integer(), nullable=False),
        sa.Column('label', sa.Text(), nullable=False),
        sa.Column('real_name', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['file_id'], ['files.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('file_id', 'label', name='uq_speaker_file_label'),
    )

    op.execute(
        """
        CREATE TABLE chunks (
            id               SERIAL PRIMARY KEY,
            file_id          INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            speaker_id       INTEGER NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
            chunk_index      INTEGER NOT NULL,
            start_sec        DOUBLE PRECISION NOT NULL,
            end_sec          DOUBLE PRECISION NOT NULL,
            text             TEXT NOT NULL,
            embedding_input  TEXT NOT NULL,
            source           VARCHAR(16) NOT NULL,
            embedding        vector(1536),
            tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
            UNIQUE (file_id, chunk_index)
        )
        """
    )

    op.execute("CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv)")
    op.execute(
        "CREATE INDEX chunks_text_trgm_idx ON chunks USING GIN (text gin_trgm_ops)"
    )
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw_idx "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS chunks_embedding_hnsw_idx")
    op.execute("DROP INDEX IF EXISTS chunks_text_trgm_idx")
    op.execute("DROP INDEX IF EXISTS chunks_tsv_idx")
    op.drop_table('chunks')
    op.drop_table('speakers')
    op.drop_table('files')
