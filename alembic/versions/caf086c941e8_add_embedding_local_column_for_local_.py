"""add embedding_local column for local-embedder ablation

Revision ID: caf086c941e8
Revises: 5b5438aaf806
Create Date: 2026-09-26 15:10:00.000000

"""
from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'caf086c941e8'
down_revision: Union[str, Sequence[str], None] = '5b5438aaf806'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'chunks',
        sa.Column('embedding_local', pgvector.sqlalchemy.Vector(dim=384), nullable=True),
    )
    op.add_column(
        'chunks', sa.Column('embedding_local_model', sa.String(length=128), nullable=True)
    )
    op.execute(
        "CREATE INDEX chunks_embedding_local_hnsw_idx "
        "ON chunks USING hnsw (embedding_local vector_cosine_ops)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS chunks_embedding_local_hnsw_idx")
    op.drop_column('chunks', 'embedding_local_model')
    op.drop_column('chunks', 'embedding_local')
