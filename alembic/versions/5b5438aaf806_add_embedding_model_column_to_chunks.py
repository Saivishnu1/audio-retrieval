"""add embedding_model column to chunks

Revision ID: 5b5438aaf806
Revises: 853cc0600816
Create Date: 2026-09-26 13:23:57.833608

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '5b5438aaf806'
down_revision: Union[str, Sequence[str], None] = '853cc0600816'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'chunks', sa.Column('embedding_model', sa.String(length=128), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('chunks', 'embedding_model')
