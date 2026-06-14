"""add mobile device capabilities

Revision ID: 2e87ef73f2d4
Revises: 7b0b1fbfc69b
Create Date: 2026-06-14 00:00:01.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "2e87ef73f2d4"
down_revision: str | None = "7b0b1fbfc69b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "mobile_devices",
        sa.Column(
            "capabilities",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("mobile_devices", "capabilities")
