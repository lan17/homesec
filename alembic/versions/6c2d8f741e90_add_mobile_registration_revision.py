"""Add a registration revision independent of APNs delivery bookkeeping.

Revision ID: 6c2d8f741e90
Revises: 7b0b1fbfc69b
Create Date: 2026-10-04 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "6c2d8f741e90"
down_revision: str | None = "7b0b1fbfc69b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "mobile_devices",
        sa.Column(
            "registration_revision", sa.BigInteger(), server_default=sa.text("1"), nullable=False
        ),
    )


def downgrade() -> None:
    op.drop_column("mobile_devices", "registration_revision")
