"""add mobile device registry

Revision ID: 7b0b1fbfc69b
Revises: d936851f725a
Create Date: 2026-06-14 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b0b1fbfc69b"
down_revision: str | None = "d936851f725a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "mobile_devices",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("platform", sa.Text(), nullable=False),
        sa.Column("apns_token_hash", sa.Text(), nullable=False),
        sa.Column("apns_token_encrypted", sa.Text(), nullable=False),
        sa.Column("apns_environment", sa.Text(), nullable=False),
        sa.Column("bundle_id", sa.Text(), nullable=False),
        sa.Column("device_name", sa.Text(), nullable=True),
        sa.Column("app_version", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_push_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_push_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("apns_token_hash", name="uq_mobile_devices_apns_token_hash"),
    )
    op.create_index("idx_mobile_devices_enabled", "mobile_devices", ["enabled"], unique=False)
    op.create_index(
        "idx_mobile_devices_platform_environment",
        "mobile_devices",
        ["platform", "apns_environment"],
        unique=False,
    )
    op.create_index(
        "idx_mobile_devices_updated_at_desc",
        "mobile_devices",
        [sa.literal_column("updated_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_mobile_devices_updated_at_desc", table_name="mobile_devices")
    op.drop_index("idx_mobile_devices_platform_environment", table_name="mobile_devices")
    op.drop_index("idx_mobile_devices_enabled", table_name="mobile_devices")
    op.drop_table("mobile_devices")
