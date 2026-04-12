"""backfill alert decision timestamps

Revision ID: 1c3a6f9d2b10
Revises: d936851f725a
Create Date: 2026-04-12 18:05:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1c3a6f9d2b10"
down_revision: str | None = "d936851f725a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        WITH decision_times AS (
            SELECT clip_id, max(timestamp) AS alert_decision_at
            FROM clip_events
            WHERE event_type = 'alert_decision_made'
            GROUP BY clip_id
        )
        UPDATE clip_states
        SET data = jsonb_set(
            data,
            '{alert_decision_at}',
            to_jsonb(
                to_char(
                    timezone('UTC', decision_times.alert_decision_at),
                    'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
                )
            ),
            true
        )
        FROM decision_times
        WHERE clip_states.clip_id = decision_times.clip_id
        """
    )
    op.create_index(
        "idx_clip_states_alert_decision_at",
        "clip_states",
        [sa.literal_column("jsonb_extract_path_text(data, 'alert_decision_at')")],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_clip_states_alert_decision_at", table_name="clip_states")
    op.execute(
        """
        UPDATE clip_states
        SET data = data - 'alert_decision_at'
        WHERE jsonb_extract_path_text(data, 'alert_decision_at') IS NOT NULL
        """
    )
