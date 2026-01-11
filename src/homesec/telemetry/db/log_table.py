from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, MetaData, Table, func
from sqlalchemy.dialects.postgresql import JSONB

metadata = MetaData()

logs = Table(
    "logs",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("payload", JSONB, nullable=False),
)

Index("logs_ts_idx", logs.c.ts.desc())
