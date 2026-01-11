from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_DOTENV = Path(__file__).resolve().parents[3] / ".env"


class PostgresConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", _REPO_DOTENV),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    db_dsn: str | None = None  # postgresql+asyncpg://user:pass@host:5432/db
    db_log_level: str = "INFO"
    db_log_queue_size: int = 5000
    db_log_batch_size: int = 100
    db_log_flush_s: float = 1.0
    db_log_backoff_initial_s: float = 1.0
    db_log_backoff_max_s: float = 30.0
    db_log_drop_policy: Literal["drop_new", "drop_oldest"] = "drop_new"

    @field_validator("db_log_level")
    @classmethod
    def _normalize_level(cls, value: str) -> str:
        return str(value).upper()

    @property
    def enabled(self) -> bool:
        return bool(self.db_dsn)

    @property
    def sync_dsn(self) -> str | None:
        if not self.db_dsn:
            return None
        return self.db_dsn.replace("+asyncpg", "")
