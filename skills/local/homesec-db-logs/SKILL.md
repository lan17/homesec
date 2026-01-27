---
name: homesec-db-logs
description: Query the HomeSec Postgres logs table to inspect recent runtime issues, counts by level/logger, and recurring error messages. Use when asked to check database logs in the telemetry DB, scan the last N hours for errors, or summarize log patterns. Enforce read-only access (SELECT only) and redact secrets.
---

# HomeSec DB Log Queries

## Quick Start

1. Confirm the DSN and time window; use SELECT-only queries.
2. Confirm the logs table schema (from SQLAlchemy models) and time window.
3. Pull counts by level/logger, then top error/warn messages.
4. Inspect a few recent payloads to confirm fields.
5. Summarize findings; avoid dumping large payloads or secrets.

## Canonical Queries

Use psql with `-Atc` for clean output. Replace the DSN as needed.

- List tables:

```bash
psql "$DSN" -Atc "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;"
```

- Inspect columns:

```bash
psql "$DSN" -Atc "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='public' AND table_name='logs' ORDER BY ordinal_position;"
```

- Severity counts (logs table with JSON payload):

```bash
psql "$DSN" -Atc "SELECT payload->>'level' AS level, count(*) FROM logs WHERE ts >= now() - interval '12 hours' GROUP BY level ORDER BY count DESC;"
```

- Top error messages (logs table):

```bash
psql "$DSN" -Atc "SELECT payload->>'message' AS message, count(*) FROM logs WHERE ts >= now() - interval '12 hours' AND payload->>'level'='ERROR' GROUP BY message ORDER BY count DESC LIMIT 10;"
```

- Top warn/error by logger (logs table):

```bash
psql "$DSN" -Atc "SELECT payload->>'logger' AS logger, count(*) FROM logs WHERE ts >= now() - interval '12 hours' AND payload->>'level' IN ('ERROR','WARNING') GROUP BY logger ORDER BY count DESC;"
```

- Recent samples (logs table):

```bash
psql "$DSN" -Atc "SELECT payload FROM logs WHERE ts >= now() - interval '12 hours' ORDER BY ts DESC LIMIT 3;"
```

## Table Structure Notes

- `logs`: generic JSONB log sink (defined in `src/homesec/telemetry/db/log_table.py`).
  - Columns: `id` (bigint), `ts` (timestamptz), `payload` (jsonb).
  - Common payload keys: `ts`, `kind`, `level`, `message`, `fields`, `logger`, `module`, `pathname`, `event_type`, `camera_name`, `recording_id`.
  - Typical filters use JSONB: `payload->>'level'`, `payload->>'logger'`, `payload->>'message'`.

## Guardrails

- Use SELECT statements only. Never run INSERT/UPDATE/DELETE/DDL.
- If output contains secrets (RTSP URLs with credentials, tokens), redact before reporting.
- Keep results compact: aggregate counts first, then sample rows with LIMIT.
- State the time window explicitly (use absolute dates when summarizing).

## Interpretation Tips

- For repeated ffmpeg/ffprobe errors, check for missing flags or unsupported options.
- If warnings/errors cluster around a single logger, focus analysis there before expanding scope.
