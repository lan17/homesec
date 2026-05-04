# Dump Env Command

> 27 nodes · cohesion 0.09

## Key Concepts

- **postgres_backup.py** (18 connections) — `src/homesec/maintenance/postgres_backup.py`
- **build_pg_dump_env()** (9 connections) — `src/homesec/maintenance/postgres_backup.py`
- **_BackupManifest** (8 connections) — `src/homesec/maintenance/postgres_backup.py`
- **test_pg_dump_command_uses_env_for_credentials()** (5 connections) — `tests/homesec/test_postgres_backup.py`
- **build_pg_dump_command()** (4 connections) — `src/homesec/maintenance/postgres_backup.py`
- **resolve_postgres_backup_dsn()** (4 connections) — `src/homesec/maintenance/postgres_backup.py`
- **test_pg_dump_env_ignores_ambient_libpq_variables()** (3 connections) — `tests/homesec/test_postgres_backup.py`
- **test_pg_dump_env_maps_asyncpg_tls_query_params()** (3 connections) — `tests/homesec/test_postgres_backup.py`
- **_copy_query_env()** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **_copy_ssl_query_env()** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **_normalize_libpq_scheme()** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **.__init__()** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **run_pg_dump_available()** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **run_pg_dump_version()** (2 connections) — `src/homesec/maintenance/postgres_backup.py`
- **pg_dump invocation should keep credentials out of command args.** (1 connections) — `tests/homesec/test_postgres_backup.py`
- **pg_dump should only use target values derived from the configured DSN.** (1 connections) — `tests/homesec/test_postgres_backup.py`
- **pg_dump should preserve TLS intent from asyncpg-style state-store DSNs.** (1 connections) — `tests/homesec/test_postgres_backup.py`
- **enabled()** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **_now_utc()** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **_private_file_opener()** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **Periodic Postgres backup manager.** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **Resolve the state-store DSN using the app runtime precedence.** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **Build credential-bearing libpq environment variables from a Postgres DSN.** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **Return the credential-free pg_dump command.** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- **Return whether pg_dump is available in PATH.** (1 connections) — `src/homesec/maintenance/postgres_backup.py`
- *... and 2 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/maintenance/postgres_backup.py`
- `tests/homesec/test_postgres_backup.py`

## Audit Trail

- EXTRACTED: 66 (84%)
- INFERRED: 13 (16%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*