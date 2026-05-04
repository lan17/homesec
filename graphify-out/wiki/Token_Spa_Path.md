# Token Spa Path

> 15 nodes · cohesion 0.15

## Key Concepts

- **server.py** (11 connections) — `src/homesec/api/server.py`
- **_TokenRedactingAccessLogFilter** (6 connections) — `src/homesec/api/server.py`
- **_ensure_access_log_redaction()** (4 connections) — `src/homesec/api/server.py`
- **_configure_ui_serving()** (3 connections) — `src/homesec/api/server.py`
- **_redact_token_query_param_from_url()** (3 connections) — `src/homesec/api/server.py`
- **.filter()** (3 connections) — `src/homesec/api/server.py`
- **_is_reserved_spa_path()** (2 connections) — `src/homesec/api/server.py`
- **_is_allowed_root_static_file()** (1 connections) — `src/homesec/api/server.py`
- **_is_spa_route_path()** (1 connections) — `src/homesec/api/server.py`
- **FastAPI server wiring for HomeSec.** (1 connections) — `src/homesec/api/server.py`
- **Serve built SPA assets from FastAPI using configured ui_dist_dir.** (1 connections) — `src/homesec/api/server.py`
- **Return True when SPA fallback must not capture this path.** (1 connections) — `src/homesec/api/server.py`
- **Return a URL/path string with token query parameters removed.** (1 connections) — `src/homesec/api/server.py`
- **Strip token query parameters from uvicorn access log paths.** (1 connections) — `src/homesec/api/server.py`
- **Install the token-redacting access-log filter once per process.** (1 connections) — `src/homesec/api/server.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/server.py`

## Audit Trail

- EXTRACTED: 37 (92%)
- INFERRED: 3 (8%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*