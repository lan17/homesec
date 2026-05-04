# Token Access Log

> 16 nodes · cohesion 0.12

## Key Concepts

- **test_api_server.py** (15 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_raises_when_server_exits_before_started()** (3 connections) — `tests/homesec/test_api_server.py`
- **test_redact_token_query_param_from_url_leaves_non_token_paths_unchanged()** (2 connections) — `tests/homesec/test_api_server.py`
- **test_redact_token_query_param_from_url_strips_only_token_values()** (2 connections) — `tests/homesec/test_api_server.py`
- **test_spa_path_helpers_cover_edge_cases()** (2 connections) — `tests/homesec/test_api_server.py`
- **test_token_redacting_access_log_filter_leaves_nonstandard_records_untouched()** (2 connections) — `tests/homesec/test_api_server.py`
- **test_token_redacting_access_log_filter_rewrites_uvicorn_path_argument()** (2 connections) — `tests/homesec/test_api_server.py`
- **test_token_redacting_access_log_filter_supports_mutable_args_lists()** (2 connections) — `tests/homesec/test_api_server.py`
- **Tests for APIServer lifecycle behavior.** (1 connections) — `tests/homesec/test_api_server.py`
- **SPA helper predicates should classify reserved/static/route paths correctly.** (1 connections) — `tests/homesec/test_api_server.py`
- **Access-log URL redaction should drop token query parameters and preserve the res** (1 connections) — `tests/homesec/test_api_server.py`
- **Access-log URL redaction should leave unrelated paths and empty queries untouche** (1 connections) — `tests/homesec/test_api_server.py`
- **Access-log filter should redact token query params from uvicorn access records.** (1 connections) — `tests/homesec/test_api_server.py`
- **Access-log filter should redact token query params when uvicorn passes args as a** (1 connections) — `tests/homesec/test_api_server.py`
- **Access-log filter should no-op when the record does not look like a uvicorn acce** (1 connections) — `tests/homesec/test_api_server.py`
- **APIServer should fail when server task exits cleanly before startup flag is set.** (1 connections) — `tests/homesec/test_api_server.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_server.py`

## Audit Trail

- EXTRACTED: 37 (97%)
- INFERRED: 1 (3%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*