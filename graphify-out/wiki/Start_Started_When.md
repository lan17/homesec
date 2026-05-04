# Start Started When

> 8 nodes · cohesion 0.39

## Key Concepts

- **APIServer** (6 connections) — `tests/homesec/test_api_server.py`
- **FastAPI** (5 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_and_stop_waits_for_started** (2 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_is_idempotent** (2 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_raises_when_server_exits_before_started** (2 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_raises_when_uvicorn_crashes** (2 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_times_out_when_started_flag_never_set** (2 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_stop_swallows_server_exit_exception** (1 connections) — `tests/homesec/test_api_server.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_server.py`

## Audit Trail

- EXTRACTED: 20 (91%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 2 (9%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*