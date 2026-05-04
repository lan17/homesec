# Start Apiserver Stop

> 20 nodes · cohesion 0.11

## Key Concepts

- **APIServer** (17 connections) — `src/homesec/api/server.py`
- **.start()** (4 connections) — `src/homesec/api/server.py`
- **._wait_until_started()** (3 connections) — `src/homesec/api/server.py`
- **test_api_server_start_and_stop_waits_for_started()** (3 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_is_idempotent()** (3 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_raises_when_uvicorn_crashes()** (3 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_start_times_out_when_started_flag_never_set()** (3 connections) — `tests/homesec/test_api_server.py`
- **test_api_server_stop_swallows_server_exit_exception()** (3 connections) — `tests/homesec/test_api_server.py`
- **test_wait_until_started_requires_initialized_server()** (3 connections) — `tests/homesec/test_api_server.py`
- **.__init__()** (1 connections) — `src/homesec/api/server.py`
- **.stop()** (1 connections) — `src/homesec/api/server.py`
- **Manages the API server lifecycle.** (1 connections) — `src/homesec/api/server.py`
- **Start the API server in the background.** (1 connections) — `src/homesec/api/server.py`
- **Wait until uvicorn reports startup success or startup fails.** (1 connections) — `src/homesec/api/server.py`
- **APIServer should time out if startup never completes.** (1 connections) — `tests/homesec/test_api_server.py`
- **APIServer.start should not create a second uvicorn server when already running.** (1 connections) — `tests/homesec/test_api_server.py`
- **APIServer.stop should log and swallow task exceptions raised during shutdown.** (1 connections) — `tests/homesec/test_api_server.py`
- **APIServer should wait for startup and stop cleanly.** (1 connections) — `tests/homesec/test_api_server.py`
- **_wait_until_started should fail fast when start() has not initialized internals.** (1 connections) — `tests/homesec/test_api_server.py`
- **APIServer should fail fast when uvicorn fails before startup completes.** (1 connections) — `tests/homesec/test_api_server.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/server.py`
- `tests/homesec/test_api_server.py`

## Audit Trail

- EXTRACTED: 36 (68%)
- INFERRED: 17 (32%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*