# Thread Background Hook

> 25 nodes · cohesion 0.09

## Key Concepts

- **ThreadedClipSource** (35 connections) — `src/homesec/sources/base.py`
- **.is_healthy()** (3 connections) — `src/homesec/sources/base.py`
- **.ping()** (3 connections) — `src/homesec/sources/base.py`
- **._thread_is_healthy()** (3 connections) — `src/homesec/sources/base.py`
- **.last_heartbeat()** (2 connections) — `src/homesec/sources/base.py`
- **._on_start()** (2 connections) — `src/homesec/sources/base.py`
- **._on_started()** (2 connections) — `src/homesec/sources/base.py`
- **._on_stop()** (2 connections) — `src/homesec/sources/base.py`
- **._on_stopped()** (2 connections) — `src/homesec/sources/base.py`
- **.register_callback()** (2 connections) — `src/homesec/sources/base.py`
- **.shutdown()** (2 connections) — `src/homesec/sources/base.py`
- **Hook called before starting the background thread.** (1 connections) — `src/homesec/sources/base.py`
- **Hook called after starting the background thread.** (1 connections) — `src/homesec/sources/base.py`
- **Hook called before stopping the background thread.** (1 connections) — `src/homesec/sources/base.py`
- **Hook called after stopping the background thread.** (1 connections) — `src/homesec/sources/base.py`
- **Base class for clip sources that run in a background thread.** (1 connections) — `src/homesec/sources/base.py`
- **Register callback to be invoked when a new clip is ready.** (1 connections) — `src/homesec/sources/base.py`
- **Async wrapper for stopping the background thread.** (1 connections) — `src/homesec/sources/base.py`
- **Default health check: thread is alive (if started).** (1 connections) — `src/homesec/sources/base.py`
- **Return monotonic timestamp of last heartbeat update.** (1 connections) — `src/homesec/sources/base.py`
- **Health check - verify source is operational.          Returns True if:         -** (1 connections) — `src/homesec/sources/base.py`
- **._emit_clip()** (1 connections) — `src/homesec/sources/base.py`
- **.__init__()** (1 connections) — `src/homesec/sources/base.py`
- **._stop_timeout()** (1 connections) — `src/homesec/sources/base.py`
- **._touch_heartbeat()** (1 connections) — `src/homesec/sources/base.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/base.py`

## Audit Trail

- EXTRACTED: 58 (81%)
- INFERRED: 14 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*