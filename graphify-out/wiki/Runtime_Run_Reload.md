# Runtime Run Reload

> 19 nodes · cohesion 0.12

## Key Concepts

- **._create_components()** (13 connections) — `src/homesec/app.py`
- **._run_bootstrap()** (8 connections) — `src/homesec/app.py`
- **.run()** (7 connections) — `src/homesec/app.py`
- **.request_runtime_reload()** (6 connections) — `src/homesec/app.py`
- **._validate_config()** (6 connections) — `src/homesec/app.py`
- **.shutdown()** (5 connections) — `src/homesec/app.py`
- **._setup_signal_handlers()** (4 connections) — `src/homesec/app.py`
- **.activate_setup_config()** (3 connections) — `src/homesec/app.py`
- **._create_runtime_controller()** (3 connections) — `src/homesec/app.py`
- **._require_config()** (3 connections) — `src/homesec/app.py`
- **_classify_reload_config_error()** (2 connections) — `src/homesec/app.py`
- **config()** (2 connections) — `src/homesec/app.py`
- **Run the application.          Loads config, creates components, and runs until s** (1 connections) — `src/homesec/app.py`
- **Start in bootstrap mode with API/UI only.** (1 connections) — `src/homesec/app.py`
- **Create all components based on config.** (1 connections) — `src/homesec/app.py`
- **Apply setup-finalized config and start runtime without restarting FastAPI.** (1 connections) — `src/homesec/app.py`
- **Request a runtime reload using the latest persisted config.** (1 connections) — `src/homesec/app.py`
- **Set up signal handlers for graceful shutdown.** (1 connections) — `src/homesec/app.py`
- **Graceful shutdown of all components.** (1 connections) — `src/homesec/app.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/app.py`

## Audit Trail

- EXTRACTED: 56 (81%)
- INFERRED: 13 (19%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*