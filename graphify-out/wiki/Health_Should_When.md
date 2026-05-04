# Health Should When

> 16 nodes · cohesion 0.12

## Key Concepts

- **_StubSource** (24 connections) — `tests/homesec/test_api_routes.py`
- **test_health_endpoints_degrade_when_postgres_unavailable()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_list_cameras_includes_health_fields()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_list_cameras_serializes_model_config()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_root_health_endpoint_matches_versioned_health()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_stats_endpoint_includes_camera_counts_and_uptime()** (9 connections) — `tests/homesec/test_api_routes.py`
- **test_diagnostics_reports_degraded_when_storage_ping_raises()** (8 connections) — `tests/homesec/test_api_routes.py`
- **GET /stats should include camera totals and uptime.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /health should expose the same probe payload as /api/v1/health.** (1 connections) — `tests/homesec/test_api_routes.py`
- **Health endpoints should report degraded when pipeline runs but DB is unavailable** (1 connections) — `tests/homesec/test_api_routes.py`
- **Diagnostics should degrade when component ping raises and preserve error detail.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /cameras should include health fields.** (1 connections) — `tests/homesec/test_api_routes.py`
- **GET /cameras should serialize BaseModel configs.** (1 connections) — `tests/homesec/test_api_routes.py`
- **.__init__()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.is_healthy()** (1 connections) — `tests/homesec/test_api_routes.py`
- **.last_heartbeat()** (1 connections) — `tests/homesec/test_api_routes.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/test_api_routes.py`

## Audit Trail

- EXTRACTED: 72 (84%)
- INFERRED: 14 (16%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*