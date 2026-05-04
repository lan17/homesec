# App Contract Openapi

> 19 nodes · cohesion 0.12

## Key Concepts

- **create_app** (5 connections) — `src/homesec/api/server.py`
- **create_contract_app** (5 connections) — `src/homesec/api/server.py`
- **_configure_ui_serving** (4 connections) — `src/homesec/api/server.py`
- **APIServer create_app create_contract_app exports** (3 connections) — `src/homesec/api/__init__.py`
- **build_openapi_schema** (3 connections) — `src/homesec/api/openapi_export.py`
- **create_contract_app** (2 connections) — `src/homesec/api/openapi_export.py`
- **main** (2 connections) — `src/homesec/api/openapi_export.py`
- **write_openapi_schema** (2 connections) — `src/homesec/api/openapi_export.py`
- **register_exception_handlers** (2 connections) — `src/homesec/api/server.py`
- **register_routes** (2 connections) — `src/homesec/api/server.py`
- **src/homesec/api/server.py** (2 connections) — `src/homesec/api/server.py`
- **src/homesec/api/__init__.py** (1 connections) — `src/homesec/api/__init__.py`
- **_build_parser** (1 connections) — `src/homesec/api/openapi_export.py`
- **src/homesec/api/openapi_export.py** (1 connections) — `src/homesec/api/openapi_export.py`
- **CORSMiddleware** (1 connections) — `src/homesec/api/server.py`
- **FastAPIServerConfig** (1 connections) — `src/homesec/api/server.py`
- **FileResponse** (1 connections) — `src/homesec/api/server.py`
- **_is_allowed_root_static_file** (1 connections) — `src/homesec/api/server.py`
- **StaticFiles** (1 connections) — `src/homesec/api/server.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/__init__.py`
- `src/homesec/api/openapi_export.py`
- `src/homesec/api/server.py`

## Audit Trail

- EXTRACTED: 40 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*