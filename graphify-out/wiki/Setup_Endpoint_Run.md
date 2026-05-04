# Setup Endpoint Run

> 10 nodes · cohesion 0.20

## Key Concepts

- **setup.py** (5 connections) — `src/homesec/api/routes/setup.py`
- **finalize_setup_endpoint()** (3 connections) — `src/homesec/api/routes/setup.py`
- **get_setup_status_endpoint()** (3 connections) — `src/homesec/api/routes/setup.py`
- **run_setup_preflight_endpoint()** (3 connections) — `src/homesec/api/routes/setup.py`
- **test_setup_connection_endpoint()** (3 connections) — `src/homesec/api/routes/setup.py`
- **Setup onboarding endpoints.** (1 connections) — `src/homesec/api/routes/setup.py`
- **Return setup completion status for onboarding UX.** (1 connections) — `src/homesec/api/routes/setup.py`
- **Run setup preflight checks for onboarding UX.** (1 connections) — `src/homesec/api/routes/setup.py`
- **Run a non-persistent connection test for setup-managed integrations.** (1 connections) — `src/homesec/api/routes/setup.py`
- **Persist finalized setup config and activate runtime in-process.** (1 connections) — `src/homesec/api/routes/setup.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/api/routes/setup.py`

## Audit Trail

- EXTRACTED: 18 (82%)
- INFERRED: 4 (18%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*