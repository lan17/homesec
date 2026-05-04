# Clippipeline Retention Prune

> 13 nodes · cohesion 0.17

## Key Concepts

- **ClipPipeline** (9 connections) — `src/homesec/pipeline/core.py`
- **ClipPipeline.request_retention_prune** (3 connections) — `src/homesec/pipeline/core.py`
- **RetentionPruner** (3 connections) — `src/homesec/pipeline/core.py`
- **ClipPipeline._run_retention_prune** (2 connections) — `src/homesec/pipeline/core.py`
- **AlertPolicy** (1 connections) — `src/homesec/pipeline/core.py`
- **ClipRepository** (1 connections) — `src/homesec/pipeline/core.py`
- **Config** (1 connections) — `src/homesec/pipeline/core.py`
- **Notifier** (1 connections) — `src/homesec/pipeline/core.py`
- **ObjectFilter** (1 connections) — `src/homesec/pipeline/core.py`
- **ClipPipeline._on_retention_prune_done** (1 connections) — `src/homesec/pipeline/core.py`
- **StorageBackend** (1 connections) — `src/homesec/pipeline/core.py`
- **VLMAnalyzer** (1 connections) — `src/homesec/pipeline/core.py`
- **ClipPipeline** (1 connections) — `src/homesec/pipeline/__init__.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/pipeline/__init__.py`
- `src/homesec/pipeline/core.py`

## Audit Trail

- EXTRACTED: 26 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*