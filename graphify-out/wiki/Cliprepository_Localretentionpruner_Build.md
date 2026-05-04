# Cliprepository Localretentionpruner Build

> 13 nodes · cohesion 0.17

## Key Concepts

- **ClipRepository** (5 connections) — `src/homesec/repository/clip_repository.py`
- **LocalRetentionPruner** (4 connections) — `src/homesec/retention/pruner.py`
- **build_local_retention_pruner** (4 connections) — `src/homesec/retention/wiring.py`
- **ClipRepository** (2 connections) — `src/homesec/retention/pruner.py`
- **ClipRepository** (2 connections) — `src/homesec/retention/wiring.py`
- **LocalRetentionPruner** (2 connections) — `src/homesec/retention/wiring.py`
- **is_retryable_pg_error** (1 connections) — `src/homesec/repository/clip_repository.py`
- **RetryConfig** (1 connections) — `src/homesec/repository/clip_repository.py`
- **ClipRepository** (1 connections) — `src/homesec/repository/__init__.py`
- **build_local_retention_pruner** (1 connections) — `src/homesec/retention/__init__.py`
- **LocalRetentionPruner** (1 connections) — `src/homesec/retention/__init__.py`
- **ClipStatus** (1 connections) — `src/homesec/retention/pruner.py`
- **RetentionConfig** (1 connections) — `src/homesec/retention/wiring.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/repository/__init__.py`
- `src/homesec/repository/clip_repository.py`
- `src/homesec/retention/__init__.py`
- `src/homesec/retention/pruner.py`
- `src/homesec/retention/wiring.py`

## Audit Trail

- EXTRACTED: 26 (100%)
- INFERRED: 0 (0%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*