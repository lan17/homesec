# Disabled Vlm Build

> 10 nodes · cohesion 0.20

## Key Concepts

- **DisabledVLMAnalyzer** (14 connections) — `src/homesec/runtime/disabled_vlm.py`
- **.build_bundle()** (10 connections) — `src/homesec/runtime/assembly.py`
- **.analyze()** (2 connections) — `src/homesec/runtime/disabled_vlm.py`
- **disabled_vlm.py** (2 connections) — `src/homesec/runtime/disabled_vlm.py`
- **VLMAnalyzer** (2 connections)
- **Build a runtime bundle and clean up partial state on failure.** (1 connections) — `src/homesec/runtime/assembly.py`
- **.ping()** (1 connections) — `src/homesec/runtime/disabled_vlm.py`
- **.shutdown()** (1 connections) — `src/homesec/runtime/disabled_vlm.py`
- **Disabled VLM analyzer implementation for run_mode=never.** (1 connections) — `src/homesec/runtime/disabled_vlm.py`
- **Runtime-only analyzer used when VLM is disabled.** (1 connections) — `src/homesec/runtime/disabled_vlm.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/runtime/assembly.py`
- `src/homesec/runtime/disabled_vlm.py`

## Audit Trail

- EXTRACTED: 19 (54%)
- INFERRED: 16 (46%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*