# Derive Url Detect

> 10 nodes · cohesion 0.31

## Key Concepts

- **derive_detect_rtsp_url()** (10 connections) — `src/homesec/sources/rtsp/url_derivation.py`
- **derive_probe_candidate_urls()** (5 connections) — `src/homesec/sources/rtsp/url_derivation.py`
- **url_derivation.py** (5 connections) — `src/homesec/sources/rtsp/url_derivation.py`
- **test_derive_detect_rtsp_url_from_stream1()** (3 connections) — `tests/homesec/rtsp/test_helpers.py`
- **_replace_stream_suffix()** (3 connections) — `src/homesec/sources/rtsp/url_derivation.py`
- **_replace_subtype_value()** (3 connections) — `src/homesec/sources/rtsp/url_derivation.py`
- **DerivedRTSPUrl** (2 connections) — `src/homesec/sources/rtsp/url_derivation.py`
- **Derive detect stream from trailing /stream1 paths.** (1 connections) — `tests/homesec/rtsp/test_helpers.py`
- **Derive a likely lower-cost detect stream URL from a primary URL.** (1 connections) — `src/homesec/sources/rtsp/url_derivation.py`
- **Derive additional RTSP candidates worth probing during startup preflight.** (1 connections) — `src/homesec/sources/rtsp/url_derivation.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/url_derivation.py`
- `tests/homesec/rtsp/test_helpers.py`

## Audit Trail

- EXTRACTED: 27 (79%)
- INFERRED: 7 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*