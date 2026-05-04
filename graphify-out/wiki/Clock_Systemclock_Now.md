# Clock Systemclock Now

> 8 nodes · cohesion 0.25

## Key Concepts

- **Clock** (4 connections) — `src/homesec/sources/rtsp/clock.py`
- **SystemClock** (4 connections) — `src/homesec/sources/rtsp/clock.py`
- **Clock** (3 connections) — `src/homesec/sources/rtsp/recorder.py`
- **Clock** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **SystemClock** (2 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **SystemClock.now** (1 connections) — `src/homesec/sources/rtsp/clock.py`
- **SystemClock.sleep** (1 connections) — `src/homesec/sources/rtsp/clock.py`
- **Clock** (1 connections) — `src/homesec/sources/rtsp/live_publisher.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/clock.py`
- `src/homesec/sources/rtsp/frame_pipeline.py`
- `src/homesec/sources/rtsp/live_publisher.py`
- `src/homesec/sources/rtsp/recorder.py`

## Audit Trail

- EXTRACTED: 16 (89%)
- INFERRED: 2 (11%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*