# Frame Pipeline When

> 31 nodes · cohesion 0.09

## Key Concepts

- **FfmpegFramePipeline** (35 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **FakeProcess** (11 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **FakeClock** (8 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **test_frame_pipeline.py** (8 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **FakeStdout** (7 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **_make_pipeline()** (7 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **ErrorStdout** (6 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **test_frame_pipeline_retries_without_timeouts_when_unsupported()** (6 connections) — `tests/homesec/rtsp/test_runtime.py`
- **test_frame_pipeline_drops_oldest_when_full()** (5 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **test_frame_pipeline_reader_error_stops_loop()** (5 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.stop()** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **._stop_process()** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **.__init__()** (2 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.exit_code()** (1 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **.is_running()** (1 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **.read_frame()** (1 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **.set_motion_profile()** (1 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **.read()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.__init__()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.now()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.sleep()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.__init__()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.kill()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.poll()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- **.terminate()** (1 connections) — `tests/homesec/rtsp/test_frame_pipeline.py`
- *... and 6 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/frame_pipeline.py`
- `tests/homesec/rtsp/test_frame_pipeline.py`
- `tests/homesec/rtsp/test_runtime.py`

## Audit Trail

- EXTRACTED: 78 (64%)
- INFERRED: 44 (36%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*