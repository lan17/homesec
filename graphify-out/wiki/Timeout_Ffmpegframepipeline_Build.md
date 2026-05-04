# Timeout Ffmpegframepipeline Build

> 17 nodes · cohesion 0.14

## Key Concepts

- **FfmpegRecorder.start** (7 connections) — `src/homesec/sources/rtsp/recorder.py`
- **FfmpegFramePipeline._get_frame_pipe** (5 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **_build_timeout_attempts** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **_format_cmd** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **_is_timeout_option_error** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **_redact_rtsp_url** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **FfmpegFramePipeline.start** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **_build_timeout_attempts** (2 connections) — `src/homesec/sources/rtsp/recorder.py`
- **_format_cmd** (2 connections) — `src/homesec/sources/rtsp/recorder.py`
- **_is_timeout_option_error** (2 connections) — `src/homesec/sources/rtsp/recorder.py`
- **_redact_rtsp_url** (2 connections) — `src/homesec/sources/rtsp/recorder.py`
- **RTSPTimeoutCapabilities** (2 connections) — `src/homesec/sources/rtsp/recorder.py`
- **_build_timeout_attempts** (2 connections) — `src/homesec/sources/rtsp/utils.py`
- **_format_cmd** (2 connections) — `src/homesec/sources/rtsp/utils.py`
- **_is_timeout_option_error** (2 connections) — `src/homesec/sources/rtsp/utils.py`
- **_redact_rtsp_url** (2 connections) — `src/homesec/sources/rtsp/utils.py`
- **FfmpegFramePipeline.read_frame** (1 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/frame_pipeline.py`
- `src/homesec/sources/rtsp/recorder.py`
- `src/homesec/sources/rtsp/utils.py`

## Audit Trail

- EXTRACTED: 31 (76%)
- INFERRED: 2 (5%)
- AMBIGUOUS: 8 (20%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*