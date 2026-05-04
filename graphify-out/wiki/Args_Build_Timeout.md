# Args Build Timeout

> 20 nodes · cohesion 0.13

## Key Concepts

- **live_publisher.py** (17 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **Enum** (16 connections)
- **get_global_rtsp_timeout_capabilities()** (8 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **._build_codec_plans()** (7 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **capabilities.py** (5 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **_build_timeout_args()** (4 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **TimeoutOptionSupport** (4 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **_format_segment_duration()** (4 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **.build_ffmpeg_timeout_args()** (3 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **.build_ffmpeg_timeout_args_for_user_flags()** (3 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **_build_audio_codec_args()** (3 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **_hardware_h264_transcode_args()** (3 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **._probe_stream_info()** (3 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **_software_h264_transcode_args()** (3 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **.build_ffprobe_timeout_args()** (2 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **.__init__()** (2 connections) — `src/homesec/sources/rtsp/frame_pipeline.py`
- **_is_hls_browser_audio_copy_compatible()** (2 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **.probe()** (2 connections) — `src/homesec/sources/rtsp/live_publisher.py`
- **Build timeout args while respecting explicit user timeout options.** (1 connections) — `src/homesec/sources/rtsp/capabilities.py`
- **Support status for ffmpeg/ffprobe RTSP timeout options.** (1 connections) — `src/homesec/sources/rtsp/capabilities.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/capabilities.py`
- `src/homesec/sources/rtsp/frame_pipeline.py`
- `src/homesec/sources/rtsp/live_publisher.py`

## Audit Trail

- EXTRACTED: 85 (91%)
- INFERRED: 8 (9%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*