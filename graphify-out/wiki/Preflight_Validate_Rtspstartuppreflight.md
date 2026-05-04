# Preflight Validate Rtspstartuppreflight

> 144 nodes · cohesion 0.03

## Key Concepts

- **RTSPStartupPreflight** (62 connections) — `src/homesec/sources/rtsp/preflight.py`
- **list** (45 connections)
- **ProbeStreamInfo** (44 connections) — `src/homesec/sources/rtsp/discovery.py`
- **CameraProbeResult** (34 connections) — `src/homesec/sources/rtsp/discovery.py`
- **FfprobeStreamDiscovery** (32 connections) — `src/homesec/sources/rtsp/discovery.py`
- **_FakeDiscovery** (31 connections) — `tests/homesec/test_rtsp_preflight.py`
- **ProbeError** (26 connections) — `src/homesec/sources/rtsp/discovery.py`
- **RecordingProfile** (23 connections) — `src/homesec/sources/rtsp/recording_profile.py`
- **test_rtsp_preflight.py** (22 connections) — `tests/homesec/test_rtsp_preflight.py`
- **preflight.py** (20 connections) — `src/homesec/sources/rtsp/preflight.py`
- **.run()** (15 connections) — `src/homesec/sources/rtsp/preflight.py`
- **RecordingValidationSignals** (14 connections) — `src/homesec/sources/rtsp/preflight.py`
- **._validate_concurrent_stream_opens()** (14 connections) — `src/homesec/sources/rtsp/preflight.py`
- **._validate_recording_profile()** (13 connections) — `src/homesec/sources/rtsp/preflight.py`
- **discovery.py** (13 connections) — `src/homesec/sources/rtsp/discovery.py`
- **RecordingValidationResult** (12 connections) — `src/homesec/sources/rtsp/preflight.py`
- **SelectionError** (12 connections) — `src/homesec/sources/rtsp/preflight.py`
- **NegotiationError** (11 connections) — `src/homesec/sources/rtsp/preflight.py`
- **_SessionOpenSpec** (11 connections) — `src/homesec/sources/rtsp/preflight.py`
- **ConcurrentStreamOpenResult** (10 connections) — `src/homesec/sources/rtsp/preflight.py`
- **StreamDiscovery** (10 connections) — `src/homesec/sources/rtsp/preflight.py`
- **_is_timeout_option_error()** (9 connections) — `src/homesec/sources/rtsp/utils.py`
- **._probe_single()** (8 connections) — `src/homesec/sources/rtsp/discovery.py`
- **_build_timeout_attempts()** (8 connections) — `src/homesec/sources/rtsp/utils.py`
- **Exception** (7 connections)
- *... and 119 more nodes in this community*

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/notifiers/multiplex.py`
- `src/homesec/plugins/alert_policies/default.py`
- `src/homesec/sources/rtsp/core.py`
- `src/homesec/sources/rtsp/discovery.py`
- `src/homesec/sources/rtsp/frame_pipeline.py`
- `src/homesec/sources/rtsp/live_publisher.py`
- `src/homesec/sources/rtsp/preflight.py`
- `src/homesec/sources/rtsp/recorder.py`
- `src/homesec/sources/rtsp/recording_profile.py`
- `src/homesec/sources/rtsp/utils.py`
- `tests/homesec/rtsp/test_live_publisher.py`
- `tests/homesec/test_rtsp_preflight.py`

## Audit Trail

- EXTRACTED: 473 (56%)
- INFERRED: 368 (44%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*