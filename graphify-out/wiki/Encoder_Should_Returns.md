# Encoder Should Returns

> 20 nodes · cohesion 0.10

## Key Concepts

- **test_hardware.py** (10 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_select_h264_encoder_adds_vaapi_device_and_hwupload()** (3 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_select_h264_encoder_returns_matching_videotoolbox_encoder()** (3 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_select_h264_encoder_returns_none_when_encoder_is_unavailable()** (3 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_test_decode_logs_backend_name_on_unexpected_exception()** (3 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_test_decode_returns_false_on_known_errors()** (3 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_test_decode_treats_timeout_as_success()** (3 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_check_nvidia_returns_false_on_error()** (2 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_detect_returns_software_when_no_accel_available()** (2 connections) — `tests/homesec/rtsp/test_hardware.py`
- **test_test_hwaccel_returns_false_when_ffmpeg_missing()** (2 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Tests for hardware acceleration detection.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Detector should fall back to software when no accel works.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Encoder selection should map VideoToolbox decode to VideoToolbox H.264 encode.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Missing ffmpeg should disable hwaccel detection.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **VAAPI selection should keep the render device and upload requirement explicit.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Missing ffmpeg encoder support should fall back to software encode.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Known decode errors should return False.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Timeouts should be treated as a successful decode check.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **Unexpected decode probe failures should log the active backend name.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`
- **NVIDIA check should return False when nvidia-smi fails.** (1 connections) — `tests/homesec/rtsp/test_hardware.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `tests/homesec/rtsp/test_hardware.py`

## Audit Trail

- EXTRACTED: 38 (86%)
- INFERRED: 6 (14%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*