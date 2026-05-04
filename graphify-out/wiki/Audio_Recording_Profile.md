# Audio Recording Profile

> 14 nodes · cohesion 0.18

## Key Concepts

- **build_recording_profile_candidates()** (8 connections) — `src/homesec/sources/rtsp/recording_profile.py`
- **recording_profile.py** (5 connections) — `src/homesec/sources/rtsp/recording_profile.py`
- **test_rtsp_recording_profile.py** (5 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **test_candidates_include_passthrough_vsync_defaults()** (3 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **test_candidates_prefer_audio_copy_for_compatible_codec()** (3 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **test_candidates_skip_audio_copy_for_incompatible_codec()** (3 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **test_default_profile_includes_passthrough_vsync_defaults()** (3 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **is_mp4_audio_copy_compatible()** (2 connections) — `src/homesec/sources/rtsp/recording_profile.py`
- **Tests for RTSP recording profile candidate generation.** (1 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **Incompatible audio codecs should not produce mp4 audio-copy candidate.** (1 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **MP4-safe audio codecs should keep copy-audio as first candidate.** (1 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **All generated recording profiles should default to passthrough timing mode.** (1 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **Default recording profile should also preserve passthrough timing mode.** (1 connections) — `tests/homesec/test_rtsp_recording_profile.py`
- **Build startup-only negotiation candidates in deterministic order.** (1 connections) — `src/homesec/sources/rtsp/recording_profile.py`

## Relationships

- No strong cross-community connections detected

## Source Files

- `src/homesec/sources/rtsp/recording_profile.py`
- `tests/homesec/test_rtsp_recording_profile.py`

## Audit Trail

- EXTRACTED: 30 (79%)
- INFERRED: 8 (21%)
- AMBIGUOUS: 0 (0%)

---

*Part of the graphify knowledge wiki. See [[index]] to navigate.*