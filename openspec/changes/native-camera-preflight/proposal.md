# Proposal

## Why

Camera startup and setup still depend on the operating system's FFmpeg tools even when Rust owns media ingestion. On Lenovo, an older `ffmpeg` rejected `-fps_mode` during startup negotiation, preventing otherwise configured cameras from reaching native ingestion; pinning the libraries does not fix that dependency while probing still shells out.

## What Changes

- Replace `ffprobe` metadata discovery, detect-stream recovery probes, and HLS metadata/audio planning with bounded Rust operations using the pinned FFmpeg libraries.
- Replace CLI-based recording-profile trials, decode/session checks, and codec/hardware capability inspection used by RTSP startup and setup Test Connection.
- Distinguish camera metadata, linked-build capabilities, and successful validation of a selected operation. A reachable camera advertising another codec is not thereby offline, and metadata discovery is not proof that a recorder can handle it.
- Preserve candidate ordering, stream selection, recording-profile negotiation, timestamp retry policy, existing session classifications, background startup retries, and recording priority. Bound helper lifetime, camera sessions, output, and temporary trial files on every exit path.
- Make setup report native probing readiness rather than finding an arbitrary executable on `PATH`; keep the existing response shape and `ffmpeg` check identity.
- Keep shared-session accounting in ticket 192 and remaining media execution replacement in ticket 195. This ticket removes CLI use from inspection and preflight; epic-wide removal of media subprocesses and runtime packaging remains ticket 196.

### Confirmed compatibility scope

The current pinned build cannot validate every profile the existing CLI path can attempt: it lacks HEVC decoding/parsing, AAC encoding, and hardware support. Leonidas confirmed that ticket 194 must expand the pinned build as needed for probing and validation while preserving existing camera support. An incomplete native validator must not silently choose no audio, claim a camera is offline, or fall back to a CLI probe. Remaining recording and preview execution replacement stays in ticket 195.

## Capabilities

### New Capabilities

None. Camera inspection belongs to the existing capture and setup contracts.

### Modified Capabilities

- `capture-recording`: require CLI-independent RTSP discovery and startup/recovery validation, truthful capability reporting, preserved negotiation behavior, and bounded probe cleanup.
- `setup-configuration`: make the media readiness check and registered RTSP connection test use the native inspection boundary without saving configuration or overstating readiness.

## Impact

- Python RTSP source modules: `discovery.py`, `preflight.py`, `hardware.py`, recovery in `core.py`, `live_publisher.py` metadata consumers, and `setup_probe.py`; setup readiness in `services/setup.py` must respect the source/runtime boundary.
- Native helper inspection/validation operations, shared input/deadline primitives, typed private IPC models, and the pinned build recipe/capability coverage.
- Protocol fixtures, helper/source/setup tests, build verification, and operator/reference evidence. Existing preview/talk contracts remain constraints; preview transport, high-level pipeline policy, Python YOLO, repository persistence, and public configuration are outside this change.
- Tracking: [ticket 194](https://app.notion.com/p/3fdd8336c59f813eac7effebd396fb79), under the [Rust migration epic](https://app.notion.com/p/3efd8336c59f8195b6b5c50c71c1d5e7). This PR prepares planning artifacts only; it does not claim an implemented runtime fix.
