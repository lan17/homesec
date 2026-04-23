# Preview Deployment Notes

`preview.config.storage_dir` is a scratch directory intended for HLS preview output
(short-lived playlists and segments). It is not durable clip storage. Keep it
separate from `recordings/` and any long-term storage backend.

Current implementation status: RTSP sources can serve on-demand HLS preview when
`preview.enabled` is true and the runtime can attach or start preview for the
camera. Artifacts are only written while preview is active and are cleaned up on
stop or runtime shutdown. If preview is disabled (or the source backend does not
support preview), this directory remains unused.

By default, preview yields to active recording with:

```yaml
preview:
  recording_policy: stop_on_recording
```

You can opt into best-effort concurrent preview during recording with:

```yaml
preview:
  recording_policy: allow_during_recording
```

That mode can open an extra direct RTSP session to the camera. Use it only if
your camera tolerates concurrent motion, recording, and preview consumers.

## Recommended Storage

Use a tmpfs mount for `preview.config.storage_dir` when possible, and cap its size.

Why:

- preview segments are short-lived scratch data
- tmpfs avoids unnecessary disk churn
- preview I/O stays isolated from recording and upload paths
- tmpfs data is disposable and cleared on host reboot or container restart
- tmpfs uses RAM (and may count against container memory limits)
- capping size bounds preview RAM usage

The default preview storage path is:

```yaml
preview:
  config:
    storage_dir: /tmp/homesec-preview
```

If you use a different mount point, keep the config value aligned with the mounted
path inside the HomeSec process or container.

## Sizing Guidance

Preview footprint scales with:

- `segment_duration_ms`
- `live_window_segments`
- number of simultaneously active preview publishers
- actual encoded preview bitrate

Approximate active-window size per camera:

```text
encoded_bitrate_bytes_per_second
  * segment_duration_seconds
  * live_window_segments
```

Plan extra headroom for muxing overhead, playlist files, and delete lag.

The exact bitrate depends on source content and codec handling:

- `video_codec: copy` preserves the upstream video bitrate
- `video_codec: h264` or `audio_codec: aac` can reduce or increase output size
  depending on the source
- upstream H.264 sources are the cheapest path for browser playback; H.265/HEVC
  sources will transcode to H.264 for preview and therefore increase CPU or
  hardware-encoder load

At the default `1000 ms` segments and `4` retained segments, a conservative
starting point is `64 MiB` of tmpfs per actively previewed camera. Increase that
if you:

- copy high-bitrate upstream video
- keep a longer live window
- preview multiple cameras at the same time

## Docker Guidance

The bundled `docker-compose.yml` now mounts the default preview scratch
directory as tmpfs:

```yaml
services:
  homesec:
    tmpfs:
      - /tmp/homesec-preview:size=${HOMESEC_PREVIEW_TMPFS_SIZE:-256m}
```

That matches the default `preview.config.storage_dir`. The bundled default is
`256 MiB`, which is conservative headroom for a few simultaneous previews at the
default segment/window settings without forcing operators to edit the Compose
file on day one. If you change the path in config, change the tmpfs mount path
too. If you know your expected concurrency, set `HOMESEC_PREVIEW_TMPFS_SIZE`
explicitly in `.env` to match it.

The HomeSec Docker image runs as a non-root `homesec` user created by:

```dockerfile
RUN useradd --create-home --shell /bin/bash homesec
```

On the current `python:3.14-slim-bookworm` base image, that resolves to
`uid=1000,gid=1000`, but the Dockerfile does not pin numeric IDs. If you want
to set tmpfs ownership and mode explicitly, inspect the image you deploy instead
of assuming those numbers will stay fixed:

```bash
docker run --rm --entrypoint id leva/homesec:latest homesec
```

Example with explicit ownership and mode:

```yaml
services:
  homesec:
    tmpfs:
      - /tmp/homesec-preview:uid=1000,gid=1000,mode=1700,size=${HOMESEC_PREVIEW_TMPFS_SIZE:-256m}
```

Replace `uid`/`gid` with the user/group reported for the image you actually run.

## Bare-Metal Guidance

On bare-metal Linux, point `preview.config.storage_dir` at a tmpfs-backed path
such as `/run/homesec-preview`, `/dev/shm/homesec-preview`, or a dedicated tmpfs
mount managed by your init system.

Operator checklist:

- create the directory before starting HomeSec
- ensure the HomeSec process can read and write it
- restrict permissions; HLS playlists and segments contain camera audio/video
- keep it on tmpfs, not the same persistent disk used for clips
- size it for your expected number of concurrent previews

## Operational Notes

- Preview storage is disposable. Losing it interrupts live preview but should not
  delete recorded clips.
- Treat preview storage as sensitive media. Keep it private to the HomeSec process.
- Keep preview storage independent from `output_dir`, uploaded clip storage, and
  Postgres data paths.
- Larger `segment_duration_ms` or `live_window_segments` values increase memory
  use linearly.
- If preview storage fills up, treat that as a preview-capacity issue first. Do
  not solve it by moving preview segments into the same durable path as recordings.
