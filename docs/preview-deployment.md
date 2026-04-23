# Preview Deployment Notes

`preview.config.storage_dir` is the scratch directory used by the HLS preview publisher
for short-lived playlists and segments. It is not durable clip storage. Keep it
separate from `recordings/` and any long-term storage backend.

Preview artifacts are only written when a preview publisher is available and active.
If preview is disabled (or the source backend does not support preview yet), this
directory will be unused. When preview is unavailable, activation may be refused as
`preview_temporarily_unavailable` and nothing will be written.

## Recommended Storage

Use a tmpfs mount for `preview.config.storage_dir` when possible, and cap its size.

Why:

- preview segments are short-lived scratch data
- tmpfs avoids unnecessary disk churn
- preview I/O stays isolated from recording and upload paths
- tmpfs data is disposable and cleared on host reboot
- capping size prevents preview from consuming unbounded RAM

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

At the default `1000 ms` segments and `4` retained segments, a conservative
starting point is `64 MiB` of tmpfs per actively previewed camera. Increase that
if you:

- copy high-bitrate upstream video
- keep a longer live window
- preview multiple cameras at the same time

## Docker Guidance

The bundled `docker-compose.yml` does not add a preview tmpfs mount yet. If you
enable preview in Docker, add a tmpfs mount that matches `preview.config.storage_dir`.

Example:

```yaml
services:
  homesec:
    tmpfs:
      - /tmp/homesec-preview:size=64m
```

If you change the path in config, change the tmpfs mount path too.

## Bare-Metal Guidance

On bare-metal Linux, point `preview.config.storage_dir` at a tmpfs-backed path
such as `/run/homesec-preview`, `/dev/shm/homesec-preview`, or a dedicated tmpfs
mount managed by your init system.

Operator checklist:

- create the directory before starting HomeSec
- ensure the HomeSec process can read and write it
- keep it on tmpfs, not the same persistent disk used for clips
- size it for your expected number of concurrent previews

## Operational Notes

- Preview storage is disposable. Losing it interrupts live preview but should not
  delete recorded clips.
- Keep preview storage independent from `output_dir`, uploaded clip storage, and
  Postgres data paths.
- Larger `segment_duration_ms` or `live_window_segments` values increase memory
  use linearly.
- If preview storage fills up, treat that as a preview-capacity issue first. Do
  not solve it by moving preview segments into the same durable path as recordings.
