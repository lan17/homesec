# Rust WebRTC preview

WebRTC is the default backend for live camera viewing. Preview remains disabled
until enabled in config. HomeSec supervises a `homesec-webrtc` Rust helper for
each active camera. Viewers share one preview RTSP input per camera. Video-only
H.264 copy mode uses native Rust RTSP ingestion; transcoding and audio-enabled
configurations use FFmpeg for H.264 video and optional Opus audio. Recording,
motion detection, and push-to-talk retain their existing paths and camera
session requirements.

The initial deployment scope is direct UDP connectivity over LAN or VPN. HTTP
signaling uses the existing HomeSec server and authentication. Media travels
directly between the helper and browser, so an HTTP reverse proxy alone does not
provide connectivity. Optional browser STUN/TURN configuration is supported,
but public internet access, relay connectivity, and real-browser/device latency
must be validated separately before relying on them.

## Configuration

Enable preview in the existing `preview` section and configure the server IP:

```yaml
preview:
  enabled: true
  backend: webrtc
  token_ttl_s: 60
  idle_timeout_s: 30
  recording_policy: stop_on_recording
  config:
    helper_path: homesec-webrtc
    advertised_ip: "192.0.2.10" # Replace with this host's client-reachable LAN/VPN IP.
    udp_port_start: 8189
    udp_port_end: 8199
    max_viewers: 4
    negotiation_timeout_s: 10
    max_session_duration_s: 3600
    audio_enabled: true
    video_codec: h264
    ice_servers: []
```

`advertised_ip` is required when WebRTC preview is enabled; it may remain unset
while preview is disabled. It must identify the HomeSec host from the viewer's
network. Do not advertise a Docker bridge address, loopback address, or camera
address. A LAN address must also be routable from VPN clients; otherwise use a
VPN address and validate connectivity from every supported client network.

Explicit `backend: hls` settings remain HLS. Older configs that omit `backend`
but include HLS-specific settings such as `storage_dir`, `segment_duration_ms`,
`live_window_segments`, `audio_codec`, or `video_codec: auto` also retain HLS.
For new configurations, omitting `backend` selects WebRTC.
If an older enabled preview has no backend or HLS-specific settings, add
`backend: hls` to retain HLS, or configure `advertised_ip` to use WebRTC.

For clients that require STUN or a TURN relay, supply `ice_servers` entries with
`urls` (a list), optional `username`, and optional `credential_env` naming an
environment variable containing the credential. HomeSec returns resolved
credentials only in the authorized, short-lived preview descriptor. The helper
still needs a reachable advertised UDP address; this configuration does not
give the server its own TURN allocation. Validate the relay path from the target
network before enabling public access.

The UDP range is shared by active camera helpers. Match the configured range to
firewall rules and Docker port mappings. The helper's FFmpeg ingestion uses
separate loopback RTP sockets, which should not be published externally.

`max_viewers` limits peers for each camera, including sessions negotiating a
connection. Authentication leases are renewed by the UI. Expired leases and
failed connections close the associated peer; `max_session_duration_s` also caps
the lifetime of a session. After the final viewer leaves, `idle_timeout_s` bounds
how long the preview input and helper remain active.

The default `stop_on_recording` policy yields preview resources to recording.
`allow_during_recording` is best-effort and can consume another direct camera
session. Existing camera preflight and concurrency refusal behavior still apply.
A preview failure must not prevent recording or upload.
WebRTC preview temporarily refuses activation until background camera discovery
finishes, so discovered audio and camera session policy are applied before it
opens a preview input.

## Codecs and playback

`video_codec: h264` transcodes to a browser-compatible H.264 stream without
B-frames. This uses CPU and is the compatibility-first setting. `video_codec:
copy` avoids video transcoding but requires a compatible camera H.264 profile,
packetization, and keyframe cadence; validate new viewer joins and loss recovery
with the actual camera and browser before using it. H.265 and H.264 with B-frames
require transcoding for this path.

With `video_codec: copy` and `audio_enabled: false`, the helper connects directly
to the camera over RTSP/TCP using Retina. It does not launch FFmpeg for preview.
The same camera input feeds every preview viewer. Camera credentials travel only
through the private Python-to-helper control pipe. RTSP authentication, keepalive,
and teardown are handled by the client library.

Native copy supports H.264 Baseline, Main, and High profiles through level 3.1,
without B-frames. Negotiation checks the actual source profile and selects a
compatible receiver level at least as high as the camera's. SDP parameter sets and camera RTP
payload types are honored. Damaged or oversized RTP access units are discarded,
with delivery resuming at a valid keyframe. Unsupported video, changed profiles,
reversed video timestamps, or an overflowing source queue stop preview; select
the default `video_codec: h264` to transcode an incompatible camera.
Native connection setup is bounded to five seconds, media reads to ten seconds,
and stop/parent disconnect cancels the source. It never waits on camera I/O in
the WebRTC control loop.

Native media assembly and delivery queues have byte/count limits. Retina's RTSP
control-response parser currently exposes no response-size limit, so oversized
camera responses can consume memory before the I/O deadline. This first mode
is intended for operator-configured cameras on a trusted LAN/VPN; it does not
provide a total memory bound against a malicious RTSP server. A library-level
response cap is required before migrating shared recording into this process.

This is the first step of the [shared Rust media plan](shared-rust-media.md).
Recording and motion still use their existing independent inputs. FFmpeg and
ffprobe remain required for those paths and for preview transcoding/audio.

The preview input decoder uses slice threading rather than frame threading.
Frame threading queues future frames and can add about a second of delay at
15 fps on a many-core host; slice threading avoids that queue while preserving
source frame reordering. Encoding already uses FFmpeg's zero-latency tuning.

When `audio_enabled` is true and the camera supplies audio, FFmpeg converts it to
Opus. Camera AAC is not passed directly to WebRTC. The UI preserves its existing
mute behavior and push-to-talk coordination; changing the preview transport does
not change microphone-to-camera transport.

Closing a viewer detaches only that peer. The camera-level force-stop operation
and recording-priority shedding stop the shared preview and all its peers.
Runtime replacement also invalidates old sessions.

## Docker

The Docker image builds the helper with the pinned Rust toolchain and installs it
at `/usr/local/bin/homesec-webrtc`. Rust tooling is confined to the build stage;
FFmpeg is already part of the runtime image.

The bundled Compose file publishes UDP `8189-8199` alongside HTTP `8081`. The UDP
ports are used only by the WebRTC backend. Remove that mapping for HLS-only
deployments, or adjust both the mapping and preview config when choosing another
range. Docker must preserve the advertised UDP port numbers:

```yaml
ports:
  - "8081:8081"
  - "8189-8199:8189-8199/udp"
```

The HLS tmpfs mount can remain in place. WebRTC does not use it for its media.
Neither raw media nor signaling should be persisted as troubleshooting output.

## Local development and Python installations

The Python wheel does not contain a native helper. Build or install it from the
same HomeSec source revision used by the Python application.

For a source checkout on Linux or macOS, install these developer tools first:

- Rust via [rustup](https://rust-lang.org/tools/install/). The repository's
  `rust-toolchain.toml` selects the compiler and required components.
- A C compiler/linker for the bundled native crypto dependency. On macOS, use
  Xcode Command Line Tools (`xcode-select --install`).
- [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 20.19+
  or 22.12+, and pnpm 10.15.1 (the version in `ui/package.json`).
- FFmpeg, including `ffprobe`, on `PATH`.

On Debian/Ubuntu, install the compiler tools, FFmpeg, and libraries required by
the existing OpenCV dependency with:

```bash
sudo apt-get update
sudo apt-get install build-essential ffmpeg libgl1 libglib2.0-0
```

On Ubuntu 24.04, use `libglib2.0-0t64` in place of `libglib2.0-0`.

Then prepare the checkout:

```bash
make dev-setup
```

This checks the tools before syncing Python and UI dependencies from their
lockfiles, building the Rust helper, and building the UI. It does not install
global tools or OS packages, start services, or run database migrations.
Rustup may download the repository's pinned toolchain on its first use.

`make run` adds the checkout's release-helper directory to the application
`PATH`, so `preview.config.helper_path: homesec-webrtc` works without a global
helper installation. It retains its existing database-migration step; configure
the intended database and application config before running it. If invoking the
Python CLI directly, set `preview.config.helper_path` to the absolute
`native/webrtc/target/release/homesec-webrtc` path, or add its directory to `PATH`.
`CARGO_TARGET_DIR` is respected when choosing the release-helper directory.

After Rust edits, run `make rust-build` and stop/start preview to launch the new
helper. Python and UI changes do not require a Rust rebuild. For UI hot reload,
use `make ui-run-local VITE_API_PROXY_TARGET=http://127.0.0.1:8081`, replacing
the proxy URL with the backend's address.

For Python installations outside a source checkout, `cargo install --path
native/webrtc --locked` builds and installs the helper, normally in
`~/.cargo/bin`. Add that directory to the HomeSec service's `PATH`, or configure
an absolute helper path. An absent or incompatible helper makes WebRTC preview
unavailable; it does not affect the HLS backend.

```bash
make rust-check # Formatting, Clippy, and Rust tests.
make check      # Rust, Python, and UI checks.
```

Prebuilt standalone helper distribution is not provided by the Python release
workflow. Docker includes the helper; other installations need the source build
above.

## Validation and troubleshooting

An SDP answer proves signaling succeeded, not that media can reach the browser.
For a connected client with no picture, check the advertised IP, UDP mapping,
host/VPN firewall, camera input availability, codec compatibility, and receipt of
a decodable keyframe. A connection setup timeout should terminate the peer rather
than leave a camera input running indefinitely.

Compare HLS and WebRTC using the same camera, browser, and network. Record
camera-to-screen latency, startup time, CPU/memory, and the chosen codec mode.
Exercise two viewers, late joins, one viewer closing, network loss/reconnect,
camera stalls, helper termination, runtime reload, and recording plus preview
plus talk. Validate Chrome, Firefox, Safari, and the actual iOS WebView/device;
automated fixture results do not establish physical-device behavior.

Logs should contain stable errors and bounded operational context. Do not capture
API keys, preview tokens, RTSP credentials, SDP/ICE credentials, or raw media.

To roll back, select `preview.backend: hls` and replace `preview.config` with the
HLS configuration described in [preview deployment notes](preview-deployment.md).
Backend-specific config fields cannot be mixed. Runtime reload closes existing
WebRTC sessions; viewers can then attach using HLS.
