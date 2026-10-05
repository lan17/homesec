# Rust WebRTC preview

WebRTC is the default backend for live camera viewing. Preview remains disabled
until enabled in config. HomeSec supervises a `homesec-webrtc` Rust helper for
each active camera. Viewers share one preview RTSP input per camera. Video-only
H.264 copy mode uses native Rust RTSP ingestion; transcoding and audio-enabled
configurations use FFmpeg for H.264 video and optional Opus audio. Recording,
motion detection, and push-to-talk retain separate camera inputs. Eligible CPU
H.264 motion detection now uses the Rust helper and linked FFmpeg/OpenCV libraries;
recording still uses the existing FFmpeg process and Python recording policy.

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
On initial connection, the first observed picture is discarded because the
camera may have started sending midway through it. Playback waits for the next
complete keyframe; startup can therefore take one camera GOP. Use a short camera
keyframe interval that fits within the ten-second media deadline.
Native connection setup is bounded to five seconds, media reads to ten seconds,
and stop/parent disconnect cancels the source. It never waits on camera I/O in
the WebRTC control loop. Python requests graceful stop and allows camera teardown
to finish before falling back to bounded process-group termination.

Native media assembly and delivery queues have byte/count limits. Retina's RTSP
client options currently expose no response-size limit, so oversized
camera responses can consume memory before the I/O deadline. This first mode
is intended for operator-configured cameras on a trusted LAN/VPN; it does not
provide a total memory bound against a malicious RTSP server. A library-level
response cap is required before migrating shared recording into this process.

This is the first step of the [shared Rust media plan](shared-rust-media.md).
Recording and motion still own independent inputs. FFmpeg and ffprobe remain
required for recording, compatible motion fallback, and preview transcoding/audio.

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

## Native motion detection

HomeSec prefers Rust motion detection for compatible H.264 CPU inputs when the
helper is available. Rust calls the bundled FFmpeg decode and filter libraries
directly, then calls statically linked OpenCV 4.12.0 for Gaussian blur, absolute
difference, thresholding, and changed-pixel counting. The detector borrows each
prepared input frame for one call and reuses its owned image buffers.
It prepares the same 320x240 grayscale frames at 10 fps and uses the existing
motion settings, including blur normalization and recording sensitivity.
If startup cannot supply its first frame within the source's existing read or
readiness deadline, HomeSec releases the native helper and selects compatibility
motion. This avoids repeated native restarts for cameras with longer keyframe
intervals. Once native input supplies a frame, missing frames continue through
the existing source stall and reconnect policy.

The RTSP source supervises the motion helper using the selected motion stream,
independent of preview viewers. Python receives typed motion observations rather
than raw pixels in JSON and retains recording, reconnect, stall, and upload policy.
Rust does not record video or audio in this stage.

Hardware decoding, custom FFmpeg input flags, unsupported camera inputs, or an
unavailable native helper use the existing FFmpeg/OpenCV path. This compatibility
fallback preserves the existing configuration; it does not require a second set
of motion settings. A helper failure must release its camera input before fallback
opens another one. Operational logs identify the selected path and stable failure
reasons without exposing camera credentials or media.

## Docker

The Docker image builds the helper with the pinned Rust toolchain and installs it
at `/usr/local/bin/homesec-webrtc`. The same `make rust-build` recipe used locally
builds verified FFmpeg 8.1.3 and OpenCV 4.12.0 source and statically links their
libraries into the helper. Rust, native headers, CMake, and libclang are confined
to the build stage. The image checks that the runtime can load the helper and
that it has no shared OpenCV or libav dependency. Standard C/C++ runtime libraries
remain OS dependencies.
The system FFmpeg command-line tools remain in the image for recording, preview
transcoding/audio, and compatibility motion.

The bundled FFmpeg license and source/build notice are installed in
`/usr/share/licenses/homesec-ffmpeg/`. The
[notice](../native/webrtc/FFMPEG-NOTICE.md) identifies the exact source archive,
checksum, and recipe for rebuilding the helper. OpenCV, its bundled zlib, and the
Rust binding license/notice are installed in `/usr/share/licenses/homesec-opencv/`;
the [OpenCV notice](../native/webrtc/OPENCV-NOTICE.md) identifies their sources.

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
- Python 3, Make, C and C++ compilers/linkers, Clang, and the libclang shared library for
  native bindings.
  On macOS, use Xcode Command Line Tools (`xcode-select --install`).
- [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 20.19+
  or 22.12+, and pnpm 10.15.1 (the version in `ui/package.json`).
- FFmpeg, including `ffprobe`, on `PATH` for recording, preview transcoding/audio,
  and compatibility motion.
- `pkg-config` so the Rust build can find the privately built libraries.
- CMake to compile the private OpenCV libraries.
- Optional NASM on x86 hosts for FFmpeg assembly optimizations. The build disables
  x86 assembly when NASM is unavailable.

On Debian/Ubuntu, install the compiler tools, FFmpeg, and libraries required by
the existing OpenCV dependency with:

```bash
sudo apt-get update
sudo apt-get install build-essential python3 cmake clang libclang-dev pkg-config ffmpeg nasm \
  libgl1 libglib2.0-0
```

On Ubuntu 24.04, use `libglib2.0-0t64` in place of `libglib2.0-0`.

On macOS with Xcode Command Line Tools installed:

```bash
brew install python cmake ffmpeg pkg-config
```

`make rust-build` and `make rust-check` download exact
[FFmpeg 8.1.3](https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz) and
[OpenCV 4.12.0](https://github.com/opencv/opencv/tree/4.12.0) source archives,
verify their pinned SHA-256 checksums, and build the required libraries privately.
OpenCV builds only `core` and `imgproc`, with bundled static zlib. Cargo pins the
Rust bindings to `opencv = "=0.101.0"`; the native version and archive checksum
are recorded in `native/webrtc/Cargo.toml` metadata and enforced by the build.
System FFmpeg/OpenCV development packages are unnecessary. The helper links
private static archives and does not load system OpenCV or libav shared libraries.
Use the Make targets or build wrapper; direct Cargo builds without the pinned
private build environment are refused.

Verified downloads and compiled libraries remain under the operating system's
temporary directory in `homesec-native-<uid>/`.
Matching platform, compiler, and build recipes reuse the caches. Clearing them
causes a fresh download and build. No libraries are installed globally.

Each private FFmpeg installation retains `COPYING.LGPLv2.1`. The
[bundled-component notice](../native/webrtc/FFMPEG-NOTICE.md) records its source
and rebuild recipe; keep the license and notice with a separately packaged helper.
The OpenCV installation retains `LICENSE` and `ZLIB-LICENSE`; distribute these,
`native/webrtc/OPENCV-BINDINGS-LICENSE`, and the
[OpenCV notice](../native/webrtc/OPENCV-NOTICE.md) with the helper too.

The first build needs internet access and compiles both libraries using the available CPU
count. Limit CPU and memory use on a shared host with:

```bash
FFMPEG_JOBS=4 make rust-build
```

The existing `FFMPEG_JOBS` setting limits both native builds and also applies to
`make rust-check` and `make dev-setup`. If bindgen
cannot locate libclang, set `LIBCLANG_PATH` to the directory containing
`libclang.so` or `libclang.dylib`; on a standard macOS Command Line Tools
installation this is `/Library/Developer/CommandLineTools/usr/lib`.

Then prepare the checkout:

```bash
make dev-setup
```

This checks tools before syncing Python and UI dependencies from their lockfiles,
building the bundled native libraries and Rust helper, and building the UI. The
native build also verifies that libclang can generate bindings and that the
private FFmpeg/OpenCV headers and static libraries can link. It does not install
global tools or OS packages, start services, or run database migrations.
Rustup may download the repository's pinned toolchain on its first use.

`make run` adds the checkout's release-helper directory to the application
`PATH`, so `preview.config.helper_path: homesec-webrtc` works without a global
helper installation. It retains its existing database-migration step; configure
the intended database and application config before running it. If invoking the
Python CLI directly, set `preview.config.helper_path` to the absolute
`native/webrtc/target/release/homesec-webrtc` path, or add its directory to `PATH`.
`CARGO_TARGET_DIR` is respected when choosing the release-helper directory.

After Rust edits, run `make rust-build` and restart HomeSec to launch the new
motion helper. Stop/start preview also launches a rebuilt preview helper. Python
and UI changes do not require a Rust rebuild. For UI hot reload,
use `make ui-run-local VITE_API_PROXY_TARGET=http://127.0.0.1:8081`, replacing
the proxy URL with the backend's address.

For Python installations outside a source checkout, use the same bundled-library
build wrapper from a matching HomeSec checkout to install the helper:

```bash
python3 native/webrtc/build_native.py -- cargo install --path native/webrtc --locked
```

This normally installs it in `~/.cargo/bin`. Add that directory to the HomeSec
service's `PATH`, or configure an absolute helper path. An absent or incompatible
helper makes WebRTC preview unavailable and selects compatible legacy motion;
it does not affect the HLS backend.

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
