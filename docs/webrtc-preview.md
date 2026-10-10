# Rust WebRTC preview

WebRTC is the default backend for live camera viewing. Preview remains disabled
until enabled in config. HomeSec supervises a `homesec-webrtc` Rust helper for
each eligible RTSP source. On compatible CPU H.264 sources, native motion,
copied MP4 recording with optional AAC-LC audio, and preview share one pinned
FFmpeg library RTSP input per selected URL. All preview viewers share that input.
Rust copies compatible H.264 or transcodes it through pinned x264, and converts
supported camera audio through pinned libopus. Hardware motion, unsupported or
custom recording profiles, and push-to-talk retain their established paths.
Python continues to own recording policy, supervision, and finalized clip delivery.

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
firewall rules and Docker port mappings. Compatibility FFmpeg preview uses
separate loopback RTP sockets, which should not be published externally. Shared
motion and recording can run with preview disabled and do not require an
advertised IP or an externally published WebRTC port.

`max_viewers` limits peers for each camera, including sessions negotiating a
connection. Authentication leases are renewed by the UI. Expired leases and
failed connections close the associated peer; `max_session_duration_s` also caps
the lifetime of a session. After the final viewer leaves, `idle_timeout_s` bounds
how long the preview consumer remains active. A shared helper/input stays active
while motion or recording still needs it; RTSP source cleanup owns helper shutdown.

The default `stop_on_recording` policy yields preview resources to recording.
`allow_during_recording` is best-effort and can consume another direct camera
session for separate selected streams or compatibility
paths. Compatible native consumers using the same selected URL share one RTSP
PLAY. Existing camera preflight and concurrency refusal behavior still apply;
sharing does not override a preflight downgrade or the selected recording policy.
Preflight still conservatively probes separate camera sessions and can refuse
concurrent preview even when native consumers could share one selected input.
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
require transcoding for this path. H.264 with B-frames can use native transcoding;
other input video codecs currently use the existing FFmpeg compatibility path.

Eligible CPU H.264 RTSP sources use the shared helper's pinned FFmpeg library
demuxer over RTSP/TCP for both preview modes, with or without audio. This path does
not launch an FFmpeg command-line process for preview. Matching motion and recording
URLs reuse the same input, and the same encoded video feeds every preview viewer.
When the source cannot use shared mode, standalone native copy preview retains
the Retina adapter. Camera credentials travel only through private
Python-to-helper control pipes. RTSP authentication, keepalive, and teardown are
handled by the client libraries.

Native copy supports H.264 Baseline, Main, and High profiles through level 3.1,
without B-frames. Negotiation checks the actual source profile and selects a
compatible receiver level at least as high as the camera's. SDP parameter sets and camera RTP
payload types are honored. Damaged or oversized RTP access units are discarded,
with delivery resuming at a valid keyframe. Unsupported video, changed profiles,
reversed video timestamps, or an overflowing source queue stop preview; select
the default `video_codec: h264` to transcode an incompatible camera.
Shared input preparation waits for a complete timestamped IDR and uses actual
in-band SPS/PPS when initial SDP values are stale. A late consumer also waits for
a complete keyframe. The standalone Retina adapter discards its first observed
picture because the camera may have started midway through it. Startup can
therefore take one camera GOP; use a short keyframe interval within the source's
configured deadline. Native connection setup and reads use the RTSP source's
configured connect and I/O timeouts, with standalone defaults of five and ten
seconds. Stop or parent disconnect cancels camera I/O without waiting on it in
the WebRTC control loop. Python requests graceful stop and allows camera teardown
to finish before bounded process-group termination.

The native H.264 encoder preserves the existing baseline YUV420p,
veryfast/zerolatency settings, with no B-frames and periodic one-second IDRs.
Encoder selection and options live in private adapters in
`native/webrtc/src/preview.rs`; changing the encoder requires updating those
adapters and the native build recipe, without changing the source or Python IPC.
Preview owns its decoder/filters independently of motion preparation and copied
recording, so encoder failure cannot stall those consumers.

Shared helpers advertise `native_preview` in their ready response. Older helpers
keep the existing start payloads. On native startup or later media failure,
HomeSec requires an acknowledged `stop_preview` before starting the compatibility
FFmpeg input through the same shared helper. Failed or lost detach acknowledgement
prevents replacement; compatibility selection stays pinned for that publisher's
lifetime. Hardware/custom configurations retain the existing selector.

Native media assembly and delivery queues have byte/count limits. The pinned
Retina and FFmpeg recipes cap aggregate RTSP response headers/body at 256 KiB
during reads, before allocating an oversized body. The FFmpeg H.264 parser and
delivered compressed packets have a 2 MiB access-unit limit. Each shared consumer
has its own 16-packet queue; a slow or overflowing preview consumer fails without
blocking a healthy recording consumer. Native preview outputs use separate
eight-unit video and 64-packet audio queues with fair draining. Each encoded
video unit is limited to 2 MiB and each Opus packet to 4000 bytes. The FFmpeg fallback
drains its loopback RTP sockets on an independent receiver thread, so WebRTC
encryption or a slow viewer cannot delay receiving the rest of a fragmented
picture. Only complete H.264 access units cross that receiver boundary: its video
queue holds at most eight units, each limited to 2 MiB plus parameter sets, and
its audio queue holds at most 64 Opus packets of 4000 bytes each. Both queues are
drained fairly; overflow stops preview rather than forwarding compressed frames
with missing dependencies. Stop, parent disconnect, source failure, and timeout
cancel the receiver and reap FFmpeg without requiring global socket-buffer tuning.
These limits bound individual buffers and queues, not the entire process.
Cameras remain operator-configured sources on a trusted LAN/VPN.

See the [shared Rust media plan](shared-rust-media.md) for ingestion, recording,
and lifecycle details. FFmpeg and ffprobe command-line tools remain required for
preflight and compatibility recording, motion, and preview.

The preview input decoder uses slice threading rather than frame threading.
Frame threading queues future frames and can add about a second of delay at
15 fps on a many-core host; slice threading avoids that queue while preserving
source frame reordering. Encoding already uses FFmpeg's zero-latency tuning.

When `audio_enabled` is true and the camera supplies AAC, G.711 A-law/mu-law, or
Opus, native preview converts it to 48 kHz stereo Opus at 64 kbps with 20 ms
frames. Unsupported audio selects compatibility preview. A camera without audio
still supplies video. Source timestamps, including encoder lookahead, map to a
common wallclock for WebRTC audio/video synchronization. Media that arrives early
waits until its presentation time in a bounded slot per track, while the other
track and control requests continue. Camera AAC is not passed
directly to WebRTC. The UI preserves its existing
mute behavior and push-to-talk coordination; changing the preview transport does
not change microphone-to-camera transport.

Closing a viewer detaches only that peer. The camera-level preview force-stop
operation and recording-priority shedding detach preview and close all its peers;
they do not stop a healthy shared recording or motion consumer. Runtime
replacement invalidates old sessions and cleans up the source-owned helper.

## Native motion detection

HomeSec prefers Rust motion detection for compatible H.264 CPU inputs when the
helper is available. Rust calls the bundled FFmpeg decode and filter libraries
directly, then calls statically linked OpenCV 4.12.0 for Gaussian blur, absolute
difference, thresholding, and changed-pixel counting. The detector borrows each
prepared input frame for one call and reuses its owned image buffers.
It prepares the same 320x240 grayscale frames at 10 fps and uses the existing
motion settings, including blur normalization and recording sensitivity.
If startup cannot supply its first frame within the source's existing read or
readiness deadline, HomeSec detaches native motion and selects compatibility
motion. Other shared consumers retain their input. This avoids repeated native
restarts for cameras with longer keyframe
intervals. Once native input supplies a frame, missing frames continue through
the existing source stall and reconnect policy.

The RTSP source supervises the shared helper using the selected motion stream,
independent of preview viewers. Matching recording/preview URLs reuse this input;
a separate detection substream keeps its own input. Python receives typed motion
observations rather than raw pixels in JSON and retains recording, reconnect,
stall, and upload policy. Rust writes eligible compressed video/audio recordings
through the linked MP4 muxer, without routing them through prepared motion pixels.

Hardware decoding, custom FFmpeg input flags, unsupported camera inputs, or an
unavailable native helper use the existing FFmpeg/OpenCV path. This compatibility
fallback preserves the existing configuration; it does not require a second set
of motion settings. A consumer failure must detach its camera input before fallback
opens another one. If a recording's startup or stop reply is uncertain, HomeSec
retains the recording owner and confirms writer closure or owned helper death
before retrying through FFmpeg. Operational logs identify the selected path and
stable failure reasons without exposing camera credentials or media.

## Shared recording lifecycle

Eligible CPU H.264 sources use native recording for canonical copied MP4 profiles
with no audio or AAC-LC audio. Custom FFmpeg flags, wall-clock timestamp profiles,
unsupported codecs/AAC extensions, and audio transcoding retain the selected
compatibility FFmpeg profile. The shared helper permits two selected stream URLs
and two recording IDs, supporting the existing overlap during clip rotation.
Per-consumer queues and stop operations are independent; stopping preview or
motion leaves an active recording attached.

Each native clip starts at a real IDR, preserves copied compressed packets and
their presentation/decode timestamps, and keeps one audio/video epoch. Native
MP4 recording can retain H.264 B-frames even though copy preview cannot display
them. Invalid timing, zero packet duration, changed SPS/PPS, queue overflow, disk
failure, and the limit of fewer than one million accepted packets fail explicitly.

The helper writes `<final-name>.partial`, then drains accepted packets, writes
the trailer, flushes, and syncs on stop. It publishes the final filename atomically
without replacing an existing file. Python hands the clip to the existing pipeline
only after successful finalization is confirmed. Failed or forcibly interrupted
recording leaves a `.partial` file excluded from callbacks and replay. This is
conventional MP4; an unfinalized partial has no guaranteed playback or automatic
crash recovery. Inspect or remove abandoned partials only after confirming their
writer has stopped. Successfully finalized files retain the existing replay path.

## Docker

The Docker image builds the helper with the pinned Rust toolchain and installs it
at `/usr/local/bin/homesec-webrtc`. The same `make rust-build` recipe used locally
builds verified FFmpeg 8.1.3, OpenCV 4.12.0, x264 API 165, and Opus 1.6.1 source and statically links their
libraries into the helper. Rust, native headers, CMake, and libclang are confined
to the build stage. The image checks that the runtime can load the helper and
that it has no shared OpenCV, libav, x264, or libopus dependency. Standard C/C++ runtime libraries
remain OS dependencies.
The system FFmpeg command-line tools remain in the image for preflight,
compatibility recording, motion, and preview.

The bundled FFmpeg license and source/build notice are installed in
`/usr/share/licenses/homesec-ffmpeg/`. The
[notice](../native/webrtc/FFMPEG-NOTICE.md) identifies the exact source archive,
checksum, and recipe for rebuilding the helper. OpenCV, its bundled zlib, and the
Rust binding license/notice are installed in `/usr/share/licenses/homesec-opencv/`;
the [OpenCV notice](../native/webrtc/OPENCV-NOTICE.md) identifies their sources.
The x264 and Opus notices/licenses are installed alongside them. Static x264
makes the combined helper GPLv3; HomeSec source remains Apache-2.0.
The image includes corresponding source and rebuild instructions under
`/usr/share/homesec-native/source/`. Preserve the complete source bundle and
licenses when distributing a separately packaged helper.

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
- FFmpeg, including `ffprobe`, on `PATH` for preflight, compatibility
  recording, motion, and preview.
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
[OpenCV 4.12.0](https://github.com/opencv/opencv/tree/4.12.0),
[x264 API 165](https://code.videolan.org/videolan/x264/-/tree/b35605ace3ddf7c1a5d67a2eb553f034aef41d55), and
[Opus 1.6.1](https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz) source archives,
verify their pinned SHA-256 checksums, and build the required libraries privately.
OpenCV builds only `core` and `imgproc`, with bundled static zlib. Cargo pins the
Rust bindings to `opencv = "=0.101.0"`; the native version and archive checksum
are recorded in `native/webrtc/Cargo.toml` metadata and enforced by the build.
System FFmpeg/OpenCV development packages are unnecessary. The helper links
private static archives and does not load system OpenCV, libav, x264, or libopus shared libraries.
Use the Make targets or build wrapper; direct Cargo builds without the pinned
private build environment are refused.

Verified downloads and compiled libraries remain under the operating system's
temporary directory in `homesec-native-<uid>/`.
Matching platform, compiler, and build recipes reuse the caches. Clearing them
causes a fresh download and build. No libraries are installed globally.

Each private FFmpeg installation retains `COPYING.GPLv3`. The
[bundled-component notice](../native/webrtc/FFMPEG-NOTICE.md) records its source
and rebuild recipe; keep the license and notice with a separately packaged helper.
The OpenCV installation retains `LICENSE` and `ZLIB-LICENSE`; distribute these,
`native/webrtc/OPENCV-BINDINGS-LICENSE`, and the
[OpenCV notice](../native/webrtc/OPENCV-NOTICE.md) with the helper too.
Keep the x264 `COPYING`, Opus `COPYING`, and their bundled notices as well.

The first build needs internet access and compiles these libraries using the available CPU
count. Limit CPU and memory use on a shared host with:

```bash
FFMPEG_JOBS=4 make rust-build
```

The existing `FFMPEG_JOBS` setting limits all native builds and also applies to
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
source-owned helper. Stop/start preview alone reuses a helper already serving
motion or recording, so it does not load a rebuilt shared binary. Python
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
