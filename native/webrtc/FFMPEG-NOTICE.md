# Bundled FFmpeg notice

The HomeSec native helper statically links FFmpeg 8.1.3, x264 and libopus for
camera demuxing, decoding, motion preparation, recording and preview encoding.
The FFmpeg build enables `--enable-gpl --enable-version3 --enable-libx264`, so the
combined helper binary is distributed under the GNU General Public License,
version 3 or later (SPDX: `GPL-3.0-or-later`). The HomeSec source files retain
their Apache-2.0 license; this notice does not change the Python application's
or other source files' license.

The full FFmpeg license texts are provided in `COPYING.GPLv3` and
`COPYING.LGPLv2.1` beside this notice in the Docker image. Copyright notices for
the FFmpeg authors are included in the corresponding source. x264 and Opus
have their own source pins and notices in `native/webrtc/X264-NOTICE.md` and
`native/webrtc/OPUS-NOTICE.md`.

Source archive: [ffmpeg-8.1.3.tar.xz](https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz)

SHA-256: `7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3`

`native/webrtc/build_native.py` in the corresponding
[HomeSec source revision](https://github.com/lan17/homesec) contains the complete
configure/build recipe and the narrow HomeSec changes to
`libavformat/rtsp.c` and `libavcodec/h264_parser.c`. Those changes bound RTSP
control messages and incomplete H.264 access units; the source is not
unmodified upstream FFmpeg.

From that checkout, run `make rust-build` to verify the source archives,
reapply the source changes, build the private static libraries and rebuild the
helper. Downloads and build workspaces use an owned temporary cache; immutable
installations retain the license texts. Docker installs this notice and the
FFmpeg license texts under `/usr/share/licenses/homesec-ffmpeg/`.
The Docker image also contains the verified FFmpeg/x264/Opus/OpenCV source
archives and the corresponding HomeSec native source/build recipe under
`/usr/share/homesec-native/source/`. Extract `homesec-native-source.tar.gz`
to obtain the helper's source, Cargo manifest/lockfile, vendored Retina changes,
the locked registry dependency sources, a portable `.cargo/config.toml`,
Makefile and pinned Rust toolchain. The FFmpeg changes are applied by the
included build script to the verified upstream archive. The OpenCV archive
also includes the zlib source compiled into the helper.

To reuse the supplied native archives while rebuilding, place them in the
owned temporary cache printed by `build_native.py`, using their original
archive names from that script and `Cargo.toml`. The bootstrap verifies
their checksums before building. Run `make rust-build` from the extracted
source root so Cargo loads the included registry source configuration.
