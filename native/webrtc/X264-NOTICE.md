# Bundled x264 notice

The native helper statically links x264 API 165 for low-latency H.264 preview
encoding. The pinned upstream source revision is
`b35605ace3ddf7c1a5d67a2eb553f034aef41d55` from VideoLAN's stable branch.
x264 is licensed under the GNU General Public License, version 2 or later
(SPDX: `GPL-2.0-or-later`); the combined helper uses GPL-3.0-or-later as
described in `FFMPEG-NOTICE.md`. The full x264 license text is retained in
`COPYING` and installed beside this notice in the Docker image.

Source archive:
[x264-b35605ace3ddf7c1a5d67a2eb553f034aef41d55.tar.gz](https://code.videolan.org/videolan/x264/-/archive/b35605ace3ddf7c1a5d67a2eb553f034aef41d55/x264-b35605ace3ddf7c1a5d67a2eb553f034aef41d55.tar.gz)

SHA-256: `cd71a7515b0e9a012e1ac9b1f8415bebcaf6fc97d4db32286642ac4c0fbe24f9`

The source is unmodified. `native/webrtc/Cargo.toml` pins its revision and
checksum, and `native/webrtc/build_native.py` supplies the static, PIC,
8-bit/YUV420 build recipe. Run `make rust-build` from the corresponding
[HomeSec source revision](https://github.com/lan17/homesec) to reproduce it.
No system x264 library is used. Docker installs the license and this notice
under `/usr/share/licenses/homesec-x264/`.
The exact upstream archive is included as
`/usr/share/homesec-native/source/x264.tar.gz` in the image.
