# Bundled Opus notice

The native helper statically links libopus 1.6.1 for Opus preview audio
encoding. The source uses a BSD-style license (SPDX: `BSD-3-Clause`), with
the additional copyright and patent notices included in its `COPYING` file.
The complete file is retained in the private installation and installed
beside this notice in the Docker image.

Source archive:
[opus-1.6.1.tar.gz](https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz)

SHA-256: `6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1`

The source is unmodified. `native/webrtc/Cargo.toml` pins its version and
checksum, and `native/webrtc/build_native.py` supplies the static/PIC CMake
build recipe. Run `make rust-build` from the corresponding
[HomeSec source revision](https://github.com/lan17/homesec) to reproduce it.
No system Opus library is used. Docker installs the license and this notice
under `/usr/share/licenses/homesec-opus/`.
The exact upstream archive is included as
`/usr/share/homesec-native/source/opus.tar.gz` in the image.
