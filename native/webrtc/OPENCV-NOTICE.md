# Bundled OpenCV notice

The HomeSec native helper statically links OpenCV 4.12.0 `core` and `imgproc`,
built from unmodified upstream source. The native version and source checksum
are pinned in `Cargo.toml` metadata; the Rust bindings are pinned to `opencv`
0.101.0 in the same manifest and locked in `Cargo.lock`.

Source archive: [OpenCV 4.12.0](https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.12.0)

SHA-256: `44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d`

OpenCV uses the Apache License, version 2.0. Its `LICENSE` retains the upstream
copyright and attribution text. The static build also includes the zlib source
bundled in that archive; `ZLIB-LICENSE` retains its license and attribution.
The [Rust OpenCV bindings](https://crates.io/crates/opencv/0.101.0) use the MIT
license, retained in `OPENCV-BINDINGS-LICENSE` beside this notice.

The build recipe is `native/webrtc/build_native.py` in the corresponding
[HomeSec source revision](https://github.com/lan17/homesec). Run `make rust-build`
from that checkout to verify the archives, build private static libraries in
temporary storage, and rebuild the helper. Platform runtime libraries remain
dynamically linked; OpenCV and its bundled zlib are compiled into the helper.

Docker installs this notice and the three licenses under
`/usr/share/licenses/homesec-opencv/`. Keep them with separately packaged helpers.
The exact upstream archive, including its bundled zlib source, is also included
as `/usr/share/homesec-native/source/opencv.tar.gz` in the image. The native
source bundle in that directory contains the locked Rust OpenCV binding sources
and the remaining registry dependencies alongside the helper build recipe.
