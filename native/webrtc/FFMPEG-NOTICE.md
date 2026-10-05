# Bundled FFmpeg notice

The HomeSec native helper uses FFmpeg 8.1.3 decode and filter libraries, statically
linked from unmodified official FFmpeg source. This build uses the GNU Lesser
General Public License, version 2.1 or later (SPDX: `LGPL-2.1-or-later`). The full
license is provided in `COPYING.LGPLv2.1` beside this notice in the Docker image.
Copyright notices for the FFmpeg authors are included in the corresponding source.

Source archive: [ffmpeg-8.1.3.tar.xz](https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz)

SHA-256: `7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3`

The build recipe and configuration are in `native/webrtc/build_ffmpeg.py` in the
corresponding [HomeSec source revision](https://github.com/lan17/homesec). From that
checkout, run `make rust-build` to verify the source archive, build the private
FFmpeg libraries, and rebuild the helper. The script retains `COPYING.LGPLv2.1`
in the private FFmpeg installation. Docker installs that license and this notice
under `/usr/share/licenses/homesec-ffmpeg/`.
