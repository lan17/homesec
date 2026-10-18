# Native hardware adapter notice

Linux builds statically link the pinned libdrm, libva DRM adapter, and oneVPL
dispatcher. They include NVIDIA's nv-codec-headers to load CUDA/NVDEC/NVENC
interfaces dynamically. These libraries do not include a GPU driver or establish
that hardware can decode a camera stream. The host must supply the applicable
driver and device access. macOS builds use Apple's VideoToolbox framework from
the platform SDK; Apple frameworks are not redistributed by this build.

The exact versions and SHA-256 pins are maintained in
[`Cargo.toml`](Cargo.toml), with their complete isolated build recipe in
[`build_native.py`](build_native.py):

| Dependency | Corresponding source | License |
| --- | --- | --- |
| libdrm 2.4.124 | [release archive](https://dri.freedesktop.org/libdrm/libdrm-2.4.124.tar.xz) | MIT; notices retained in the installed `xf86drm.h` |
| libva 2.22.0 | [Intel source tag](https://github.com/intel/libva/tree/2.22.0) | MIT; `COPYING` retained |
| libvpl 2.15.0 | [Intel source tag](https://github.com/intel/libvpl/tree/v2.15.0) | MIT; `LICENSE` and `third-party-programs.txt` retained |
| nv-codec-headers n8.1.24.15 | [FFmpeg source tag](https://github.com/FFmpeg/nv-codec-headers/tree/n8.1.24.15) | MIT; notices retained in each installed header |

The NVIDIA header branch is the oldest accepted by pinned FFmpeg 8.1.3,
preserving its H.264/HEVC API compatibility rather than requiring the newest
SDK's driver floor. Its upstream README requires Linux driver 390.25 or newer;
optional CUDA 10 features require 410.48 or newer. These are build/API
prerequisites, not evidence that an old driver, GPU, or codec worked. Actual
CUDA/NVENC device validation remains a separate hardware QA obligation.
CUDA decoding uses libav's native parsers and NVDEC hardware accelerators;
the separate CUVID parser/decoder wrappers are disabled because their source
requires newer SDK struct fields even when configure accepts older headers.
This does not disable CUDA hardware decoding or H.264 NVENC encoding.

Docker installs these notices and license-bearing headers under
`/usr/share/licenses/homesec-hardware/` and retains every verified source archive
under `/usr/share/homesec-native/source/`. Preserve those files and the HomeSec
native source/build bundle when distributing a separately packaged helper.
The build script omits libdrm's disabled test trees, which contain upstream
symlinks, and changes libva's six Meson `shared_library()` constructors to
`library()` so `--default-library=static` is honored. The original verified
archives remain available in the source bundle; the complete changes are in the
included build script.
The oneVPL static pkg-config file also gains its required private C++ runtime
and pthread linkage, which upstream omits from that file.
The combined helper remains GPL-3.0-or-later because it statically links x264
and GPL-enabled FFmpeg; see [`FFMPEG-NOTICE.md`](FFMPEG-NOTICE.md).
