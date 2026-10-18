"""Build pinned, private static media libraries before invoking Cargo (Linux/macOS)."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Literal, TextIO


@dataclass(frozen=True)
class NativeDependency:
    name: str
    version: str
    archive: str
    url: str
    sha256: str
    required: tuple[str, ...]

    @property
    def source_directory(self) -> str:
        return f"{self.name.lower()}-{self.version}"


FFMPEG = NativeDependency(
    name="FFmpeg",
    version="8.1.3",
    archive="ffmpeg-8.1.3.tar.xz",
    url="https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz",
    sha256="7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3",
    required=(
        "bin/ffmpeg",
        "COPYING.LGPLv2.1",
        "COPYING.GPLv3",
        *(
            path
            for library in ("avcodec", "avformat", "avfilter", "avutil", "swscale", "swresample")
            for path in (
                f"lib/lib{library}.a",
                f"lib/pkgconfig/lib{library}.pc",
                f"include/lib{library}/{library}.h",
            )
        ),
    ),
)
FFMPEG_CONFIGURE = (
    "--disable-everything",
    "--disable-autodetect",
    "--enable-network",
    "--disable-shared",
    "--enable-static",
    "--enable-pic",
    "--disable-doc",
    "--disable-debug",
    "--disable-avdevice",
    "--enable-swresample",
    "--disable-ffplay",
    "--disable-ffprobe",
    "--enable-ffmpeg",
    "--enable-gpl",
    "--enable-version3",
    "--enable-libx264",
    "--enable-libopus",
    "--pkg-config-flags=--static",
    # RTSP compatibility never imposed a video-codec allowlist. Enable native
    # decoders/parsers rather than inventing a narrower camera codec policy.
    # Autodetection remains disabled: no optional system codec library is used.
    "--enable-decoders",
    "--enable-parsers",
    "--enable-demuxer=h264,hevc,mov,rtsp,rtp",
    "--enable-protocol=file,pipe,tcp,udp,rtp",
    "--enable-bsf=extract_extradata",
    "--enable-filter=buffer,buffersink,fps,scale,format,abuffer,abuffersink,aresample,anull,aformat",
    "--enable-encoder=rawvideo,libx264,libopus,aac",
    "--enable-muxer=rawvideo,mp4",
)
# These adapters load the host's GPU driver at runtime. Building them does not
# require a GPU and never establishes that a device can decode camera media.
FFMPEG_LINUX_CONFIGURE = (
    "--enable-libdrm",
    "--enable-vaapi",
    "--enable-libvpl",
    "--enable-ffnvcodec",
    "--enable-cuda",
    # Use libav's native parsers + NVDEC hwaccels, not the separate CUVID
    # parser/decoder wrappers. The latter unconditionally need newer SDK fields
    # despite configure accepting this compatible older NVIDIA API branch.
    "--disable-cuvid",
    "--enable-nvenc",
    "--enable-hwaccel=h264_vaapi,hevc_vaapi,h264_cuda,hevc_cuda,h264_qsv,hevc_qsv",
    "--enable-decoder=h264_qsv,hevc_qsv",
    "--enable-encoder=h264_vaapi,h264_nvenc,h264_qsv",
)
FFMPEG_DARWIN_CONFIGURE = (
    "--enable-videotoolbox",
    "--enable-audiotoolbox",
    "--enable-hwaccel=h264_videotoolbox,hevc_videotoolbox",
    "--enable-encoder=h264_videotoolbox",
)
LIBDRM_CONFIGURE = (
    "--buildtype=release",
    "--default-library=static",
    "--libdir=lib",
    "--wrap-mode=nodownload",
    "-Db_staticpic=true",
    "-Dauto_features=disabled",
    "-Dtests=false",
)
LIBVA_CONFIGURE = (
    "--buildtype=release",
    "--default-library=static",
    "--libdir=lib",
    "--wrap-mode=nodownload",
    "-Db_staticpic=true",
    "-Dwith_x11=no",
    "-Dwith_glx=no",
    "-Dwith_wayland=no",
    "-Dwith_win32=no",
    "-Denable_docs=false",
    # Drivers are OS resources, not part of this private library installation.
    "-Ddriverdir=/usr/lib/dri:/usr/lib64/dri:/usr/lib/x86_64-linux-gnu/dri:/usr/lib/aarch64-linux-gnu/dri",
)
LIBVPL_CONFIGURE = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_LIBDIR=lib",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DINSTALL_EXAMPLES=OFF",
)
X264_CONFIGURE = (
    "--enable-static",
    "--enable-pic",
    "--disable-cli",
    "--disable-opencl",
    "--disable-avs",
    "--disable-lavf",
    "--disable-ffms",
    "--disable-swscale",
    "--disable-gpac",
    "--disable-lsmash",
    "--bit-depth=8",
    "--chroma-format=420",
)
OPUS_CONFIGURE = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_LIBDIR=lib",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DOPUS_BUILD_SHARED_LIBRARY=OFF",
    "-DOPUS_BUILD_TESTING=OFF",
    "-DOPUS_BUILD_PROGRAMS=OFF",
    "-DOPUS_INSTALL_PKG_CONFIG_MODULE=ON",
)
OPENCV_CONFIGURE = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_LIBDIR=lib",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DBUILD_SHARED_LIBS=OFF",
    "-DBUILD_LIST=core,imgproc",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_PERF_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_DOCS=OFF",
    "-DBUILD_JAVA=OFF",
    "-DBUILD_opencv_apps=OFF",
    "-DBUILD_opencv_python2=OFF",
    "-DBUILD_opencv_python3=OFF",
    "-DBUILD_ZLIB=ON",
    "-DWITH_ADE=OFF",
    "-DWITH_JPEG=OFF",
    "-DWITH_OPENJPEG=OFF",
    "-DWITH_JASPER=OFF",
    "-DWITH_PNG=OFF",
    "-DWITH_TIFF=OFF",
    "-DWITH_WEBP=OFF",
    "-DWITH_OPENEXR=OFF",
    "-DWITH_AVIF=OFF",
    "-DWITH_PROTOBUF=OFF",
    "-DWITH_IPP=OFF",
    "-DWITH_ITT=OFF",
    "-DWITH_OPENCL=OFF",
    "-DWITH_LAPACK=OFF",
    "-DWITH_EIGEN=OFF",
    "-DWITH_TBB=OFF",
    "-DWITH_OPENMP=OFF",
    "-DWITH_CAROTENE=OFF",
    "-DWITH_KLEIDICV=OFF",
    "-DWITH_FFMPEG=OFF",
    "-DWITH_GSTREAMER=OFF",
    "-DPARALLEL_ENABLE_PLUGINS=OFF",
    "-DENABLE_PRECOMPILED_HEADERS=OFF",
    "-DENABLE_CCACHE=OFF",
)


DependencyName = Literal["OpenCV", "x264", "Opus", "nv-codec-headers", "libdrm", "libva", "libvpl"]


def native_dependency(name: DependencyName, cargo: str = "cargo") -> NativeDependency:
    # Cargo pins native implementations alongside the Rust bindings. Consume
    # that metadata rather than duplicating versions/checksums in this recipe.
    manifest = Path(__file__).with_name("Cargo.toml")
    packages = json.loads(
        subprocess.check_output(
            [
                cargo,
                "metadata",
                "--no-deps",
                "--locked",
                "--format-version",
                "1",
                "--manifest-path",
                str(manifest),
            ],
            text=True,
        )
    )["packages"]
    try:
        package = next(
            package
            for package in packages
            if Path(package["manifest_path"]).resolve() == manifest.resolve()
        )
        metadata = package["metadata"]["native-dependencies"][name.lower()]
        version = metadata["version"]
        checksum = metadata["sha256"]
    except (KeyError, StopIteration, TypeError) as error:
        raise RuntimeError(f"Cargo.toml must pin the {name} release and SHA256") from error
    version_pattern = {
        "x264": r"[0-9a-f]{40}",
        "nv-codec-headers": r"n\d+\.\d+\.\d+\.\d+",
    }.get(name, r"\d+\.\d+\.\d+")
    if (
        not isinstance(version, str)
        or not re.fullmatch(version_pattern, version)
        or not isinstance(checksum, str)
        or not re.fullmatch(r"[0-9a-f]{64}", checksum)
    ):
        raise RuntimeError(f"Cargo.toml must pin the {name} release and SHA256")
    if name == "x264":
        return NativeDependency(
            name=name,
            version=version,
            archive=f"x264-{version}.tar.gz",
            url=f"https://code.videolan.org/videolan/x264/-/archive/{version}/x264-{version}.tar.gz",
            sha256=checksum,
            required=("lib/libx264.a", "lib/pkgconfig/x264.pc", "include/x264.h", "COPYING"),
        )
    if name == "Opus":
        return NativeDependency(
            name=name,
            version=version,
            archive=f"opus-{version}.tar.gz",
            url=f"https://downloads.xiph.org/releases/opus/opus-{version}.tar.gz",
            sha256=checksum,
            required=("lib/libopus.a", "lib/pkgconfig/opus.pc", "include/opus/opus.h", "COPYING"),
        )
    if name in ("nv-codec-headers", "libdrm", "libva", "libvpl"):
        required = {
            "nv-codec-headers": (
                "lib/pkgconfig/ffnvcodec.pc",
                "include/ffnvcodec/dynlink_loader.h",
                "include/ffnvcodec/nvEncodeAPI.h",
            ),
            "libdrm": ("lib/libdrm.a", "lib/pkgconfig/libdrm.pc", "include/xf86drm.h"),
            "libva": (
                "lib/libva.a",
                "lib/libva-drm.a",
                "lib/pkgconfig/libva.pc",
                "lib/pkgconfig/libva-drm.pc",
                "include/va/va.h",
                "COPYING",
            ),
            "libvpl": (
                "lib/libvpl.a",
                "lib/pkgconfig/vpl.pc",
                "include/vpl/mfxvideo.h",
                "LICENSE",
            ),
        }[name]
        archive = f"{name}-{version}.tar.{'xz' if name == 'libdrm' else 'gz'}"
        url = {
            "nv-codec-headers": f"https://codeload.github.com/FFmpeg/nv-codec-headers/tar.gz/refs/tags/{version}",
            "libdrm": f"https://dri.freedesktop.org/libdrm/{archive}",
            "libva": f"https://codeload.github.com/intel/libva/tar.gz/refs/tags/{version}",
            "libvpl": f"https://codeload.github.com/intel/libvpl/tar.gz/refs/tags/v{version}",
        }[name]
        return NativeDependency(name, version, archive, url, checksum, required)
    return NativeDependency(
        name="OpenCV",
        version=version,
        archive=f"opencv-{version}.tar.gz",
        url=f"https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/{version}",
        sha256=checksum,
        required=(
            "lib/libopencv_imgproc.a",
            "lib/libopencv_core.a",
            "lib/opencv4/3rdparty/libzlib.a",
            "include/opencv4/opencv2/core/version.hpp",
            "include/opencv4/opencv2/imgproc.hpp",
            "LICENSE",
            "ZLIB-LICENSE",
        ),
    )


def available_jobs() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def verify_archive(path: Path, dependency: NativeDependency) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != dependency.sha256:
        raise RuntimeError(f"{dependency.name} archive checksum mismatch: remove {path} and retry")


def download_archive(root: Path, dependency: NativeDependency) -> Path:
    archive = root / dependency.archive
    if not archive.exists():
        print(f"Downloading {dependency.name} {dependency.version} to {root}", flush=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=root) as download:
            temporary = Path(download) / dependency.archive
            with (
                urllib.request.urlopen(dependency.url, timeout=60) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            verify_archive(temporary, dependency)
            temporary.replace(archive)
    verify_archive(archive, dependency)
    return archive


def extract_archive(archive: Path, workspace: Path, dependency: NativeDependency) -> Path:
    # Only checksum-verified archives reach extraction. Reject links and
    # traversal as well, including on Python versions without tar filters.
    with tarfile.open(archive) as source:
        members = []
        for member in source.getmembers():
            parts = PurePosixPath(member.name)
            # libdrm's release contains symlinks only in disabled test trees.
            # Omit those trees entirely; never relax link extraction safety.
            if dependency.name == "libdrm" and (
                parts.parts[1:2] == ("tests",) or parts.parts[1:3] == ("intel", "tests")
            ):
                continue
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or not (member.isdir() or member.isfile())
                or not parts.parts
                or parts.parts[0] != dependency.source_directory
            ):
                raise RuntimeError(f"Unsafe {dependency.name} archive member")
            members.append(member)
        source.extractall(workspace, members=members)  # noqa: S202 (verified checksum and validated members)
    return workspace / dependency.source_directory


def private_environment(
    ffmpeg: Path,
    opencv: Path | None = None,
    opencv_version: str | None = None,
    *,
    encoders: tuple[Path, ...] = (),
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("FFMPEG_DIR", "OpenCV_DIR", "CMAKE_PREFIX_PATH", "CMAKE_MODULE_PATH")
        and "PKG_CONFIG" not in key
        and not key.startswith(("BINDGEN_EXTRA_CLANG_ARGS", "OPENCV_", "HOMESEC_OPENCV_"))
    }
    search_paths = os.pathsep.join(str(prefix / "lib/pkgconfig") for prefix in (ffmpeg, *encoders))
    environment["PKG_CONFIG_PATH"] = search_paths
    environment["PKG_CONFIG_LIBDIR"] = search_paths
    # The parity oracle needs the same FFmpeg build; recording and other
    # fixtures retain the full system CLI through the unchanged PATH.
    environment["HOMESEC_FFMPEG_REFERENCE"] = str(ffmpeg / "bin/ffmpeg")
    environment["HOMESEC_FFMPEG_PREFIX"] = str(ffmpeg)
    environment["HOMESEC_NATIVE_BUILD_ID"] = ffmpeg.name
    if opencv is not None:
        environment["HOMESEC_OPENCV_PREFIX"] = str(opencv)
        if opencv_version is None:
            raise RuntimeError("A private OpenCV installation requires its pinned version")
        environment["HOMESEC_OPENCV_VERSION"] = opencv_version
        environment["OPENCV_INCLUDE_PATHS"] = str(opencv / "include/opencv4")
        environment["OPENCV_LINK_PATHS"] = ",".join(
            (str(opencv / "lib"), str(opencv / "lib/opencv4/3rdparty"))
        )
        runtime_libraries = (
            ("c++",) if platform.system() == "Darwin" else ("stdc++", "m", "dl", "pthread", "rt")
        )
        environment["OPENCV_LINK_LIBS"] = ",".join(
            ("static=opencv_imgproc", "static=opencv_core", "static=zlib", *runtime_libraries)
        )
        # If private discovery fails, fail instead of probing an OS OpenCV.
        environment["OPENCV_DISABLE_PROBES"] = "pkg_config,cmake,vcpkg_cmake,vcpkg"
    return environment


def build_ffmpeg(
    source: Path,
    prefix: Path,
    options: tuple[str, ...],
    jobs: int,
    log: TextIO,
    *,
    encoders: tuple[Path, ...] = (),
) -> None:
    patch_ffmpeg_rtsp(source)
    for command in (
        ["./configure", f"--prefix={prefix}", *options],
        ["make", f"-j{jobs}"],
        ["make", "install"],
    ):
        subprocess.run(
            command,
            cwd=source,
            env=private_environment(prefix, encoders=encoders),
            stdout=log,
            stderr=log,
            check=True,
        )
    for license_name in ("COPYING.LGPLv2.1", "COPYING.GPLv3"):
        shutil.copyfile(source / license_name, prefix / license_name)


def build_x264(
    source: Path, prefix: Path, options: tuple[str, ...], jobs: int, log: TextIO
) -> None:
    for command in (
        ["./configure", f"--prefix={prefix}", *options],
        ["make", f"-j{jobs}"],
        ["make", "install-lib-static"],
    ):
        subprocess.run(
            command, cwd=source, env=private_environment(prefix), stdout=log, stderr=log, check=True
        )
    shutil.copyfile(source / "COPYING", prefix / "COPYING")


FFMPEG_RTSP_PATCH: tuple[tuple[str, str], ...] = (
    (
        "                av_strlcpy(reply->reason, p, sizeof(reply->reason));",
        "                av_strlcpy(reply->reason, p, sizeof(reply->reason));\n"
        "                /* HomeSec preserves bounded explicit camera capacity reasons. */\n"
        "                if (reply->status_code >= 400 &&\n"
        '                    (av_stristr(reply->reason, "too many clients") ||\n'
        '                     av_stristr(reply->reason, "too many connections") ||\n'
        '                     av_stristr(reply->reason, "too many sessions") ||\n'
        '                     av_stristr(reply->reason, "max number of clients") ||\n'
        '                     av_stristr(reply->reason, "maximum number of clients") ||\n'
        '                     av_stristr(reply->reason, "max number of connections") ||\n'
        '                     av_stristr(reply->reason, "maximum number of connections") ||\n'
        '                     av_stristr(reply->reason, "session limit") ||\n'
        '                     av_stristr(reply->reason, "max sessions") ||\n'
        '                     av_stristr(reply->reason, "maximum sessions")))\n'
        "                    reply->status_code = RTSP_STATUS_BANDWIDTH;",
    ),
    (
        "        reply->content_length = strtol(p, NULL, 10);",
        "        long content_length = strtol(p, NULL, 10);\n"
        "        reply->content_length = content_length < 0 || content_length > 262144\n"
        "                              ? -1 : content_length;",
    ),
    (
        "    int ret, content_length, line_count, request;",
        "    int ret, content_length, line_count, request;\n    int control_bytes;",
    ),
    (
        "start:\n    line_count = 0;",
        "start:\n    control_bytes = 0;\n    line_count = 0;",
    ),
    (
        "            ret = ffurl_read_complete(rt->rtsp_hd, &ch, 1);",
        "            if (control_bytes >= 262144)\n"
        "                return AVERROR_INVALIDDATA;\n"
        "            ret = ffurl_read_complete(rt->rtsp_hd, &ch, 1);",
    ),
    (
        '            av_log(s, AV_LOG_TRACE, "ret=%d c=%02x [%c]\\n", ret, ch, ch);',
        "            control_bytes++;\n"
        '            av_log(s, AV_LOG_TRACE, "ret=%d c=%02x [%c]\\n", ret, ch, ch);',
    ),
    (
        "            if (ch == '$' && q == buf) {",
        "            if (ch == '$' && q == buf) {\n"
        "                control_bytes--; /* Interleaved media has its own u16 bound. */",
    ),
    (
        "    content_length = reply->content_length;\n    if (content_length > 0) {",
        "    content_length = reply->content_length;\n"
        "    if (content_length < 0 || content_length > 262144 - control_bytes)\n"
        "        return AVERROR_INVALIDDATA;\n"
        "    if (content_length > 0) {",
    ),
)
FFMPEG_RTSP_STATUS_PATCH = (
    "    return ff_http_averror(status_code, default_averror);",
    "    /* HomeSec preserves explicit RTSP capacity evidence without log parsing. */\n"
    "    if (status_code == RTSP_STATUS_BANDWIDTH)\n"
    "        return -0x35345348; /* Private FFERRTAG('H', 'S', '4', '5'). */\n"
    "    return ff_http_averror(status_code, default_averror);",
)


def patch_ffmpeg_rtsp(source: Path) -> None:
    """Apply the narrow control-message guard to the verified pinned source."""
    path = source / "libavformat/rtsp.c"
    text = path.read_text()
    for original, replacement in FFMPEG_RTSP_PATCH:
        if text.count(original) != 1:
            raise RuntimeError("Pinned FFmpeg RTSP patch context differs from verified source")
        text = text.replace(original, replacement)
    path.write_text(text)
    codes = source / "libavformat/rtspcodes.h"
    text = codes.read_text()
    original, replacement = FFMPEG_RTSP_STATUS_PATCH
    if text.count(original) != 1:
        raise RuntimeError("Pinned FFmpeg RTSP status patch context differs from verified source")
    codes.write_text(text.replace(original, replacement))
    parser = source / "libavcodec/h264_parser.c"
    text = parser.read_text()
    original = "        if (ff_combine_frame(pc, next, &buf, &buf_size) < 0) {"
    replacement = (
        "        /* HomeSec bounds incomplete access units before parser allocation. */\n"
        "        if (buf_size > 2097152 || pc->index > 2097152 - buf_size) {\n"
        "            av_freep(&pc->buffer);\n"
        "            pc->buffer_size = 0;\n"
        "            pc->index = pc->last_index = pc->overread = pc->overread_index = 0;\n"
        "            pc->frame_start_found = 0;\n"
        "            pc->state = -1;\n"
        "            *poutbuf = NULL;\n"
        "            *poutbuf_size = 0;\n"
        "            return buf_size;\n"
        "        }\n\n" + original
    )
    if text.count(original) != 1:
        raise RuntimeError("Pinned FFmpeg H.264 patch context differs from verified source")
    parser.write_text(text.replace(original, replacement))


def build_cmake(
    source: Path, prefix: Path, options: tuple[str, ...], jobs: int, log: TextIO
) -> None:
    workspace = source / "native-build"
    for command in (
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(workspace),
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            *options,
        ],
        ["cmake", "--build", str(workspace), "--parallel", str(jobs)],
        ["cmake", "--install", str(workspace)],
    ):
        subprocess.run(command, env=private_environment(prefix), stdout=log, stderr=log, check=True)


def build_opencv(
    source: Path, prefix: Path, options: tuple[str, ...], jobs: int, log: TextIO
) -> None:
    build_cmake(source, prefix, options, jobs, log)
    shutil.copyfile(source / "LICENSE", prefix / "LICENSE")
    shutil.copyfile(source / "3rdparty/zlib/LICENSE", prefix / "ZLIB-LICENSE")


def build_opus(
    source: Path, prefix: Path, options: tuple[str, ...], jobs: int, log: TextIO
) -> None:
    build_cmake(source, prefix, options, jobs, log)
    shutil.copyfile(source / "COPYING", prefix / "COPYING")


def build_nvcodec(
    source: Path, prefix: Path, options: tuple[str, ...], jobs: int, log: TextIO
) -> None:
    subprocess.run(
        ["make", f"-j{jobs}", f"PREFIX={prefix}", "install"],
        cwd=source,
        env=private_environment(prefix),
        stdout=log,
        stderr=log,
        check=True,
    )


def build_meson(
    source: Path,
    prefix: Path,
    options: tuple[str, ...],
    jobs: int,
    log: TextIO,
    *,
    dependencies: tuple[Path, ...] = (),
) -> None:
    if source.name.startswith("libva-"):
        # Upstream hard-codes shared_library(), ignoring default_library. Use
        # Meson's normal selector so the pinned adapter honors our static build.
        path = source / "va/meson.build"
        text = path.read_text()
        if text.count("shared_library(") != 6:
            raise RuntimeError(
                "Pinned libva static-build patch context differs from verified source"
            )
        path.write_text(text.replace("shared_library(", "library("))
    workspace = source / "native-build"
    for command in (
        ["meson", "setup", str(workspace), str(source), f"--prefix={prefix}", *options],
        ["meson", "compile", "-C", str(workspace), "-j", str(jobs)],
        ["meson", "install", "-C", str(workspace)],
    ):
        subprocess.run(
            command,
            env=private_environment(prefix, encoders=dependencies),
            stdout=log,
            stderr=log,
            check=True,
        )
    if (source / "COPYING").is_file():
        shutil.copyfile(source / "COPYING", prefix / "COPYING")


def build_vpl(source: Path, prefix: Path, options: tuple[str, ...], jobs: int, log: TextIO) -> None:
    build_cmake(source, prefix, options, jobs, log)
    shutil.copyfile(source / "LICENSE", prefix / "LICENSE")
    # Upstream's static pkg-config file omits the dispatcher's C++ runtime.
    # FFmpeg links its C feature checks with cc, so retain that transitive link.
    path = prefix / "lib/pkgconfig/vpl.pc"
    text = path.read_text()
    if text.count("Libs.private:") != 1:
        raise RuntimeError("Pinned oneVPL static pkg-config format differs from verified source")
    path.write_text(text.replace("Libs.private:", "Libs.private: -lstdc++ -pthread"))


def ensure_hardware(jobs: int, cargo: str) -> tuple[Path, ...]:
    if platform.system() != "Linux":
        return ()
    drm = ensure_dependency(native_dependency("libdrm", cargo), jobs, LIBDRM_CONFIGURE, build_meson)
    va = ensure_dependency(
        native_dependency("libva", cargo),
        jobs,
        LIBVA_CONFIGURE,
        partial(build_meson, dependencies=(drm,)),
        dependencies=(drm,),
    )
    vpl = ensure_dependency(native_dependency("libvpl", cargo), jobs, LIBVPL_CONFIGURE, build_vpl)
    nvcodec = ensure_dependency(
        native_dependency("nv-codec-headers", cargo), jobs, (), build_nvcodec
    )
    return drm, va, vpl, nvcodec


def complete(prefix: Path, dependency: NativeDependency) -> bool:
    if not (prefix / ".complete").is_file() or not all(
        (prefix / path).is_file() for path in dependency.required
    ):
        return False
    # A cache with shared media libraries must never satisfy a static build.
    return not any(
        path.name.endswith((".dylib", ".so")) or ".so." in path.name
        for path in (prefix / "lib").rglob("*")
    )


def ensure_dependency(
    dependency: NativeDependency,
    jobs: int,
    options: tuple[str, ...],
    build: Callable[[Path, Path, tuple[str, ...], int, TextIO], None],
    *,
    dependencies: tuple[Path, ...] = (),
) -> Path:
    if platform.system() not in ("Linux", "Darwin"):
        raise RuntimeError("The bundled native build supports Linux and macOS")
    root = Path(tempfile.gettempdir()) / f"homesec-native-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"Native cache must be an owned private directory: {root}")
    compiler_keys = ("CC", "CXX") if dependency.name in ("OpenCV", "libvpl") else ("CC",)
    toolchain: dict[str, object] = {}
    for variable in compiler_keys:
        compiler = shlex.split(os.environ.get(variable, "c++" if variable == "CXX" else "cc"))
        identity = subprocess.check_output([*compiler, "--version"], text=True)
        toolchain[variable] = [compiler, identity]
    if dependency.name in ("OpenCV", "Opus", "libvpl"):
        toolchain["cmake"] = subprocess.check_output(["cmake", "--version"], text=True)
    if dependency.name in ("libdrm", "libva"):
        toolchain["meson"] = subprocess.check_output(["meson", "--version"], text=True)
        toolchain["ninja"] = subprocess.check_output(["ninja", "--version"], text=True)
    fingerprint = {
        "archive": dependency.sha256,
        "recipe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system": platform.system(),
        "machine": platform.machine(),
        "compiler": toolchain,
        "options": options,
        "dependencies": [str(prefix) for prefix in dependencies],
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CFLAGS",
                "CXXFLAGS",
                "CPPFLAGS",
                "LDFLAGS",
                "SDKROOT",
                "MACOSX_DEPLOYMENT_TARGET",
            )
        },
    }
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    prefix = root / f"{dependency.source_directory}-{key}"
    # One lock also serializes archive publication. Cargo can run independently
    # once the immutable installations have been published.
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete(prefix, dependency):
            print(f"Using cached {dependency.name} {dependency.version}: {prefix}", flush=True)
            return prefix
        archive = download_archive(root, dependency)
        if prefix.exists():
            shutil.rmtree(prefix)
        prefix.mkdir()
        log_path = root / f"{dependency.source_directory}-{key}.log"
        print(
            f"Building {dependency.name} {dependency.version} with {jobs} parallel jobs; log: {log_path}",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="build-", dir=root) as workspace:
            source = extract_archive(archive, Path(workspace), dependency)
            with log_path.open("w") as log:
                build(source, prefix, options, jobs, log)
        (prefix / ".complete").touch()
        if not complete(prefix, dependency):
            (prefix / ".complete").unlink()
            raise RuntimeError(f"{dependency.name} installation is incomplete; see {log_path}")
    return prefix


def ensure_ffmpeg(jobs: int, encoders: tuple[Path, ...]) -> Path:
    options = FFMPEG_CONFIGURE + (
        FFMPEG_LINUX_CONFIGURE if platform.system() == "Linux" else FFMPEG_DARWIN_CONFIGURE
    )
    if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686") and not shutil.which(
        "nasm"
    ):
        options += ("--disable-x86asm",)
    return ensure_dependency(
        FFMPEG,
        jobs,
        options,
        partial(build_ffmpeg, encoders=encoders),
        dependencies=encoders,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=available_jobs())
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a Cargo command after --")
    try:
        cargo = command[0] if Path(command[0]).name == "cargo" else os.environ.get("CARGO", "cargo")
        opencv_pin = native_dependency("OpenCV", cargo)
        x264_pin = native_dependency("x264", cargo)
        opus_pin = native_dependency("Opus", cargo)
        x264_options: tuple[str, ...] = X264_CONFIGURE
        if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686") and not shutil.which(
            "nasm"
        ):
            x264_options += ("--disable-asm",)
        encoders = (
            ensure_dependency(x264_pin, args.jobs, x264_options, build_x264),
            ensure_dependency(opus_pin, args.jobs, OPUS_CONFIGURE, build_opus),
        )
        dependencies = (*encoders, *ensure_hardware(args.jobs, cargo))
        ffmpeg = ensure_ffmpeg(args.jobs, dependencies)
        opencv = ensure_dependency(opencv_pin, args.jobs, OPENCV_CONFIGURE, build_opencv)
        return subprocess.call(
            command,
            env=private_environment(ffmpeg, opencv, opencv_pin.version, encoders=dependencies),
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Bundled native build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
