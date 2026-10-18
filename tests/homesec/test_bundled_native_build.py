"""Behavioral checks for the private static native build bootstrap."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
from native.webrtc import build_native

OPENCV_FIXTURE = build_native.NativeDependency(
    name="OpenCV",
    version="4.12.0",
    archive="opencv-4.12.0.tar.gz",
    url="https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.12.0",
    sha256="44c106d5bb47efec04e531fd93008b3fcd1d27138985c5baf4eafac0e1ec9e9d",
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
X264_FIXTURE = build_native.NativeDependency(
    name="x264",
    version="b35605ace3ddf7c1a5d67a2eb553f034aef41d55",
    archive="x264-b35605ace3ddf7c1a5d67a2eb553f034aef41d55.tar.gz",
    url="https://code.videolan.org/videolan/x264/-/archive/b35605ace3ddf7c1a5d67a2eb553f034aef41d55/x264-b35605ace3ddf7c1a5d67a2eb553f034aef41d55.tar.gz",
    sha256="cd71a7515b0e9a012e1ac9b1f8415bebcaf6fc97d4db32286642ac4c0fbe24f9",
    required=("lib/libx264.a", "lib/pkgconfig/x264.pc", "include/x264.h", "COPYING"),
)
OPUS_FIXTURE = build_native.NativeDependency(
    name="Opus",
    version="1.6.1",
    archive="opus-1.6.1.tar.gz",
    url="https://downloads.xiph.org/releases/opus/opus-1.6.1.tar.gz",
    sha256="6ffcb593207be92584df15b32466ed64bbec99109f007c82205f0194572411a1",
    required=("lib/libopus.a", "lib/pkgconfig/opus.pc", "include/opus/opus.h", "COPYING"),
)
HARDWARE_FIXTURES = tuple(
    build_native.NativeDependency(
        name, version, f"{name}-{version}.tar.{archive_type}", url, checksum, required
    )
    for name, version, archive_type, url, checksum, required in (
        (
            "nv-codec-headers",
            "n8.1.24.15",
            "gz",
            "https://codeload.github.com/FFmpeg/nv-codec-headers/tar.gz/refs/tags/n8.1.24.15",
            "1e1600606cde2c04f5b1a71cb3b2e966b3ffe4e252a6ddec5781151d982e259f",
            (
                "lib/pkgconfig/ffnvcodec.pc",
                "include/ffnvcodec/dynlink_loader.h",
                "include/ffnvcodec/nvEncodeAPI.h",
            ),
        ),
        (
            "libdrm",
            "2.4.124",
            "xz",
            "https://dri.freedesktop.org/libdrm/libdrm-2.4.124.tar.xz",
            "ac36293f61ca4aafaf4b16a2a7afff312aa4f5c37c9fbd797de9e3c0863ca379",
            ("lib/libdrm.a", "lib/pkgconfig/libdrm.pc", "include/xf86drm.h"),
        ),
        (
            "libva",
            "2.22.0",
            "gz",
            "https://codeload.github.com/intel/libva/tar.gz/refs/tags/2.22.0",
            "467c418c2640a178c6baad5be2e00d569842123763b80507721ab87eb7af8735",
            (
                "lib/libva.a",
                "lib/libva-drm.a",
                "lib/pkgconfig/libva.pc",
                "lib/pkgconfig/libva-drm.pc",
                "include/va/va.h",
                "COPYING",
            ),
        ),
        (
            "libvpl",
            "2.15.0",
            "gz",
            "https://codeload.github.com/intel/libvpl/tar.gz/refs/tags/v2.15.0",
            "7218c3b8206b123204c3827ce0cf7c008d5c693c1f58ab461958d05fe6f847b3",
            ("lib/libvpl.a", "lib/pkgconfig/vpl.pc", "include/vpl/mfxvideo.h", "LICENSE"),
        ),
    )
)
NATIVE_FIXTURES = (
    build_native.FFMPEG,
    OPENCV_FIXTURE,
    X264_FIXTURE,
    OPUS_FIXTURE,
    *HARDWARE_FIXTURES,
)


@pytest.mark.parametrize("dependency", NATIVE_FIXTURES)
def test_download_rejects_checksum_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dependency: build_native.NativeDependency
) -> None:
    # Given: A release server returns bytes that do not match the pinned checksum.
    monkeypatch.setattr(build_native.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))

    # When: Downloading a native dependency.
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_native.download_archive(tmp_path, dependency)

    # Then: No unverified archive or partial download is retained.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dependency", NATIVE_FIXTURES)
@pytest.mark.parametrize("name", ["../outside", "/outside"])
def test_extraction_rejects_traversal(
    tmp_path: Path, dependency: build_native.NativeDependency, name: str
) -> None:
    # Given: An archive member points outside the temporary source directory.
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        out.addfile(tarfile.TarInfo(name))

    # When: Extracting it.
    with pytest.raises(RuntimeError, match="Unsafe"):
        build_native.extract_archive(archive, tmp_path / "source", dependency)

    # Then: The source tree has not been created.
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_extraction_rejects_links(tmp_path: Path, link_type: bytes) -> None:
    # Given: An archive contains a link that could escape the source directory.
    archive = tmp_path / "source.tar.gz"
    member = tarfile.TarInfo(f"{OPENCV_FIXTURE.source_directory}/include")
    member.type = link_type
    member.linkname = "/outside"
    with tarfile.open(archive, "w:gz") as out:
        out.addfile(member)

    # When: Extracting the archive.
    with pytest.raises(RuntimeError, match="Unsafe"):
        build_native.extract_archive(archive, tmp_path / "source", OPENCV_FIXTURE)

    # Then: The source tree has not been created.
    assert not (tmp_path / "source").exists()


def test_libdrm_omits_disabled_test_trees_without_extracting_links(tmp_path: Path) -> None:
    # Given: The pinned libdrm release's disabled test trees contain symlinks.
    dependency = next(item for item in HARDWARE_FIXTURES if item.name == "libdrm")
    archive = tmp_path / "source.tar.xz"
    with tarfile.open(archive, "w:xz") as out:
        for name in ("tests/amdgpu/.editorconfig", "intel/tests/gen4-3d.batch.sh"):
            member = tarfile.TarInfo(f"{dependency.source_directory}/{name}")
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside"
            out.addfile(member)
        member = tarfile.TarInfo(f"{dependency.source_directory}/xf86drm.c")
        member.size = len(b"source")
        out.addfile(member, io.BytesIO(b"source"))

    # When: Extracting source for a build that explicitly disables all tests.
    source = build_native.extract_archive(archive, tmp_path / "source", dependency)

    # Then: Only actual build inputs are extracted; no link or test tree is created.
    assert (source / "xf86drm.c").read_bytes() == b"source"
    assert not (source / "tests").exists()
    assert not (source / "intel").exists()


@pytest.mark.parametrize("system,runtime", [("Darwin", "c++"), ("Linux", "stdc++,m,dl,pthread,rt")])
def test_private_discovery_preserves_system_cli_and_refuses_opencv_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, runtime: str
) -> None:
    # Given: The shell points at arbitrary shared FFmpeg and OpenCV installations.
    for key, value in {
        "FFMPEG_DIR": "/usr/local",
        "PKG_CONFIG_PATH": "/usr/local/lib/pkgconfig",
        "HOST_PKG_CONFIG_LIBDIR": "/other/lib",
        "PKG_CONFIG_PATH_aarch64_unknown_linux_gnu": "/target/lib",
        "BINDGEN_EXTRA_CLANG_ARGS": "-I/usr/local/include",
        "OPENCV_LINK_LIBS": "+opencv_world",
        "OPENCV_LINK_PATHS": "/usr/local/lib",
        "OPENCV_INCLUDE_PATHS": "/usr/local/include",
        "OPENCV_CLANG_ARGS": "-I/usr/local/include",
        "OPENCV_DISABLE_PROBES": "environment",
        "OpenCV_DIR": "/usr/local/share/opencv4",
        "CMAKE_PREFIX_PATH": "/usr/local",
        "HOMESEC_OPENCV_PREFIX": "/usr/local",
        "HOMESEC_OPENCV_VERSION": "5.0.0",
        "PATH": "/system/tools",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(build_native.platform, "system", lambda: system)
    ffmpeg = tmp_path / "ffmpeg"
    opencv = tmp_path / "opencv"
    encoders = (tmp_path / "x264", tmp_path / "opus")

    # When: Invoking Cargo with the pinned private installations.
    environment = build_native.private_environment(
        ffmpeg, opencv, OPENCV_FIXTURE.version, encoders=encoders
    )

    # Then: Only private media libraries can be discovered, with no system fallback.
    assert environment["PKG_CONFIG_PATH"].split(os.pathsep) == [
        str(prefix / "lib/pkgconfig") for prefix in (ffmpeg, *encoders)
    ]
    assert environment["PKG_CONFIG_LIBDIR"] == environment["PKG_CONFIG_PATH"]
    assert environment["HOMESEC_FFMPEG_REFERENCE"] == str(ffmpeg / "bin/ffmpeg")
    assert environment["HOMESEC_OPENCV_PREFIX"] == str(opencv)
    assert environment["HOMESEC_OPENCV_VERSION"] == "4.12.0"
    assert environment["OPENCV_INCLUDE_PATHS"] == str(opencv / "include/opencv4")
    assert environment["OPENCV_LINK_PATHS"] == f"{opencv}/lib,{opencv}/lib/opencv4/3rdparty"
    assert environment["OPENCV_LINK_LIBS"] == (
        f"static=opencv_imgproc,static=opencv_core,static=zlib,{runtime}"
    )
    assert environment["OPENCV_DISABLE_PROBES"] == "pkg_config,cmake,vcpkg_cmake,vcpkg"
    assert environment["PATH"] == "/system/tools"
    for key in (
        "FFMPEG_DIR",
        "HOST_PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_PATH_aarch64_unknown_linux_gnu",
        "BINDGEN_EXTRA_CLANG_ARGS",
        "OPENCV_CLANG_ARGS",
        "OpenCV_DIR",
        "CMAKE_PREFIX_PATH",
    ):
        assert key not in environment


def _executable(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
def test_parallel_builds_share_cache_and_repair_incomplete_install(
    tmp_path: Path, system: str
) -> None:
    # Given: Verified source fixtures and a fake toolchain recording real bootstrap commands.
    cache = tmp_path / f"homesec-native-{os.getuid()}"
    cache.mkdir(mode=0o700)
    dependencies = []
    for dependency in NATIVE_FIXTURES:
        source = tmp_path / dependency.source_directory
        source.mkdir()
        if dependency.name == "FFmpeg":
            (source / "COPYING.LGPLv2.1").write_text("fixture FFmpeg license")
            (source / "COPYING.GPLv3").write_text("fixture FFmpeg GPL3 license")
            (source / "libavformat").mkdir()
            (source / "libavformat/rtsp.c").write_text(
                "\n".join(original for original, _ in build_native.FFMPEG_RTSP_PATCH)
            )
            (source / "libavformat/rtspcodes.h").write_text(
                build_native.FFMPEG_RTSP_STATUS_PATCH[0]
            )
            (source / "libavcodec").mkdir()
            (source / "libavcodec/h264_parser.c").write_text(
                "        if (ff_combine_frame(pc, next, &buf, &buf_size) < 0) {"
            )
        elif dependency.name == "OpenCV":
            (source / "LICENSE").write_text("fixture OpenCV license")
            (source / "3rdparty/zlib").mkdir(parents=True)
            (source / "3rdparty/zlib/LICENSE").write_text("fixture zlib license")
        else:
            (source / "COPYING").write_text(f"fixture {dependency.name} license")
            (source / "LICENSE").write_text(f"fixture {dependency.name} license")
            if dependency.name == "libva":
                (source / "va").mkdir()
                (source / "va/meson.build").write_text("shared_library(\n" * 6)
        if dependency.name in ("FFmpeg", "x264"):
            _executable(
                source / "configure",
                f"""import json, os, pathlib, sys
prefix = sys.argv[1].removeprefix('--prefix=')
pathlib.Path('.prefix').write_text(prefix)
pathlib.Path('.name').write_text({dependency.name.lower()!r})
pathlib.Path(prefix, 'configuration.json').write_text(json.dumps({{'arguments': sys.argv[1:], 'PKG_CONFIG_PATH': os.environ['PKG_CONFIG_PATH'], 'PKG_CONFIG_LIBDIR': os.environ['PKG_CONFIG_LIBDIR']}}))
""",
            )
        archive = cache / dependency.archive
        with tarfile.open(archive, "w:xz" if dependency.archive.endswith(".xz") else "w:gz") as out:
            out.add(source, arcname=source.name)
        dependencies.append(
            replace(dependency, sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        )
    ffmpeg, opencv, x264, opus = dependencies[:4]
    script = tmp_path / "build_native.py"
    script.write_text(
        Path(build_native.__file__)
        .read_text()
        .replace(build_native.FFMPEG.sha256, ffmpeg.sha256)
        .replace("sys.exit(main())", f"platform.system = lambda: {system!r}\n    sys.exit(main())")
    )
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls"
    compiler = _executable(tools / "cc", "print('fixture-compiler')\n")
    _executable(tools / "c++", "print('fixture-compiler')\n")
    _executable(
        tools / "cargo",
        f"""import json, sys
print(json.dumps({{'packages': [{{'manifest_path': sys.argv[-1], 'metadata': {{'native-dependencies': { {dependency.name.lower(): {"version": dependency.version, "sha256": dependency.sha256} for dependency in dependencies if dependency.name != "FFmpeg"}!r} }}}}]}}))
""",
    )
    required = {dependency.name.lower(): dependency.required for dependency in dependencies}
    _executable(
        tools / "make",
        f"""import json, os, pathlib, sys, time
with open(os.environ['BUILD_CALLS'], 'a') as out:
    out.write(json.dumps(['make', *sys.argv[1:]]) + '\\n')
time.sleep(0.05)
if sys.argv[-1] == 'install' or sys.argv[1] == 'install-lib-static':
    if any(arg.startswith('PREFIX=') for arg in sys.argv):
        prefix = next(arg.removeprefix('PREFIX=') for arg in sys.argv if arg.startswith('PREFIX='))
        kind = 'nv-codec-headers'
    else:
        prefix = pathlib.Path('.prefix').read_text()
        kind = pathlib.Path('.name').read_text()
    for name in {required!r}[kind]:
        path = pathlib.Path(prefix) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
""",
    )
    _executable(
        tools / "cmake",
        f"""import json, os, pathlib, sys, time
args = sys.argv[1:]
if args == ['--version']:
    print('fixture-cmake')
    sys.exit(0)
with open(os.environ['BUILD_CALLS'], 'a') as out:
    out.write(json.dumps(['cmake', *args]) + '\\n')
time.sleep(0.05)
if args[0] == '-S':
    workspace = pathlib.Path(args[args.index('-B') + 1])
    workspace.mkdir()
    workspace.joinpath('.prefix').write_text(next(arg.removeprefix('-DCMAKE_INSTALL_PREFIX=') for arg in args if arg.startswith('-DCMAKE_INSTALL_PREFIX=')))
    workspace.joinpath('.name').write_text('opencv' if '-DBUILD_LIST=core,imgproc' in args else 'libvpl' if '-DINSTALL_EXAMPLES=OFF' in args else 'opus')
if args[0] == '--install':
    prefix = pathlib.Path(args[1]).joinpath('.prefix').read_text()
    kind = pathlib.Path(args[1]).joinpath('.name').read_text()
    for name in {required!r}[kind]:
        path = pathlib.Path(prefix) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if kind == 'libvpl':
        pathlib.Path(prefix, 'lib/pkgconfig/vpl.pc').write_text('Libs.private: ')
""",
    )
    _executable(
        tools / "meson",
        f"""import json, os, pathlib, sys
args = sys.argv[1:]
if args == ['--version']:
    print('fixture-meson')
    sys.exit(0)
with open(os.environ['BUILD_CALLS'], 'a') as out:
    out.write(json.dumps(['meson', *args]) + '\\n')
if args[0] == 'setup':
    workspace = pathlib.Path(args[1]); workspace.mkdir()
    workspace.joinpath('.prefix').write_text(next(arg.removeprefix('--prefix=') for arg in args if arg.startswith('--prefix=')))
    workspace.joinpath('.name').write_text('libva' if pathlib.Path(args[2]).name.startswith('libva-') else 'libdrm')
    workspace.joinpath('configuration.json').write_text(json.dumps({{'arguments': args, 'PKG_CONFIG_PATH': os.environ['PKG_CONFIG_PATH']}}))
if args[0] == 'install':
    workspace = pathlib.Path(args[args.index('-C') + 1])
    prefix = pathlib.Path(workspace.joinpath('.prefix').read_text())
    kind = workspace.joinpath('.name').read_text()
    for name in {required!r}[kind]:
        path = prefix / name; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
    prefix.joinpath('configuration.json').write_text(workspace.joinpath('configuration.json').read_text())
""",
    )
    _executable(tools / "ninja", "print('fixture-ninja')\n")
    environment = {
        **os.environ,
        "TMPDIR": str(tmp_path),
        "CC": str(compiler),
        "PATH": f"{tools}:{os.environ['PATH']}",
        "CARGO": str(tools / "cargo"),
        "BUILD_CALLS": str(calls),
    }
    command = [sys.executable, str(script), "--jobs", "3", "--", sys.executable, "-c", "pass"]

    # When: Two developers/check targets request the same build simultaneously.
    processes = [
        subprocess.Popen(command, env=environment, stdout=subprocess.PIPE) for _ in range(2)
    ]
    outputs = [process.communicate(timeout=20)[0] for process in processes]

    # Then: Every dependency is built once in parallel and reused by the other caller.
    assert all(process.returncode == 0 for process in processes)
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    assert commands[0:2] == [["make", "-j3"], ["make", "install-lib-static"]]
    build_call_count = 20 if system == "Linux" else 10
    assert len(commands) == build_call_count
    assert "-DBUILD_SHARED_LIBS=OFF" in commands[2]
    assert "-DOPUS_BUILD_SHARED_LIBRARY=OFF" in commands[2]
    assert commands[3][0:2] == ["cmake", "--build"]
    assert commands[3][-2:] == ["--parallel", "3"]
    assert commands[-5:-3] == [["make", "-j3"], ["make", "install"]]
    assert "-DBUILD_LIST=core,imgproc" in commands[-3]
    assert "-DBUILD_ZLIB=ON" in commands[-3]
    assert commands[-2][-2:] == ["--parallel", "3"]
    if system == "Linux":
        meson_setups = [command for command in commands if command[:2] == ["meson", "setup"]]
        assert len(meson_setups) == 2
        assert all("--default-library=static" in command for command in meson_setups)
        assert all("--wrap-mode=nodownload" in command for command in meson_setups)
        assert any(
            command[-2:] == ["-j", "3"]
            for command in commands
            if command[:2] == ["meson", "compile"]
        )
        assert any("PREFIX=" in arg for command in commands for arg in command)
    assert any(b"Using cached" in output for output in outputs)
    assert not list(cache.glob("build-*"))
    prefixes = [path for path in cache.iterdir() if path.is_dir()]
    assert len(prefixes) == (8 if system == "Linux" else 4)
    ffmpeg_prefix = next(path for path in prefixes if path.name.startswith("ffmpeg-"))
    opencv_prefix = next(path for path in prefixes if path.name.startswith("opencv-"))
    x264_prefix = next(path for path in prefixes if path.name.startswith("x264-"))
    opus_prefix = next(path for path in prefixes if path.name.startswith("opus-"))
    assert (opencv_prefix / "LICENSE").read_text() == "fixture OpenCV license"
    assert (opencv_prefix / "ZLIB-LICENSE").read_text() == "fixture zlib license"
    assert (x264_prefix / "COPYING").read_text() == "fixture x264 license"
    assert (opus_prefix / "COPYING").read_text() == "fixture Opus license"
    assert (ffmpeg_prefix / "COPYING.GPLv3").read_text() == "fixture FFmpeg GPL3 license"
    configuration = json.loads((ffmpeg_prefix / "configuration.json").read_text())
    assert "--enable-libx264" in configuration["arguments"]
    assert "--enable-libopus" in configuration["arguments"]
    assert "--enable-gpl" in configuration["arguments"]
    assert "--enable-version3" in configuration["arguments"]
    assert "--pkg-config-flags=--static" in configuration["arguments"]
    assert "--enable-decoders" in configuration["arguments"]
    assert "--enable-parsers" in configuration["arguments"]
    assert "--disable-autodetect" in configuration["arguments"]
    assert any(
        argument.startswith("--enable-encoder=") and "aac" in argument.split("=", 1)[1].split(",")
        for argument in configuration["arguments"]
    )
    expected_hardware = (
        build_native.FFMPEG_LINUX_CONFIGURE
        if system == "Linux"
        else build_native.FFMPEG_DARWIN_CONFIGURE
    )
    assert all(option in configuration["arguments"] for option in expected_hardware)
    assert configuration["PKG_CONFIG_PATH"].split(os.pathsep) == [
        str(prefix / "lib/pkgconfig")
        for prefix in (
            ffmpeg_prefix,
            x264_prefix,
            opus_prefix,
            *(
                next(path for path in prefixes if path.name.startswith(name + "-"))
                for name in ("libdrm", "libva", "libvpl", "nv-codec-headers")
                if system == "Linux"
            ),
        )
    ]
    assert configuration["PKG_CONFIG_LIBDIR"] == configuration["PKG_CONFIG_PATH"]

    # Given: Cached installations lose static archives despite their completion markers.
    (ffmpeg_prefix / "lib/libavcodec.a").unlink()
    (opencv_prefix / "lib/libopencv_core.a").unlink()
    (x264_prefix / "lib/libx264.a").unlink()
    (opus_prefix / "lib/libopus.a").unlink()

    # When: Building again.
    subprocess.run(command, env=environment, check=True, capture_output=True, timeout=20)

    # Then: All installations are repaired using their already verified downloads.
    assert len(calls.read_text().splitlines()) == build_call_count + 10
    assert (ffmpeg_prefix / "lib/libavcodec.a").is_file()
    assert (opencv_prefix / "lib/libopencv_core.a").is_file()
    assert (x264_prefix / "lib/libx264.a").is_file()
    assert (opus_prefix / "lib/libopus.a").is_file()

    # Given: Caches are contaminated with arbitrary shared OpenCV/Opus libraries.
    (opencv_prefix / "lib/libopencv_core.so.4.12").touch()
    (opus_prefix / "lib/libopus.dylib").touch()

    # When: Building again.
    subprocess.run(command, env=environment, check=True, capture_output=True, timeout=20)

    # Then: The installation is rebuilt and only static media libraries remain.
    assert len(calls.read_text().splitlines()) == build_call_count + 16
    assert not (opencv_prefix / "lib/libopencv_core.so.4.12").exists()
    assert not (opus_prefix / "lib/libopus.dylib").exists()

    # Given: Cargo selects a different verified x264 source checksum.
    x264_source = tmp_path / x264.source_directory
    (x264_source / "COPYING").write_text("updated fixture x264 license")
    with tarfile.open(cache / x264.archive, "w:gz") as archive:
        archive.add(x264_source, arcname=x264_source.name)
    updated_checksum = hashlib.sha256((cache / x264.archive).read_bytes()).hexdigest()
    cargo_fixture = tools / "cargo"
    cargo_fixture.write_text(cargo_fixture.read_text().replace(x264.sha256, updated_checksum))

    # When: Building with that new encoder pin.
    subprocess.run(command, env=environment, check=True, capture_output=True, timeout=20)

    # Then: FFmpeg is rebuilt for the new encoder while unrelated native caches are reused.
    assert len(calls.read_text().splitlines()) == build_call_count + 20
    updated_x264 = next(
        path for path in cache.glob("x264-*") if path.is_dir() and path != x264_prefix
    )
    updated_ffmpeg = next(
        path for path in cache.glob("ffmpeg-*") if path.is_dir() and path != ffmpeg_prefix
    )
    assert (updated_x264 / "COPYING").read_text() == "updated fixture x264 license"
    configuration = json.loads((updated_ffmpeg / "configuration.json").read_text())
    assert str(updated_x264 / "lib/pkgconfig") in configuration["PKG_CONFIG_PATH"].split(os.pathsep)
    assert str(x264_prefix / "lib/pkgconfig") not in configuration["PKG_CONFIG_PATH"].split(
        os.pathsep
    )

    # Given: The next OpenCV rebuild fails at the native toolchain boundary.
    (opencv_prefix / "include/opencv4/opencv2/imgproc.hpp").unlink()
    _executable(
        tools / "cmake",
        "import sys\nif sys.argv[1:] == ['--version']: print('fixture-cmake')\nelse: sys.exit(1)\n",
    )

    # When: Attempting that rebuild.
    failure = subprocess.run(command, env=environment, capture_output=True, timeout=20)

    # Then: Failure is explicit, sources are removed, and the cache is unpublished.
    assert failure.returncode == 1
    assert b"Bundled native build failed" in failure.stderr
    assert not (opencv_prefix / ".complete").exists()
    assert not list(cache.glob("build-*"))


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_invalid_parallelism_is_refused_before_build(jobs: str) -> None:
    # Given: A caller supplies invalid build parallelism.
    # When: Invoking the bootstrap with a nonexistent command.
    result = subprocess.run(
        [sys.executable, build_native.__file__, "--jobs", jobs, "--", "not-a-tool"],
        capture_output=True,
        timeout=5,
    )

    # Then: Argument validation refuses it before downloading or executing any tool.
    assert result.returncode == 2
    assert b"--jobs must be positive" in result.stderr


@pytest.mark.parametrize(
    "dependency", [OPENCV_FIXTURE, X264_FIXTURE, OPUS_FIXTURE, *HARDWARE_FIXTURES]
)
def test_native_pin_is_read_from_cargo_metadata(
    monkeypatch: pytest.MonkeyPatch, dependency: build_native.NativeDependency
) -> None:
    # Given: Cargo reports the exact native pin from this package's manifest.
    calls: list[list[str]] = []
    manifest = Path(build_native.__file__).with_name("Cargo.toml")

    def metadata(command: list[str], **kwargs: object) -> str:
        calls.append(command)
        return json.dumps(
            {
                "packages": [
                    {
                        "manifest_path": str(manifest),
                        "metadata": {
                            "native-dependencies": {
                                dependency.name.lower(): {
                                    "version": dependency.version,
                                    "sha256": dependency.sha256,
                                }
                            }
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(build_native.subprocess, "check_output", metadata)

    # When: Preparing the native release via an explicitly selected Cargo executable.
    name: build_native.DependencyName
    match dependency.name:
        case "OpenCV":
            name = "OpenCV"
        case "x264":
            name = "x264"
        case "Opus":
            name = "Opus"
        case "nv-codec-headers":
            name = "nv-codec-headers"
        case "libdrm":
            name = "libdrm"
        case "libva":
            name = "libva"
        case _:
            name = "libvpl"
    actual = build_native.native_dependency(name, "/toolchain/cargo")

    # Then: The source tag/checksum come from Cargo, without changing the lockfile.
    assert actual == dependency
    assert calls == [
        [
            "/toolchain/cargo",
            "metadata",
            "--no-deps",
            "--locked",
            "--format-version",
            "1",
            "--manifest-path",
            str(manifest),
        ]
    ]


@pytest.mark.parametrize(
    "pin",
    [
        {},
        {"version": "4.12.0", "sha256": "unverified"},
        {"version": "../source", "sha256": OPENCV_FIXTURE.sha256},
    ],
)
@pytest.mark.parametrize(
    "name", ["OpenCV", "x264", "Opus", "nv-codec-headers", "libdrm", "libva", "libvpl"]
)
def test_missing_or_invalid_native_pin_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    pin: dict[str, str],
    name: build_native.DependencyName,
) -> None:
    # Given: Cargo metadata does not contain a valid, checksum-pinned native release.
    manifest = Path(build_native.__file__).with_name("Cargo.toml")
    payload = {
        "packages": [
            {
                "manifest_path": str(manifest),
                "metadata": {"native-dependencies": {name.lower(): pin}},
            }
        ]
    }
    monkeypatch.setattr(
        build_native.subprocess, "check_output", lambda *args, **kwargs: json.dumps(payload)
    )

    # When: Preparing native dependencies.
    with pytest.raises(RuntimeError, match="must pin"):
        build_native.native_dependency(name)

    # Then: No system discovery or unpinned release is substituted.
