"""Behavioral checks for the private native build bootstrap."""

import hashlib
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from native.webrtc import build_ffmpeg


def test_download_rejects_checksum_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: The release server returns bytes that do not match the pinned checksum.
    monkeypatch.setattr(build_ffmpeg.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))

    # When: Downloading the native dependency.
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        build_ffmpeg.download_archive(tmp_path)

    # Then: No unverified archive or partial download is retained.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["../outside", "/outside"])
def test_extraction_rejects_traversal(tmp_path: Path, name: str) -> None:
    # Given: An archive member points outside the temporary source directory.
    archive = tmp_path / "source.tar.xz"
    with tarfile.open(archive, "w:xz") as out:
        out.addfile(tarfile.TarInfo(name))

    # When: Extracting it.
    with pytest.raises(RuntimeError, match="Unsafe"):
        build_ffmpeg.extract_archive(archive, tmp_path / "source")

    # Then: The source tree has not been created.
    assert not (tmp_path / "source").exists()


def test_private_discovery_preserves_system_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: The shell points at incompatible custom FFmpeg headers and libraries.
    monkeypatch.setenv("FFMPEG_DIR", "/usr/local")
    monkeypatch.setenv("PKG_CONFIG_PATH", "/usr/local/lib/pkgconfig")
    monkeypatch.setenv("HOST_PKG_CONFIG_LIBDIR", "/other/lib")
    monkeypatch.setenv("PKG_CONFIG_PATH_aarch64_unknown_linux_gnu", "/target/lib")
    monkeypatch.setenv("BINDGEN_EXTRA_CLANG_ARGS", "-I/usr/local/include")
    monkeypatch.setenv("PATH", "/system/tools")

    # When: Invoking Cargo with the private installation.
    environment = build_ffmpeg.private_environment(tmp_path)

    # Then: Only pinned pkg-config metadata is visible, while recording retains its CLI.
    assert environment["PKG_CONFIG_PATH"] == str(tmp_path / "lib/pkgconfig")
    assert environment["PKG_CONFIG_LIBDIR"] == str(tmp_path / "lib/pkgconfig")
    assert environment["HOMESEC_FFMPEG_REFERENCE"] == str(tmp_path / "bin/ffmpeg")
    assert environment["PATH"] == "/system/tools"
    assert "FFMPEG_DIR" not in environment
    assert "HOST_PKG_CONFIG_LIBDIR" not in environment
    assert "PKG_CONFIG_PATH_aarch64_unknown_linux_gnu" not in environment
    assert "BINDGEN_EXTRA_CLANG_ARGS" not in environment


def test_parallel_builds_share_cache_and_repair_incomplete_install(tmp_path: Path) -> None:
    # Given: A verified source fixture and fake toolchain that records native commands.
    source = tmp_path / f"ffmpeg-{build_ffmpeg.VERSION}"
    source.mkdir()
    (source / "COPYING.LGPLv2.1").write_text("fixture license")
    configure = source / "configure"
    configure.write_text('#!/bin/sh\nprintf "%s" "${1#--prefix=}" > .prefix\n')
    configure.chmod(0o755)
    cache = tmp_path / f"homesec-ffmpeg-{os.getuid()}"
    cache.mkdir(mode=0o700)
    archive = cache / build_ffmpeg.ARCHIVE
    with tarfile.open(archive, "w:xz") as out:
        out.add(source, arcname=source.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    script = tmp_path / "build_ffmpeg.py"
    script.write_text(Path(build_ffmpeg.__file__).read_text().replace(build_ffmpeg.SHA256, digest))
    tools = tmp_path / "tools"
    tools.mkdir()
    compiler = tools / "cc"
    compiler.write_text("#!/bin/sh\necho fixture-compiler\n")
    compiler.chmod(0o755)
    make = tools / "make"
    make.write_text(
        '#!/bin/sh\nset -eu\nprintf "%s\\n" "$*" >> "$BUILD_CALLS"\n'
        'sleep 0.05\nif [ "$1" = install ]; then\n'
        'prefix=$(cat .prefix)\nmkdir -p "$prefix/bin" "$prefix/lib/pkgconfig"\n'
        'touch "$prefix/bin/ffmpeg"\nfor lib in avcodec avformat avfilter avutil swscale; do\n'
        'mkdir -p "$prefix/include/lib$lib"\n'
        'touch "$prefix/lib/lib$lib.a" "$prefix/lib/pkgconfig/lib$lib.pc" '
        '"$prefix/include/lib$lib/$lib.h"\ndone\nfi\n'
    )
    make.chmod(0o755)
    calls = tmp_path / "calls"
    environment = {
        **os.environ,
        "TMPDIR": str(tmp_path),
        "CC": str(compiler),
        "PATH": f"{tools}:{os.environ['PATH']}",
        "BUILD_CALLS": str(calls),
    }
    command = [sys.executable, str(script), "--jobs", "3", "--", sys.executable, "-c", "pass"]

    # When: Two developers/check targets request the same build simultaneously.
    processes = [
        subprocess.Popen(command, env=environment, stdout=subprocess.PIPE) for _ in range(2)
    ]
    outputs = [process.communicate(timeout=15)[0] for process in processes]

    # Then: One parallel native build is published and the other consumes its cache.
    assert all(process.returncode == 0 for process in processes)
    assert calls.read_text().splitlines() == ["-j3", "install"]
    assert any(b"Using cached" in output for output in outputs)
    assert not list(cache.glob("build-*"))
    prefixes = [path for path in cache.iterdir() if path.is_dir()]
    assert len(prefixes) == 1

    # Given: A cached installation has lost an archive despite its completion marker.
    (prefixes[0] / "lib/libavcodec.a").unlink()

    # When: Building again.
    subprocess.run(command, env=environment, check=True, capture_output=True, timeout=15)

    # Then: It repairs the installation using the cached verified download.
    assert calls.read_text().splitlines() == ["-j3", "install", "-j3", "install"]
    assert (prefixes[0] / "lib/libavcodec.a").is_file()

    # Given: The next rebuild fails at the native toolchain boundary.
    (prefixes[0] / "include/libavcodec/avcodec.h").unlink()
    make.write_text("#!/bin/sh\nexit 1\n")

    # When: Attempting that rebuild.
    failure = subprocess.run(command, env=environment, capture_output=True, timeout=15)

    # Then: Failure is explicit, temporary sources are removed, and the cache is unpublished.
    assert failure.returncode == 1
    assert b"Bundled FFmpeg build failed" in failure.stderr
    assert not (prefixes[0] / ".complete").exists()
    assert not list(cache.glob("build-*"))


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_invalid_parallelism_is_refused_before_build(jobs: str) -> None:
    # Given: A caller supplies invalid build parallelism.
    # When: Invoking the bootstrap with a nonexistent command.
    result = subprocess.run(
        [sys.executable, build_ffmpeg.__file__, "--jobs", jobs, "--", "not-a-tool"],
        capture_output=True,
        timeout=5,
    )

    # Then: Argument validation refuses it before downloading or executing any tool.
    assert result.returncode == 2
    assert b"--jobs must be positive" in result.stderr
