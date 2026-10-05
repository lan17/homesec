"""Build pinned, private FFmpeg libraries before invoking Cargo (Linux/macOS)."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import TextIO

VERSION = "8.1.3"
ARCHIVE = f"ffmpeg-{VERSION}.tar.xz"
URL = f"https://ffmpeg.org/releases/{ARCHIVE}"
SHA256 = "7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3"
CONFIGURE = (
    "--disable-everything",
    "--disable-autodetect",
    "--disable-network",
    "--disable-shared",
    "--enable-static",
    "--enable-pic",
    "--disable-doc",
    "--disable-debug",
    "--disable-avdevice",
    "--disable-swresample",
    "--disable-ffplay",
    "--disable-ffprobe",
    "--enable-ffmpeg",
    "--enable-decoder=h264",
    "--enable-parser=h264",
    "--enable-demuxer=h264",
    "--enable-protocol=file,pipe",
    "--enable-filter=fps,scale,format",
    "--enable-encoder=rawvideo",
    "--enable-muxer=rawvideo",
)
LIBRARIES = ("avcodec", "avformat", "avfilter", "avutil", "swscale")


def available_jobs() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1


def verify_archive(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise RuntimeError(f"FFmpeg archive checksum mismatch: remove {path} and retry")


def download_archive(root: Path) -> Path:
    archive = root / ARCHIVE
    if not archive.exists():
        print(f"Downloading FFmpeg {VERSION} to {root}", flush=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=root) as download:
            temporary = Path(download) / ARCHIVE
            with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            verify_archive(temporary)
            temporary.replace(archive)
    verify_archive(archive)
    return archive


def extract_archive(archive: Path, workspace: Path) -> Path:
    # Only the checksum-verified official archive reaches extraction. Reject
    # links and traversal as well, including on Python versions without filters.
    with tarfile.open(archive) as source:
        for member in source.getmembers():
            parts = PurePosixPath(member.name)
            if (
                parts.is_absolute()
                or ".." in parts.parts
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError("Unsafe FFmpeg archive member")
        source.extractall(workspace)  # noqa: S202 (verified checksum and validated members)
    return workspace / f"ffmpeg-{VERSION}"


def private_environment(prefix: Path) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != "FFMPEG_DIR"
        and "PKG_CONFIG" not in key
        and not key.startswith("BINDGEN_EXTRA_CLANG_ARGS")
    }
    environment["PKG_CONFIG_PATH"] = str(prefix / "lib/pkgconfig")
    environment["PKG_CONFIG_LIBDIR"] = str(prefix / "lib/pkgconfig")
    # The parity oracle needs the same FFmpeg build; other fixtures and Python
    # recording still use the full system CLI through the unchanged PATH.
    environment["HOMESEC_FFMPEG_REFERENCE"] = str(prefix / "bin/ffmpeg")
    return environment


def build(source: Path, prefix: Path, options: tuple[str, ...], jobs: int, log: TextIO) -> None:
    environment = private_environment(prefix)
    for command in (
        ["./configure", f"--prefix={prefix}", *options],
        ["make", f"-j{jobs}"],
        ["make", "install"],
    ):
        subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=log, check=True)
    shutil.copyfile(source / "COPYING.LGPLv2.1", prefix / "COPYING.LGPLv2.1")


def complete(prefix: Path) -> bool:
    required = [prefix / "bin/ffmpeg", prefix / "COPYING.LGPLv2.1"]
    for library in LIBRARIES:
        required.extend(
            (
                prefix / f"lib/lib{library}.a",
                prefix / f"lib/pkgconfig/lib{library}.pc",
                prefix / f"include/lib{library}/{library}.h",
            )
        )
    return (prefix / ".complete").is_file() and all(path.is_file() for path in required)


def ensure_ffmpeg(jobs: int) -> Path:
    if platform.system() not in ("Linux", "Darwin"):
        raise RuntimeError("The bundled FFmpeg build supports Linux and macOS")
    root = Path(tempfile.gettempdir()) / f"homesec-ffmpeg-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"FFmpeg cache must be an owned private directory: {root}")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    identity = subprocess.check_output([*compiler, "--version"], text=True)
    options: tuple[str, ...] = CONFIGURE
    if platform.machine().lower() in ("x86_64", "amd64", "i386", "i686") and not shutil.which(
        "nasm"
    ):
        options += ("--disable-x86asm",)
    fingerprint = {
        "archive": SHA256,
        "recipe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "system": platform.system(),
        "machine": platform.machine(),
        "compiler": [compiler, identity],
        "options": options,
        "environment": {
            key: os.environ.get(key, "")
            for key in ("CFLAGS", "CPPFLAGS", "LDFLAGS", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET")
        },
    }
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:20]
    prefix = root / key
    # One lock also serializes archive publication. Cargo invocations can run
    # independently once the complete immutable installation has been published.
    with (root / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete(prefix):
            print(f"Using cached FFmpeg {VERSION}: {prefix}", flush=True)
            return prefix
        archive = download_archive(root)
        if prefix.exists():
            shutil.rmtree(prefix)
        prefix.mkdir()
        log_path = root / f"{key}.log"
        print(f"Building FFmpeg {VERSION} with make -j{jobs}; log: {log_path}", flush=True)
        with tempfile.TemporaryDirectory(prefix="build-", dir=root) as workspace:
            source = extract_archive(archive, Path(workspace))
            with log_path.open("w") as log:
                build(source, prefix, options, jobs, log)
        (prefix / ".complete").touch()
        if not complete(prefix):
            (prefix / ".complete").unlink()
            raise RuntimeError(f"FFmpeg installation is incomplete; see {log_path}")
    return prefix


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
        prefix = ensure_ffmpeg(args.jobs)
        return subprocess.call(command, env=private_environment(prefix))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Bundled FFmpeg build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
