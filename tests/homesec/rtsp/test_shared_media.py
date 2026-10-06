"""Shared source consumers tested at a real executable control boundary."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import pytest

from homesec.models.clip import Clip
from homesec.models.config import WebRTCPreviewConfig
from homesec.models.preview import PreviewAnswer, PreviewOffer
from homesec.sources.rtsp.capabilities import RTSPTimeoutCapabilities
from homesec.sources.rtsp.core import RTSPSource, RTSPSourceConfig
from homesec.sources.rtsp.discovery import ProbeStreamInfo
from homesec.sources.rtsp.helper_client import HelperClient, HelperError
from homesec.sources.rtsp.motion_input import MotionObservation
from homesec.sources.rtsp.preflight import CameraPreflightDiagnostics, CameraPreflightOutcome
from homesec.sources.rtsp.recorder import RecordingHandle
from homesec.sources.rtsp.recording_profile import (
    MotionProfile,
    RecordingProfile,
    build_recording_profile_candidates,
)
from homesec.sources.rtsp.rust_motion import _MotionHelper, _MotionSettings, build_motion_input
from homesec.sources.rtsp.shared_media import (
    SharedMediaSession,
    SharedRecorder,
    _SharedReply,
    _SharedRequest,
)
from homesec.sources.rtsp.webrtc_publisher import RustWebRTCLivePublisher, _PreviewHelper

_HELPER = r"""import json, os, sys, time
from pathlib import Path

mode = MODE
trace = Path(__file__).with_suffix('.calls')
recordings = {}
peers = {}
preview = False
pending = None
motion_id = None
motion_reads = 0

def emit(value):
    print(json.dumps(value), flush=True)

emit({'event': 'ready', 'video_port': 20001, 'audio_port': 20003, 'media_port': 8189})
for line in sys.stdin:
    request = json.loads(line)
    with trace.open('a') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'request': request}) + '\n')
    command = request['command']
    reply = {'request_id': request['request_id'], 'ok': True}
    peers = {k: v for k, v in peers.items() if v > time.monotonic()}
    if command == 'start_motion':
        if motion_id is not None:
            reply.update(ok=False, error_code='motion_already_active')
        else:
            motion_id = request.get('motion_id')
            motion_reads = 0
            emit({'event': 'frame'})
    elif command == 'read_motion':
        if motion_id is None or request.get('motion_id') != motion_id:
            reply.update(ok=False, error_code='motion_generation_mismatch')
        elif mode == 'pending_motion':
            pending = request['request_id']
            continue
        else:
            motion_reads += 1
            fresh = mode == 'motion_generations' and motion_reads == 1
            reply['observation'] = {
                'motion': False if fresh else 25.0 >= request['threshold'],
                'changed_pixels': 0 if fresh else 19200,
                'changed_pct': 0.0 if fresh else 25.0,
            }
    elif command == 'discard_frame':
        if motion_id is None or request.get('motion_id') != motion_id:
            reply.update(ok=False, error_code='motion_generation_mismatch')
        else:
            reply['frame_available'] = True
    elif command == 'stop_motion':
        if mode == 'lost_motion_stop_reply':
            continue
        if mode == 'motion_stop_failure':
            reply.update(ok=False, error_code='motion_stop_failed')
        elif motion_id is not None and request.get('motion_id') != motion_id:
            reply.update(ok=False, error_code='motion_generation_mismatch')
        else:
            motion_id = None
            if pending is not None:
                emit({'request_id': pending, 'ok': False, 'error_code': 'cancelled'})
                pending = None
    elif command == 'start_recording':
        if mode == 'reject_start':
            reply.update(ok=False, error_code='unsupported_codec')
        else:
            path = Path(request['output_path'])
            if path.exists() or path.is_symlink():
                emit({'request_id': request['request_id'], 'ok': False, 'error_code': 'recording_path_exists'})
                continue
            path.with_name(path.name + '.partial').write_bytes(b'container pending finish')
            recordings[request['recording_id']] = path
            reply['recording_active'] = True
            if mode == 'publish_then_exit':
                path.with_name(path.name + '.partial').replace(path)
                sys.exit(0)
            if mode in ('lost_start_reply', 'lost_start_and_stop_replies', 'unretirable', 'publish_during_teardown'):
                continue
    elif command == 'stop_recording':
        if mode in ('lost_start_and_stop_replies', 'lost_stop_reply', 'unretirable'):
            continue
        if mode == 'missing_stop_ownership':
            emit({'request_id': request['request_id'], 'ok': False, 'error_code': 'recording_failed'})
            continue
        path = recordings.pop(request['recording_id'], None)
        if path is not None and mode != 'finish_failure':
            path.with_name(path.name + '.partial').replace(path)
            reply['recording_finalized'] = True
            if mode == 'publish_during_teardown':
                sys.exit(0)
        elif path is not None:
            reply.update(ok=False, error_code='recording_finalize_failed')
        reply['recording_active'] = False
    elif command == 'recording_status':
        if mode == 'exit_on_status':
            sys.exit(2)
        reply['recording_active'] = request['recording_id'] in recordings
    elif command == 'start':
        if preview:
            reply.update(ok=False, error_code='preview_already_active')
        else:
            preview = True
    elif command == 'offer':
        peers[request['session_id']] = time.monotonic() + request['lease_seconds']
        reply['sdp'] = 'v=0\r\ns=answer\r\n'
    elif command == 'renew':
        peers[request['session_id']] = time.monotonic() + request['lease_seconds']
    elif command == 'close':
        peers.pop(request['session_id'], None)
    elif command == 'stop_preview':
        if mode == 'lost_preview_stop_reply':
            continue
        if mode == 'preview_stop_failure':
            reply.update(ok=False, error_code='preview_stop_failed')
        else:
            preview = False
            peers.clear()
    elif command == 'stop' and mode == 'unretirable':
        continue
    reply.update(viewer_count=len(peers), active_session_count=len(peers), media_active=preview)
    emit(reply)
    if command == 'stop':
        break
"""


def helper_script(tmp_path: Path, mode: str = "normal") -> Path:
    path = tmp_path / "shared-helper"
    path.write_text(f"#!{sys.executable}\n" + _HELPER.replace("MODE", repr(mode)))
    path.chmod(0o700)
    return path


def requests(helper: Path) -> list[dict[str, object]]:
    trace = helper.with_suffix(".calls")
    return [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []


def commands(helper: Path) -> list[str]:
    return [record["request"]["command"] for record in requests(helper)]


@dataclass(frozen=True)
class LegacyHandle:
    pid: int = 1
    returncode: int | None = None


class LegacyRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.profiles: list[RecordingProfile] = []

    def configure_profile(self, profile: RecordingProfile) -> None:
        self.profiles.append(profile)

    def start(self, output_file: Path, stderr_log: Path) -> LegacyHandle:
        self.calls.append(("start", output_file))
        return LegacyHandle()

    def stop(self, handle: RecordingHandle, output_file: Path | None) -> None:
        self.calls.append(("stop", output_file))

    def is_alive(self, handle: RecordingHandle) -> bool:
        return True


class LegacyMotion:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def start(self, rtsp_url: str) -> None:
        self.calls.append(("start", rtsp_url))

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def set_motion_profile(self, profile: MotionProfile) -> None:
        self.calls.append(("profile", profile))

    def read_motion(self, timeout_s: float, threshold: float) -> MotionObservation | None:
        self.calls.append(("read", threshold))
        return None

    def discard_frame(self, timeout_s: float) -> bool:
        return False

    def is_running(self) -> bool:
        return False

    def exit_code(self) -> int | None:
        return None


def profile(audio_codec: str | None = "aac", audio_mode: str = "copy") -> RecordingProfile:
    return next(
        selected
        for selected in build_recording_profile_candidates(
            input_url="rtsp://camera/main",
            audio_codec=audio_codec,
        )
        if selected.audio_mode == audio_mode
    )


def recorder(session: SharedMediaSession, fallback: LegacyRecorder) -> SharedRecorder:
    result = SharedRecorder(
        session=session,
        fallback=fallback,
        connect_timeout_s=0.02,
        io_timeout_s=2.0,
    )
    result.configure_profile(profile(), video_codec="h264", audio_codec="aac")
    return result


def motion(session: SharedMediaSession, helper: Path, fallback: LegacyMotion):
    return build_motion_input(
        fallback=fallback,
        pixel_threshold=47,
        min_changed_pct=30.0,
        blur_kernel=7,
        recording_sensitivity_factor=2.0,
        frame_queue_size=3,
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=3.0,
        hwaccel_active=False,
        helper_path=str(helper),
        on_frame=lambda: None,
        helper_factory=session.motion_client,
    )


def start_motion(client: _MotionHelper) -> None:
    assert client.request(
        "start",
        rtsp_url="rtsp://camera/main",
        motion_config=_MotionSettings(
            pixel_threshold=47,
            min_changed_pct=30.0,
            blur_kernel=7,
            recording_sensitivity_factor=2.0,
        ),
        frame_queue_size=3,
        connect_timeout_s=2.0,
        io_timeout_s=3.0,
    ).ok


def preview(session: SharedMediaSession, config: WebRTCPreviewConfig) -> RustWebRTCLivePublisher:
    return RustWebRTCLivePublisher(
        camera_name="front",
        rtsp_url="rtsp://camera/main",
        config=config,
        idle_timeout_s=5.0,
        recording_policy="allow_during_recording",
        rtsp_connect_timeout_s=2.0,
        rtsp_io_timeout_s=3.0,
        timeout_capabilities=RTSPTimeoutCapabilities(),
        helper_factory=session.preview_client,
    )


def test_consumers_share_helper_and_detach_without_interrupting_rotation(tmp_path: Path) -> None:
    # Given: independent motion, compressed recording, and video-only preview consumers
    helper = helper_script(tmp_path)
    config = WebRTCPreviewConfig(
        helper_path=str(helper),
        advertised_ip="127.0.0.1",
        video_codec="copy",
        audio_enabled=False,
    )
    session = SharedMediaSession(helper_path=str(helper), preview_config=config)
    legacy_recording = LegacyRecorder()
    legacy_motion = LegacyMotion()
    recording = recorder(session, legacy_recording)
    detection = motion(session, helper, legacy_motion)
    publisher = preview(session, config)
    one_path, two_path = tmp_path / "one.mp4", tmp_path / "two.mp4"
    try:
        # When: rotation overlaps two writers, viewers detach, and motion reconnects
        detection.start("rtsp://camera/sub")
        assert detection.discard_frame(0.1)
        observation = detection.read_motion(0.1, 15.0)
        assert observation == MotionObservation(motion=True, changed_pixels=19200, changed_pct=25.0)
        one = recording.start(one_path, tmp_path / "one.log")
        two = recording.start(two_path, tmp_path / "two.log")
        assert one is not None and two is not None
        first = publisher.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        second = publisher.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(first, PreviewAnswer) and isinstance(second, PreviewAnswer)
        publisher.close(first.session_id)
        publisher.request_stop()
        detection.stop()
        assert recording.is_alive(one) and recording.is_alive(two)
        assert recording.stop(one, one_path)
        assert recording.is_alive(two)
        detection.start("rtsp://camera/main")
        assert detection.discard_frame(0.1)
        assert recording.stop(two, two_path)
        publisher.shutdown()
        detection.stop()

        # Then: the camera source owns one process; only completed files reach final paths
        records = requests(helper)
        assert {record["pid"] for record in records} == {one.pid, two.pid}
        assert "stop" not in commands(helper)
        starts = [
            record["request"]
            for record in records
            if record["request"]["command"] == "start_recording"
        ]
        assert len({start["recording_id"] for start in starts}) == 2
        assert all(start["audio_mode"] == "copy" for start in starts)
        settings = next(
            record["request"]["motion_config"]
            for record in records
            if record["request"]["command"] == "start_motion"
        )
        assert settings == {
            "pixel_threshold": 47,
            "min_changed_pct": 30.0,
            "blur_kernel": 7,
            "recording_sensitivity_factor": 2.0,
        }
        assert one_path.read_bytes() == two_path.read_bytes() == b"container pending finish"
        assert not list(tmp_path.glob("*.partial"))
        assert legacy_recording.calls == []
        assert not any(call[0] == "start" for call in legacy_motion.calls)
    finally:
        publisher.shutdown()
        detection.stop()
        session.shutdown()
    assert commands(helper)[-1] == "stop"


@pytest.mark.parametrize("mode", ["normal", "finish_failure", "exit_on_status"])
def test_source_hands_off_only_finalized_native_files(tmp_path: Path, mode: str) -> None:
    # Given: the actual source wiring with a negotiated H264/AAC profile and helper boundary
    helper = helper_script(tmp_path, mode)
    config = RTSPSourceConfig.model_validate(
        {
            "rtsp_url": "rtsp://camera/main",
            "detect_rtsp_url": "rtsp://camera/sub",
            "output_dir": str(tmp_path / "recordings"),
            "stream": {"disable_hwaccel": True},
            "__runtime_preview__": {
                "enabled": True,
                "backend": "webrtc",
                "recording_policy": "allow_during_recording",
                "config": {
                    "helper_path": str(helper),
                    "advertised_ip": "127.0.0.1",
                    "video_codec": "copy",
                    "audio_enabled": False,
                },
            },
        }
    )
    source = RTSPSource(config, camera_name="front")
    clips: list[Clip] = []
    source.register_callback(clips.append)
    source._apply_preflight_outcome(
        CameraPreflightOutcome(
            camera_key="synthetic-camera",
            motion_profile=MotionProfile(input_url="rtsp://camera/sub"),
            recording_profile=profile(),
            diagnostics=CameraPreflightDiagnostics(
                attempted_urls=["rtsp://camera/main"],
                probes=[
                    ProbeStreamInfo(
                        url="rtsp://camera/main",
                        probe_ok=True,
                        video_codec="h264",
                        audio_codec="aac",
                    )
                ],
            ),
        )
    )
    try:
        # When: preview yields its consumer while recording finishes or its owner fails
        source.start_recording()
        offer = source.negotiate_preview(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(offer, PreviewAnswer)
        source.stop_preview()
        if mode == "exit_on_status":
            assert not source.check_recording_health()
        else:
            assert source.check_recording_health()
            source.stop_recording()

        # Then: incomplete bytes stay outside replay paths and never reach the clip callback
        assert len(clips) == int(mode == "normal")
        final_files = list((tmp_path / "recordings").glob("*.mp4"))
        partial_files = list((tmp_path / "recordings").glob("*.partial"))
        assert len(final_files) == int(mode == "normal")
        assert len(partial_files) == int(mode != "normal")
        if clips:
            assert clips[0].local_path == final_files[0]
        assert len({record["pid"] for record in requests(helper)}) == 1
    finally:
        source.cleanup()


@pytest.mark.parametrize(
    ("video_codec", "audio_codec", "audio_mode", "input_args", "output_extra"),
    [
        ("hevc", "aac", "copy", [], []),
        ("h264", "ac3", "copy", [], []),
        ("h264", "pcm_mulaw", "aac", [], []),
        ("h264", "aac", "copy", ["-use_wallclock_as_timestamps", "1"], []),
        ("h264", "aac", "copy", [], ["-movflags", "+faststart"]),
    ],
)
def test_unsupported_profiles_preserve_legacy_recording(
    tmp_path: Path,
    video_codec: str,
    audio_codec: str,
    audio_mode: str,
    input_args: list[str],
    output_extra: list[str],
) -> None:
    # Given: a negotiated profile whose codec, audio, or timing behavior is not native-compatible
    helper = helper_script(tmp_path)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    selected = profile(audio_codec, audio_mode).model_copy(
        update={
            "ffmpeg_input_args": input_args,
            "ffmpeg_output_args": [
                *profile(audio_codec, audio_mode).ffmpeg_output_args,
                *output_extra,
            ],
        }
    )
    recording = recorder(session, fallback)
    recording.configure_profile(selected, video_codec=video_codec, audio_codec=audio_codec)
    output = tmp_path / "clip.mp4"
    try:
        # When: recording starts and stops through the same source-owned interface
        handle = recording.start(output, tmp_path / "clip.log")
        assert handle is not None
        result = recording.stop(handle, output)

        # Then: fallback receives the exact negotiated profile, including audio and workaround flags
        assert result is None
        assert fallback.profiles[-1] == selected
        assert fallback.calls == [("start", output), ("stop", output)]
        assert requests(helper) == []
    finally:
        session.shutdown()


@pytest.mark.parametrize(
    "mode", ["lost_start_reply", "lost_start_and_stop_replies", "reject_start"]
)
def test_uncertain_native_start_cannot_race_legacy_overwrite(tmp_path: Path, mode: str) -> None:
    # Given: a helper whose startup reply is lost or whose supported stream cannot be opened
    helper = helper_script(tmp_path, mode)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    output = tmp_path / "clip.mp4"
    try:
        # When: the recorder resolves writer ownership before choosing the compatibility path
        handle = recording.start(output, tmp_path / "clip.log")

        # Then: a completed writer is handed off, an unknown writer is retried, and refusal falls back
        assert commands(helper)[:2] == ["start_recording", "stop_recording"]
        if mode == "lost_start_reply":
            assert handle is not None
            assert recording.stop(handle, output) is True
            assert fallback.calls == []
            assert output.exists()
        elif mode == "lost_start_and_stop_replies":
            assert handle is None
            assert fallback.calls == []
            assert not output.exists()
            assert output.with_name(output.name + ".partial").exists()
            retired_pid = requests(helper)[0]["pid"]
            with pytest.raises(ProcessLookupError):
                os.kill(retired_pid, 0)
            retry_path = tmp_path / "retry.mp4"
            retry = recording.start(retry_path, tmp_path / "retry.log")
            assert isinstance(retry, LegacyHandle)
            assert fallback.calls == [("start", retry_path)]
        else:
            assert isinstance(handle, LegacyHandle)
            assert fallback.calls == [("start", output)]
    finally:
        session.shutdown()


@pytest.mark.parametrize("mode", ["publish_then_exit", "publish_during_teardown"])
def test_lost_reply_preserves_final_clip_published_before_helper_death(
    tmp_path: Path, mode: str
) -> None:
    # Given: an executable publishes its final clip but exits before startup or teardown replies
    helper = helper_script(tmp_path, mode)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    output = tmp_path / "published.mp4"
    try:
        # When: native recovery and a later policy retry encounter that same completed path
        assert recording.start(output, tmp_path / "published.log") is None
        assert recording.start(output, tmp_path / "published.log") is None

        # Then: completed bytes remain replayable and fallback starts only at a fresh filename
        assert output.read_bytes() == b"container pending finish"
        assert not output.with_name(output.name + ".partial").exists()
        assert fallback.calls == []
        assert commands(helper).count("start_recording") == 1
        fresh_output = tmp_path / "fresh.mp4"
        assert isinstance(recording.start(fresh_output, tmp_path / "fresh.log"), LegacyHandle)
        assert fallback.calls == [("start", fresh_output)]
        assert output.read_bytes() == b"container pending finish"
    finally:
        session.shutdown()


@pytest.mark.parametrize("existing_kind", ["file", "dangling_symlink"])
def test_native_path_collision_never_routes_occupied_output_to_legacy_recorder(
    tmp_path: Path, existing_kind: str
) -> None:
    # Given: native recording refuses an occupied final pathname
    helper = helper_script(tmp_path)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    output = tmp_path / "occupied.mp4"
    target = tmp_path / "absent-target.mp4"
    if existing_kind == "file":
        output.write_bytes(b"previous completed clip")
    else:
        output.symlink_to(target)
    try:
        # When: initial refusal and a pinned compatibility retry use the occupied filename
        assert recording.start(output, tmp_path / "occupied.log") is None
        assert recording.start(output, tmp_path / "occupied.log") is None

        # Then: fallback cannot truncate prior media or follow the existing directory entry
        assert fallback.calls == []
        assert commands(helper).count("start_recording") == 1
        if existing_kind == "file":
            assert output.read_bytes() == b"previous completed clip"
        else:
            assert output.is_symlink() and output.readlink() == target
            assert not target.exists()
    finally:
        session.shutdown()


def test_motion_cancellation_preserves_active_recording(tmp_path: Path) -> None:
    # Given: an active writer and a motion read waiting on the same helper process
    helper = helper_script(tmp_path, "pending_motion")
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    detection = motion(session, helper, LegacyMotion())
    try:
        detection.start("rtsp://camera/main")
        handle = recording.start(tmp_path / "clip.mp4", tmp_path / "clip.log")
        assert handle is not None
        with ThreadPoolExecutor(max_workers=1) as executor:
            read = executor.submit(detection.read_motion, 5.0, 30.0)
            deadline = time.monotonic() + 2.0
            while "read_motion" not in commands(helper):
                assert time.monotonic() < deadline
                time.sleep(0.005)

            # When: reconnect cancellation stops only the motion consumer
            detection.stop()

            # Then: the pending read ends promptly and recording retains its owner
            assert read.result(timeout=1.0) is None
            assert recording.is_alive(handle)
            assert "stop" not in commands(helper)
            assert recording.stop(handle, tmp_path / "clip.mp4")
    finally:
        detection.stop()
        session.shutdown()


@pytest.mark.parametrize("detach_first", [False, True])
def test_old_motion_teardown_cannot_stop_new_consumer(tmp_path: Path, detach_first: bool) -> None:
    # Given: the executable permits one motion consumer, active or already detached
    helper = helper_script(tmp_path)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    try:
        old = session.motion_client(lambda reply: None)
        start_motion(old)
        if detach_first:
            old.stop()
        current = session.motion_client(lambda reply: None)
        start_motion(current)

        # When: the retired generation repeats teardown after its successor starts
        old.stop()
        assert current.request("discard_frame", wait_timeout_s=0.1).frame_available
        result = current.request("read_motion", wait_timeout_s=0.1, threshold=15.0)

        # Then: the new lifetime remains usable and all motion commands name their exact owner
        assert result.observation == MotionObservation(
            motion=True, changed_pixels=19200, changed_pct=25.0
        )
        motion_requests = [
            record["request"]
            for record in requests(helper)
            if record["request"]["command"]
            in {"start_motion", "read_motion", "discard_frame", "stop_motion"}
        ]
        assert [request["command"] for request in motion_requests] == [
            "start_motion",
            "stop_motion",
            "start_motion",
            "discard_frame",
            "read_motion",
        ]
        old_id, new_id = motion_requests[0]["motion_id"], motion_requests[2]["motion_id"]
        assert old_id != new_id
        assert motion_requests[1]["motion_id"] == old_id
        assert all(request["motion_id"] == new_id for request in motion_requests[2:])
        assert len({record["pid"] for record in requests(helper)}) == 1
        current.stop()
    finally:
        session.shutdown()


def test_delayed_old_motion_read_cannot_advance_successor_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a consumed-frame read paused before executable dispatch, with recording active
    helper = helper_script(tmp_path, "motion_generations")
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    legacy_motion = LegacyMotion()
    detection = motion(session, helper, legacy_motion)
    recording = recorder(session, LegacyRecorder())
    paused, release = Event(), Event()
    original_request = HelperClient.request

    def delayed_request(
        client: HelperClient[_SharedRequest, _SharedReply],
        command: str,
        *,
        timeout_s: float = 2.0,
        **fields: object,
    ) -> _SharedReply:
        if command == "read_motion" and not paused.is_set():
            paused.set()
            if not release.wait(timeout=3.0):
                raise HelperError("Synthetic dispatch pause timed out")
        return original_request(client, command, timeout_s=timeout_s, **fields)

    monkeypatch.setattr(HelperClient, "request", delayed_request)
    try:
        detection.start("rtsp://camera/main")
        handle = recording.start(tmp_path / "clip.mp4", tmp_path / "clip.log")
        assert handle is not None
        with ThreadPoolExecutor(max_workers=2) as executor:
            old_read = executor.submit(detection.read_motion, 0.1, 15.0)
            assert paused.wait(timeout=1.0)

            # When: stop/start replaces motion without waiting for the paused read or detector
            replacement = executor.submit(detection.start, "rtsp://camera/main")
            try:
                replacement.result(timeout=1.0)
                assert recording.is_alive(handle)
            finally:
                release.set()
            assert old_read.result(timeout=1.0) is None
        fresh = detection.read_motion(0.1, 15.0)
        next_frame = detection.read_motion(0.1, 15.0)

        # Then: native ownership refuses the late read, leaving the successor baseline untouched
        assert fresh == MotionObservation(motion=False, changed_pixels=0, changed_pct=0.0)
        assert next_frame == MotionObservation(motion=True, changed_pixels=19200, changed_pct=25.0)
        motion_requests = [
            record["request"]
            for record in requests(helper)
            if record["request"]["command"] in {"start_motion", "read_motion", "stop_motion"}
        ]
        assert [request["command"] for request in motion_requests] == [
            "start_motion",
            "stop_motion",
            "start_motion",
            "read_motion",
            "read_motion",
            "read_motion",
        ]
        old_id, new_id = motion_requests[0]["motion_id"], motion_requests[2]["motion_id"]
        assert old_id != new_id
        assert motion_requests[1]["motion_id"] == motion_requests[3]["motion_id"] == old_id
        assert all(request["motion_id"] == new_id for request in motion_requests[4:])
        assert not any(call[0] == "start" for call in legacy_motion.calls)
        assert "stop" not in commands(helper)
        assert recording.stop(handle, tmp_path / "clip.mp4")
    finally:
        release.set()
        detection.stop()
        session.shutdown()


@pytest.mark.parametrize("mode", ["motion_stop_failure", "lost_motion_stop_reply"])
def test_failed_motion_detach_blocks_replacement_and_preserves_recording(
    tmp_path: Path, mode: str
) -> None:
    # Given: a healthy writer and motion consumer whose detach cannot be confirmed
    helper = helper_script(tmp_path, mode)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    recording = recorder(session, LegacyRecorder())
    output = tmp_path / "clip.mp4"
    try:
        handle = recording.start(output, tmp_path / "clip.log")
        assert handle is not None
        old = session.motion_client(lambda reply: None)
        start_motion(old)

        # When: a failed detach is followed by a request for a successor consumer
        with pytest.raises(HelperError):
            old.stop()
        with pytest.raises(HelperError):
            session.motion_client(lambda reply: None)

        # Then: old ownership stays retained while recording continues in its healthy process
        assert commands(helper).count("start_motion") == 1
        assert "stop" not in commands(helper)
        assert recording.is_alive(handle)
        assert recording.stop(handle, output)
        os.kill(handle.pid, 0)
    finally:
        session.shutdown()


@pytest.mark.parametrize("detach_first", [False, True])
def test_old_preview_teardown_cannot_stop_new_consumer(tmp_path: Path, detach_first: bool) -> None:
    # Given: native input permits one preview, with the old consumer active or already detached
    helper = helper_script(tmp_path)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    try:
        old = session.preview_client([str(helper)])
        assert old.request("start", rtsp_url="rtsp://camera/main").ok
        if detach_first:
            old.stop()
        current = session.preview_client([str(helper)])
        assert current.request("start", rtsp_url="rtsp://camera/main").ok

        # When: the old generation attempts teardown again
        old.stop()

        # Then: the current media remains active and the helper process is retained
        assert current.request("status").media_active
        assert commands(helper).count("stop_preview") == 1
        current.stop()
    finally:
        session.shutdown()


def test_delayed_old_offer_cannot_attach_viewer_to_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: an authorized offer paused at the control-transport boundary before executable dispatch
    helper = helper_script(tmp_path)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    recording = recorder(session, LegacyRecorder())
    paused, release, replacement_entered = Event(), Event(), Event()
    original_request = HelperClient.request

    def delayed_request(
        client: HelperClient[_SharedRequest, _SharedReply],
        command: str,
        *,
        timeout_s: float = 2.0,
        **fields: object,
    ) -> _SharedReply:
        if command == "offer" and fields.get("session_id") == "old-viewer":
            paused.set()
            if not release.wait(timeout=3.0):
                raise HelperError("Synthetic dispatch pause timed out")
        return original_request(client, command, timeout_s=timeout_s, **fields)

    monkeypatch.setattr(HelperClient, "request", delayed_request)
    try:
        handle = recording.start(tmp_path / "clip.mp4", tmp_path / "clip.log")
        assert handle is not None
        old = session.preview_client([str(helper)])
        assert old.request("start", rtsp_url="rtsp://camera/main").ok
        with ThreadPoolExecutor(max_workers=2) as executor:
            offer = executor.submit(
                old.request,
                "offer",
                session_id="old-viewer",
                sdp="v=0",
                lease_seconds=10.0,
            )
            assert paused.wait(timeout=1.0)

            def replace_preview() -> _PreviewHelper:
                replacement_entered.set()
                current = session.preview_client([str(helper)])
                assert current.request("start", rtsp_url="rtsp://camera/main").ok
                return current

            # When: preview replacement races the delayed offer while recording continues
            replacement = executor.submit(replace_preview)
            try:
                assert replacement_entered.wait(timeout=1.0)
                with pytest.raises(TimeoutError):
                    replacement.result(timeout=0.1)
                assert recording.is_alive(handle)
            finally:
                release.set()
            assert offer.result(timeout=1.0).ok
            current = replacement.result(timeout=1.0)

            # Then: replacement clears the old peer and no unsolicited viewer leaks into its lifetime
            assert current.request("status").active_session_count == 0
            assert commands(helper) == [
                "start_recording",
                "start",
                "recording_status",
                "offer",
                "stop_preview",
                "start",
                "status",
            ]
            assert recording.stop(handle, tmp_path / "clip.mp4")
            current.stop()
    finally:
        release.set()
        session.shutdown()


def test_native_process_death_never_marks_partial_file_complete(tmp_path: Path) -> None:
    # Given: a writer has only created its excluded partial path
    helper = helper_script(tmp_path)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    recording = recorder(session, LegacyRecorder())
    output = tmp_path / "clip.mp4"
    try:
        handle = recording.start(output, tmp_path / "clip.log")
        assert handle is not None

        # When: its owned helper dies before trailer, synchronization, and publication
        os.kill(handle.pid, signal.SIGKILL)
        deadline = time.monotonic() + 2.0
        while handle.returncode is None:
            assert time.monotonic() < deadline
            time.sleep(0.005)

        # Then: the source sees failure and cannot hand the incomplete file to its pipeline
        assert not recording.is_alive(handle)
        assert recording.stop(handle, output) is False
        assert not output.exists()
        assert output.with_name(output.name + ".partial").exists()
    finally:
        session.shutdown()


@pytest.mark.parametrize("mode", ["reject_start", "finish_failure", "exit_on_status"])
def test_native_failure_pins_compatible_recording_for_unchanged_profile(
    tmp_path: Path, mode: str
) -> None:
    # Given: a native startup, finalization, or runtime failure under a negotiated profile
    helper = helper_script(tmp_path, mode)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    first_output = tmp_path / "one.mp4"
    try:
        first = recording.start(first_output, tmp_path / "one.log")
        assert first is not None
        if mode == "finish_failure":
            assert recording.stop(first, first_output) is False
        elif mode == "exit_on_status":
            assert not recording.is_alive(first)
            assert recording.stop(first, first_output) is False
        native_start_count = commands(helper).count("start_recording")

        # When: policy retries with the same profile, including reapplying discovery metadata
        recording.configure_profile(profile(), video_codec="h264", audio_codec="aac")
        retry_output = tmp_path / "two.mp4"
        retry = recording.start(retry_output, tmp_path / "two.log")

        # Then: recording proceeds through the compatible path rather than losing every native clip
        assert isinstance(retry, LegacyHandle)
        assert fallback.calls[-1] == ("start", retry_output)
        assert commands(helper).count("start_recording") == native_start_count
        if mode == "finish_failure":
            assert "stop" not in commands(helper)
            os.kill(first.pid, 0)
    finally:
        session.shutdown()


def test_negotiated_video_only_profile_stays_video_only_in_native_recorder(tmp_path: Path) -> None:
    # Given: preflight selected video-only recording rather than an audio conversion
    helper = helper_script(tmp_path)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    selected = profile(None, "none")
    recording.configure_profile(selected, video_codec="h264", audio_codec=None)
    output = tmp_path / "clip.mp4"
    try:
        # When: the source requests recording with the selected profile
        handle = recording.start(output, tmp_path / "clip.log")
        assert handle is not None
        assert recording.stop(handle, output)

        # Then: audio is omitted explicitly and no legacy audio policy is silently changed
        start = next(
            record["request"]
            for record in requests(helper)
            if record["request"]["command"] == "start_recording"
        )
        assert start["audio_mode"] == "none"
        assert fallback.calls == []
        assert output.exists()
    finally:
        session.shutdown()


@pytest.mark.parametrize("mode", ["lost_stop_reply", "missing_stop_ownership"])
def test_unknown_writer_stop_retires_owner_before_rotation_fallback(
    tmp_path: Path, mode: str
) -> None:
    # Given: old and new rotation writers share a process that cannot confirm writer teardown
    helper = helper_script(tmp_path, mode)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    old_output, new_output = tmp_path / "old.mp4", tmp_path / "new.mp4"
    try:
        old = recording.start(old_output, tmp_path / "old.log")
        new = recording.start(new_output, tmp_path / "new.log")
        assert old is not None and new is not None
        assert old.pid == new.pid

        # When: stopping one writer loses the confirmation and policy must recover recording
        assert recording.stop(new, new_output) is False
        retry_output = tmp_path / "retry.mp4"
        retry = recording.start(retry_output, tmp_path / "retry.log")

        # Then: private-owner death is confirmed before fallback, including its overlapping writer
        with pytest.raises(ProcessLookupError):
            os.kill(old.pid, 0)
        assert not recording.is_alive(old)
        assert recording.stop(old, old_output) is False
        assert isinstance(retry, LegacyHandle)
        assert fallback.calls == [("start", retry_output)]
        assert commands(helper).count("start_recording") == 2
        assert not old_output.exists() and not new_output.exists()
    finally:
        session.shutdown()


def test_unconfirmed_owner_death_retains_recording_id_and_blocks_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: the operating-system boundary cannot retire an unresponsive owned helper
    helper = helper_script(tmp_path, "unretirable")
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    fallback = LegacyRecorder()
    recording = recorder(session, fallback)
    original_killpg = os.killpg

    def refuse_owned_group(group: int, requested_signal: int) -> None:
        records = requests(helper)
        if records and group == records[0]["pid"]:
            raise PermissionError("synthetic process supervisor failure")
        original_killpg(group, requested_signal)

    monkeypatch.setattr("homesec.sources.rtsp.helper_client.os.killpg", refuse_owned_group)
    try:
        # When: both native startup and teardown lose their replies, followed by a policy retry
        assert recording.start(tmp_path / "one.mp4", tmp_path / "one.log") is None
        assert recording.start(tmp_path / "two.mp4", tmp_path / "two.log") is None

        # Then: the original writer remains owned and no second recorder or writer starts
        assert fallback.calls == []
        assert commands(helper).count("start_recording") == 1
        owned_pid = requests(helper)[0]["pid"]
        os.kill(owned_pid, 0)

        # When: the process supervisor can confirm teardown again
        monkeypatch.undo()
        retry_output = tmp_path / "three.mp4"
        recovered = recording.start(retry_output, tmp_path / "three.log")

        # Then: fallback begins only after that same retained owner has died
        assert isinstance(recovered, LegacyHandle)
        assert fallback.calls == [("start", retry_output)]
        with pytest.raises(ProcessLookupError):
            os.kill(owned_pid, 0)
    finally:
        monkeypatch.undo()
        session.shutdown()


@pytest.mark.parametrize("mode", ["preview_stop_failure", "lost_preview_stop_reply"])
def test_failed_preview_detach_blocks_replacement_and_preserves_recording(
    tmp_path: Path, mode: str
) -> None:
    # Given: a healthy writer and native preview that will refuse or lose its detach confirmation
    helper = helper_script(tmp_path, mode)
    session = SharedMediaSession(helper_path=str(helper), preview_config=None)
    recording = recorder(session, LegacyRecorder())
    output = tmp_path / "clip.mp4"
    try:
        handle = recording.start(output, tmp_path / "clip.log")
        assert handle is not None
        old = session.preview_client([str(helper)])
        assert old.request("start", rtsp_url="rtsp://camera/main").ok

        # When: the source detaches preview and a subsequent activation requests a successor
        with pytest.raises(HelperError):
            old.stop()
        with pytest.raises(HelperError):
            session.preview_client([str(helper)])

        # Then: unknown preview ownership is retained without killing or interrupting recording
        assert commands(helper).count("start") == 1
        assert "stop" not in commands(helper)
        assert recording.is_alive(handle)
        assert recording.stop(handle, output)
        os.kill(handle.pid, 0)
    finally:
        session.shutdown()
