"""Behavioral supervision tests against a real helper process protocol boundary."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pytest

from homesec.models.config import WebRTCPreviewConfig
from homesec.models.preview import (
    PreviewAnswer,
    PreviewOffer,
    PreviewSessionAction,
    PreviewSessionRefusal,
    PreviewSessionRefusalReason,
)
from homesec.sources.rtsp.capabilities import RTSPTimeoutCapabilities
from homesec.sources.rtsp.live_publisher import (
    LivePublisherRefusalReason,
    LivePublisherStartRefusal,
    LivePublisherState,
)
from homesec.sources.rtsp.webrtc_publisher import (
    RustWebRTCLivePublisher,
    _HelperClient,
    _HelperError,
)


@pytest.fixture
def helper(tmp_path: Path) -> tuple[Path, Path]:
    script = tmp_path / "helper"
    observations = tmp_path / "commands.jsonl"
    script.write_text(
        f"#!{sys.executable}\n"
        + """import json,sys,time,os
peers = {}
def reply(payload):
    print(json.dumps(payload), flush=True)
reply({'event':'ready','video_port':20001,'audio_port':20003,'media_port':8189})
for raw in sys.stdin:
    command = json.loads(raw)
    now = time.monotonic()
    peers = {k:v for k,v in peers.items() if v > now}
    with open("""
        + repr(str(observations))
        + """, 'a') as stream:
        stream.write(json.dumps(command) + '\\n')
    result = {'request_id':command['request_id'],'ok':True,'viewer_count':len(peers),'media_active':True}
    if command['command'] == 'offer':
        if command['sdp'] == 'reject':
            result.update(ok=False,error_code='invalid_offer')
        elif command['sdp'] == 'disconnect':
            sys.exit(2)
        else:
            peers[command['session_id']] = now + command['lease_seconds']
            result['sdp'] = 'v=0\\r\\ns=answer\\r\\n'
    elif command['command'] == 'renew':
        if command['session_id'] not in peers:
            result.update(ok=False,error_code='session_not_found')
        else:
            peers[command['session_id']] = now + command['lease_seconds']
    elif command['command'] == 'close':
        peers.pop(command['session_id'],None)
    result['viewer_count'] = len(peers)
    result['active_session_count'] = len(peers)
    reply(result)
"""
    )
    script.chmod(0o700)
    return script, observations


def publisher(
    helper: tuple[Path, Path],
    *,
    idle: float = 5.0,
    concurrent: bool = False,
    negotiation_timeout_s: float = 0.2,
    video_codec: Literal["h264", "copy"] = "h264",
    audio_enabled: bool = True,
) -> RustWebRTCLivePublisher:
    return RustWebRTCLivePublisher(
        camera_name="front",
        rtsp_url="rtsp://user:private-password@camera/main",
        config=WebRTCPreviewConfig(
            helper_path=str(helper[0]),
            advertised_ip="127.0.0.1",
            negotiation_timeout_s=negotiation_timeout_s,
            video_codec=video_codec,
            audio_enabled=audio_enabled,
        ),
        idle_timeout_s=idle,
        recording_policy="allow_during_recording" if concurrent else "stop_on_recording",
        rtsp_connect_timeout_s=0,
        rtsp_io_timeout_s=0,
        timeout_capabilities=RTSPTimeoutCapabilities(),
    )


def wait_until(predicate: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 4
    while not predicate():
        assert time.monotonic() < deadline, "Publisher did not reach expected observable state"
        time.sleep(0.02)


def test_viewers_share_one_media_reader_and_detach_independently(helper: tuple[Path, Path]) -> None:
    # Given: A real helper process supervising one camera publisher
    preview = publisher(helper)
    try:
        # When: Two viewers attach and the first detaches
        one = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        two = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(one, PreviewAnswer) and isinstance(two, PreviewAnswer)
        preview.close(one.session_id)
        wait_until(lambda: preview.status().viewer_count == 1)
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]

        # Then: One input serves both peers, with the second peer still renewable
        assert [command["command"] for command in commands].count("start") == 1
        renewed = preview.renew(two.session_id, time.time() + 10)
        assert isinstance(renewed, PreviewSessionAction) and renewed.accepted
        assert preview.status().state == LivePublisherState.READY
        args = next(command["ffmpeg_args"] for command in commands if command["command"] == "start")
        assert args[args.index("-profile:v") + 1] == "baseline"
        assert args[args.index("-bf") + 1] == "0"
        # The input decoder must not queue future frames across decode threads.
        input_args = args[: args.index("-i")]
        assert input_args[input_args.index("-thread_type:v") + 1] == "slice"
        assert "libopus" not in args  # No empty audio output for video-only cameras.
    finally:
        preview.shutdown()


def test_audio_rtp_transcodes_when_startup_discovered_audio(helper: tuple[Path, Path]) -> None:
    # Given: The source's existing startup discovery identified camera audio
    preview = publisher(helper)
    preview.set_audio_available(True)
    try:
        # When: Preview starts
        preview.ensure_active()
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]

        # Then: One FFmpeg input sends browser-compatible H.264 and Opus loopback RTP
        args = next(command["ffmpeg_args"] for command in commands if command["command"] == "start")
        assert args.count("-i") == 1
        assert "libopus" in args
        assert "rtp://127.0.0.1:20001?pkt_size=1200" in args
        assert "rtp://127.0.0.1:20003?pkt_size=1200" in args
    finally:
        preview.shutdown()


def test_video_only_copy_starts_native_rtsp_over_helper_stdin(helper: tuple[Path, Path]) -> None:
    # Given: Audio is explicitly disabled even though the camera has an audio track.
    helper[0].write_text(
        helper[0]
        .read_text()
        .replace(
            "peers = {}", "assert not any('rtsp://' in arg for arg in sys.argv)\npeers = {}", 1
        )
    )
    preview = publisher(helper, video_codec="copy", audio_enabled=False)
    preview.set_audio_available(True)
    try:
        # When: Two viewers attach to the same camera.
        first = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        second = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        starts = [command for command in commands if command["command"] == "start"]

        # Then: One native input receives its URL privately, without an FFmpeg command.
        assert isinstance(first, PreviewAnswer)
        assert isinstance(second, PreviewAnswer)
        assert len(starts) == 1
        assert starts[0]["rtsp_url"] == "rtsp://user:private-password@camera/main"
        assert "ffmpeg_args" not in starts[0]
        assert preview.status().state == LivePublisherState.READY
    finally:
        preview.shutdown()


def test_native_rtsp_start_waits_for_camera_negotiation(helper: tuple[Path, Path]) -> None:
    # Given: Native camera startup needs longer than the ordinary control request deadline.
    helper[0].write_text(
        helper[0]
        .read_text()
        .replace(
            "    if command['command'] == 'offer':",
            "    if command['command'] == 'start':\n"
            "        time.sleep(2.2)\n"
            "    elif command['command'] == 'offer':",
            1,
        )
    )
    preview = publisher(helper, video_codec="copy", audio_enabled=False)
    try:
        # When: A viewer starts preview while the helper negotiates with the camera.
        result = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 15)

        # Then: The dedicated startup deadline allows negotiation to finish successfully.
        assert isinstance(result, PreviewAnswer)
        assert preview.status().state == LivePublisherState.READY
    finally:
        preview.shutdown()


@pytest.mark.parametrize(
    ("video_codec", "audio_enabled", "audio_available"),
    [
        ("copy", True, False),
        ("copy", True, True),
        ("h264", False, True),
        ("h264", True, False),
        ("h264", True, True),
    ],
)
def test_transcoding_or_enabled_audio_keeps_ffmpeg_input(
    helper: tuple[Path, Path],
    video_codec: Literal["h264", "copy"],
    audio_enabled: bool,
    audio_available: bool,
) -> None:
    # Given: The requested mode requires transcoding or has audio enabled in config.
    preview = publisher(helper, video_codec=video_codec, audio_enabled=audio_enabled)
    preview.set_audio_available(audio_available)
    try:
        # When: Preview starts using the camera discovery result.
        result = preview.ensure_active()
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        start = next(command for command in commands if command["command"] == "start")

        # Then: Existing FFmpeg video/audio selection is preserved without native ingest.
        assert not isinstance(result, LivePublisherStartRefusal)
        assert "rtsp_url" not in start
        args = start["ffmpeg_args"]
        assert args.count("-i") == 1
        assert args[args.index("-c:v") + 1] == ("copy" if video_codec == "copy" else "libx264")
        assert ("libopus" in args) == (audio_enabled and audio_available)
    finally:
        preview.shutdown()


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"ffmpeg_args": []},
        {"rtsp_url": ""},
        {"ffmpeg_args": ["-i", "input"], "rtsp_url": "rtsp://camera/main"},
        {"ffmpeg_args": [], "rtsp_url": "rtsp://camera/main"},
    ],
)
def test_start_rejects_missing_or_ambiguous_input_before_sending(
    helper: tuple[Path, Path], fields: dict[str, object]
) -> None:
    # Given: An available helper and a malformed media input selection.
    client = _HelperClient([str(helper[0])])
    try:
        client.wait_ready(timeout_s=2)

        # When: The client rejects startup, then sends a valid status request.
        with pytest.raises(_HelperError, match="Invalid preview helper command"):
            client.request("start", **fields)
        response = client.request("status")
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]

        # Then: Invalid input never reaches the process and the connection stays usable.
        assert response.ok
        assert [command["command"] for command in commands] == ["status"]
    finally:
        client.stop()


def test_recording_stops_all_peers_and_refuses_new_preview(helper: tuple[Path, Path]) -> None:
    # Given: A ready preview with an attached viewer
    preview = publisher(helper)
    try:
        assert isinstance(
            preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10), PreviewAnswer
        )

        # When: Recording becomes active
        preview.sync_recording_active(True)
        refusal = preview.ensure_active()

        # Then: Preview yields and cannot open another camera reader during recording
        assert preview.status().state == LivePublisherState.IDLE
        assert isinstance(refusal, LivePublisherStartRefusal)
        assert refusal.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
        preview.sync_recording_active(False)
        assert not isinstance(preview.ensure_active(), LivePublisherStartRefusal)
    finally:
        preview.shutdown()


def test_http_activity_cannot_keep_lease_or_media_alive(helper: tuple[Path, Path]) -> None:
    # Given: A short peer lease and source idle timeout
    preview = publisher(helper, idle=0.1)
    try:
        preview.ensure_active()
        assert isinstance(
            preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 0.05), PreviewAnswer
        )

        # When: The peer expires while HTTP viewer bookkeeping continues
        preview.note_viewer_activity("polling-browser")
        wait_until(lambda: preview.status().state == LivePublisherState.IDLE)

        # Then: The camera reader stops without any API polling dependence
        assert preview.status().viewer_count == 0
    finally:
        preview.shutdown()


def test_pending_negotiation_survives_idle_timeout_then_expiry_releases_input(
    helper: tuple[Path, Path],
) -> None:
    # Given: A helper distinguishes negotiating sessions from connected viewers.
    script = (
        helper[0]
        .read_text()
        .replace(
            "peers = {}",
            "peers = {}\n"
            "negotiation_timeout = float(sys.argv[sys.argv.index('--negotiation-timeout-s')+1])",
            1,
        )
        .replace(
            "now + command['lease_seconds']",
            "now + min(command['lease_seconds'], negotiation_timeout)",
        )
    )
    script = script.replace("result['viewer_count'] = len(peers)", "result['viewer_count'] = 0")
    helper[0].write_text(script)
    preview = publisher(helper, idle=0.1, negotiation_timeout_s=1.8)
    try:
        answer = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(answer, PreviewAnswer)

        # When: Multiple maintenance polls observe a pending peer after the idle deadline.
        def status_polls() -> int:
            return sum(
                json.loads(raw)["command"] == "status" for raw in helper[1].read_text().splitlines()
            )

        wait_until(lambda: status_polls() >= 2)

        # Then: Negotiation keeps the input alive without counting as a connected viewer.
        assert preview.status().state == LivePublisherState.READY
        assert preview.status().viewer_count == 0
        assert preview.status().idle_shutdown_at is None
        wait_until(lambda: preview.status().state == LivePublisherState.IDLE)
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        assert [command["command"] for command in commands].count("start") == 1
    finally:
        preview.shutdown()


def test_accepted_offer_outlives_stale_zero_session_status(helper: tuple[Path, Path]) -> None:
    # Given: A helper delays its no-session status until a new offer has been accepted.
    script = helper[0].read_text().replace("peers = {}", "peers = {}\ndeferred_status = None", 1)
    script = script.replace(
        "    reply(result)",
        "    if command['command'] == 'status' and not peers and deferred_status is None:\n"
        "        deferred_status = result\n"
        "        continue\n"
        "    reply(result)\n"
        "    if command['command'] == 'offer' and deferred_status is not None:\n"
        "        time.sleep(0.3)\n"
        "        reply(deferred_status)\n"
        "        deferred_status = None",
        1,
    )
    helper[0].write_text(script)
    preview = publisher(helper, idle=0.1)
    try:
        preview.ensure_active()

        def status_polls() -> int:
            return sum(
                json.loads(raw)["command"] == "status" for raw in helper[1].read_text().splitlines()
            )

        wait_until(lambda: status_polls() == 1)

        # When: An offer succeeds, then the older zero-session status arrives after the idle delay.
        answer = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(answer, PreviewAnswer)
        wait_until(lambda: status_polls() >= 2)

        # Then: The new viewer and original shared input survive the stale idle observation.
        assert preview.status().state == LivePublisherState.READY
        renewed = preview.renew(answer.session_id, time.time() + 10)
        assert isinstance(renewed, PreviewSessionAction) and renewed.accepted
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        assert [command["command"] for command in commands].count("start") == 1
    finally:
        preview.shutdown()


def test_failed_offer_returns_stable_reason_without_transport_details(
    helper: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    # Given: A helper that rejects the browser offer
    preview = publisher(helper)
    try:
        # When: Negotiation fails at the protocol boundary
        refusal = preview.negotiate(PreviewOffer(sdp="reject"), time.time() + 10)

        # Then: Only the refusal reason crosses the boundary; no camera credentials are logged
        assert isinstance(refusal, PreviewSessionRefusal)
        assert refusal.reason.value == "invalid_offer"
        assert "private-password" not in caplog.text
        assert "rtsp://" not in caplog.text
    finally:
        preview.shutdown()


def test_helper_disconnect_is_cleaned_up_without_breaking_source(helper: tuple[Path, Path]) -> None:
    # Given: A preview whose helper exits while answering an offer
    preview = publisher(helper)
    try:
        # When: The helper crashes
        result = preview.negotiate(PreviewOffer(sdp="disconnect"), time.time() + 10)
        wait_until(lambda: preview.status().state == LivePublisherState.ERROR)

        # Then: Negotiation fails safely and a later explicit start recovers
        assert isinstance(result, PreviewSessionRefusal)
        assert not isinstance(preview.ensure_active(), LivePublisherStartRefusal)
    finally:
        preview.shutdown()


@pytest.mark.parametrize("failure", ["missing_executable", "rejected_start"])
@pytest.mark.parametrize("native", [False, True])
def test_failed_helper_startup_is_redacted_and_later_activation_recovers(
    helper: tuple[Path, Path], failure: str, native: bool, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: The helper is missing or refuses media startup.
    original = helper[0].read_text()
    if failure == "missing_executable":
        helper[0].unlink()
    else:
        helper[0].write_text(
            original.replace(
                "    if command['command'] == 'offer':",
                "    if command['command'] == 'start':\n"
                "        result.update(ok=False)\n"
                "    elif command['command'] == 'offer':",
                1,
            )
        )
    preview = publisher(helper, video_codec="copy" if native else "h264", audio_enabled=not native)
    try:
        # When: A viewer attempts to attach, then the executable is repaired.
        failed = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(failed, PreviewSessionRefusal)
        assert failed.reason == PreviewSessionRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
        assert preview.status().state == LivePublisherState.ERROR
        helper[0].write_text(original)
        helper[0].chmod(0o700)
        recovered = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)

        # Then: Failure stays local, credentials stay out of logs, and a new helper succeeds.
        assert isinstance(recovered, PreviewAnswer)
        assert "private-password" not in caplog.text
        assert "rtsp://" not in caplog.text
    finally:
        preview.shutdown()


def test_authorization_expiring_during_startup_never_creates_peer(
    helper: tuple[Path, Path],
) -> None:
    # Given: Helper startup takes longer than the caller's remaining authorization.
    helper[0].write_text(
        helper[0]
        .read_text()
        .replace("reply({'event':'ready'", "time.sleep(0.2)\nreply({'event':'ready'", 1)
    )
    preview = publisher(helper, idle=0.1)
    try:
        # When: A queued viewer reaches signaling after startup completes.
        result = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 0.05)

        # Then: Media startup emits no unauthorized offer and unused input is cleaned up.
        assert isinstance(result, PreviewSessionRefusal)
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        operations = [command["command"] for command in commands]
        assert operations.count("start") == 1
        assert "offer" not in operations
        wait_until(lambda: preview.status().state == LivePublisherState.IDLE)
    finally:
        preview.shutdown()


def test_missing_answer_closes_peer_and_renewal_cannot_restore_expired_authorization(
    helper: tuple[Path, Path],
) -> None:
    # Given: A helper accepts a peer but omits the required SDP answer.
    helper[0].write_text(
        helper[0]
        .read_text()
        .replace("            result['sdp'] = 'v=0\\r\\ns=answer\\r\\n'", "            pass", 1)
    )
    preview = publisher(helper)
    try:
        # When: Negotiation fails, the closed peer renews, and expired authorization retries.
        result = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(result, PreviewSessionRefusal)
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        offer = next(command for command in commands if command["command"] == "offer")
        closed = next(command for command in commands if command["command"] == "close")
        assert closed["session_id"] == offer["session_id"]
        unknown = preview.renew(offer["session_id"], time.time() + 10)
        expired = preview.renew(offer["session_id"], time.time() - 1)
        preview.request_stop()
        first_close = preview.close(offer["session_id"])
        repeated_close = preview.close(offer["session_id"])

        # Then: Cleanup is idempotent and neither unknown peers nor expired leases revive media.
        assert isinstance(unknown, PreviewSessionRefusal)
        assert unknown.reason == PreviewSessionRefusalReason.SESSION_NOT_FOUND
        assert isinstance(expired, PreviewSessionRefusal)
        assert expired.reason == PreviewSessionRefusalReason.PREVIEW_TEMPORARILY_UNAVAILABLE
        assert first_close == repeated_close == PreviewSessionAction(accepted=True)
        assert preview.status().state == LivePublisherState.IDLE
    finally:
        preview.shutdown()


@pytest.mark.parametrize("failure", ["media_inactive", "helper_exited"])
def test_maintenance_releases_peers_when_media_or_helper_fails(
    helper: tuple[Path, Path], failure: str
) -> None:
    # Given: A helper loses its input or exits while a viewer is attached.
    action = (
        "        result.update(media_active=False,state='error')"
        if failure == "media_inactive"
        else "        sys.exit(2)"
    )
    helper[0].write_text(
        helper[0]
        .read_text()
        .replace(
            "    reply(result)",
            "    if command['command'] == 'status':\n" + action + "\n    reply(result)",
            1,
        )
    )
    preview = publisher(helper)
    try:
        assert isinstance(
            preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10), PreviewAnswer
        )

        # When: Maintenance observes the failed media boundary.
        wait_until(lambda: preview.status().state == LivePublisherState.ERROR)

        # Then: Peers are released within the bounded poll loop and errors remain stable.
        assert preview.status().viewer_count == 0
        assert preview.status().last_error in {
            "WebRTC preview media is unavailable",
            "WebRTC preview helper exited",
        }
    finally:
        preview.shutdown()


def test_concurrent_preview_downgrade_preempts_recording_viewers(helper: tuple[Path, Path]) -> None:
    # Given: Concurrent preview is allowed and an active recording has a viewer.
    preview = publisher(helper, concurrent=True)
    try:
        preview.sync_recording_active(True)
        assert isinstance(
            preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10), PreviewAnswer
        )

        # When: Camera session discovery downgrades concurrent preview support.
        preview.downgrade_concurrent_preview("camera-session-limit")
        refused = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)

        # Then: Recording keeps priority and preview recovers in degraded mode after recording.
        assert preview.status().state == LivePublisherState.IDLE
        assert isinstance(refused, PreviewSessionRefusal)
        assert refused.reason == PreviewSessionRefusalReason.RECORDING_PRIORITY
        preview.sync_recording_active(False)
        recovered = preview.ensure_active()
        assert not isinstance(recovered, LivePublisherStartRefusal)
        assert recovered.state == LivePublisherState.DEGRADED
        assert recovered.degraded_reason == "camera-session-limit"
    finally:
        preview.shutdown()


def test_source_shutdown_preempts_slow_helper_startup(
    helper: tuple[Path, Path], tmp_path: Path
) -> None:
    # Given: A helper which is alive but has not emitted its ready event
    marker = tmp_path / "spawned"
    helper[0].write_text(
        helper[0]
        .read_text()
        .replace(
            "reply({'event':'ready'",
            f"open({str(marker)!r},'w').write(str(os.getpid()))\ntime.sleep(10)\nreply({{'event':'ready'",
            1,
        )
    )
    preview = publisher(helper)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            activation = executor.submit(preview.ensure_active)
            wait_until(marker.exists)

            # When: Recording takes priority during startup
            began = time.monotonic()
            preview.sync_recording_active(True)
            result = activation.result(timeout=3)

            # Then: Startup is cancelled promptly with a stable recording refusal
            assert time.monotonic() - began < 3
            assert isinstance(result, LivePublisherStartRefusal)
            assert result.reason == LivePublisherRefusalReason.RECORDING_PRIORITY
            assert preview.status().state == LivePublisherState.IDLE
    finally:
        preview.shutdown()


def test_starting_media_gets_bounded_warmup_grace(helper: tuple[Path, Path]) -> None:
    # Given: A helper awaiting its first complete frame before it reports active media
    script = helper[0].read_text().replace("peers = {}", "peers = {}\nborn = time.monotonic()", 1)
    script = script.replace(
        "    reply(result)",
        "    if command['command'] == 'status' and time.monotonic()-born < 1.2:\n        result.update(state='starting',media_active=False)\n    reply(result)",
    )
    helper[0].write_text(script)
    preview = publisher(helper)
    try:
        # When: Input is initially starting, then becomes healthy
        preview.ensure_active()
        wait_until(lambda: preview.status().state == LivePublisherState.STARTING)
        wait_until(lambda: preview.status().state == LivePublisherState.READY)

        # Then: Startup grace preserves the single helper and input instead of killing them
        commands = [json.loads(raw) for raw in helper[1].read_text().splitlines()]
        assert [command["command"] for command in commands].count("start") == 1
    finally:
        preview.shutdown()


def test_expired_authorization_does_not_start_camera_input(helper: tuple[Path, Path]) -> None:
    # Given: A source with no active input and expired viewer authorization
    preview = publisher(helper)
    try:
        # When: A queued offer reaches the source after its authorization expires
        result = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() - 1)

        # Then: No helper or media input starts
        assert isinstance(result, PreviewSessionRefusal)
        assert not helper[1].exists()
        assert preview.status().state == LivePublisherState.IDLE
    finally:
        preview.shutdown()


def test_frozen_helper_requests_and_shutdown_remain_bounded(tmp_path: Path) -> None:
    # Given: A real helper that announces ready and then never reads its command pipe
    helper = tmp_path / "frozen-helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json,time\n"
        "print(json.dumps({'event':'ready','video_port':20001,'audio_port':20003}),flush=True)\n"
        "time.sleep(30)\n"
    )
    helper.chmod(0o700)
    client = _HelperClient([str(helper)])
    try:
        client.wait_ready(timeout_s=2)

        # When: Valid large offers fill the pipe, including callers queued on its write lock
        began = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            requests = [
                executor.submit(
                    client.request,
                    "offer",
                    timeout_s=0.2,
                    session_id=str(index),
                    sdp="v=0\r\n" + "a" * 47_900,
                    lease_seconds=10,
                )
                for index in range(4)
            ]
            for request in requests:
                with pytest.raises(_HelperError, match="timed out"):
                    request.result(timeout=1)

        # Then: Both pipe backpressure and lock contention respect the request deadline
        assert time.monotonic() - began < 1
        client.stop()
        assert client.process.poll() is not None
        assert time.monotonic() - began < 2
    finally:
        client.stop()


def test_late_offer_reply_cannot_survive_recording_preemption(
    helper: tuple[Path, Path], tmp_path: Path
) -> None:
    # Given: A helper that ignores TERM long enough to answer an in-flight offer
    marker = tmp_path / "offer-received"
    script = (
        helper[0]
        .read_text()
        .replace(
            "peers = {}",
            "import signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\npeers = {}",
            1,
        )
    )
    script = script.replace(
        "    if command['command'] == 'offer':",
        "    if command['command'] == 'offer':\n"
        f"        open({str(marker)!r},'w').close()\n"
        "        time.sleep(0.3)",
        1,
    )
    helper[0].write_text(script)
    preview = publisher(helper)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            offer = executor.submit(preview.negotiate, PreviewOffer(sdp="v=0"), time.time() + 10)
            wait_until(marker.exists)

            # When: Recording takes ownership before the helper's successful answer arrives
            recording = executor.submit(preview.sync_recording_active, True)
            result = offer.result(timeout=1)
            recording.result(timeout=3)

        # Then: The source refuses the stale answer and all peers are stopped
        assert isinstance(result, PreviewSessionRefusal)
        assert result.reason.value == "recording_priority"
        assert preview.status().state == LivePublisherState.IDLE
    finally:
        preview.shutdown()


def test_replacement_releases_child_of_abruptly_exited_helper(
    helper: tuple[Path, Path], tmp_path: Path
) -> None:
    # Given: A helper's media child holds a loopback socket independently of its parent.
    children = tmp_path / "children.jsonl"
    maintenance_started = tmp_path / "maintenance-started"
    child_code = (
        "import json,os,socket,time\n"
        "server=socket.socket()\n"
        "server.bind(('127.0.0.1',0))\n"
        "server.listen()\n"
        f"with open({str(children)!r},'a') as stream:\n"
        "    stream.write(json.dumps({'helper':os.getppid(),'port':server.getsockname()[1]})+'\\n')\n"
        "time.sleep(60)\n"
    )
    helper[0].write_text(
        helper[0]
        .read_text()
        .replace("import json,sys,time,os", "import json,sys,time,os,subprocess", 1)
        .replace(
            "    if command['command'] == 'offer':",
            "    if command['command'] == 'start':\n"
            f"        subprocess.Popen([sys.executable,'-c',{child_code!r}])\n"
            f"    elif command['command'] == 'status' and os.path.exists({str(children)!r}):\n"
            f"        open({str(maintenance_started)!r},'w').close()\n"
            "        time.sleep(10)\n"
            "    elif command['command'] == 'offer':",
            1,
        )
    )
    preview = publisher(helper)
    try:
        preview.ensure_active()
        wait_until(lambda: children.exists() and children.stat().st_size > 0)
        original = json.loads(children.read_text().splitlines()[0])
        wait_until(maintenance_started.exists)

        # When: The helper dies while maintenance awaits its reply, then activation replaces it.
        os.kill(original["helper"], signal.SIGKILL)
        os.waitpid(original["helper"], 0)
        replacement = preview.ensure_active()

        # Then: Replacement reclaims the old media socket and owns a usable new input.
        assert not isinstance(replacement, LivePublisherStartRefusal)

        def media_socket_released() -> bool:
            try:
                with socket.socket() as reclaimed:
                    reclaimed.bind(("127.0.0.1", original["port"]))
                return True
            except OSError:
                return False

        wait_until(media_socket_released)
        answer = preview.negotiate(PreviewOffer(sdp="v=0"), time.time() + 10)
        assert isinstance(answer, PreviewAnswer)
        preview.request_stop()
        assert preview.status().state == LivePublisherState.IDLE
    finally:
        preview.shutdown()
        if children.exists():
            for raw in children.read_text().splitlines():
                try:
                    os.killpg(json.loads(raw)["helper"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
