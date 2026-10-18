"""Setup uses source-owned inspection context and waits for concrete cleanup."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from homesec.models.config import PreviewConfig
from homesec.models.setup import TestConnectionRequest as ConnectionRequest
from homesec.plugins import discover_all_plugins
from homesec.services import setup
from homesec.services.setup_probes import SetupProbeContext, SetupProbeRegistry
from homesec.sources.rtsp.helper_client import HelperClient, HelperError
from homesec.sources.rtsp.inspection import InspectionFailure, InspectionGroup
from homesec.sources.rtsp.preflight import PreflightError, RTSPStartupPreflight
from homesec.sources.rtsp.setup_probe import _run_owned_inspection
from tests.homesec.inspection_helpers import inspection_helper, inspection_requests


@pytest.fixture(autouse=True)
def _plugins() -> None:
    discover_all_plugins()


def _app(preview: PreviewConfig | None, *, saved: PreviewConfig | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        bootstrap_mode=preview is None,
        config=SimpleNamespace(preview=preview),
        config_manager=SimpleNamespace(
            get_config=lambda: SimpleNamespace(preview=saved or PreviewConfig())
        ),
        setup_test_connection_lock=asyncio.Lock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("preview_mode", ["bootstrap", "enabled", "disabled", "hls", "active"])
async def test_setup_readiness_and_connection_use_the_same_source_helper(
    preview_mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: Existing preview configuration, no media CLIs, and a real protocol peer.
    helper = inspection_helper(tmp_path)
    (tmp_path / "homesec-webrtc").symlink_to(helper)
    monkeypatch.setenv("PATH", str(tmp_path))
    custom = PreviewConfig(
        enabled=preview_mode == "enabled",
        config={"helper_path": str(helper), "advertised_ip": "127.0.0.1"},
    )
    if preview_mode == "bootstrap":
        app = _app(None)
    elif preview_mode == "active":
        app = _app(custom, saved=PreviewConfig(config={"helper_path": "/missing/saved-helper"}))
    elif preview_mode == "hls":
        app = _app(PreviewConfig(backend="hls"))
    else:
        app = _app(None, saved=custom)
    request = ConnectionRequest(
        type="camera", backend="rtsp", config={"rtsp_url": "rtsp://camera/main"}
    )

    # When: Checking readiness and testing a proposed camera without saving/activation APIs.
    readiness = await setup._ffmpeg_check(app)
    after_readiness = inspection_requests(tmp_path)
    response = await setup.test_connection(request, app)

    # Then: Readiness opens no camera; Test Connection uses the same selected helper
    # and preserves the public check identity and detail fields with no tool discovery.
    assert readiness.name == "ffmpeg"
    assert readiness.passed, readiness.message
    assert "Native inspection ready" in readiness.message
    assert "8.0" in readiness.message
    assert all(r["command"] in {"inspect_capabilities", "stop"} for r in after_readiness)
    assert response.success, response.message
    assert response.details is not None
    assert response.details["probed_streams"] == 1
    assert response.details["selected_recording_profile"] == "mp4:v=copy:a=copy"
    assert response.details["session_mode"] == "dual_stream"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "incompatible", "insufficient"])
async def test_readiness_rejects_invalid_override_without_default_fallback(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A valid default helper and a configured override that cannot inspect.
    default_dir = tmp_path / "default"
    default_dir.mkdir()
    default = inspection_helper(default_dir)
    (default_dir / "homesec-webrtc").symlink_to(default)
    monkeypatch.setenv("PATH", str(default_dir))
    invalid_dir = tmp_path / "invalid"
    invalid_dir.mkdir()
    if failure == "missing":
        override = invalid_dir / "missing-helper"
    else:
        override = inspection_helper(
            invalid_dir,
            ready_version=2 if failure == "incompatible" else 1,
            behavior=(
                "if command == 'inspect_capabilities':\n"
                "    inspection = dict(kind='capabilities',library_version='8.0',build_id='incomplete',"
                "decoders=[],encoders=[],parsers=[],muxers=[],filters=[],hwaccels=[])"
            )
            if failure == "insufficient"
            else "",
        )
    app = _app(PreviewConfig(enabled=False, config={"helper_path": str(override)}))

    # When: Checking inspection readiness against the configured executable.
    result = await setup._ffmpeg_check(app)

    # Then: A separate default cannot hide a local missing/incompatible build.
    assert not result.passed
    assert "inspection" in result.message.lower()
    assert inspection_requests(default_dir) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_setup_timeout_and_cancel_reap_trial_writer_before_returning(
    cancel: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: A real helper process stalls while owning a private trial writer.
    marker = tmp_path / "writer.json"
    helper = inspection_helper(
        tmp_path,
        behavior=(
            "if command == 'inspect_recording':\n"
            "    pathlib.Path(request['output_path']).write_bytes(b'pending-media')\n"
            f"    pathlib.Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),path=request['output_path'])))\n"
            "    time.sleep(30)"
        ),
    )
    monkeypatch.setattr("homesec.sources.rtsp.setup_probe.RTSP_TEST_CONNECTION_TIMEOUT_S", 1.5)
    app = _app(PreviewConfig(config={"helper_path": str(helper)}))
    request = ConnectionRequest(
        type="camera", backend="rtsp", config={"rtsp_url": "rtsp://camera/main"}
    )
    started = time.monotonic()
    task = asyncio.create_task(setup.test_connection(request, app))
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists(), "fixture did not reach the recording operation"
    writer = json.loads(marker.read_text())

    # When: The request expires or the actual request task is cancelled.
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        response = await task
        assert not response.success

    # Then: No helper/writer survives, no later camera operation starts, and output
    # disappears only after the process is reaped, within the outer cleanup budget.
    assert time.monotonic() - started < 3.0
    with pytest.raises(ProcessLookupError):
        os.kill(writer["pid"], 0)
    assert not Path(writer["path"]).exists()
    commands = [r["command"] for r in inspection_requests(tmp_path)]
    assert "inspect_overlap" not in commands


@pytest.mark.asyncio
async def test_contextual_registration_preserves_unrelated_probe_signatures() -> None:
    # Given: A contextual source probe and an unrelated callback in the same registry.
    registry = SetupProbeRegistry()
    calls: list[str] = []

    async def legacy(*, config: dict[str, object]) -> object:
        calls.append("legacy")
        return config

    async def contextual(*, config: dict[str, object], context: SetupProbeContext) -> object:
        calls.append(context.preview.backend)
        return config

    registry.register("camera", "legacy")(legacy)
    registry.register("camera", "native", contextual=True)(contextual)
    context = SetupProbeContext(preview=PreviewConfig(backend="hls"))

    # When: Dispatching through the extended existing registration seam.
    for backend in ["legacy", "native"]:
        probe = registry.get("camera", backend, context=context)
        assert probe is not None
        await probe(config={})

    # Then: Existing callbacks keep their signature and only opted-in sources get context.
    assert calls == ["legacy", "hls"]


@pytest.mark.asyncio
async def test_initial_supervision_cleanup_failure_still_joins_worker_before_reporting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a native trial writer and an initial supervisor stop/reap failure
    marker = tmp_path / "owned-writer.json"
    helper = inspection_helper(
        tmp_path,
        behavior=(
            "if command == 'inspect_recording':\n"
            "    pathlib.Path(request['output_path']).write_bytes(b'pending')\n"
            f"    pathlib.Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),path=request['output_path'])))\n"
            "    time.sleep(30)"
        ),
    )
    inspector = InspectionGroup(helper_path=str(helper), total_timeout_s=2)
    target = inspector.trial_path()
    original_stop = HelperClient.stop
    stops: list[str] = []

    def fail_initial_stop(self: HelperClient, *, timeout_s: float | None = None) -> None:
        stops.append("stop")
        if len(stops) == 1:
            raise HelperError("Synthetic supervisor failure")
        original_stop(self, timeout_s=timeout_s)

    monkeypatch.setattr("homesec.sources.rtsp.helper_client.HelperClient.stop", fail_initial_stop)
    task = asyncio.create_task(
        _run_owned_inspection(
            inspector,
            lambda: inspector.request(
                "inspect_recording",
                timeout_s=0.8,
                rtsp_url="rtsp://camera/live",
                output_path=str(target),
                audio_mode="none",
                timestamp_mode="stream",
                duration_s=1,
            ),
            timeout_s=2,
        )
    )
    deadline = time.monotonic() + 2
    while not marker.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert marker.exists()
    # When: request cancellation starts cleanup before the bounded command expires
    task.cancel()
    with pytest.raises(InspectionFailure, match="cleanup_failed"):
        await task
    # Then: the worker's later transport deadline reaps the process before failure is returned
    writer = json.loads(marker.read_text())
    assert len(stops) >= 2
    with pytest.raises(ProcessLookupError):
        os.kill(writer["pid"], 0)
    assert not target.exists()


@pytest.mark.asyncio
async def test_transport_deadline_is_reported_as_timeout_with_failure_details(
    tmp_path: Path,
) -> None:
    # Given: native startup does not reply within the overall inspection deadline
    helper = inspection_helper(tmp_path, behavior="time.sleep(30)")
    inspector = InspectionGroup(helper_path=str(helper), total_timeout_s=1.2)
    preflight = RTSPStartupPreflight(
        output_dir=tmp_path, rtsp_connect_timeout_s=1, rtsp_io_timeout_s=1, inspection=inspector
    )
    # When: the finite group budget expires while metadata owns camera work
    outcome = await _run_owned_inspection(
        inspector,
        lambda: preflight.run(
            camera_name="test",
            primary_rtsp_url="rtsp://camera/live",
            detect_rtsp_url="rtsp://camera/live",
        ),
        timeout_s=2,
    )
    # Then: the timeout is not misreported as incompatible/missing local helper support
    assert isinstance(outcome, PreflightError)
    assert outcome.stage == "inspection"
    assert outcome.message == "Native inspection: timeout"
