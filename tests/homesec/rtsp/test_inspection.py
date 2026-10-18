"""Inspection ownership tests exercise the actual pipe/process boundary."""

import json
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tests.homesec.inspection_helpers import inspection_helper, inspection_requests

from homesec.sources.rtsp.inspection import InspectionFailure, InspectionGroup


def test_cancel_stops_native_writer_before_return_and_prevents_new_inputs(tmp_path: Path) -> None:
    # Given: an active native request holds a private trial writer and ignores graceful stop
    marker = tmp_path / "writer.pid"
    helper = inspection_helper(
        tmp_path,
        behavior=f"signal.signal(signal.SIGTERM, signal.SIG_IGN)\npathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\npathlib.Path(request['output_path']).write_bytes(b'writer')\nwhile True: time.sleep(.01)",
    )
    group = InspectionGroup(helper_path=str(helper), total_timeout_s=4)
    with ThreadPoolExecutor(max_workers=1) as executor:
        group.__enter__()
        target = group.trial_path()
        future = executor.submit(
            group.request,
            "inspect_recording",
            timeout_s=3,
            rtsp_url="rtsp://camera/live",
            output_path=str(target),
            audio_mode="none",
            timestamp_mode="stream",
            duration_s=2,
        )
        deadline = time.monotonic() + 2
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        # When: another thread cancels the owner
        started = time.monotonic()
        group.cancel()
        with pytest.raises(InspectionFailure):
            future.result(timeout=1)
        # Then: writers are dead and artifacts removed before cancellation completes
        assert time.monotonic() - started < 1.5
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
        assert not target.exists()
        with pytest.raises(InspectionFailure):
            group.request("inspect_metadata", timeout_s=1, rtsp_url="rtsp://later")
        assert len(inspection_requests(tmp_path)) == 1


def test_group_exit_cannot_restart_helper_or_open_more_camera_inputs(tmp_path: Path) -> None:
    # Given: a successfully closed native inspection lifetime
    helper = inspection_helper(tmp_path)
    with InspectionGroup(helper_path=str(helper), total_timeout_s=3) as group:
        group.request("inspect_metadata", timeout_s=1, rtsp_url="rtsp://camera/live")
    # When: a worker tries to issue a later request
    with pytest.raises(InspectionFailure):
        group.request("inspect_metadata", timeout_s=1, rtsp_url="rtsp://later")
    # Then: no later camera input is opened
    assert [request["command"] for request in inspection_requests(tmp_path)] == [
        "inspect_metadata",
        "stop",
    ]


def test_wrong_typed_evidence_stops_writer_before_removing_trial(tmp_path: Path) -> None:
    # Given: A helper emits valid metadata for a recording request while retaining its writer.
    marker = tmp_path / "writer.pid"
    termination = tmp_path / "writer-termination.json"
    helper = inspection_helper(
        tmp_path,
        behavior=(
            "writer = open(request['output_path'], 'wb')\n"
            "writer.write(b'active-writer')\n"
            "writer.flush()\n"
            f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\n"
            "def terminating(_signal, _frame):\n"
            f"    pathlib.Path({str(termination)!r}).write_text(json.dumps(dict(target_exists=pathlib.Path(request['output_path']).exists())))\n"
            "signal.signal(signal.SIGTERM, terminating)\n"
            "print(json.dumps(dict(request_id=request['request_id'],ok=True,inspection=dict(kind='metadata',video_codec='h264',width=160,height=120))),flush=True)\n"
            "while True: time.sleep(.01)"
        ),
    )
    with InspectionGroup(helper_path=str(helper), total_timeout_s=3) as group:
        target = group.trial_path()
        # When: The process boundary receives the unexpected but otherwise valid evidence.
        with pytest.raises(InspectionFailure, match="helper_failure"):
            group.request(
                "inspect_recording",
                timeout_s=1,
                rtsp_url="rtsp://camera/live",
                output_path=str(target),
                audio_mode="none",
                timestamp_mode="stream",
                duration_s=1,
            )
        # Then: The writer still owned its target during termination and is reaped before
        # artifacts disappear or any caller can issue another camera operation.
        assert json.loads(termination.read_text())["target_exists"]
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
        assert not target.exists()
        with pytest.raises(InspectionFailure):
            group.request("inspect_metadata", timeout_s=1, rtsp_url="rtsp://later")
        assert len(inspection_requests(tmp_path)) == 1


@pytest.mark.parametrize(
    "response", ["print('invalid-json', flush=True)", "print('x' * 128001, flush=True)"]
)
def test_invalid_control_output_is_bounded_and_reaped(tmp_path: Path, response: str) -> None:
    # Given: native work emits invalid or oversized control output then stalls
    helper = inspection_helper(tmp_path, behavior=response + "\nwhile True: time.sleep(.01)")
    # When: the owner encounters a protocol failure
    with (
        InspectionGroup(helper_path=str(helper), total_timeout_s=2) as group,
        pytest.raises(InspectionFailure, match="helper_failure"),
    ):
        group.request("inspect_metadata", timeout_s=0.5, rtsp_url="rtsp://user:secret@camera/live")
    # Then: the diagnostic is fixed and no process continues camera work
    assert len(inspection_requests(tmp_path)) == 1


def test_cancel_before_start_cannot_create_helper(tmp_path: Path) -> None:
    # Given: a group is cancelled before its worker starts
    helper = inspection_helper(tmp_path)
    group = InspectionGroup(helper_path=str(helper), total_timeout_s=2)
    group.cancel()
    # When: its thread enters and attempts an operation
    with group, pytest.raises(InspectionFailure, match="cancelled"):
        group.request("inspect_metadata", timeout_s=1, rtsp_url="rtsp://camera/live")
    # Then: the helper was never launched
    assert inspection_requests(tmp_path) == []


def test_failed_readiness_cannot_start_an_inspection_operation(tmp_path: Path) -> None:
    # Given: A protocol peer advertises the version but explicitly fails readiness.
    helper = inspection_helper(tmp_path)
    helper.write_text(
        helper.read_text().replace("event='ready', ok=True", "event='ready', ok=False")
    )

    # When: The owner attempts its first camera-free operation.
    with (
        InspectionGroup(helper_path=str(helper), total_timeout_s=2) as group,
        pytest.raises(InspectionFailure, match="local_capability_unavailable"),
    ):
        group.request("inspect_capabilities", timeout_s=1)

    # Then: A failed ready message cannot establish support or open camera work.
    assert all(request["command"] == "stop" for request in inspection_requests(tmp_path))


def test_invalid_request_validation_keeps_private_input_out_of_chained_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Given: a camera URL exceeds the typed control contract and includes short credentials
    helper = inspection_helper(tmp_path)
    url = "rtsp://u:p@cam/token?" + "private-path" * 1000
    # When: source-style exception logging includes the transport failure traceback
    with InspectionGroup(helper_path=str(helper), total_timeout_s=3) as group:
        try:
            group.request("inspect_metadata", timeout_s=1, rtsp_url=url)
        except InspectionFailure as exc:
            logging.getLogger(__name__).warning("Probe failed", exc_info=True)
            details = "".join(traceback.format_exception(exc))
        else:
            pytest.fail("oversized URL unexpectedly passed typed request validation")
    # Then: the bounded failure does not retain Pydantic's sensitive input in its cause
    assert "helper_failure" in details
    assert "Invalid inspection helper command" in details
    assert "rtsp://u:p@cam" not in details
    assert "private-path" not in details
    assert "rtsp://u:p@cam" not in caplog.text
    assert "private-path" not in caplog.text
    assert [request["command"] for request in inspection_requests(tmp_path)] == ["stop"]
