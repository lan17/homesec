"""Executable protocol peer for testing inspection process ownership."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def inspection_helper(tmp_path: Path, *, behavior: str = "", ready_version: int = 1) -> Path:
    helper = tmp_path / "inspection-helper"
    helper.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, signal, sys, time\n"
        f"log = pathlib.Path({str(tmp_path / 'inspection-requests.jsonl')!r})\n"
        f"print(json.dumps(dict(event='ready', ok=True, inspection_version={ready_version})), flush=True)\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    with log.open('a') as handle: handle.write(json.dumps(request) + '\\n')\n"
        "    command = request['command']\n"
        "    if command == 'stop': break\n"
        "    error = None\n"
        "    inspection = None\n"
        + "\n".join("    " + line for line in behavior.splitlines())
        + "\n"
        "    if inspection is None and error is None:\n"
        "        if command == 'inspect_metadata':\n"
        "            inspection = dict(kind='metadata',video_codec='hevc',audio_codec='aac',width=1280,height=720,fps=15.0,fps_raw='15/1')\n"
        "        elif command == 'inspect_recording':\n"
        "            pathlib.Path(request['output_path']).write_bytes(b'trial-media')\n"
        "            inspection = dict(kind='recording',finalized=True,video_packets=3,audio_packets=2,output_bytes=11,signals={})\n"
        "        elif command == 'inspect_overlap':\n"
        "            inspection = dict(kind='overlap',status='supported',roles=[dict(role=r['role'],ok=True,frames_decoded=5,video_packets=3,audio_packets=1) for r in request['roles']])\n"
        "        elif command == 'inspect_decode':\n"
        "            inspection = dict(kind='decode',frames_decoded=5,backend=request['backend'])\n"
        "        elif command == 'inspect_capabilities':\n"
        "            inspection = dict(kind='capabilities',library_version='8.0',build_id='test',decoders=['h264','hevc','aac','aac_latm','pcm_alaw','pcm_mulaw','opus','ac3','eac3','alac','mp2','mp3'],encoders=['aac','libx264','libopus','h264_videotoolbox'],parsers=['h264','hevc','aac'],muxers=['mp4'],filters=['aresample'],hwaccels=['videotoolbox'],device=dict(backend=request.get('backend','software'),compiled=True,available=True))\n"
        "    print(json.dumps(dict(request_id=request['request_id'],ok=error is None,error_code=error,inspection=inspection)),flush=True)\n"
    )
    helper.chmod(0o700)
    return helper


def inspection_requests(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "inspection-requests.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
