"""ONVIF RTSP audio-backchannel session orchestration."""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import dataclass

from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    CameraRejectedTalkSessionError,
    CameraTalkStreamFailedError,
    UnsupportedTalkCodecError,
)
from homesec.sources.rtsp.talk.g711 import encode_pcmu
from homesec.sources.rtsp.talk.models import ONVIFBackchannelConfig
from homesec.sources.rtsp.talk.resample import resample_pcm_s16le_mono
from homesec.sources.rtsp.talk.rtp import RTPPacketizer
from homesec.sources.rtsp.talk.rtsp_client import (
    RTSPClient,
    RTSPConnectionConfig,
    parse_interleaved_channels,
)
from homesec.sources.rtsp.talk.sdp import SDPCodec, SelectedBackchannel, select_audio_backchannel

logger = logging.getLogger(__name__)
_ONVIF_BACKCHANNEL_REQUIRE = "www.onvif.org/ver20/backchannel"


@dataclass(slots=True)
class ONVIFTalkSession:
    """Active ONVIF backchannel session for one camera talk stream."""

    session_id: str
    camera_name: str
    client: RTSPClient
    selected: SelectedBackchannel
    input_sample_rate: int = 16000
    rtp_channel: int = 0
    packetizer: RTPPacketizer | None = None
    _closed: bool = False

    @property
    def selected_codec(self) -> str:
        return self.selected.selected_codec

    async def write_pcm_frame(self, pcm_s16le: bytes) -> None:
        """Encode and send one browser PCM frame to the camera speaker."""
        if self._closed:
            raise CameraTalkStreamFailedError("Cannot send audio on a closed talk session")
        packetizer = self.packetizer
        if packetizer is None:
            packetizer = RTPPacketizer(
                payload_type=self.selected.payload_type,
                clock_rate=self.selected.codec.clock_rate,
            )
            self.packetizer = packetizer

        pcm_for_camera = resample_pcm_s16le_mono(
            pcm_s16le,
            input_rate=self.input_sample_rate,
            output_rate=self.selected.codec.clock_rate,
        )
        payload = _encode_for_codec(pcm_for_camera, self.selected.codec)
        rtp_packet = packetizer.packetize(payload, timestamp_increment=len(payload))
        try:
            await self.client.send_interleaved_frame(self.rtp_channel, rtp_packet)
        except Exception as exc:
            raise CameraTalkStreamFailedError(str(exc)) from exc

    async def close(self) -> None:
        """Tear down the RTSP talk session best-effort and close its TCP socket."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.client.session_id is not None:
                await self.client.teardown(headers=_require_header())
        except Exception as exc:
            logger.warning("RTSP talk TEARDOWN failed: %s", exc)
        finally:
            await self.client.close()


class ONVIFBackchannelAdapter:
    """Open ONVIF RTSP audio-backchannel sessions for one source."""

    def __init__(
        self,
        config: ONVIFBackchannelConfig,
        *,
        camera_name: str,
        rtsp_url: str | None = None,
    ) -> None:
        self._config = config
        self._rtsp_url = rtsp_url or config.resolve_rtsp_url()
        self._credentials = config.resolve_credentials()
        self._camera_name = camera_name

    async def open_session(
        self,
        *,
        session_id: str,
        input_sample_rate: int = 16000,
    ) -> ONVIFTalkSession:
        """Negotiate DESCRIBE/SETUP/PLAY and return a ready talk session."""
        client = RTSPClient(
            RTSPConnectionConfig(
                url=self._rtsp_url,
                credentials=self._credentials,
                user_agent=self._config.user_agent,
                connect_timeout_s=self._config.connect_timeout_s,
                io_timeout_s=self._config.io_timeout_s,
            )
        )
        setup_completed = False
        await client.connect()
        try:
            describe = await client.describe(headers=_require_header())
            if describe.status_code == 551:
                raise CameraBackchannelUnsupportedError(
                    f"Camera rejected backchannel DESCRIBE with RTSP {describe.status_code}"
                )
            if describe.status_code != 200:
                raise CameraRejectedTalkSessionError(
                    f"Camera rejected backchannel DESCRIBE with RTSP {describe.status_code}"
                )
            base_control_url = (
                describe.header("content-base")
                or describe.header("content-location")
                or self._rtsp_url
            )
            selected = select_audio_backchannel(
                describe.body.decode("utf-8", errors="replace"),
                preferred_codecs=self._config.preferred_codecs,
                base_control_url=base_control_url,
            )
            setup = await client.setup_interleaved(
                selected.control,
                rtp_channel=0,
                headers=_require_header(),
            )
            if setup.status_code != 200:
                raise CameraRejectedTalkSessionError(
                    f"Camera rejected backchannel SETUP with RTSP {setup.status_code}"
                )
            if not client.session_id:
                raise CameraRejectedTalkSessionError(
                    "Camera accepted backchannel SETUP without an RTSP Session header"
                )
            setup_completed = True
            rtp_channel = _rtp_channel_from_setup(setup.header("transport"))
            play = await client.play(headers=_require_header())
            if play.status_code != 200:
                raise CameraRejectedTalkSessionError(
                    f"Camera rejected backchannel PLAY with RTSP {play.status_code}"
                )
            return ONVIFTalkSession(
                session_id=session_id,
                camera_name=self._camera_name,
                client=client,
                selected=selected,
                input_sample_rate=input_sample_rate,
                rtp_channel=rtp_channel,
            )
        except Exception:
            if setup_completed:
                await _best_effort_teardown(client)
            await client.close()
            raise


def _require_header() -> dict[str, str]:
    return {"Require": _ONVIF_BACKCHANNEL_REQUIRE}


async def _best_effort_teardown(client: RTSPClient) -> None:
    with suppress(Exception):
        await client.teardown(headers=_require_header())


def _rtp_channel_from_setup(transport_header: str | None) -> int:
    if transport_header is None:
        return 0
    channels = parse_interleaved_channels(transport_header)
    if channels is None:
        return 0
    return channels[0]


def _encode_for_codec(pcm_s16le: bytes, codec: SDPCodec) -> bytes:
    normalized = codec.normalized_name.upper()
    if normalized == "PCMU/8000":
        return encode_pcmu(pcm_s16le)
    raise UnsupportedTalkCodecError(f"Unsupported talk codec selected: {codec.normalized_name}")
