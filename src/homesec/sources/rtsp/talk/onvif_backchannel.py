"""ONVIF RTSP audio-backchannel session orchestration."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass

from homesec.models.talk import (
    TalkCapabilityProbeResult,
    TalkCapabilityState,
    TalkRefusalReason,
)
from homesec.sources.rtsp.talk.errors import (
    CameraBackchannelUnsupportedError,
    CameraRejectedTalkSessionError,
    CameraTalkStreamFailedError,
    TalkProtocolError,
    UnsupportedTalkCodecError,
)
from homesec.sources.rtsp.talk.g711 import encode_pcma, encode_pcmu
from homesec.sources.rtsp.talk.models import ONVIFBackchannelConfig
from homesec.sources.rtsp.talk.resample import resample_pcm_s16le_mono
from homesec.sources.rtsp.talk.rtp import RTPPacketizer
from homesec.sources.rtsp.talk.rtsp_client import (
    RTSPClient,
    RTSPConnectionConfig,
    parse_interleaved_channels,
)
from homesec.sources.rtsp.talk.sdp import (
    SDPCodec,
    SDPDescription,
    SelectedBackchannel,
    advertised_audio_backchannel_codecs,
    parse_sdp,
    select_audio_backchannel,
)

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


@dataclass(slots=True)
class BackchannelDescription:
    """Parsed ONVIF backchannel DESCRIBE response used by probe and open."""

    description: SDPDescription
    offered_codecs: list[str]
    base_control_url: str


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

    async def probe(self) -> TalkCapabilityProbeResult:
        """Probe camera SDP for ONVIF audio backchannel capability."""
        client = self._client()
        try:
            await client.connect()
            backchannel = await self._describe_backchannel(client)
            try:
                selected = self._select_backchannel(backchannel)
            except CameraBackchannelUnsupportedError as exc:
                return TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.UNSUPPORTED,
                    offered_codecs=backchannel.offered_codecs,
                    refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
                    message=str(exc),
                )
            except UnsupportedTalkCodecError as exc:
                return TalkCapabilityProbeResult(
                    capability=TalkCapabilityState.UNSUPPORTED_CODEC,
                    offered_codecs=backchannel.offered_codecs,
                    refusal_reason=TalkRefusalReason.UNSUPPORTED_CODEC,
                    message=str(exc),
                )
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.SUPPORTED,
                offered_codecs=backchannel.offered_codecs,
                selected_codec=selected.selected_codec,
            )
        except CameraBackchannelUnsupportedError as exc:
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.UNSUPPORTED,
                refusal_reason=TalkRefusalReason.UNSUPPORTED_CAMERA,
                message=str(exc),
            )
        except CameraRejectedTalkSessionError as exc:
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.ERROR,
                refusal_reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
                message=str(exc),
            )
        except TalkProtocolError as exc:
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.ERROR,
                refusal_reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
                message=str(exc) or type(exc).__name__,
            )
        except Exception as exc:
            return TalkCapabilityProbeResult(
                capability=TalkCapabilityState.ERROR,
                refusal_reason=TalkRefusalReason.CAMERA_BACKCHANNEL_FAILED,
                message=str(exc) or type(exc).__name__,
            )
        finally:
            with suppress(Exception):
                await client.close()

    async def open_session(
        self,
        *,
        session_id: str,
        input_sample_rate: int = 16000,
    ) -> ONVIFTalkSession:
        """Negotiate DESCRIBE/SETUP/PLAY and return a ready talk session."""
        client = self._client()
        setup_completed = False
        await client.connect()
        try:
            backchannel = await self._describe_backchannel(client)
            selected = self._select_backchannel(backchannel)
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
        except (Exception, asyncio.CancelledError):
            if setup_completed:
                await _best_effort_teardown(client)
            await client.close()
            raise

    async def _describe_backchannel(self, client: RTSPClient) -> BackchannelDescription:
        """Run ONVIF backchannel DESCRIBE and parse SDP diagnostics once."""
        describe = await client.describe(headers=_require_header())
        if describe.status_code == 551:
            raise CameraBackchannelUnsupportedError("Camera rejected ONVIF backchannel DESCRIBE")
        if describe.status_code != 200:
            raise CameraRejectedTalkSessionError(
                f"Camera rejected ONVIF backchannel DESCRIBE with RTSP {describe.status_code}"
            )

        description = parse_sdp(describe.body.decode("utf-8", errors="replace"))
        base_control_url = (
            describe.header("content-base") or describe.header("content-location") or self._rtsp_url
        )
        return BackchannelDescription(
            description=description,
            offered_codecs=advertised_audio_backchannel_codecs(description),
            base_control_url=base_control_url,
        )

    def _select_backchannel(self, backchannel: BackchannelDescription) -> SelectedBackchannel:
        return select_audio_backchannel(
            backchannel.description,
            preferred_codecs=self._config.preferred_codecs,
            base_control_url=backchannel.base_control_url,
        )

    def _client(self) -> RTSPClient:
        return RTSPClient(
            RTSPConnectionConfig(
                url=self._rtsp_url,
                credentials=self._credentials,
                user_agent=self._config.user_agent,
                connect_timeout_s=self._config.connect_timeout_s,
                io_timeout_s=self._config.io_timeout_s,
            )
        )


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
    if normalized == "PCMA/8000":
        return encode_pcma(pcm_s16le)
    raise UnsupportedTalkCodecError(f"Unsupported talk codec selected: {codec.normalized_name}")
