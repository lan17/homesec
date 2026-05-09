"""Tapo local talk session implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from homesec.sources.rtsp.talk.g711 import encode_pcma
from homesec.sources.rtsp.talk.resample import resample_pcm_s16le_mono
from homesec.talk.tapo.client import TapoLocalClient, TapoProtocolError
from homesec.talk.tapo.mpegts import TapoPCMATransportStreamMuxer

TAPO_LOCAL_CODEC = "PCMA/8000"
_TAPO_SAMPLE_RATE = 8000


@dataclass(slots=True)
class TapoLocalTalkSession:
    """Active HomeSec talk session over a Tapo local multipart stream."""

    session_id: str
    camera_name: str
    client: TapoLocalClient = field(repr=False)
    input_sample_rate: int
    muxer: TapoPCMATransportStreamMuxer = field(
        default_factory=TapoPCMATransportStreamMuxer,
        repr=False,
    )
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    async def create(
        cls,
        *,
        session_id: str,
        camera_name: str,
        client: TapoLocalClient,
        input_sample_rate: int,
    ) -> TapoLocalTalkSession:
        """Create a session and send the MPEG-TS stream header."""
        session = cls(
            session_id=session_id,
            camera_name=camera_name,
            client=client,
            input_sample_rate=input_sample_rate,
        )
        try:
            await session.client.write_audio_mp2t(session.muxer.header())
        except BaseException:
            await session.close()
            raise
        return session

    @property
    def selected_codec(self) -> str:
        """Return the camera-side codec emitted by this backend."""
        return TAPO_LOCAL_CODEC

    async def write_pcm_frame(self, frame: bytes) -> None:
        """Encode one browser PCM frame to Tapo PCMA MPEG-TS and send it."""
        if self._closed:
            raise TapoProtocolError("Cannot write to closed Tapo talk session")
        try:
            pcm_8k = resample_pcm_s16le_mono(
                frame,
                input_rate=self.input_sample_rate,
                output_rate=_TAPO_SAMPLE_RATE,
            )
            pcma = encode_pcma(pcm_8k)
            payload = self.muxer.audio_payload(pcma)
        except ValueError as exc:
            raise TapoProtocolError("Invalid PCM frame for Tapo local talk") from exc
        await self.client.write_audio_mp2t(payload)

    async def close(self) -> None:
        """Close the underlying Tapo local stream."""
        if self._closed:
            return
        self._closed = True
        await self.client.close()
