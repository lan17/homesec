from __future__ import annotations

import pytest
from pydantic import ValidationError

from homesec.sources.rtsp.talk.models import ONVIFBackchannelConfig
from homesec.sources.rtsp.talk.rtsp_auth import RTSPCredentials


def test_onvif_backchannel_config_resolves_url_and_credentials_from_env() -> None:
    config = ONVIFBackchannelConfig(
        rtsp_url_env="TALK_RTSP_URL",
        username_env="TALK_RTSP_USER",
        password_env="TALK_RTSP_PASSWORD",
    )
    environ = {
        "TALK_RTSP_URL": "rtsp://camera.local/talk",
        "TALK_RTSP_USER": "alice",
        "TALK_RTSP_PASSWORD": "secret",
    }

    assert config.resolve_rtsp_url(environ) == "rtsp://camera.local/talk"
    assert config.resolve_credentials(environ) == RTSPCredentials(
        username="alice", password="secret"
    )


def test_onvif_backchannel_config_requires_credential_pairs() -> None:
    with pytest.raises(ValidationError, match="username and password"):
        ONVIFBackchannelConfig(rtsp_url="rtsp://camera.local/talk", username="alice")

    with pytest.raises(ValidationError, match="username_env and password_env"):
        ONVIFBackchannelConfig(rtsp_url="rtsp://camera.local/talk", username_env="USER")


def test_onvif_backchannel_config_rejects_deferred_codecs() -> None:
    with pytest.raises(ValidationError, match="PCMU/8000"):
        ONVIFBackchannelConfig(
            rtsp_url="rtsp://camera.local/talk",
            preferred_codecs=["PCMA/8000"],
        )
