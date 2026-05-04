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


def test_onvif_backchannel_config_prefers_explicit_url_and_credentials() -> None:
    config = ONVIFBackchannelConfig(
        rtsp_url="rtsp://camera.local/explicit",
        rtsp_url_env="TALK_RTSP_URL",
        username="direct-user",
        password="direct-pass",
        username_env="TALK_RTSP_USER",
        password_env="TALK_RTSP_PASSWORD",
    )
    environ = {
        "TALK_RTSP_URL": "rtsp://camera.local/env",
        "TALK_RTSP_USER": "env-user",
        "TALK_RTSP_PASSWORD": "env-pass",
    }

    assert config.resolve_rtsp_url(environ) == "rtsp://camera.local/explicit"
    assert config.resolve_credentials(environ) == RTSPCredentials(
        username="direct-user", password="direct-pass"
    )


def test_onvif_backchannel_config_requires_rtsp_url_source() -> None:
    with pytest.raises(ValidationError, match="rtsp_url_env or rtsp_url"):
        ONVIFBackchannelConfig()


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


def test_onvif_backchannel_config_normalizes_supported_codec_names() -> None:
    config = ONVIFBackchannelConfig(
        rtsp_url="rtsp://camera.local/talk",
        preferred_codecs=["pcmu/8000/1"],
    )

    assert config.preferred_codecs == ["PCMU/8000"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("transport", "udp", "rtsp_tcp_interleaved"),
        ("connect_timeout_s", 0, "greater than 0"),
        ("io_timeout_s", 31, "less than or equal to 30"),
        ("extra_field", True, "Extra inputs"),
    ],
)
def test_onvif_backchannel_config_rejects_invalid_shape(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs: dict[str, object] = {"rtsp_url": "rtsp://camera.local/talk", field: value}

    with pytest.raises(ValidationError, match=match):
        ONVIFBackchannelConfig(**kwargs)


def test_onvif_backchannel_config_reports_missing_env_values() -> None:
    config = ONVIFBackchannelConfig(
        rtsp_url_env="TALK_RTSP_URL",
        username_env="TALK_RTSP_USER",
        password_env="TALK_RTSP_PASSWORD",
    )

    with pytest.raises(ValueError, match="TALK_RTSP_URL"):
        config.resolve_rtsp_url({})

    with pytest.raises(ValueError, match="TALK_RTSP_USER"):
        config.resolve_credentials({"TALK_RTSP_URL": "rtsp://camera.local/talk"})
