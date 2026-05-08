"""Tests for Tapo local talk backend config helpers."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from homesec.models.config import CameraTalkConfig, TalkConfig
from homesec.talk.backends import TalkBackendConfigError, TalkBackendContext
from homesec.talk.tapo.config import (
    TapoCredential,
    TapoLocalTalkConfig,
    resolve_tapo_credential,
    resolve_tapo_host,
)


def _context(
    *,
    source_uri: str | None = "rtsp://admin:secret@192.168.1.33:554/stream1",
    source_host: str | None = None,
    env: dict[str, str] | None = None,
) -> TalkBackendContext:
    values = env or {}
    return TalkBackendContext(
        camera_name="office",
        source_backend="rtsp",
        runtime_talk=TalkConfig(),
        camera_talk=CameraTalkConfig(),
        source_host=source_host,
        source_uri=source_uri,
        resolved_source_uri=source_uri,
        resolve_env=lambda name: values.get(name),
    )


def test_camera_talk_config_accepts_tapo_backend_and_config_block() -> None:
    """Camera talk config should preserve Tapo backend config for runtime selection."""
    # Given: A camera configured with an explicit Tapo local backend
    # When: Validating the generic camera talk config
    talk = CameraTalkConfig(
        backend="TAPO_LOCAL",
        backends={
            "TAPO_LOCAL": {
                "host": "192.168.1.33",
                "password_sha256_env": "OFFICE_TAPO_SHA256",
            }
        },
    )

    # Then: Backend IDs are normalized and backend-specific config is preserved
    assert talk.backend == "tapo_local"
    assert talk.config_for_backend("tapo_local") == {
        "host": "192.168.1.33",
        "password_sha256_env": "OFFICE_TAPO_SHA256",
    }


def test_tapo_config_uses_safe_defaults() -> None:
    """Tapo backend config should default to the local C120 protocol shape."""
    # Given: Minimal Tapo backend config
    # When: Validating the config model
    config = TapoLocalTalkConfig.model_validate({"password_sha256_env": "OFFICE_SHA"})

    # Then: Defaults match the local Tapo talk endpoint assumptions
    assert config.port == 8800
    assert config.username == "admin"
    assert config.mode == "aec"
    assert config.audio_codec == "PCMA/8000"


def test_tapo_config_rejects_unknown_fields() -> None:
    """Tapo backend config should reject unknown fields instead of ignoring mistakes."""
    # Given: Config containing a misspelled or unsupported field
    # When: Validating the config model
    # Then: Validation fails before runtime selection can use the wrong shape
    with pytest.raises(ValidationError, match="extra"):
        TapoLocalTalkConfig.model_validate(
            {"password_sha256_env": "OFFICE_SHA", "cloud_password": "not-supported"}
        )


def test_tapo_config_rejects_unsafe_env_var_names() -> None:
    """Tapo env-var references should be safe to mention in diagnostics."""
    # Given: A credential env var name containing unsafe characters
    # When: Validating the config model
    # Then: The config is rejected before that name reaches public diagnostics
    with pytest.raises(ValidationError, match="safe shell identifiers"):
        TapoLocalTalkConfig.model_validate({"password_sha256_env": "bad env name"})


def test_tapo_host_prefers_explicit_config_host() -> None:
    """Tapo host resolution should prefer the backend-specific host."""
    # Given: A Tapo config with an explicit host and a different RTSP source URI
    config = TapoLocalTalkConfig(host="192.168.1.50", password_sha256_env="OFFICE_SHA")

    # When: Resolving the Tapo endpoint host
    host = resolve_tapo_host(config, _context())

    # Then: The explicit backend host wins
    assert host == "192.168.1.50"


def test_tapo_host_can_be_derived_from_rtsp_source_uri_without_credentials() -> None:
    """Tapo host resolution should derive only the host from the RTSP source URI."""
    # Given: A Tapo config without host and an RTSP URL containing credentials
    config = TapoLocalTalkConfig(password_sha256_env="OFFICE_SHA")

    # When: Resolving the Tapo endpoint host
    host = resolve_tapo_host(config, _context())

    # Then: Only the credential-free host is returned
    assert host == "192.168.1.33"


def test_tapo_host_prefers_source_host_before_parsing_source_uri() -> None:
    """Tapo host resolution should use the source host hint when available."""
    # Given: A Tapo config without host and a source host hint
    config = TapoLocalTalkConfig(password_sha256_env="OFFICE_SHA")

    # When: Resolving the Tapo endpoint host
    host = resolve_tapo_host(
        config,
        _context(source_host="camera.local", source_uri="rtsp://192.168.1.33/stream1"),
    )

    # Then: The explicit context host wins over parsing the URI
    assert host == "camera.local"


def test_tapo_host_missing_from_config_and_source_uri_reports_safe_error() -> None:
    """Tapo host resolution should fail safely when no host is available."""
    # Given: No explicit Tapo host and no source URI host
    config = TapoLocalTalkConfig(password_sha256_env="OFFICE_SHA")

    # When: Resolving the Tapo endpoint host
    # Then: A structured public config error explains the missing host
    with pytest.raises(TalkBackendConfigError) as exc_info:
        resolve_tapo_host(config, _context(source_uri=None))
    assert exc_info.value.public_message == "Tapo local backend requires host or source URI host"


def test_tapo_credential_requires_hash_env_config() -> None:
    """Tapo credentials should require env or source URL password material."""
    # Given: Tapo config without hash env references and without RTSP password
    config = TapoLocalTalkConfig()

    # When: Resolving credential material
    # Then: A safe config error explains the missing password source
    with pytest.raises(TalkBackendConfigError) as exc_info:
        resolve_tapo_credential(config, _context(source_uri="rtsp://192.168.1.33/stream1"))
    assert (
        exc_info.value.public_message
        == "Tapo local backend requires password hash env var or RTSP URL password"
    )


def test_tapo_credential_reports_missing_env_var_by_name() -> None:
    """Tapo credential resolution should report missing env vars without secret values."""
    # Given: A Tapo config that references an unset credential env var
    config = TapoLocalTalkConfig(password_sha256_env="OFFICE_TAPO_SHA256")

    # When: Resolving credential material
    # Then: The public error names only the missing env var
    with pytest.raises(TalkBackendConfigError) as exc_info:
        resolve_tapo_credential(config, _context())
    assert (
        exc_info.value.public_message
        == "Required Tapo local environment variable is not set: OFFICE_TAPO_SHA256"
    )


def test_tapo_credential_defaults_to_admin_and_rtsp_sha256_password() -> None:
    """Tapo credentials should default to admin plus RTSP password material."""
    # Given: Tapo config without credential overrides and a credentialed RTSP URL
    config = TapoLocalTalkConfig()

    # When: Resolving credential material for a SHA256 Tapo challenge
    credential = resolve_tapo_credential(
        config,
        _context(source_uri="rtsp://camera%40home:secret@192.168.1.33/stream1"),
    )

    # Then: The Tapo username defaults to admin and only password comes from RTSP
    assert credential.username == "admin"
    assert credential.password_hash == hashlib.sha256(b"secret").hexdigest().upper()
    assert credential.hash_kind == "sha256"


def test_tapo_credential_defaults_to_rtsp_md5_password_when_challenge_prefers_md5() -> None:
    """Tapo credential defaults should match the hash kind requested by the camera."""
    # Given: Tapo config without credential overrides and a credentialed RTSP URL
    config = TapoLocalTalkConfig()

    # When: Resolving credential material for an older MD5 Tapo challenge
    credential = resolve_tapo_credential(
        config,
        _context(),
        preferred_hash_kind="md5",
    )

    # Then: The password hash is derived from the RTSP password using MD5
    assert credential.username == "admin"
    assert (
        credential.password_hash
        == hashlib.md5(
            b"secret",
            usedforsecurity=False,
        )
        .hexdigest()
        .upper()
    )
    assert credential.hash_kind == "md5"


def test_tapo_password_can_be_derived_from_percent_encoded_rtsp_url() -> None:
    """Tapo password defaults should decode RTSP URL password info."""
    # Given: Tapo config without credential overrides and encoded RTSP credentials
    config = TapoLocalTalkConfig()

    # When: Resolving credential material
    credential = resolve_tapo_credential(
        config,
        _context(source_uri="rtsp://camera%40home:p%40ss%3Aword@192.168.1.33/stream1"),
    )

    # Then: The decoded RTSP password is used with the Tapo admin username
    assert credential.username == "admin"
    assert credential.password_hash == hashlib.sha256(b"p@ss:word").hexdigest().upper()


def test_tapo_explicit_credential_env_wins_over_rtsp_url_password() -> None:
    """Explicit Tapo credential env vars should override RTSP URL password defaults."""
    # Given: A Tapo config with explicit hash env and a different RTSP password
    config = TapoLocalTalkConfig(password_sha256_env="OFFICE_TAPO_SHA256")
    sha256 = "A" * 64

    # When: Resolving credential material
    credential = resolve_tapo_credential(
        config,
        _context(env={"OFFICE_TAPO_SHA256": sha256}),
    )

    # Then: The explicit env hash wins over any derived source password hash
    assert credential.username == "admin"
    assert credential.password_hash == sha256
    assert credential.password_hash != hashlib.sha256(b"secret").hexdigest().upper()


def test_tapo_explicit_username_wins_over_admin_default() -> None:
    """Explicit Tapo usernames should override the admin default."""
    # Given: A Tapo config with an explicit username and source URL credentials
    config = TapoLocalTalkConfig(username="speaker", password_sha256_env="OFFICE_TAPO_SHA256")

    # When: Resolving credential material
    credential = resolve_tapo_credential(
        config,
        _context(env={"OFFICE_TAPO_SHA256": "A" * 64}),
    )

    # Then: The explicitly configured username wins over the admin default
    assert credential.username == "speaker"


def test_tapo_credential_normalizes_sha256_hash_from_env() -> None:
    """Tapo credential resolution should normalize configured SHA256 hash material."""
    # Given: A lowercase SHA256 hash in the configured env var
    sha256 = "a" * 64
    config = TapoLocalTalkConfig(password_sha256_env="OFFICE_TAPO_SHA256")

    # When: Resolving credential material
    credential = resolve_tapo_credential(
        config,
        _context(env={"OFFICE_TAPO_SHA256": sha256}),
    )

    # Then: The hash is uppercased and tagged as SHA256
    assert credential.username == "admin"
    assert credential.password_hash == "A" * 64
    assert credential.hash_kind == "sha256"


def test_tapo_credential_repr_does_not_expose_hash_value() -> None:
    """Tapo credential repr should not expose hash material by accident."""
    # Given: Resolved Tapo credential hash material
    secret_hash = "A" * 64
    credential = TapoCredential(username="admin", password_hash=secret_hash, hash_kind="sha256")

    # When: Formatting the credential for debugging
    text = repr(credential)

    # Then: The hash value is not included in the repr
    assert secret_hash not in text
    assert "password_hash" not in text


def test_tapo_credential_rejects_invalid_hash_shape_without_exposing_value() -> None:
    """Tapo credential resolution should reject invalid hash values safely."""
    # Given: A configured env var containing a malformed hash
    config = TapoLocalTalkConfig(password_sha256_env="OFFICE_TAPO_SHA256")

    # When: Resolving credential material
    # Then: The error names the env var but not the invalid value
    with pytest.raises(TalkBackendConfigError) as exc_info:
        resolve_tapo_credential(
            config,
            _context(env={"OFFICE_TAPO_SHA256": "not-a-hash"}),
        )
    assert exc_info.value.public_message == (
        "Tapo local credential hash in OFFICE_TAPO_SHA256 is invalid"
    )
    assert "not-a-hash" not in exc_info.value.public_message
