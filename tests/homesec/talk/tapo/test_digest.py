"""Tests for Tapo local Digest authentication helpers."""

from __future__ import annotations

import pytest

from homesec.talk.tapo.digest import (
    TapoDigestError,
    build_digest_authorization_header,
    digest_authorization_matches,
    parse_digest_authorization,
    parse_www_authenticate,
)


def test_parse_www_authenticate_reads_encrypt_type_hint() -> None:
    """Digest challenge parsing should preserve Tapo SHA256 hint metadata."""
    # Given: A Tapo Digest challenge with encrypt_type=3
    header = 'Digest realm="tapo", nonce="abc", qop="auth", encrypt_type="3"'

    # When: Parsing the challenge
    challenge = parse_www_authenticate(header)

    # Then: The challenge prefers SHA256 credential material
    assert challenge.scheme == "digest"
    assert challenge.realm == "tapo"
    assert challenge.nonce == "abc"
    assert challenge.encrypt_type == "3"
    assert challenge.preferred_hash_kind == "sha256"


def test_parse_www_authenticate_defaults_to_md5_without_encrypt_hint() -> None:
    """Digest challenge parsing should default old Tapo challenges to MD5 hashes."""
    # Given: A Digest challenge without the newer encrypt_type hint
    header = 'Digest realm="tapo", nonce="abc", qop="auth"'

    # When: Parsing the challenge
    challenge = parse_www_authenticate(header)

    # Then: MD5 credential material is preferred
    assert challenge.preferred_hash_kind == "md5"


def test_build_digest_authorization_header_matches_expected_response() -> None:
    """Digest authorization should be deterministic for a fixed cnonce and nonce count."""
    # Given: A fixed Tapo Digest challenge and credential hash material
    challenge = parse_www_authenticate('Digest realm="tapo", nonce="abc", qop="auth"')

    # When: Building the Authorization header
    header = build_digest_authorization_header(
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="A" * 64,
        nonce_count=1,
        cnonce="client-nonce",
    )

    # Then: The header is parseable and validates against the same challenge
    values = parse_digest_authorization(header)
    assert values["username"] == "admin"
    assert values["uri"] == "/stream"
    assert values["qop"] == "auth"
    assert values["nc"] == "00000001"
    assert values["cnonce"] == "client-nonce"
    assert digest_authorization_matches(
        authorization_header=header,
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="A" * 64,
    )


def test_build_digest_authorization_supports_legacy_no_qop_challenge() -> None:
    """Digest authorization should support old challenges without qop."""
    # Given: A Digest challenge without qop
    challenge = parse_www_authenticate('Digest realm="tapo", nonce="abc"')

    # When: Building the Authorization header
    header = build_digest_authorization_header(
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="B" * 32,
    )

    # Then: The response validates without nc/cnonce fields
    values = parse_digest_authorization(header)
    assert "qop" not in values
    assert digest_authorization_matches(
        authorization_header=header,
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="B" * 32,
    )


def test_build_digest_authorization_rejects_non_digest_challenge() -> None:
    """Tapo Digest helpers should reject unsupported auth schemes."""
    # Given: A Basic challenge
    challenge = parse_www_authenticate('Basic realm="tapo"')

    # When/Then: Building a Tapo authorization header fails safely
    with pytest.raises(TapoDigestError, match="Digest"):
        build_digest_authorization_header(
            challenge=challenge,
            method="POST",
            uri="/stream",
            username="admin",
            password_material="A" * 64,
        )


def test_build_digest_authorization_rejects_unsupported_qop() -> None:
    """Digest authorization should fail closed for unsupported qop challenges."""
    # Given: A Digest challenge that does not offer qop=auth
    challenge = parse_www_authenticate('Digest realm="tapo", nonce="abc", qop="auth-int"')

    # When/Then: Building a Tapo authorization header fails safely
    with pytest.raises(TapoDigestError, match="qop"):
        build_digest_authorization_header(
            challenge=challenge,
            method="POST",
            uri="/stream",
            username="admin",
            password_material="A" * 64,
        )


def test_digest_mismatch_does_not_validate() -> None:
    """Digest validation should reject wrong credential material."""
    # Given: A header generated with one credential hash
    challenge = parse_www_authenticate('Digest realm="tapo", nonce="abc", qop="auth"')
    header = build_digest_authorization_header(
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="A" * 64,
        cnonce="client-nonce",
    )

    # When: Validating with a different credential hash
    valid = digest_authorization_matches(
        authorization_header=header,
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="B" * 64,
    )

    # Then: The header does not match
    assert valid is False


def test_digest_validation_rejects_invalid_nonce_count() -> None:
    """Digest validation should fail closed for malformed qop counters."""
    # Given: A valid Digest header whose nc value is corrupted
    challenge = parse_www_authenticate('Digest realm="tapo", nonce="abc", qop="auth"')
    header = build_digest_authorization_header(
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="A" * 64,
        cnonce="client-nonce",
    ).replace("nc=00000001", "nc=nothex")

    # When: Validating the corrupted header
    valid = digest_authorization_matches(
        authorization_header=header,
        challenge=challenge,
        method="POST",
        uri="/stream",
        username="admin",
        password_material="A" * 64,
    )

    # Then: The helper rejects it instead of raising
    assert valid is False
