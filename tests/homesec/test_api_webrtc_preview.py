"""WebRTC preview authorization, leases and signaling API behavior."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from homesec.api.preview_tokens import issue_camera_preview_token
from homesec.models.config import FastAPIServerConfig, PreviewConfig, WebRTCPreviewConfig
from homesec.models.preview import (
    PreviewAnswer,
    PreviewOffer,
    PreviewSessionAction,
    PreviewSessionRefusal,
    PreviewSessionRefusalReason,
)
from tests.homesec.test_api_preview_routes import _client, _StubPreviewApp


class _WebRTCApp(_StubPreviewApp):
    def __init__(
        self, *, auth: bool = True, ice_servers: list[dict[str, object]] | None = None
    ) -> None:
        super().__init__(
            server_config=FastAPIServerConfig(auth_enabled=auth, api_key_env="HOMESEC_API_KEY"),
            preview_config=PreviewConfig(
                enabled=True,
                backend="webrtc",
                config=WebRTCPreviewConfig(
                    advertised_ip="127.0.0.1", ice_servers=ice_servers or []
                ),
            ),
        )
        self.calls: list[tuple[str, str, float | None]] = []
        self.refusal: PreviewSessionRefusal | None = None

    async def negotiate_camera_preview(
        self, camera_name: str, *, offer: PreviewOffer, lease_expires_at: float
    ) -> PreviewAnswer | PreviewSessionRefusal:
        self.calls.append(("offer", camera_name, lease_expires_at - time.time()))
        return self.refusal or PreviewAnswer(session_id=str(uuid4()), sdp="v=0\r\ns=answer\r\n")

    async def renew_camera_preview_session(
        self, camera_name: str, *, session_id: str, lease_expires_at: float
    ) -> PreviewSessionAction | PreviewSessionRefusal:
        self.calls.append(("renew", session_id, lease_expires_at - time.time()))
        return self.refusal or PreviewSessionAction(accepted=True)

    async def close_camera_preview_session(
        self, camera_name: str, *, session_id: str
    ) -> PreviewSessionAction | PreviewSessionRefusal:
        self.calls.append(("close", session_id, None))
        return PreviewSessionAction(accepted=True)


def test_webrtc_session_snapshot_preserves_hls_and_returns_signaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: An authorized WebRTC preview backend
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    client = _client(_WebRTCApp())

    # When: Attaching to preview via the existing control endpoint
    response = client.post(
        "/api/v1/preview/cameras/front", headers={"Authorization": "Bearer secret"}
    )

    # Then: The transport is explicit and signaling never exposes a camera URL
    assert response.status_code == 200
    assert response.json()["transport"] == "webrtc"
    assert response.json()["playlist_url"] is None
    assert response.json()["signaling_url"] == "/api/v1/preview/cameras/front/sessions"
    assert response.json()["token"]
    assert response.json()["ice_servers"] == []


def test_peer_lease_cannot_outlive_current_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A token minted earlier with about ten seconds remaining
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _WebRTCApp()
    client = _client(app)
    token, _ = issue_camera_preview_token(
        api_key="secret",
        camera_name="front",
        ttl_s=60,
        now=datetime.now(UTC) - timedelta(seconds=50),
    )

    # When: Negotiating and renewing a peer with that token
    offer = client.post(
        "/api/v1/preview/cameras/front/sessions",
        params={"token": token},
        json={"type": "offer", "sdp": "v=0"},
    )
    session_id = offer.json()["session_id"]
    renewed = client.patch(
        f"/api/v1/preview/cameras/front/sessions/{session_id}", params={"token": token}
    )
    closed = client.delete(
        f"/api/v1/preview/cameras/front/sessions/{session_id}", params={"token": token}
    )

    # Then: Both leases are bounded by token expiry, and close targets only that peer
    assert offer.status_code == renewed.status_code == closed.status_code == 200
    assert 0 < app.calls[0][2] <= 10
    assert 0 < app.calls[1][2] <= 10
    assert app.calls[2] == ("close", session_id, None)
    assert renewed.json() == closed.json() == {"accepted": True}


@pytest.mark.parametrize("camera,token", [("front", None), ("front", "invalid")])
def test_signaling_rejects_unauthorized_requests(
    monkeypatch: pytest.MonkeyPatch, camera: str, token: str | None
) -> None:
    # Given: An auth-enabled camera
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _WebRTCApp()
    client = _client(app)

    # When: Signaling lacks valid authorization
    response = client.post(
        f"/api/v1/preview/cameras/{camera}/sessions",
        params={"token": token} if token else {},
        json={"type": "offer", "sdp": "v=0"},
    )

    # Then: No runtime media session is opened
    assert response.status_code == 401
    assert app.calls == []


def test_camera_scoped_token_cannot_signal_other_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A token scoped to another camera
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _WebRTCApp()
    token, _ = issue_camera_preview_token(api_key="secret", camera_name="other")

    # When: The token is used for the front camera
    response = _client(app).post(
        "/api/v1/preview/cameras/front/sessions",
        params={"token": token},
        json={"type": "offer", "sdp": "v=0"},
    )

    # Then: Camera scope is enforced before negotiation
    assert response.status_code == 401
    assert app.calls == []


def test_offer_size_and_refusal_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A backend that rejects malformed browser offers
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    app = _WebRTCApp()
    app.refusal = PreviewSessionRefusal(
        reason=PreviewSessionRefusalReason.INVALID_OFFER, message="Offer rejected"
    )
    client = _client(app)
    headers = {"Authorization": "Bearer secret"}

    # When: The client sends an oversized offer and then a bounded but invalid offer
    oversized = client.post(
        "/api/v1/preview/cameras/front/sessions",
        headers=headers,
        json={"type": "offer", "sdp": "x" * 48_001},
    )
    invalid = client.post(
        "/api/v1/preview/cameras/front/sessions",
        headers=headers,
        json={"type": "offer", "sdp": "v=0"},
    )

    # Then: Oversized input never reaches the runtime, and the refusal reason is stable
    assert oversized.status_code == 422
    assert invalid.status_code == 400
    assert len(app.calls) == 1
    assert invalid.json()["reason"] == "invalid_offer"


def test_turn_credentials_resolve_only_for_authorized_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: TURN credentials stored as environment references
    monkeypatch.setenv("HOMESEC_API_KEY", "secret")
    monkeypatch.setenv("HOMESEC_TURN_PASSWORD", "turn-secret")
    app = _WebRTCApp(
        ice_servers=[
            {
                "urls": ["turn:relay:3478"],
                "username": "viewer",
                "credential_env": "HOMESEC_TURN_PASSWORD",
            }
        ]
    )
    client = _client(app)

    # When: An authorized viewer attaches
    response = client.post(
        "/api/v1/preview/cameras/front", headers={"Authorization": "Bearer secret"}
    )

    # Then: The browser gets usable credentials while persisted config contains only the env name
    assert response.json()["ice_servers"] == [
        {"urls": ["turn:relay:3478"], "username": "viewer", "credential": "turn-secret"}
    ]
    assert "turn-secret" not in app.config.preview.model_dump_json()


def test_backend_config_is_explicit_and_hls_defaults_are_unchanged() -> None:
    # Given: The existing HLS config and an explicitly selected WebRTC backend
    hls = PreviewConfig(enabled=True)

    # When: Loading a WebRTC config and malformed port range
    webrtc = PreviewConfig.model_validate(
        {"enabled": True, "backend": "webrtc", "config": {"advertised_ip": "127.0.0.1"}}
    )
    with pytest.raises(ValidationError, match="udp_port_end"):
        WebRTCPreviewConfig(advertised_ip="127.0.0.1", udp_port_start=9000, udp_port_end=8189)

    # Then: Existing installations stay HLS; only explicit selection enables WebRTC
    assert hls.backend == "hls"
    assert hls.config.segment_duration_ms == 1000
    assert isinstance(webrtc.config, WebRTCPreviewConfig)


def test_auth_disabled_webrtc_snapshot_exposes_renewal_deadline() -> None:
    # Given: A trusted-network deployment with authentication disabled
    app = _WebRTCApp(auth=False)

    # When: A browser attaches without a token
    response = _client(app).post("/api/v1/preview/cameras/front")

    # Then: The viewer still knows when to renew its bounded peer lease
    payload = response.json()
    assert response.status_code == 200
    assert payload["token"] is None
    assert payload["token_expires_at"] is None
    deadline = datetime.fromisoformat(payload["lease_expires_at"])
    assert 55 <= (deadline - datetime.now(UTC)).total_seconds() <= 60
