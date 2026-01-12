"""Tests for MQTT notifier plugin."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from homesec.models.alert import Alert
from homesec.models.config import MQTTAuthConfig, MQTTConfig
from homesec.plugins.notifiers.mqtt import MQTTNotifier


class _FakePublishResult:
    def __init__(self) -> None:
        self.waited = False

    def wait_for_publish(self) -> None:
        self.waited = True


class _FakeClient:
    def __init__(self) -> None:
        self.on_connect = None
        self.on_disconnect = None
        self.connected = False
        self.published: list[tuple[str, str, int, bool]] = []
        self.username: str | None = None
        self.password: str | None = None

    def username_pw_set(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def connect(
        self, _host: str, _port: int, _keepalive: int | None = None, **_kwargs: Any
    ) -> None:
        self.connected = True
        if self.on_connect is not None:
            self.on_connect(self, None, {}, 0)

    def loop_start(self) -> None:
        return None

    def loop_stop(self) -> None:
        return None

    def disconnect(self) -> None:
        self.connected = False
        if self.on_disconnect is not None:
            self.on_disconnect(self, None, 0)

    def is_connected(self) -> bool:
        return self.connected

    def publish(self, topic: str, payload: str, qos: int, retain: bool) -> _FakePublishResult:
        self.published.append((topic, payload, qos, retain))
        return _FakePublishResult()


class _FakeClientNoConnect(_FakeClient):
    def connect(
        self, _host: str, _port: int, _keepalive: int | None = None, **_kwargs: Any
    ) -> None:
        self.connected = False


def _sample_alert() -> Alert:
    return Alert(
        clip_id="clip_1",
        camera_name="front",
        storage_uri="mock://clip_1",
        view_url="http://example/clip_1",
        risk_level="low",
        activity_type="passerby",
        notify_reason="risk_level=low",
        summary="summary",
        ts=datetime.now(),
        dedupe_key="clip_1",
        upload_failed=False,
    )


@pytest.mark.asyncio
async def test_mqtt_notifier_publishes_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    """MQTT notifier should publish JSON payload to templated topic."""
    # Given a notifier with stubbed MQTT client and auth env vars
    fake_client = _FakeClient()
    monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
    monkeypatch.setenv("MQTT_USER", "user")
    monkeypatch.setenv("MQTT_PASS", "pass")
    config = MQTTConfig(
        host="localhost",
        topic_template="alerts/{camera_name}",
        auth=MQTTAuthConfig(username_env="MQTT_USER", password_env="MQTT_PASS"),
    )
    notifier = MQTTNotifier(config)

    # When sending an alert
    await notifier.send(_sample_alert())

    # Then publish is called with JSON payload and templated topic
    assert fake_client.username == "user"
    assert fake_client.password == "pass"
    assert len(fake_client.published) == 1
    topic, payload, qos, retain = fake_client.published[0]
    assert topic == "alerts/front"
    assert '"clip_id":"clip_1"' in payload
    assert qos == config.qos
    assert retain == config.retain


@pytest.mark.asyncio
async def test_mqtt_notifier_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """MQTT notifier should raise when connection times out."""
    # Given a notifier whose client never connects
    fake_client = _FakeClientNoConnect()
    monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
    config = MQTTConfig(host="localhost", connection_timeout=0.01)
    notifier = MQTTNotifier(config)

    # When sending an alert
    with pytest.raises(RuntimeError):
        # Then a connection timeout error is raised
        await notifier.send(_sample_alert())


class TestMQTTNotifierPing:
    """Tests for ping method."""

    @pytest.mark.asyncio
    async def test_ping_returns_true_when_connected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True when MQTT client is connected."""
        # Given: A connected MQTT notifier
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")
        notifier = MQTTNotifier(config)

        # When: Calling ping
        result = await notifier.ping()

        # Then: Returns True
        assert result is True

    @pytest.mark.asyncio
    async def test_ping_returns_false_after_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False after shutdown is called."""
        # Given: A notifier that has been shut down
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")
        notifier = MQTTNotifier(config)
        await notifier.shutdown()

        # When: Calling ping
        result = await notifier.ping()

        # Then: Returns False
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_waits_for_connection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ping waits for connection event when not yet connected."""
        # Given: A notifier that hasn't connected yet
        fake_client = _FakeClientNoConnect()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost", connection_timeout=0.01)
        notifier = MQTTNotifier(config)

        # When: Calling ping before connection
        result = await notifier.ping()

        # Then: Returns False (connection never established)
        assert result is False


class TestMQTTNotifierShutdown:
    """Tests for shutdown method."""

    @pytest.mark.asyncio
    async def test_shutdown_disconnects_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown disconnects the MQTT client."""
        # Given: A connected notifier
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")
        notifier = MQTTNotifier(config)
        assert fake_client.connected is True

        # When: Shutting down
        await notifier.shutdown()

        # Then: Client is disconnected
        assert fake_client.connected is False

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shutdown can be called multiple times safely."""
        # Given: A connected notifier
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")
        notifier = MQTTNotifier(config)

        # When: Calling shutdown multiple times
        await notifier.shutdown()
        await notifier.shutdown()
        await notifier.shutdown()

        # Then: No exception raised
        assert notifier._shutdown_called is True

    @pytest.mark.asyncio
    async def test_send_fails_after_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Send raises RuntimeError after shutdown."""
        # Given: A notifier that has been shut down
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")
        notifier = MQTTNotifier(config)
        await notifier.shutdown()

        # When/Then: Send raises RuntimeError
        with pytest.raises(RuntimeError, match="shut down"):
            await notifier.send(_sample_alert())


class TestMQTTNotifierConnection:
    """Tests for connection handling."""

    @pytest.mark.asyncio
    async def test_connection_failure_in_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Handles connection failure during initialization."""
        # Given: A client that raises on connect

        class _FakeClientConnectFails(_FakeClient):
            def connect(
                self,
                _host: str,
                _port: int,
                _keepalive: int | None = None,
                **_kwargs: Any,
            ) -> None:
                raise OSError("Connection refused")

        fake_client = _FakeClientConnectFails()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")

        # When: Creating notifier (connection fails)
        notifier = MQTTNotifier(config)

        # Then: Notifier is created but not connected
        assert notifier._connected is False

    @pytest.mark.asyncio
    async def test_on_connect_failure_sets_not_connected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Connection callback with non-zero rc sets not connected."""
        # Given: A client that calls on_connect with error code

        class _FakeClientConnectError(_FakeClient):
            def connect(
                self,
                _host: str,
                _port: int,
                _keepalive: int | None = None,
                **_kwargs: Any,
            ) -> None:
                self.connected = False
                if self.on_connect is not None:
                    self.on_connect(self, None, {}, 5)  # rc=5 = not authorized

        fake_client = _FakeClientConnectError()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")

        # When: Creating notifier
        notifier = MQTTNotifier(config)

        # Then: Not connected due to error code
        assert notifier._connected is False

    @pytest.mark.asyncio
    async def test_unexpected_disconnect_sets_not_connected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unexpected disconnect callback sets not connected."""
        # Given: A connected notifier
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost")
        notifier = MQTTNotifier(config)
        assert notifier._connected is True

        # When: Unexpected disconnect occurs (rc != 0)
        assert fake_client.on_disconnect is not None
        fake_client.on_disconnect(fake_client, None, 1)  # rc=1 = unexpected

        # Then: Not connected
        assert notifier._connected is False


class TestMQTTNotifierAuth:
    """Tests for authentication handling."""

    @pytest.mark.asyncio
    async def test_no_auth_when_not_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No authentication when auth config is None."""
        # Given: A config without auth
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost", auth=None)

        # When: Creating notifier
        notifier = MQTTNotifier(config)

        # Then: No credentials set
        assert notifier.username is None
        assert notifier.password is None
        assert fake_client.username is None
        assert fake_client.password is None

    @pytest.mark.asyncio
    async def test_warning_when_username_env_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logs warning when username env var is not set."""
        # Given: Auth config with env var that doesn't exist
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        monkeypatch.delenv("MISSING_USER_VAR", raising=False)
        config = MQTTConfig(
            host="localhost",
            auth=MQTTAuthConfig(username_env="MISSING_USER_VAR"),
        )

        # When: Creating notifier
        with caplog.at_level("WARNING"):
            notifier = MQTTNotifier(config)

        # Then: Warning logged and username is None
        assert notifier.username is None
        assert "MISSING_USER_VAR" in caplog.text

    @pytest.mark.asyncio
    async def test_warning_when_password_env_missing(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Logs warning when password env var is not set."""
        # Given: Auth config with password env var that doesn't exist
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        monkeypatch.setenv("MQTT_USER", "user")
        monkeypatch.delenv("MISSING_PASS_VAR", raising=False)
        config = MQTTConfig(
            host="localhost",
            auth=MQTTAuthConfig(username_env="MQTT_USER", password_env="MISSING_PASS_VAR"),
        )

        # When: Creating notifier
        with caplog.at_level("WARNING"):
            notifier = MQTTNotifier(config)

        # Then: Warning logged and password is None
        assert notifier.username == "user"
        assert notifier.password is None
        assert "MISSING_PASS_VAR" in caplog.text

    @pytest.mark.asyncio
    async def test_credentials_not_set_when_partial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client credentials not set when only one is available."""
        # Given: Only username is set, password is missing
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        monkeypatch.setenv("MQTT_USER", "user")
        monkeypatch.delenv("MQTT_PASS", raising=False)
        config = MQTTConfig(
            host="localhost",
            auth=MQTTAuthConfig(username_env="MQTT_USER", password_env="MQTT_PASS"),
        )

        # When: Creating notifier
        MQTTNotifier(config)

        # Then: Client credentials not set (both required)
        assert fake_client.username is None
        assert fake_client.password is None


class TestMQTTNotifierConfig:
    """Tests for configuration handling."""

    @pytest.mark.asyncio
    async def test_custom_qos_and_retain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom QoS and retain settings are used."""
        # Given: A config with custom QoS and retain
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost", qos=2, retain=True)
        notifier = MQTTNotifier(config)

        # When: Sending an alert
        await notifier.send(_sample_alert())

        # Then: Custom settings are used
        _, _, qos, retain = fake_client.published[0]
        assert qos == 2
        assert retain is True

    @pytest.mark.asyncio
    async def test_topic_template_formatting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Topic template is formatted with alert fields."""
        # Given: A config with custom topic template
        fake_client = _FakeClient()
        monkeypatch.setattr("homesec.plugins.notifiers.mqtt.mqtt.Client", lambda: fake_client)
        config = MQTTConfig(host="localhost", topic_template="homesec/cameras/{camera_name}/alerts")
        notifier = MQTTNotifier(config)

        # When: Sending an alert
        await notifier.send(_sample_alert())

        # Then: Topic is formatted correctly
        topic, _, _, _ = fake_client.published[0]
        assert topic == "homesec/cameras/front/alerts"
