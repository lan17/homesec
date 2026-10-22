"""MQTT notifier plugin for Home Assistant integration."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any

mqtt: Any

try:
    import paho.mqtt.client as _mqtt
except Exception:
    mqtt = None
else:
    mqtt = _mqtt

from pydantic import BaseModel

from homesec.interfaces import Notifier
from homesec.models.alert import Alert
from homesec.plugins.notifiers.mqtt_home_assistant import (
    AVAILABLE_PAYLOAD,
    NOT_AVAILABLE_PAYLOAD,
    HomeAssistantConfig,
    build_discovery,
    build_event_payload,
    event_topic,
)
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)

_SHUTDOWN_PUBLISH_TIMEOUT_S = 2.0
_HOME_ASSISTANT_PUBLISH_TIMEOUT_S = 10.0
_DISCOVERY_CONNECTION_ATTEMPTS = 3


class MQTTAuthConfig(BaseModel):
    """MQTT auth configuration using env var names."""

    username_env: str | None = None
    password_env: str | None = None


class MQTTConfig(BaseModel):
    """MQTT notifier configuration."""

    host: str
    port: int = 1883
    auth: MQTTAuthConfig | None = None
    topic_template: str = "homecam/alerts/{camera_name}"
    qos: int = 1
    retain: bool = False
    connection_timeout: float = 10.0
    home_assistant: HomeAssistantConfig | None = None


@plugin(plugin_type=PluginType.NOTIFIER, name="mqtt")
class MQTTNotifier(Notifier):
    """MQTT notifier for Home Assistant alerts.

    Publishes alert messages to configured topics with QoS settings.
    """

    config_cls = MQTTConfig

    @classmethod
    def create(cls, config: MQTTConfig) -> Notifier:
        return cls(config)

    def __init__(self, config: MQTTConfig) -> None:
        """Initialize MQTT notifier with config validation.

        Args:
            config: MQTTConfig instance
        """
        if mqtt is None:
            raise RuntimeError(
                "Missing dependency: paho-mqtt. Install with: uv pip install paho-mqtt"
            )

        self.host = config.host
        self.port = int(config.port)
        self.topic_template = config.topic_template
        self.qos = int(config.qos)
        self.retain = bool(config.retain)
        self.connection_timeout = float(config.connection_timeout)
        self._home_assistant = config.home_assistant
        self._discovered_cameras: dict[str, int] = {}
        self._connection_epoch = 0
        self._discovery_locks: dict[str, asyncio.Lock] = {}

        # Get credentials from env if provided
        self.username: str | None = None
        self.password: str | None = None

        if config.auth and config.auth.username_env:
            username_var = config.auth.username_env
            self.username = os.getenv(username_var)
            if not self.username:
                logger.warning("MQTT username not found in env: %s", username_var)

        if config.auth and config.auth.password_env:
            password_var = config.auth.password_env
            self.password = os.getenv(password_var)
            if not self.password:
                logger.warning("MQTT password not found in env: %s", password_var)

        # Initialize MQTT client
        self.client = mqtt.Client()

        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)

        if self._home_assistant is not None:
            self.client.will_set(
                self._home_assistant.availability_topic,
                NOT_AVAILABLE_PAYLOAD,
                qos=1,
                retain=True,
            )

        # Connection state
        self._connected = False
        self._connected_event = threading.Event()
        self._shutdown_called = False
        self._loop_started = False

        def _on_connect(
            client: mqtt.Client, userdata: object, flags: dict[str, object], rc: int
        ) -> None:
            if rc == 0:
                self._connected = True
                self._connection_epoch += 1
                self._connected_event.set()
                logger.info("MQTTNotifier connected: %s:%d", self.host, self.port)
                if self._home_assistant is not None:
                    # Must not wait for the publish: this runs on the client's network thread.
                    client.publish(
                        self._home_assistant.availability_topic,
                        AVAILABLE_PAYLOAD,
                        qos=1,
                        retain=True,
                    )
                return
            self._connected = False
            logger.warning("MQTTNotifier connection failed: rc=%s", rc)

        def _on_disconnect(client: mqtt.Client, userdata: object, rc: int) -> None:
            self._connected = False
            if rc != 0:
                logger.warning("MQTTNotifier disconnected unexpectedly: rc=%s", rc)

        self.client.on_connect = _on_connect
        self.client.on_disconnect = _on_disconnect

        # Connect to broker
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            self._loop_started = True
        except Exception as e:
            logger.error("Failed to connect to MQTT broker: %s", e, exc_info=True)
            self._connected = False

    async def send(self, alert: Alert) -> None:
        """Send alert notification to MQTT topic."""
        await self._ensure_connected()

        # Format topic
        topic = self.topic_template.format(camera_name=alert.camera_name)

        # Serialize alert to JSON
        payload = alert.model_dump_json()

        # Publish message
        await asyncio.to_thread(
            self._publish,
            topic,
            payload,
            self.qos,
            self.retain,
        )

        logger.info(
            "Published alert to MQTT: topic=%s, clip_id=%s",
            topic,
            alert.clip_id,
        )

        if self._home_assistant is not None:
            await self._send_home_assistant_event(alert, self._home_assistant)

    async def _send_home_assistant_event(
        self, alert: Alert, home_assistant: HomeAssistantConfig
    ) -> None:
        """Publish the Home Assistant event for an alert, discovering the camera first."""
        await self._ensure_discovery(alert.camera_name, home_assistant)
        topic = event_topic(alert.camera_name, home_assistant)
        payload = build_event_payload(alert).model_dump_json()
        await asyncio.to_thread(
            self._publish, topic, payload, self.qos, False, _HOME_ASSISTANT_PUBLISH_TIMEOUT_S
        )
        logger.info(
            "Published Home Assistant event",
            extra={"mqtt_topic": topic, "recording_id": alert.clip_id},
        )

    async def _ensure_discovery(
        self, camera_name: str, home_assistant: HomeAssistantConfig
    ) -> None:
        lock = self._discovery_locks.setdefault(camera_name, asyncio.Lock())
        async with lock:
            # Do not cache an announcement made on a connection that disappeared
            # during publication or settling. Other cameras can settle independently.
            for _ in range(_DISCOVERY_CONNECTION_ATTEMPTS):
                epoch = self._connection_epoch
                if self._discovered_cameras.get(camera_name) == epoch:
                    return
                await self._ensure_connected()
                topic, discovery = build_discovery(camera_name, home_assistant)
                await asyncio.to_thread(
                    self._publish,
                    topic,
                    discovery.model_dump_json(),
                    1,
                    True,
                    _HOME_ASSISTANT_PUBLISH_TIMEOUT_S,
                )
                await asyncio.sleep(home_assistant.discovery_settle_seconds)
                if epoch == self._connection_epoch:
                    self._discovered_cameras[camera_name] = epoch
                    return
            raise RuntimeError("MQTT connection changed repeatedly during Home Assistant discovery")

    def _publish(
        self, topic: str, payload: str, qos: int, retain: bool, timeout: float | None = None
    ) -> None:
        """Publish message (blocking operation)."""
        result = self.client.publish(topic, payload, qos=qos, retain=retain)
        if timeout is None:
            result.wait_for_publish()
        else:
            result.wait_for_publish(timeout)
            if not result.is_published():
                raise TimeoutError("MQTT publish acknowledgement timed out")

    async def ping(self) -> bool:
        """Health check - verify MQTT connection."""
        if self._shutdown_called:
            return False
        if self._connected and self.client.is_connected():
            return True
        await asyncio.to_thread(self._connected_event.wait, 2.0)
        return self._connected and self.client.is_connected()

    async def shutdown(self, timeout: float | None = None) -> None:
        """Cleanup resources - disconnect from broker."""
        _ = timeout
        if self._shutdown_called:
            return

        self._shutdown_called = True
        logger.info("Shutting down MQTTNotifier...")

        # A clean disconnect does not trigger the last will, so announce offline explicitly.
        if self._home_assistant is not None and self._connected:
            try:
                await asyncio.to_thread(
                    self._publish,
                    self._home_assistant.availability_topic,
                    NOT_AVAILABLE_PAYLOAD,
                    1,
                    True,
                    _SHUTDOWN_PUBLISH_TIMEOUT_S,
                )
            except Exception as exc:
                logger.warning("Failed to publish MQTT offline availability", exc_info=exc)

        # Stop loop if it was started (prevents thread leak even if never connected)
        if self._loop_started:
            await asyncio.to_thread(self.client.loop_stop)
            await asyncio.to_thread(self.client.disconnect)

        logger.info("MQTTNotifier shutdown complete")

    async def _ensure_connected(self) -> None:
        if self._shutdown_called:
            raise RuntimeError("Notifier has been shut down")
        if self._connected:
            return
        # Wait for connection with timeout
        connected = await asyncio.to_thread(self._connected_event.wait, self.connection_timeout)
        if not connected or not self._connected:
            raise RuntimeError(
                f"MQTT broker not connected after {self.connection_timeout}s timeout"
            )
