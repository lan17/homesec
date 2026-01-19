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

from homesec.interfaces import Notifier
from homesec.models.alert import Alert
from homesec.models.config import MQTTConfig
from homesec.plugins.registry import PluginType, plugin

logger = logging.getLogger(__name__)


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
                self._connected_event.set()
                logger.info("MQTTNotifier connected: %s:%d", self.host, self.port)
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

    def _publish(self, topic: str, payload: str, qos: int, retain: bool) -> None:
        """Publish message (blocking operation)."""
        result = self.client.publish(topic, payload, qos=qos, retain=retain)
        result.wait_for_publish()

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
