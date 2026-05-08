"""Built-in talk backend registry construction."""

from __future__ import annotations

from homesec.sources.rtsp.talk.backend import onvif_rtsp_talk_backend_registration
from homesec.talk.backends import TalkBackendRegistry


def build_default_talk_backend_registry() -> TalkBackendRegistry:
    """Build the built-in talk backend registry."""
    registry = TalkBackendRegistry()
    registry.register(onvif_rtsp_talk_backend_registration())
    return registry
