"""Built-in talk backend registry construction."""

from __future__ import annotations

from homesec.talk.backends import TalkBackendRegistry


def build_default_talk_backend_registry() -> TalkBackendRegistry:
    """Build the built-in talk backend registry."""
    from homesec.sources.rtsp.talk.backend import onvif_rtsp_talk_backend_registration
    from homesec.talk.tapo.backend import tapo_local_talk_backend_registration

    registry = TalkBackendRegistry()
    registry.register(onvif_rtsp_talk_backend_registration())
    registry.register(tapo_local_talk_backend_registration())
    return registry
