"""Structural checks for the Home Assistant blueprint shipped with HomeSec."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from homesec.models.enums import RiskLevel
from homesec.plugins.notifiers.mqtt_home_assistant import UNANALYZED_EVENT_TYPE

BLUEPRINT = (
    Path(__file__).resolve().parents[2]
    / "integrations"
    / "home-assistant"
    / "homesec_alert_notification.yaml"
)


class _InputRef(str):
    """Marks a value that the blueprint takes from a declared input."""


class _BlueprintLoader(yaml.SafeLoader):
    pass


_BlueprintLoader.add_constructor(
    "!input",
    lambda loader, node: _InputRef(loader.construct_scalar(node)),  # type: ignore[arg-type]
)


def _load() -> dict[str, Any]:
    loaded = yaml.load(BLUEPRINT.read_text(encoding="utf-8"), Loader=_BlueprintLoader)  # noqa: S506
    assert isinstance(loaded, dict)
    return loaded


def _input_refs(node: object) -> set[str]:
    if isinstance(node, _InputRef):
        return {str(node)}
    if isinstance(node, dict):
        return set().union(*(_input_refs(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_input_refs(v) for v in node))
    return set()


@pytest.fixture(scope="module")
def blueprint() -> dict[str, Any]:
    return _load()


def test_blueprint_declares_an_automation_blueprint(blueprint: dict[str, Any]) -> None:
    # Given: The shipped blueprint file
    # When: Reading its metadata
    meta = blueprint["blueprint"]

    # Then: It is an automation blueprint with a name, description, and minimum version
    assert meta["domain"] == "automation"
    assert meta["name"]
    assert meta["description"]
    assert meta["homeassistant"]["min_version"]


def test_every_referenced_input_is_declared(blueprint: dict[str, Any]) -> None:
    # Given: The inputs declared in the blueprint header
    declared = set(blueprint["blueprint"]["input"])

    # When: Collecting every !input reference in the automation body
    used = _input_refs({k: v for k, v in blueprint.items() if k != "blueprint"})

    # Then: All references resolve and no declared input is unused
    assert used <= declared
    assert declared <= used


def test_risk_options_match_homesec_risk_levels(blueprint: dict[str, Any]) -> None:
    # Given: The blueprint's minimum-risk selector and the code's risk levels
    options = blueprint["blueprint"]["input"]["minimum_risk"]["selector"]["select"]["options"]
    default = blueprint["blueprint"]["input"]["minimum_risk"]["default"]

    # When/Then: Options are exactly the risk level names, lowest first, with a valid default
    assert options == [str(level) for level in sorted(RiskLevel)]
    assert default in options


def test_blueprint_handles_the_unanalyzed_event_type(blueprint: dict[str, Any]) -> None:
    # Given: The blueprint's conditions
    conditions = str(blueprint["conditions"])

    # When/Then: It branches on the same unanalyzed type the notifier publishes
    assert UNANALYZED_EVENT_TYPE in conditions


def test_blueprint_sends_a_mobile_app_notification(blueprint: dict[str, Any]) -> None:
    # Given: The blueprint's actions
    notify = next(a for a in blueprint["actions"] if a.get("type") == "notify")

    # When/Then: It targets the mobile_app device and replaces repeated notifications per clip
    assert notify["domain"] == "mobile_app"
    assert isinstance(notify["device_id"], _InputRef)
    assert "clip_id" in notify["data"]["tag"]
