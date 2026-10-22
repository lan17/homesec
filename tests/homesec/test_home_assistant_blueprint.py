"""Structure and rendered behavior of the shipped Home Assistant blueprint."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from jinja2 import Environment, StrictUndefined

from homesec.models.enums import RiskLevel
from homesec.plugins.notifiers.mqtt_home_assistant import UNANALYZED_EVENT_TYPE

BLUEPRINT = (
    Path(__file__).resolve().parents[2]
    / "integrations"
    / "home-assistant"
    / "homesec_alert_notification.yaml"
)
NOW = datetime(2026, 10, 21, 12, tzinfo=timezone.utc)


def _as_datetime(value: object, default: object = None) -> object:
    """The documented HA conversion subset needed by this blueprint's inputs."""
    if not isinstance(value, str):
        return default
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return default


def _environment() -> Environment:
    env = Environment(undefined=StrictUndefined)
    env.globals.update(now=lambda: NOW, as_datetime=_as_datetime)
    return env


def _context(risk: str = "high", minimum: str = "medium") -> dict[str, Any]:
    attributes = {
        "event_type": risk,
        "clip_id": "clip-new",
        "ts": NOW.isoformat(),
        "camera_name": "Front Door",
        "summary": "Person at the door",
        "notify_reason": "person detected",
        "view_url": "https://example.org/clip",
    }
    return {
        "trigger": SimpleNamespace(
            from_state=SimpleNamespace(state="unknown", attributes={}),
            to_state=SimpleNamespace(state=NOW.isoformat(), attributes=attributes),
        ),
        "minimum_risk": minimum,
        "notify_unanalyzed": True,
        "risk_order": ["low", "medium", "high", "critical"],
    }


def _allows_notification(blueprint: dict[str, Any], context: dict[str, Any]) -> bool:
    env = _environment()
    return all(
        env.from_string(condition["value_template"]).render(**context).strip() == "True"
        for condition in blueprint["conditions"]
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


@pytest.mark.parametrize("minimum", ["low", "medium", "high", "critical"])
@pytest.mark.parametrize("risk", ["low", "medium", "high", "critical"])
def test_risk_threshold_is_inclusive(blueprint: dict[str, Any], minimum: str, risk: str) -> None:
    # Given: Every combination of configured threshold and incoming risk.
    context = _context(risk, minimum)

    # When: Rendering the shipped conditions rather than reimplementing the comparison.
    allowed = _allows_notification(blueprint, context)

    # Then: Only risks at or above the selected threshold pass.
    assert allowed == (RiskLevel.from_string(risk) >= RiskLevel.from_string(minimum))


@pytest.mark.parametrize("enabled", [True, False])
def test_unanalyzed_respects_the_option(blueprint: dict[str, Any], enabled: bool) -> None:
    # Given: An unanalyzed event, independently of the highest risk threshold.
    context = _context("unanalyzed", "critical")
    context["notify_unanalyzed"] = enabled

    # When: Rendering the blueprint's decision.
    allowed = _allows_notification(blueprint, context)

    # Then: The option alone controls this event type.
    assert allowed is enabled


def test_unknown_event_type_is_ignored(blueprint: dict[str, Any]) -> None:
    # Given: An event outside the declared risk taxonomy.
    context = _context("unexpected")

    # When: Rendering the conditions.
    allowed = _allows_notification(blueprint, context)

    # Then: The event cannot trigger a notification or an invalid index lookup.
    assert allowed is False


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (NOW.isoformat(), True),
        ((NOW - timedelta(seconds=299)).isoformat(), True),
        ((NOW - timedelta(seconds=300)).isoformat(), False),
        ((NOW - timedelta(minutes=10)).isoformat(), False),
        ((NOW + timedelta(seconds=1)).isoformat(), False),
        (NOW.astimezone(timezone(timedelta(hours=9))).isoformat(), True),
        (NOW.isoformat().replace("+00:00", "Z"), True),
        (NOW.replace(tzinfo=None).isoformat(), False),
        ("not-a-date", False),
        (None, False),
    ],
)
def test_freshness_uses_payload_time_and_fails_closed(
    blueprint: dict[str, Any], timestamp: str | None, expected: bool
) -> None:
    # Given: HA just received an event whose payload time may be old or malformed.
    context = _context()
    context["trigger"].to_state.attributes["ts"] = timestamp

    # When: Evaluating freshness with a fixed clock and the real shipped template.
    allowed = _allows_notification(blueprint, context)

    # Then: Receipt time cannot refresh an old alert; invalid/ambiguous times fail closed.
    assert allowed is expected


@pytest.mark.parametrize(
    ("from_state", "to_state", "old_clip", "expected"),
    [
        (None, "new", "clip-old", False),
        ("unknown", "new", None, True),
        ("unavailable", "new", "clip-new", False),
        ("unavailable", "new", "clip-old", True),
        ("new", "new", "clip-new", False),
        ("old", "new", "clip-old", True),
        ("old", "unavailable", "clip-old", False),
        ("old", "unknown", "clip-old", False),
        ("old", None, "clip-old", False),
    ],
)
def test_state_transitions_require_a_new_event(
    blueprint: dict[str, Any],
    from_state: str | None,
    to_state: str | None,
    old_clip: str | None,
    expected: bool,
) -> None:
    # Given: Availability recovery, restoration, attribute-only changes, or a real new event.
    context = _context()
    trigger = context["trigger"]
    trigger.from_state = (
        None
        if from_state is None
        else SimpleNamespace(state=from_state, attributes={"clip_id": old_clip})
    )
    if to_state is None:
        trigger.to_state = None
    else:
        trigger.to_state.state = to_state

    # When: Evaluating the shipped guard against that state transition.
    allowed = _allows_notification(blueprint, context)

    # Then: Recovered old events are skipped, while first/new events remain deliverable.
    assert allowed is expected


@pytest.mark.parametrize("risk", ["low", "high", "critical", "unanalyzed"])
def test_rendered_push_contains_summary_clip_link_tag_and_urgency(
    blueprint: dict[str, Any], risk: str
) -> None:
    # Given: The real action templates and the alert details exposed by HA's event entity.
    env = _environment()
    attributes = _context(risk)["trigger"].to_state.attributes
    variables = {
        "attributes": attributes,
        "event_type": risk,
        "camera": attributes["camera_name"],
        "urgent": risk in {"high", "critical"},
    }
    action = blueprint["actions"][-1]

    # When: Rendering the fields sent to the companion app.
    def render(template: str) -> str:
        return env.from_string(template).render(**variables)

    # Then: Summary, link and clip identity survive, with urgency only for high/critical.
    assert render(action["message"]) == "Person at the door"
    assert render(action["title"]) == (
        f"Front Door: {'alert' if risk == 'unanalyzed' else risk + ' risk'}"
    )
    assert render(action["data"]["url"]) == attributes["view_url"]
    assert render(action["data"]["clickAction"]) == attributes["view_url"]
    assert render(action["data"]["tag"]) == "homesec-clip-new"
    assert render(action["data"]["push"]["interruption-level"]) == (
        "time-sensitive" if variables["urgent"] else "active"
    )


@pytest.mark.parametrize(
    ("summary", "reason", "expected"),
    [(None, "person detected", "person detected"), ("", "", "Camera alert")],
)
def test_push_has_readable_fallbacks_without_summary_or_link(
    blueprint: dict[str, Any], summary: str | None, reason: str, expected: str
) -> None:
    # Given: An alert with no uploaded link and an empty or missing analysis summary.
    attributes = _context()["trigger"].to_state.attributes
    attributes.update(summary=summary, notify_reason=reason, view_url=None)
    action = blueprint["actions"][-1]
    env = _environment()

    # When: Rendering the notification fields.
    message = env.from_string(action["message"]).render(attributes=attributes)
    url = env.from_string(action["data"]["url"]).render(attributes=attributes)

    # Then: The message remains useful and the link opens the Home Assistant root.
    assert message == expected
    assert url == "/"
