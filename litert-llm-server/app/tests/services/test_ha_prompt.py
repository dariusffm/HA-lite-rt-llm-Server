"""Parsing and rendering of Home Assistant's Assist prompt fragments."""

from __future__ import annotations

import json

import yaml

from litert_server.services.ha_prompt import (
    FILTER_NOTE,
    STATIC_MARKER,
    compact_live_context,
    entity_areas,
    entity_names,
    render_static_context,
    split_static_context,
)

HEAD = "You are a voice assistant for Home Assistant.\nAnswer truthfully.\n"
ENTITIES_YAML = (
    "- names: Wohnzimmer Lampe\n"
    "  domain: light\n"
    "  areas: Wohnzimmer\n"
    "- names: Bad Temperatur\n"
    "  domain: sensor\n"
    "  areas: Bad\n"
    "  aliases:\n"
    "  - Badthermometer\n"
    "- names: Küche Steckdose\n"
    "  domain: switch\n"
)
PROMPT = HEAD + STATIC_MARKER + "\n" + ENTITIES_YAML


def test_split_static_context_returns_head_entities_tail():
    ctx = split_static_context(PROMPT)

    assert ctx is not None
    assert ctx.head == HEAD
    assert [e["domain"] for e in ctx.entities] == ["light", "sensor", "switch"]
    assert ctx.tail == ""


def test_split_static_context_keeps_text_after_the_list_as_tail():
    ctx = split_static_context(PROMPT + "Answer in German.\n")

    assert ctx is not None
    assert ctx.tail == "Answer in German.\n"
    assert len(ctx.entities) == 3


def test_split_static_context_returns_none_without_marker():
    assert split_static_context("You are a helpful assistant.") is None


def test_split_static_context_returns_none_for_broken_yaml():
    assert split_static_context(HEAD + STATIC_MARKER + "\n- names: [unclosed\n") is None


def test_render_static_context_keeps_head_and_tail_and_adds_note():
    ctx = split_static_context(PROMPT + "Answer in German.\n")
    assert ctx is not None

    out = render_static_context(ctx, [ctx.entities[0]])

    assert out.startswith(HEAD + STATIC_MARKER + "\n" + FILTER_NOTE + "\n")
    assert out.endswith("Answer in German.\n")
    assert "Wohnzimmer Lampe" in out and "Bad Temperatur" not in out
    assert split_static_context(out) is not None  # still parseable


def test_render_static_context_round_trips_a_blank_line_before_the_tail():
    """MINOR 2: spec §5 requires the rest of the prompt to stay
    character-identical; a blank line between the entity list and the tail
    (e.g. a following instruction paragraph) must not be swallowed."""
    entities = [{"names": "Wohnzimmer Lampe", "domain": "light", "areas": "Wohnzimmer"}]
    body = yaml.safe_dump(entities, allow_unicode=True, sort_keys=False, default_flow_style=False)
    text = HEAD + STATIC_MARKER + "\n" + FILTER_NOTE + "\n" + body + "\nAnswer in German.\n"

    ctx = split_static_context(text)
    assert ctx is not None

    out = render_static_context(ctx, ctx.entities)

    assert out == text


def test_render_static_context_with_no_entities_omits_the_yaml_list():
    """A stage-1 selection of zero entities must not render as a bare
    ``[]`` line — that reads as broken output inside the prompt."""
    ctx = split_static_context(PROMPT + "Answer in German.\n")
    assert ctx is not None

    out = render_static_context(ctx, [])

    assert out == HEAD + STATIC_MARKER + "\n" + FILTER_NOTE + "\nAnswer in German.\n"
    assert "[]" not in out


def test_entity_names_includes_aliases_and_entity_areas_accepts_str_or_list():
    e = {"names": "Bad Temperatur", "aliases": ["Badthermometer"], "areas": "Bad"}

    assert entity_names(e) == ["Bad Temperatur", "Badthermometer"]
    assert entity_areas(e) == ["Bad"]
    assert entity_areas({"names": "x", "areas": ["A", "B"]}) == ["A", "B"]
    assert entity_areas({"names": "x"}) == []


LIVE = (
    "Live Context: An overview of the areas and the devices in this smart home:\n"
    "- names: Wohnzimmer Lampe\n  domain: light\n  state: 'on'\n  areas: Wohnzimmer\n"
    "  attributes:\n    brightness: 180\n"
    "- names: Bad Temperatur\n  domain: sensor\n  state: '21.4'\n  areas: Bad\n"
)


def test_compact_live_context_rewrites_one_line_per_entity():
    out = compact_live_context(LIVE)

    assert out == (
        "Live Context (compact):\n"
        "Wohnzimmer Lampe [light, Wohnzimmer]: on, brightness=180\n"
        "Bad Temperatur [sensor, Bad]: 21.4"
    )


def test_compact_live_context_handles_ha_json_envelope():
    content = json.dumps({"success": True, "result": LIVE})

    out = compact_live_context(content)

    assert out is not None
    data = json.loads(out)
    assert data["success"] is True
    assert data["result"].startswith("Live Context (compact):\n")


def test_compact_live_context_translates_binary_sensor_window_on_to_open():
    live = (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: GaesteWC-Window\n  domain: binary_sensor\n  state: 'on'\n"
        "  attributes:\n    device_class: window\n"
    )

    out = compact_live_context(live)

    assert out == ("Live Context (compact):\nGaesteWC-Window [binary_sensor]: open")


def test_compact_live_context_translates_binary_sensor_motion_off_to_clear():
    live = (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: Flur Bewegung\n  domain: binary_sensor\n  state: 'off'\n"
        "  attributes:\n    device_class: motion\n"
    )

    out = compact_live_context(live)

    assert out == ("Live Context (compact):\nFlur Bewegung [binary_sensor]: clear")


def test_compact_live_context_translated_binary_sensor_keeps_other_attributes():
    """Only the now-redundant ``device_class`` is dropped from the tail —
    other attributes on a translated binary_sensor are kept."""
    live = (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: GaesteWC-Window\n  domain: binary_sensor\n  state: 'on'\n"
        "  attributes:\n    device_class: window\n    battery_level: 87\n"
    )

    out = compact_live_context(live)

    assert out == (
        "Live Context (compact):\nGaesteWC-Window [binary_sensor]: open, battery_level=87"
    )


def test_compact_live_context_leaves_unavailable_binary_sensor_state_as_is():
    live = (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: GaesteWC-Window\n  domain: binary_sensor\n  state: unavailable\n"
        "  attributes:\n    device_class: window\n"
    )

    out = compact_live_context(live)

    assert out == (
        "Live Context (compact):\nGaesteWC-Window [binary_sensor]: unavailable, device_class=window"
    )


def test_compact_live_context_leaves_binary_sensor_on_off_as_is_for_unknown_device_class():
    live = (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: Mystery Sensor\n  domain: binary_sensor\n  state: 'on'\n"
        "  attributes:\n    device_class: unknown_class\n"
    )

    out = compact_live_context(live)

    assert out == (
        "Live Context (compact):\nMystery Sensor [binary_sensor]: on, device_class=unknown_class"
    )


def test_compact_live_context_leaves_binary_sensor_on_off_as_is_without_device_class():
    live = (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: No Class Sensor\n  domain: binary_sensor\n  state: 'on'\n"
    )

    out = compact_live_context(live)

    assert out == "Live Context (compact):\nNo Class Sensor [binary_sensor]: on"


def test_compact_live_context_translates_unquoted_yaml_boolean_binary_sensor_state():
    live = (
        "Live Context: An overview of the areas and the devices in this smart home:\n"
        "- names: GaesteWC-Window\n  domain: binary_sensor\n  state: on\n"
        "  attributes:\n    device_class: door\n"
    )

    out = compact_live_context(live)

    assert out == ("Live Context (compact):\nGaesteWC-Window [binary_sensor]: open")


def test_compact_live_context_returns_none_for_other_tool_results():
    assert compact_live_context("The weather is sunny.") is None
    assert compact_live_context(json.dumps({"success": True, "result": "done"})) is None


def test_compact_live_context_returns_none_for_broken_yaml():
    assert compact_live_context("Live Context: An overview…\n- names: [oops\n") is None


def test_split_static_context_returns_none_for_unexpected_line_before_yaml():
    # Marker followed by unexpected text (not FILTER_NOTE, not blank) before the YAML block
    prompt_with_foreign_text = (
        HEAD + STATIC_MARKER + "\nSome other text\n- names: Wohnzimmer Lampe\n  domain: light\n"
    )
    assert split_static_context(prompt_with_foreign_text) is None


def test_filter_note_is_one_line_and_names_sensor_domain():
    # One line: _split_yaml_block skips the note by exact line match.
    assert "\n" not in FILTER_NOTE
    assert "domain sensor" in FILTER_NOTE
