"""Stage-1 relevance routing: prompt, parsing, selection — no model involved."""

from __future__ import annotations

import re

from litert_server.services.relevance import (
    STAGE_ONE_PARAMS,
    STAGE_ONE_PATTERN,
    RelevanceQuery,
    available_areas,
    available_domains,
    build_stage_one_turns,
    parse_stage_one,
    select_entities,
)

ENTITIES = [
    {"names": "Wohnzimmer Lampe", "domain": "light", "areas": "Wohnzimmer"},
    {"names": "Bad Temperatur", "domain": "sensor", "areas": "Bad", "aliases": ["Badthermometer"]},
    {"names": "Küche Steckdose", "domain": "switch"},
    {"names": "Fernseher", "domain": "media_player", "areas": ["Wohnzimmer"]},
]


def test_available_domains_and_areas_are_sorted_and_unique():
    assert available_domains(ENTITIES) == ["light", "media_player", "sensor", "switch"]
    assert available_areas(ENTITIES) == ["Bad", "Wohnzimmer"]


def test_build_stage_one_turns_lists_domains_and_areas_and_question():
    turns = build_stage_one_turns("Welche Lampen sind an?", ["light", "switch"], ["Bad"])

    assert [t.role for t in turns] == ["system", "user"]
    assert "Domains available: light, switch" in turns[0].content
    assert "Areas available: Bad" in turns[0].content
    assert turns[1].content == "Welche Lampen sind an?"


def test_build_stage_one_turns_prefixes_the_previous_question_when_given():
    turns = build_stage_one_turns(
        "Und welche davon sind gerade eingeschaltet?",
        ["light", "switch"],
        ["Bad"],
        previous_question="Welche Geräte gibt es in der Küche?",
    )

    assert turns[1].content == (
        "Previous question: Welche Geräte gibt es in der Küche?\n"
        "Question: Und welche davon sind gerade eingeschaltet?"
    )


def test_build_stage_one_turns_only_permits_empty_lists_for_non_home_questions():
    """Since stage-1's empty selection compacts to zero entities (no full-prompt
    fallback), the model must not read an empty list as a safe default for an
    unsure home question — only for questions that aren't about the home."""
    turns = build_stage_one_turns("Welche Lampen sind an?", ["light", "switch"], ["Bad"])

    assert "Use an empty list when unsure." not in turns[0].content
    assert (
        "Return empty lists only if the question is not about the home "
        "(small talk, math, general knowledge)." in turns[0].content
    )


def test_stage_one_params_force_pattern_and_determinism():
    assert STAGE_ONE_PARAMS.temperature == 0.0
    assert STAGE_ONE_PARAMS.max_tokens == 96
    assert STAGE_ONE_PARAMS.response_pattern == STAGE_ONE_PATTERN


def test_stage_one_pattern_accepts_compact_json_and_rejects_whitespace():
    good = '{"domains":["light","switch"],"areas":[],"names":["Rollläden"]}'
    assert re.fullmatch(STAGE_ONE_PATTERN, good)
    assert re.fullmatch(STAGE_ONE_PATTERN, '{"domains":[],"areas":[],"names":[]}')
    assert not re.fullmatch(STAGE_ONE_PATTERN, '{"domains": ["light"],"areas":[],"names":[]}')
    assert not re.fullmatch(STAGE_ONE_PATTERN, '{"domains":["a]"],"areas":[],"names":[]}')


def test_parse_stage_one_lowercases_and_drops_blanks():
    q = parse_stage_one('{"domains": ["Light", ""], "areas": ["bad"], "names": []}')

    assert q == RelevanceQuery(
        domains=frozenset({"light"}), areas=frozenset({"bad"}), names=frozenset()
    )


def test_parse_stage_one_returns_none_for_invalid_input():
    assert parse_stage_one("not json") is None
    assert parse_stage_one('["light"]') is None
    assert parse_stage_one('{"domains": "light", "areas": [], "names": []}') is None


def test_parse_stage_one_repairs_truncated_output_and_treats_missing_lists_as_empty():
    # whitespace padding + missing closing brace (JSON-schema-mode failure seen in the spike)
    q = parse_stage_one('{"domains":["light"],"areas":[]\n\t\n\t')
    assert q == RelevanceQuery(domains=frozenset({"light"}), areas=frozenset(), names=frozenset())
    # unterminated string inside a list
    q = parse_stage_one('{"domains":["climate"],"areas":["Bad')
    assert q == RelevanceQuery(
        domains=frozenset({"climate"}), areas=frozenset({"bad"}), names=frozenset()
    )


def test_parse_stage_one_preserves_spaces_in_multi_word_areas_and_names():
    q = parse_stage_one('{"domains":[],"areas":["Bad Oben"],"names":["Bad Temperatur"]}')

    assert q == RelevanceQuery(
        domains=frozenset(), areas=frozenset({"bad oben"}), names=frozenset({"bad temperatur"})
    )
    entity = {"names": "Bad Temperatur", "domain": "sensor", "areas": "Bad Oben"}
    assert select_entities([entity], q) == [entity]


def test_parse_stage_one_empty_lists_is_empty_query():
    q = parse_stage_one('{"domains": [], "areas": [], "names": []}')
    assert q == RelevanceQuery(domains=frozenset(), areas=frozenset(), names=frozenset())


def test_select_entities_matches_domain_or_area_or_name_substring():
    q = RelevanceQuery(
        domains=frozenset({"light"}),
        areas=frozenset({"bad"}),
        names=frozenset({"fernseh"}),
    )

    picked = select_entities(ENTITIES, q)

    assert [e["names"] for e in picked] == ["Wohnzimmer Lampe", "Bad Temperatur", "Fernseher"]


def test_select_entities_matches_alias_case_insensitively():
    q = RelevanceQuery(domains=frozenset(), areas=frozenset(), names=frozenset({"badthermo"}))

    assert [e["names"] for e in select_entities(ENTITIES, q)] == ["Bad Temperatur"]


def test_select_entities_preserves_order_and_returns_empty_on_no_match():
    q = RelevanceQuery(domains=frozenset({"cover"}), areas=frozenset(), names=frozenset())
    assert select_entities(ENTITIES, q) == []
