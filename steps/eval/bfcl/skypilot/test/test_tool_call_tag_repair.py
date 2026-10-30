import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from tool_call_tag_repair import (  # noqa: E402
    is_well_formed_tool_call,
    patch_granite_handler,
    repair_unclosed_tool_calls,
)

# Verbatim malformed samples from the granite-4.2-30b-fp8 full run (see
# tool_call_tag_repair.py's module docstring for the root-cause writeup).
CLEAN_EARLY_STOP = '\n<tool_call>\n{"name": "get_user_info", "arguments": {"user_id": 7890, "special": "black"}}\n'
SECOND_TAG_OPENED_INSTEAD_OF_CLOSE = (
    '\n<tool_call>\n{"name": "github_star", "arguments": '
    '{"repos": "ShishirPatil/gorilla,gorilla-llm/gorilla-cli", "aligned": true}}\n<tool_call>\n'
)
DRIFTS_INTO_TEMPLATE_TOKENS = (
    '\n<tool_call>\n{"name": "uber.ride", "arguments": '
    '{"loc": "2020 Addison Street, Berkeley, CA, USA", "type": "comfort", "time": 600}}\n'
    "<tool_call>\n<|start_of_role|>user<parameter>follow_up_1</parameter><|"
)
ALREADY_WELL_FORMED = '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>'
NO_TOOL_CALL = "just some plain text response"
OPENED_BUT_NOT_JSON = "<tool_call>\nnot json at all"

# Back-to-back tags with no real close in between: repair closes the first
# tag right after its own balanced JSON object, which here is a fragment
# that parses as JSON but isn't a `{"name": ..., "arguments": ...}` call --
# see is_well_formed_tool_call's docstring / the module docstring's second
# fix. The second, real call is unaffected.
MALFORMED_FRAGMENT_FOLLOWED_BY_REAL_CALL = (
    '<tool_call>\n{"arguments": {"foo": "bar"}}\n'
    '<tool_call>\n{"name": "real_func", "arguments": {}}\n</tool_call>'
)


def _extract_via_original_regex(text: str) -> list:
    import json
    import re

    matches = re.findall(r"<tool_call>\n(.*?)\n</tool_call>", text, re.DOTALL)
    result = []
    for match in matches:
        try:
            result.append(json.loads(match))
        except Exception:
            pass
    return result


def test_inserts_missing_close_after_clean_early_stop():
    repaired = repair_unclosed_tool_calls(CLEAN_EARLY_STOP)
    assert repaired.rstrip("\n").endswith("</tool_call>")
    calls = _extract_via_original_regex(repaired)
    assert calls == [
        {"name": "get_user_info", "arguments": {"user_id": 7890, "special": "black"}}
    ]


def test_recovers_first_call_when_second_tag_opened_instead_of_closing():
    calls = _extract_via_original_regex(
        repair_unclosed_tool_calls(SECOND_TAG_OPENED_INSTEAD_OF_CLOSE)
    )
    assert calls == [
        {
            "name": "github_star",
            "arguments": {
                "repos": "ShishirPatil/gorilla,gorilla-llm/gorilla-cli",
                "aligned": True,
            },
        }
    ]


def test_recovers_call_when_generation_drifts_into_template_tokens():
    calls = _extract_via_original_regex(
        repair_unclosed_tool_calls(DRIFTS_INTO_TEMPLATE_TOKENS)
    )
    assert calls == [
        {
            "name": "uber.ride",
            "arguments": {
                "loc": "2020 Addison Street, Berkeley, CA, USA",
                "type": "comfort",
                "time": 600,
            },
        }
    ]


def test_leaves_already_well_formed_call_unchanged():
    assert repair_unclosed_tool_calls(ALREADY_WELL_FORMED) == ALREADY_WELL_FORMED


def test_leaves_plain_text_unchanged():
    assert repair_unclosed_tool_calls(NO_TOOL_CALL) == NO_TOOL_CALL


def test_does_not_fabricate_a_call_when_no_json_follows_the_tag():
    repaired = repair_unclosed_tool_calls(OPENED_BUT_NOT_JSON)
    assert _extract_via_original_regex(repaired) == []


def test_is_well_formed_tool_call():
    assert is_well_formed_tool_call({"name": "a", "arguments": {}}) is True
    assert (
        is_well_formed_tool_call({"name": "a", "arguments": "json string ok too"})
        is True
    )
    assert is_well_formed_tool_call({"arguments": {"foo": "bar"}}) is False  # no name
    assert is_well_formed_tool_call({"name": "a"}) is False  # no arguments
    assert (
        is_well_formed_tool_call({"name": "", "arguments": {}}) is False
    )  # empty name
    assert (
        is_well_formed_tool_call({"name": 7, "arguments": {}}) is False
    )  # non-string name
    assert is_well_formed_tool_call("not a dict") is False
    assert is_well_formed_tool_call(["not", "a", "dict"]) is False


def test_repair_can_close_a_shape_malformed_fragment_when_a_real_call_follows():
    # repair_unclosed_tool_calls alone doesn't validate shape -- that's the
    # extractor wrapper's job (next test). This just confirms the raw
    # fragment survives repair and both objects come through the original
    # unmodified regex/json.loads path, one malformed and one not.
    repaired = repair_unclosed_tool_calls(MALFORMED_FRAGMENT_FOLLOWED_BY_REAL_CALL)
    calls = _extract_via_original_regex(repaired)
    assert calls == [
        {"arguments": {"foo": "bar"}},
        {"name": "real_func", "arguments": {}},
    ]


def test_patch_granite_handler_recovers_calls_through_the_real_extractor():
    # The harness is a dependency of this step's image, not of this repository (see
    # test/conftest.py) -- everything above exercises repair_unclosed_tool_calls without
    # it; only the two tests that patch the REAL handler need it present.
    granite_4 = pytest.importorskip("bfcl_eval.model_handler.local_inference.granite_4")
    Granite4FCHandler = granite_4.Granite4FCHandler

    original = Granite4FCHandler._extract_tool_calls
    try:
        patch_granite_handler(Granite4FCHandler)
        assert Granite4FCHandler._extract_tool_calls(CLEAN_EARLY_STOP) == [
            {
                "name": "get_user_info",
                "arguments": {"user_id": 7890, "special": "black"},
            }
        ]
        assert Granite4FCHandler._extract_tool_calls(NO_TOOL_CALL) == []
    finally:
        Granite4FCHandler._extract_tool_calls = original


def test_patch_granite_handler_drops_shape_malformed_dicts_that_would_otherwise_crash_the_next_turn():
    granite_4 = pytest.importorskip("bfcl_eval.model_handler.local_inference.granite_4")
    Granite4FCHandler = granite_4.Granite4FCHandler

    original = Granite4FCHandler._extract_tool_calls
    try:
        patch_granite_handler(Granite4FCHandler)
        # Without the shape filter this would include the nameless
        # {"arguments": {"foo": "bar"}} fragment, which later crashes
        # _format_prompt with KeyError: 'name' on the following turn.
        assert Granite4FCHandler._extract_tool_calls(
            MALFORMED_FRAGMENT_FOLLOWED_BY_REAL_CALL
        ) == [{"name": "real_func", "arguments": {}}]
    finally:
        Granite4FCHandler._extract_tool_calls = original
