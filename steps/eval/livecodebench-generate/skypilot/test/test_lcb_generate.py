# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generate step's bundled Python (``../src/lcb_generate.py``).

These run in Mode 1 only, under the repo venv, which has no LiveCodeBench, vLLM or
transformers. LiveCodeBench's two library functions the script calls are stubbed
with copies of their documented behaviour, so the tests pin what this step owns:
reasoning handling, the empty-extraction count, and prompt assembly.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"


def _install_lcb_stubs():
    """Minimal stand-ins for the LiveCodeBench modules the script imports."""
    lm_styles = types.ModuleType("lcb_runner.lm_styles")
    lm_styles.LMStyle = types.SimpleNamespace(OpenAIChat="OpenAIChat")

    extraction = types.ModuleType("lcb_runner.utils.extraction_utils")

    def extract_code(model_output, lmstyle):
        # LiveCodeBench's non-GenericBase path: text between the last two ``` lines.
        lines = model_output.split("\n")
        idx = [i for i, line in enumerate(lines) if "```" in line]
        if len(idx) < 2:
            return ""
        return "\n".join(lines[idx[-2] + 1 : idx[-1]])

    extraction.extract_code = extract_code

    prompts = types.ModuleType("lcb_runner.prompts.code_generation")
    prompts.PromptConstants = types.SimpleNamespace(SYSTEM_MESSAGE_GENERIC="SYS")
    prompts.get_generic_question_template_answer = lambda p: f"Q:{p.question_id}"

    for name, mod in {
        "lcb_runner": types.ModuleType("lcb_runner"),
        "lcb_runner.utils": types.ModuleType("lcb_runner.utils"),
        "lcb_runner.prompts": types.ModuleType("lcb_runner.prompts"),
        "lcb_runner.lm_styles": lm_styles,
        "lcb_runner.utils.extraction_utils": extraction,
        "lcb_runner.prompts.code_generation": prompts,
    }.items():
        sys.modules.setdefault(name, mod)


def _load(name):
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    spec = importlib.util.spec_from_file_location(name, _SRC / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_install_lcb_stubs()
gen = _load("lcb_generate")
FENCE = "```python\n{}\n```"


def _problem(qid):
    return types.SimpleNamespace(question_id=qid)


@pytest.mark.parametrize(
    "prompt, opens",
    [
        (
            "<|im_start|>assistant\n<think>\n",
            True,
        ),  # thinking on: the prompt opens the block
        (
            "<|im_start|>assistant\n<think></think>",
            False,
        ),  # thinking off: opened and closed
        ("<|im_start|>assistant\n", False),  # no reasoning in the template
    ],
)
def test_prompt_opens_reasoning(prompt, opens):
    assert gen.prompt_opens_reasoning(prompt) is opens


def test_code_after_think_wins_over_drafts():
    out = (
        "draft:\n"
        + FENCE.format("print('draft')")
        + "\n</think>\nanswer:\n"
        + FENCE.format("print('final')")
    )
    assert gen.extract(out, reasoning_open=True) == "print('final')"


def test_cut_off_reasoning_has_no_answer():
    # The prompt opened <think>; no </think> came back: the model never answered.
    out = "thinking...\n" + FENCE.format("print('draft')")
    assert gen.extract(out, reasoning_open=True) == ""


def test_no_reasoning_template_keeps_the_fence():
    assert (
        gen.extract("Here:\n" + FENCE.format("x = 1"), reasoning_open=False) == "x = 1"
    )


def test_unclosed_think_in_output_has_no_answer():
    assert gen.extract("<think>\n" + FENCE.format("x = 1"), reasoning_open=False) == ""


def test_no_fence_is_empty():
    assert gen.extract("I cannot solve this.") == ""


def test_assemble_counts_empty_extractions():
    probs = [_problem("a"), _problem("b")]
    outs = [
        [FENCE.format("print(1)"), "no code"],
        ["</think>" + "\n" + FENCE.format("print(2)")] * 2,
    ]
    completions, raw, stats = gen.assemble(probs, outs)
    assert completions == [
        {"question_id": "a", "code_list": ["print(1)", ""]},
        {"question_id": "b", "code_list": ["print(2)", "print(2)"]},
    ]
    assert raw[0]["output_list"][1] == "no code"
    assert stats == {
        "problems": 2,
        "samples": 4,
        "empty_extractions": 1,
        "empty_fraction": 0.25,
    }


def test_assemble_refuses_a_short_output_list():
    with pytest.raises(RuntimeError, match="1 of 2"):
        gen.assemble([_problem("a"), _problem("b")], [["x"]])


def test_messages_use_lcb_generic_prompt():
    assert gen.build_messages(_problem("q1")) == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q:q1"},
    ]


class _Tok:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append(kw)
        return "rendered"


@pytest.mark.parametrize(
    "flag, passed",
    [
        (None, {}),
        (False, {"enable_thinking": False}),
        (True, {"enable_thinking": True}),
    ],
)
def test_enable_thinking_is_passed_only_when_set(flag, passed):
    tok = _Tok()
    assert gen.render_prompts(tok, [_problem("a")], flag) == ["rendered"]
    assert tok.calls == [{"tokenize": False, "add_generation_prompt": True, **passed}]


@pytest.mark.parametrize(
    "value, parsed", [("", None), ("true", True), ("False", False), ("0", False)]
)
def test_parse_bool(value, parsed):
    assert gen.parse_bool(value) is parsed


def test_parse_bool_rejects_other_values():
    with pytest.raises(ValueError, match="enable_thinking"):
        gen.parse_bool("maybe")


def test_hash_matches_the_score_step():
    """The score step recomputes this hash; both copies must agree."""
    spec = importlib.util.spec_from_file_location(
        "lcb_score_for_hash",
        _SRC.parents[2] / "livecodebench-score" / "skypilot" / "src" / "lcb_score.py",
    )
    other = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(other)
    ids = ["b", "a", "c"]
    assert gen.question_ids_sha256(ids) == other.question_ids_sha256(ids)
