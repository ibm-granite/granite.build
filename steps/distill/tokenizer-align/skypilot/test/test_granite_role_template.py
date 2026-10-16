"""The granite-native ``{% generation %}`` template, and the one property it must have.

``align_config.chat_template`` installs a template on the retagged student, and the
exported model carries it — so ``full-eval`` renders its prompts with it. A granite-4.0
model distilled under a template that renders *differently* from the one its SFT stage
used would be scored out of format, and the whole point of comparing against a recorded
after-SFT row would be lost.

``{% generation %}`` / ``{% endgeneration %}`` are a transformers extension that emit
nothing: they only mark the span ``return_assistant_tokens_mask`` reports. So a template
derived from a working one by adding *only* those markers is render-identical, and this
module asserts exactly that, by stripping the markers back out and diffing the output
over a set of conversations that exercise every branch of the message loop.

Plain jinja2 cannot parse ``{% generation %}`` — that is why the comparison is done on a
marker-stripped copy rather than by rendering the marked template directly. The
environment settings mirror ``transformers._compile_jinja_template``.
"""

import hashlib
import re
from pathlib import Path

import jinja2
import jinja2.ext
import pytest

_TESTDATA = Path(__file__).resolve().parent.parent / "test-data"
_BASE = _TESTDATA / "granite_4_role_base.jinja"
_MARKED = _TESTDATA / "granite_4_role_generation.jinja"

# The SFT mixture's own chat template, verified on the cluster 2026-09-22:
#
#     sha256sum /proj/granite-build/g4os/chat_template.jinja
#     9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce
#
# That is byte-identical to the vendored base below, which is the Hub's copy of
# granite-4.0-350m's chat_template.jinja. It is the fact the whole comparison rests on:
# the distilled student must render prompts the way the model behind the recorded
# after-SFT eval row did, and full-eval renders with the model's OWN template.
#
# Pinned as a hash rather than described in prose because the base is a vendored copy
# of a file that lives somewhere else. Editing it — or re-vendoring from a different
# granite release — silently changes what the generation-marked template is derived
# FROM, and every render-identity test below would still pass while comparing the new
# file against itself.
_SFT_TEMPLATE_SHA256 = (
    "9524df67b77a7b25a2dfee898f75b316a157eb9d855b51e32aeac79d7c8a83ce"
)

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

# One case per branch of the message loop, plus the two content shapes.
_CASES = {
    "user_only_with_generation_prompt": dict(
        messages=[{"role": "user", "content": "hi"}], add_generation_prompt=True
    ),
    "user_assistant": dict(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        add_generation_prompt=False,
    ),
    "explicit_system_multiturn": dict(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"},
        ],
        add_generation_prompt=False,
    ),
    "default_system_message_is_injected": dict(
        # The branch that distinguishes the 350m template from the 3b teacher's.
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        add_generation_prompt=True,
    ),
    "assistant_with_tool_calls": dict(
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}
                ],
            },
            {"role": "tool", "content": "21C"},
            {"role": "assistant", "content": "21 degrees"},
        ],
        tools=[_TOOL],
        add_generation_prompt=False,
    ),
    "assistant_content_and_tool_calls": dict(
        messages=[
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "let me check",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Rome"}',
                        }
                    }
                ],
            },
        ],
        tools=[_TOOL],
        add_generation_prompt=False,
    ),
    "consecutive_tool_messages": dict(
        messages=[
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": {"city": "Rome"}}}
                ],
            },
            {"role": "tool", "content": "a1"},
            {"role": "tool", "content": "a2"},
            {"role": "assistant", "content": "done"},
        ],
        tools=[_TOOL],
        add_generation_prompt=False,
    ),
    "documents": dict(
        messages=[
            {"role": "user", "content": "summarise"},
            {"role": "assistant", "content": "ok"},
        ],
        documents=[{"title": "t", "text": "body"}],
        add_generation_prompt=False,
    ),
    "list_content": dict(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "c"}]},
        ],
        add_generation_prompt=False,
    ),
}


def _env():
    # Mirrors transformers._compile_jinja_template.
    return jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )


def _strip_markers(text):
    """Remove the generation tags, leaving everything else byte-for-byte.

    The tags sit alone on their own lines with a whitespace-control dash, and
    lstrip_blocks/trim_blocks means such a line contributes nothing to the output —
    so dropping the whole line is the correct inverse of adding it.
    """
    out = re.sub(r"^[ \t]*\{%-\s*(?:end)?generation\s*%\}\n", "", text, flags=re.M)
    assert "generation %}" not in out.replace("add_generation_prompt", "")
    return out


@pytest.fixture(scope="module")
def base_tpl():
    return _env().from_string(_BASE.read_text())


@pytest.fixture(scope="module")
def marked_stripped_tpl():
    return _env().from_string(_strip_markers(_MARKED.read_text()))


class TestRenderIdentity:
    @pytest.mark.parametrize("name", sorted(_CASES))
    def test_marked_template_renders_identically(
        self, name, base_tpl, marked_stripped_tpl
    ):
        """The comparison against a recorded after-SFT eval row depends on this."""
        kwargs = _CASES[name]
        assert marked_stripped_tpl.render(**kwargs) == base_tpl.render(**kwargs)

    def test_the_cases_actually_exercise_the_assistant_branch(self, base_tpl):
        """A render-identity suite that never emits an assistant turn proves nothing."""
        for name, kwargs in _CASES.items():
            if any(m["role"] == "assistant" for m in kwargs["messages"]):
                assert "<|start_of_role|>assistant<|end_of_role|>" in base_tpl.render(
                    **kwargs
                ), name

    def test_the_default_system_message_branch_is_covered(self, base_tpl):
        """This is the branch where the student's template and the 3b teacher's differ,
        so deriving from the teacher's would silently change every prompt."""
        rendered = base_tpl.render(**_CASES["default_system_message_is_injected"])
        assert (
            "You are a helpful assistant. Please ensure responses are professional"
            in rendered
        )


class TestMarkerPlacement:
    @pytest.fixture(scope="class")
    def marked(self):
        return _MARKED.read_text()

    def test_exactly_one_generation_block(self, marked):
        assert len(re.findall(r"\{%-\s*generation\s*%\}", marked)) == 1
        assert len(re.findall(r"\{%-\s*endgeneration\s*%\}", marked)) == 1

    def test_the_block_opens_after_the_role_header(self, marked):
        """The turn header is prompt, not label. Inside the span it would be trained on."""
        head, tail = marked.split("{%- generation %}", 1)
        assert head.rstrip().endswith(
            "{{- '<|start_of_role|>' + message.role + '<|end_of_role|>' }}"
        )

    def test_the_block_closes_after_the_stop_token(self, marked):
        """<|end_of_text|> must be INSIDE the span, mirroring the ChatML reference
        template. Outside it, the student never learns to stop — that is retag v1
        (run-align.sh's post-condition comment)."""
        body = marked.split("{%- generation %}", 1)[1].split("{%- endgeneration %}", 1)[
            0
        ]
        assert "'<|end_of_text|>\\n'" in body

    def test_the_block_is_inside_the_assistant_branch_only(self, marked):
        body = marked.split("{%- generation %}", 1)[1].split("{%- endgeneration %}", 1)[
            0
        ]
        for role in ("'user'", "'system'", "'tool'"):
            assert f"message.role == {role}" not in body

    def test_tool_calls_are_inside_the_span(self, marked):
        """A tool call is something the assistant produced, so it is a label."""
        body = marked.split("{%- generation %}", 1)[1].split("{%- endgeneration %}", 1)[
            0
        ]
        assert "message.tool_calls" in body
        assert "<tool_call>" in body


class TestTheBaseIsTheSftTemplate:
    """The render-identity suite proves the marked template matches the base. This
    proves the BASE is the right file — the one the SFT mixture was rendered with."""

    def test_the_vendored_base_matches_the_clusters_sft_template(self):
        digest = hashlib.sha256(_BASE.read_bytes()).hexdigest()
        assert digest == _SFT_TEMPLATE_SHA256, (
            "the vendored base no longer matches /proj/granite-build/g4os/chat_template.jinja "
            "as verified 2026-09-22. Re-derive granite_4_role_generation.jinja from the "
            "current file and update this hash, or the distilled student will render "
            "prompts differently from the model behind the recorded eval row."
        )

    def test_the_only_edit_is_the_two_markers_and_one_split_emission(self):
        """Structural, complementing the render-identity suite above.

        The markers cannot be inserted without touching an existing line: the block
        must open BETWEEN the role header and the content, and the base emits both in
        a single expression. So the derivation is exactly (a) split that emission in
        two and (b) add the two marker lines — three added lines, one removed.

        Asserting the shape of that edit is what a render-identity pass alone cannot
        do: it would tolerate a compensating change elsewhere in the template that
        happens to produce the same output on the nine cases sampled here.
        """
        import difflib

        base = _BASE.read_text().splitlines()
        stripped = _strip_markers(_MARKED.read_text()).splitlines()

        removed = [l[1:] for l in difflib.ndiff(base, stripped) if l.startswith("- ")]
        added = [l[1:] for l in difflib.ndiff(base, stripped) if l.startswith("+ ")]
        assert len(removed) == 1, f"expected one line replaced, got {removed}"
        assert len(added) == 2, f"expected it split in two, got {added}"

        # The split must preserve the emission: header first, content second, and
        # nothing dropped between them.
        assert "'<|start_of_role|>' + message.role + '<|end_of_role|>'" in added[0]
        assert "content.val" in added[1]
        assert (
            "content.val" not in added[0]
        ), "content is inside the span, not before it"
        assert "+ content.val" in removed[0], "the base should emit both together"

        # And the marker lines themselves are the only other difference.
        assert len(_MARKED.read_text().splitlines()) - len(base) == 3
