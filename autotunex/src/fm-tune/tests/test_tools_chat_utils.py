"""Tests for autotune.tools.chat_utils — chat-template plumbing and prompts.

The Granite generations consume ``documents=`` incompatibly and
``apply_chat_template`` never raises when a template ignores the kwarg, so these
tests pin the per-template behaviour with template stand-ins (no model download).
"""

import json
from unittest.mock import MagicMock

import pytest

from autotune.tools.chat_utils import (
    _GUARDIAN_SPECS,
    ATOMIZER_CRITERIA_ID,
    ATOMIZER_TEXT,
    CRITERIA,
    CRITERIA_ID,
    DOC_STYLE_GRANITE3,
    DOC_STYLE_GRANITE4,
    DOC_STYLE_INLINE,
    DOC_STYLE_TOOL,
    FACT_CORRECTION_GUARDIAN_TEXT,
    FACT_CORRECTION_GUARDIAN_TEXT_JSON,
    FACT_DETECTION_GUARDIAN_TEXT,
    FACT_DETECTION_GUARDIAN_TEXT_JSON,
    NLI_CRITERIA_ID,
    NLI_TEXT,
    IdentityTokenizer,
    attach_documents,
    build_documents,
    detect_document_style,
    documents_ride_in_messages,
    extract_assistant_text,
    fact_guardian_text,
    get_atomizer_messages,
    get_atomizer_prompt,
    get_correction_messages,
    get_detection_messages,
    get_detection_prompt,
    get_nli_messages,
    get_nli_prompt,
    guardian_text,
    inline_documents_message,
    insert_documents_as_tool,
    is_chatml_template,
    normalize_document,
    num_tokens,
    output_name_for,
    print_token_stats,
    render_chat_input,
    resolve_formats,
    split_documents,
    summarize,
    template_supports_kwarg,
    tokenizer_document_style,
    tokenizer_is_chatml,
    tool_documents_message,
    uses_json_schema,
)

# Minimal stand-ins for the discriminating parts of each real template.
GRANITE3_TEMPLATE = "{%- if documents %}{{- document['doc_id'] }}{{- document['text'] }}{%- endif %}"
GRANITE4_TEMPLATE = "{%- if documents %}{%- set x = x + (document | tojson) %}{%- endif %}"
GRANITE42_TEMPLATE = "{%- if enable_thinking %}<|im_start|>assistant<think>{%- endif %}"
PLAIN_TEMPLATE = "{%- for message in messages %}{{ message.content }}{%- endfor %}"


def _tok(template):
    """A tokenizer stand-in that records the kwargs it was rendered with."""
    tok = MagicMock()
    tok.chat_template = template
    tok.apply_chat_template.return_value = "RENDERED"
    return tok


class TestDetectDocumentStyle:
    def test_granite3_reads_doc_id(self):
        assert detect_document_style(GRANITE3_TEMPLATE) == DOC_STYLE_GRANITE3

    def test_granite4_uses_tojson(self):
        assert detect_document_style(GRANITE4_TEMPLATE) == DOC_STYLE_GRANITE4

    def test_granite42_uses_tool_message(self):
        # 4.2 dropped documents= and moved to ChatML -> documents ride in a tool message.
        assert detect_document_style(GRANITE42_TEMPLATE) == DOC_STYLE_TOOL

    def test_chatml_wins_over_document_substrings(self):
        # A 4.2 template that also mentions doc_id/documents must not fall through to
        # a native style, which would silently drop the context.
        assert detect_document_style(GRANITE42_TEMPLATE + GRANITE3_TEMPLATE) == DOC_STYLE_TOOL
        assert detect_document_style(GRANITE42_TEMPLATE + GRANITE4_TEMPLATE) == DOC_STYLE_TOOL

    def test_plain_template(self):
        assert detect_document_style(PLAIN_TEMPLATE) == DOC_STYLE_INLINE

    def test_missing_template(self):
        assert detect_document_style(None) == DOC_STYLE_INLINE
        assert detect_document_style("") == DOC_STYLE_INLINE

    def test_from_tokenizer(self):
        assert tokenizer_document_style(_tok(GRANITE4_TEMPLATE)) == DOC_STYLE_GRANITE4
        assert tokenizer_document_style(IdentityTokenizer()) == DOC_STYLE_INLINE


class TestTemplateSupportsKwarg:
    def test_present(self):
        assert template_supports_kwarg(GRANITE42_TEMPLATE, "enable_thinking") is True

    def test_absent(self):
        assert template_supports_kwarg(GRANITE4_TEMPLATE, "enable_thinking") is False

    def test_no_template(self):
        assert template_supports_kwarg(None, "enable_thinking") is False


class TestBuildDocuments:
    def test_granite3_includes_doc_id(self):
        assert build_documents(["a", "b"], DOC_STYLE_GRANITE3) == [
            {"doc_id": "0", "text": "a"},
            {"doc_id": "1", "text": "b"},
        ]

    def test_granite4_omits_doc_id(self):
        docs = build_documents(["a", "b"], DOC_STYLE_GRANITE4)
        assert docs == [{"text": "a"}, {"text": "b"}]
        assert all("doc_id" not in d for d in docs)

    def test_one_document_per_passage(self):
        assert len(build_documents(["a", "b", "c"], DOC_STYLE_GRANITE4)) == 3

    def test_tool_style_merges_all_passages_into_one(self):
        # 4.2 renders a tool message as one <tool_response> JSON array, so a single
        # merged document keeps the block equivalent to 4.1's one <documents> block.
        assert build_documents(["a", "b", "c"], DOC_STYLE_TOOL) == [{"text": "a\n\nb\n\nc"}]

    def test_tool_style_single_passage_unchanged(self):
        assert build_documents(["only"], DOC_STYLE_TOOL) == [{"text": "only"}]

    def test_tool_style_omits_doc_id(self):
        assert all("doc_id" not in d for d in build_documents(["a", "b"], DOC_STYLE_TOOL))

    def test_empty(self):
        assert build_documents([], DOC_STYLE_GRANITE4) == []
        assert build_documents([], DOC_STYLE_TOOL) == []


class TestSplitDocuments:
    def test_inline_style_returns_inline_only(self):
        inline, template = split_documents(["a"], DOC_STYLE_INLINE)
        assert inline == [{"text": "a"}]
        assert template is None

    def test_tool_style_returns_message_docs_only(self):
        message_docs, template = split_documents(["a"], DOC_STYLE_TOOL)
        assert message_docs == [{"text": "a"}]
        assert template is None

    def test_tool_style_merges_multiple_passages(self):
        message_docs, template = split_documents(["a", "b"], DOC_STYLE_TOOL)
        assert message_docs == [{"text": "a\n\nb"}]
        assert template is None

    def test_native_style_returns_template_only(self):
        inline, template = split_documents(["a"], DOC_STYLE_GRANITE4)
        assert inline == []
        assert template == [{"text": "a"}]


class TestChatmlDetection:
    """ChatML detection: template probe first, vocab as the fallback."""

    def test_template_marker(self):
        assert is_chatml_template(GRANITE42_TEMPLATE) is True
        assert is_chatml_template(GRANITE4_TEMPLATE) is False
        assert is_chatml_template(None) is False
        assert is_chatml_template("") is False

    def test_tokenizer_via_template(self):
        assert tokenizer_is_chatml(_tok(GRANITE42_TEMPLATE)) is True

    def test_tokenizer_via_vocab_only(self):
        # A real 4.2 tokenizer resolves even if the template text stops matching.
        tok = MagicMock()
        tok.chat_template = PLAIN_TEMPLATE
        tok.get_vocab.return_value = {"<|im_start|>": 100256, "<|im_end|>": 100257}
        assert tokenizer_is_chatml(tok) is True
        assert tokenizer_document_style(tok) == DOC_STYLE_TOOL

    def test_tokenizer_via_added_tokens(self):
        tok = MagicMock()
        tok.chat_template = PLAIN_TEMPLATE
        tok.get_vocab.return_value = {}
        tok.added_tokens_encoder = {"<|im_start|>": 100256}
        assert tokenizer_is_chatml(tok) is True

    def test_no_false_positive_on_unknown_token(self):
        # convert_tokens_to_ids would return the unk id (often 0) and false-positive;
        # membership testing must not.
        tok = MagicMock()
        tok.chat_template = GRANITE4_TEMPLATE
        tok.get_vocab.return_value = {"hello": 1, "world": 2}
        tok.added_tokens_encoder = {}
        tok.convert_tokens_to_ids.return_value = 0
        assert tokenizer_is_chatml(tok) is False
        assert tokenizer_document_style(tok) == DOC_STYLE_GRANITE4

    def test_vocab_failure_is_not_fatal(self):
        tok = MagicMock()
        tok.chat_template = GRANITE4_TEMPLATE
        tok.get_vocab.side_effect = RuntimeError("boom")
        tok.added_tokens_encoder = {}
        assert tokenizer_is_chatml(tok) is False

    def test_identity_tokenizer_is_not_chatml(self):
        assert tokenizer_is_chatml(IdentityTokenizer()) is False


class TestNormalizeDocument:
    def test_plain_string(self):
        assert normalize_document("PASSAGE", 3) == {
            "source": "knowledge_base",
            "document_id": "3",
            "content": "PASSAGE",
        }

    def test_text_key(self):
        assert normalize_document({"text": "P"}, 0)["content"] == "P"

    def test_content_key_and_doc_id(self):
        out = normalize_document({"content": "P", "doc_id": "994"}, 0)
        assert out["content"] == "P"
        assert out["document_id"] == "994"

    def test_blank_doc_id_falls_back_to_index(self):
        # An empty doc_id would otherwise render as an empty field in the prompt.
        assert normalize_document({"text": "P", "doc_id": ""}, 7)["document_id"] == "7"

    def test_custom_source_preserved(self):
        assert normalize_document({"text": "P", "source": "wiki"}, 0)["source"] == "wiki"


class TestToolDocumentsMessage:
    def test_role_and_json_payload(self):
        msg = tool_documents_message([{"text": "P-ONE"}, {"text": "P-TWO"}])
        assert msg["role"] == "tool"
        payload = json.loads(msg["content"])
        assert [d["content"] for d in payload] == ["P-ONE", "P-TWO"]
        assert [d["document_id"] for d in payload] == ["0", "1"]
        assert all(d["source"] == "knowledge_base" for d in payload)

    def test_content_is_pretty_printed(self):
        assert "\n" in tool_documents_message([{"text": "P"}])["content"]

    def test_merged_contexts_render_as_one_entry(self):
        # End to end for 4.2: raw contexts -> one array entry holding every passage.
        docs = build_documents(["P-ONE", "P-TWO", "P-THREE"], DOC_STYLE_TOOL)
        payload = json.loads(tool_documents_message(docs)["content"])
        assert len(payload) == 1
        assert payload[0]["content"] == "P-ONE\n\nP-TWO\n\nP-THREE"
        assert payload[0]["document_id"] == "0"

    def test_prebuilt_records_keep_their_own_entries(self):
        # Callers passing real {source, document_id, content} records keep one entry
        # each — merging would discard their ids and sources.
        payload = json.loads(
            tool_documents_message(
                [
                    {"content": "A", "document_id": "77", "source": "wiki"},
                    {"content": "B", "document_id": "88", "source": "web"},
                ]
            )["content"]
        )
        assert [d["document_id"] for d in payload] == ["77", "88"]
        assert [d["source"] for d in payload] == ["wiki", "web"]


class TestInsertDocumentsAsTool:
    JUDGE = [
        {"role": "user", "content": "q1_query"},
        {"role": "assistant", "content": "a1_response_being_judged"},
        {"role": "user", "content": "q2_guardian_instruction"},
    ]

    def test_before_first_assistant(self):
        out = insert_documents_as_tool(self.JUDGE, [{"text": "D"}])
        assert [m["role"] for m in out] == ["user", "tool", "assistant", "user"]

    def test_after_last_user_when_flag_off(self):
        out = insert_documents_as_tool(self.JUDGE, [{"text": "D"}], before_first_assistant=False)
        assert [m["role"] for m in out] == ["user", "assistant", "user", "tool"]

    def test_falls_back_when_no_assistant_turn(self):
        msgs = [{"role": "user", "content": "Q"}]
        out = insert_documents_as_tool(msgs, [{"text": "D"}])
        assert [m["role"] for m in out] == ["user", "tool"]

    def test_no_user_or_assistant_turn(self):
        out = insert_documents_as_tool([{"role": "system", "content": "S"}], [{"text": "D"}])
        assert [m["role"] for m in out] == ["system", "tool"]

    def test_does_not_mutate_input(self):
        msgs = list(self.JUDGE)
        insert_documents_as_tool(msgs, [{"text": "D"}])
        assert len(msgs) == 3


class TestAttachDocuments:
    MSGS = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "R"},
        {"role": "user", "content": "G"},
    ]

    def test_tool_style(self):
        out = attach_documents(self.MSGS, [{"text": "D"}], DOC_STYLE_TOOL)
        assert [m["role"] for m in out] == ["user", "tool", "assistant", "user"]

    def test_inline_style_unchanged(self):
        out = attach_documents(self.MSGS, [{"text": "D"}], DOC_STYLE_INLINE)
        assert out[0]["role"] == "system"
        assert "<documents>" in out[0]["content"]

    def test_native_styles_are_noop(self):
        for style in (DOC_STYLE_GRANITE3, DOC_STYLE_GRANITE4):
            assert attach_documents(self.MSGS, [{"text": "D"}], style) == self.MSGS

    def test_no_documents_is_noop(self):
        assert attach_documents(self.MSGS, [], DOC_STYLE_TOOL) == self.MSGS

    def test_ride_in_messages_predicate(self):
        assert documents_ride_in_messages(DOC_STYLE_TOOL) is True
        assert documents_ride_in_messages(DOC_STYLE_INLINE) is True
        assert documents_ride_in_messages(DOC_STYLE_GRANITE3) is False
        assert documents_ride_in_messages(DOC_STYLE_GRANITE4) is False


class TestInlineDocumentsMessage:
    def test_system_role_and_all_passages(self):
        msg = inline_documents_message([{"text": "P-ONE"}, {"text": "P-TWO"}])
        assert msg["role"] == "system"
        assert "P-ONE" in msg["content"] and "P-TWO" in msg["content"]
        assert "<documents>" in msg["content"]


class TestRenderChatInput:
    def test_forwards_documents_when_present(self):
        tok = _tok(GRANITE4_TEMPLATE)
        render_chat_input([{"role": "user", "content": "Q"}], tok, [{"text": "D"}])
        assert tok.apply_chat_template.call_args.kwargs["documents"] == [{"text": "D"}]

    def test_omits_documents_when_none(self):
        tok = _tok(GRANITE42_TEMPLATE)
        render_chat_input([{"role": "user", "content": "Q"}], tok, None)
        assert "documents" not in tok.apply_chat_template.call_args.kwargs

    def test_thinking_only_when_supported(self):
        tok42 = _tok(GRANITE42_TEMPLATE)
        render_chat_input([{"role": "user", "content": "Q"}], tok42)
        assert tok42.apply_chat_template.call_args.kwargs["enable_thinking"] is False

        tok4 = _tok(GRANITE4_TEMPLATE)
        render_chat_input([{"role": "user", "content": "Q"}], tok4)
        assert "enable_thinking" not in tok4.apply_chat_template.call_args.kwargs

    def test_thinking_can_be_enabled(self):
        tok = _tok(GRANITE42_TEMPLATE)
        render_chat_input([{"role": "user", "content": "Q"}], tok, thinking=True)
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is True


class TestFactualityPrompts:
    CTXS = ["PASSAGE-ALPHA", "PASSAGE-BETA"]

    def test_granite3_passes_doc_id(self):
        tok = _tok(GRANITE3_TEMPLATE)
        get_detection_prompt("Q", "R", self.CTXS, tok)
        docs = tok.apply_chat_template.call_args.kwargs["documents"]
        assert docs == [
            {"doc_id": "0", "text": "PASSAGE-ALPHA"},
            {"doc_id": "1", "text": "PASSAGE-BETA"},
        ]

    def test_granite4_omits_doc_id(self):
        tok = _tok(GRANITE4_TEMPLATE)
        get_detection_prompt("Q", "R", self.CTXS, tok)
        docs = tok.apply_chat_template.call_args.kwargs["documents"]
        assert docs == [{"text": "PASSAGE-ALPHA"}, {"text": "PASSAGE-BETA"}]

    def test_granite42_uses_tool_message_instead_of_kwarg(self):
        # 4.2 ignores documents=; the passages ride in a tool message instead, placed
        # before the response being judged. Eval prompts must match training.
        tok = _tok(GRANITE42_TEMPLATE)
        get_detection_prompt("Q", "R", self.CTXS, tok)
        kwargs = tok.apply_chat_template.call_args.kwargs
        assert "documents" not in kwargs
        messages = tok.apply_chat_template.call_args.args[0]
        assert [m["role"] for m in messages] == ["user", "tool", "assistant", "user"]
        assert all(c in messages[1]["content"] for c in self.CTXS)
        assert kwargs["enable_thinking"] is False

    def test_no_contexts_no_documents(self):
        tok = _tok(GRANITE4_TEMPLATE)
        get_detection_prompt("Q", "R", [], tok)
        assert "documents" not in tok.apply_chat_template.call_args.kwargs

    def test_messages_shape(self):
        msgs = get_detection_messages("Q", "R", self.CTXS)
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        # The messages path inlines contexts for plain chat APIs.
        assert "### Documents:" in msgs[-1]["content"]

    def test_detection_and_correction_schemas_differ(self):
        assert FACT_DETECTION_GUARDIAN_TEXT != FACT_CORRECTION_GUARDIAN_TEXT
        assert "'yes'" in FACT_DETECTION_GUARDIAN_TEXT
        assert "corrected version" in FACT_CORRECTION_GUARDIAN_TEXT
        corr = get_correction_messages("Q", "R", self.CTXS)
        assert "corrected version" in corr[-1]["content"]


class TestAtomizerPrompts:
    def test_three_turn_shape_with_query(self):
        msgs = get_atomizer_messages("RESP", "QUERY")
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[0]["content"] == "QUERY"
        assert msgs[1]["content"] == "RESP"
        assert msgs[2]["content"] == ATOMIZER_TEXT

    def test_two_turn_shape_without_query(self):
        msgs = get_atomizer_messages("RESP")
        assert [m["role"] for m in msgs] == ["assistant", "user"]

    def test_prompt_disables_thinking(self):
        tok = _tok(GRANITE42_TEMPLATE)
        get_atomizer_prompt("RESP", tok, query="Q")
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False


class TestNliPrompts:
    def test_single_user_turn_carries_both_texts(self):
        msgs = get_nli_messages("PREM", "HYP")
        assert [m["role"] for m in msgs] == ["user"]
        assert "PREM" in msgs[0]["content"] and "HYP" in msgs[0]["content"]

    def test_prompt_disables_thinking(self):
        tok = _tok(GRANITE42_TEMPLATE)
        get_nli_prompt("P", "H", tok)
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False


class TestExtractAssistantText:
    def test_from_message_list(self):
        assert extract_assistant_text([{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]) == "A"

    def test_plain_string(self):
        assert extract_assistant_text("TEXT") == "TEXT"

    def test_falls_back_to_first_message(self):
        assert extract_assistant_text([{"role": "user", "content": "ONLY"}]) == "ONLY"

    def test_rejects_other_types(self):
        with pytest.raises(ValueError):
            extract_assistant_text(42)


class TestStatsHelpers:
    def test_summarize_empty(self):
        assert summarize("x", []) == {"name": "x", "count": 0}

    def test_summarize_values(self):
        s = summarize("x", [1, 2, 3, 4])
        assert s["count"] == 4 and s["min"] == 1 and s["max"] == 4
        assert s["mean"] == 2.5

    def test_print_token_stats_handles_empty(self, capsys):
        print_token_stats([summarize("input", [])], "tag")
        assert "skipping token stats" in capsys.readouterr().out

    def test_print_token_stats_table(self, capsys):
        print_token_stats([summarize("input", [1, 2])], "tag")
        out = capsys.readouterr().out
        assert "token-length stats" in out and "input" in out

    def test_num_tokens(self):
        tok = MagicMock(return_value={"input_ids": [1, 2, 3]})
        assert num_tokens(tok, "abc") == 3
        assert tok.call_args.kwargs["add_special_tokens"] is False


class TestIdentityTokenizer:
    def test_raises_on_use(self):
        t = IdentityTokenizer()
        with pytest.raises(RuntimeError):
            t("x")
        with pytest.raises(RuntimeError):
            t.apply_chat_template([])


class TestFormatHelpers:
    def test_resolve_formats(self):
        assert resolve_formats("both") == ["formatted", "chat"]
        assert resolve_formats("chat") == ["chat"]


class TestOutputNameFor:
    def test_stem_plus_format(self):
        assert output_name_for("/a/b/raw_nli.json", "chat") == "raw_nli_chat.jsonl"

    def test_extra_parts_sit_between(self):
        assert output_name_for("/a/eli5_raw_train.json", "chat", "detection") == "eli5_raw_train_detection_chat.jsonl"

    def test_strips_directory_and_extension(self):
        assert output_name_for("/deep/path/bio.json", "formatted") == "bio_formatted.jsonl"

    def test_no_extension(self):
        assert output_name_for("raw", "chat") == "raw_chat.jsonl"

    def test_dotted_stem_keeps_leading_parts(self):
        assert output_name_for("a.b.json", "chat") == "a.b_chat.jsonl"


class TestGuardianCriteria:
    """The guardian block is assembled from the CRITERIA dict + a scoring schema."""

    def test_factuality_is_a_criteria_key(self):
        assert CRITERIA_ID == "factuality"
        assert "factuality" in CRITERIA
        assert "factually incorrect" in CRITERIA["factuality"]

    def test_criteria_text_is_embedded_in_the_block(self):
        block = guardian_text(variant="detection")
        assert CRITERIA["factuality"] in block

    def test_block_structure(self):
        block = guardian_text(variant="detection")
        # Preamble, criteria, and schema, in that order.
        assert block.startswith("<guardian>")
        assert block.index("### Criteria:") < block.index("### Scoring Schema:")

    def test_detection_schema(self):
        block = guardian_text(variant="detection")
        assert "'yes'" in block and "'no'" in block
        assert "corrected version" not in block

    def test_correction_schema(self):
        block = guardian_text(variant="correction")
        assert "corrected version" in block
        assert "'yes'" not in block

    def test_for_prompt_appends_json_hint(self):
        assert '"score"' in guardian_text(variant="detection", for_prompt=True)
        assert '"correction"' in guardian_text(variant="correction", for_prompt=True)

    def test_no_json_hint_by_default(self):
        assert '"score"' not in guardian_text(variant="detection")
        assert '"correction"' not in guardian_text(variant="correction")

    def test_convenience_constants_match_builders(self):
        assert FACT_DETECTION_GUARDIAN_TEXT == guardian_text(variant="detection")
        assert FACT_CORRECTION_GUARDIAN_TEXT == guardian_text(variant="correction")

    def test_unknown_criteria_id_raises(self):
        with pytest.raises(KeyError):
            guardian_text("nope", variant="detection")

    def test_additional_judging_task_can_be_added(self, monkeypatch):
        # A new judging task is a CRITERIA entry plus a _GUARDIAN_SPECS entry —
        # and no new assembly code.
        monkeypatch.setitem(CRITERIA, "toxicity", "A toxic response contains slurs.")
        monkeypatch.setitem(
            _GUARDIAN_SPECS,
            "toxicity",
            {
                "system_prompt": "SYS-TOX",
                "variants": {None: ("SCHEMA-TOX", None)},
                "default_variant": None,
                "always_hint": False,
            },
        )
        block = guardian_text("toxicity")
        assert block.startswith("<guardian>SYS-TOX")
        assert "A toxic response contains slurs." in block
        assert "### Scoring Schema: SCHEMA-TOX" in block
        assert CRITERIA["factuality"] not in block


class TestAtomizerGuardianStyle:
    """The atomizer prompt is a guardian block built from CRITERIA["atomicity"]."""

    def test_atomicity_is_a_criteria_key(self):
        assert ATOMIZER_CRITERIA_ID == "atomicity"
        assert "atomicity" in CRITERIA
        assert "atomic unit" in CRITERIA["atomicity"]

    def test_atom_definition_comes_from_criteria(self):
        assert CRITERIA["atomicity"] in guardian_text(ATOMIZER_CRITERIA_ID)

    def test_same_block_layout_as_factuality(self):
        atom = guardian_text(ATOMIZER_CRITERIA_ID)
        fact = guardian_text(variant="detection")
        for block in (atom, fact):
            assert block.startswith("<guardian>")
            assert block.index("### Criteria:") < block.index("### Scoring Schema:")

    def test_no_legacy_atomizer_markers(self):
        # The old block used an <atomizer> tag and #### headings.
        atom = guardian_text(ATOMIZER_CRITERIA_ID)
        assert "<atomizer>" not in atom
        assert "#### Criteria:" not in atom
        assert "#### Instruction:" not in atom

    def test_decomposition_schema(self):
        atom = guardian_text(ATOMIZER_CRITERIA_ID)
        assert "one unit per line" in atom
        assert "no numbering, bullets, or extra commentary" in atom

    def test_no_json_hint(self):
        # The atomizer target is newline-separated text, not JSON.
        assert "JSON" not in guardian_text(ATOMIZER_CRITERIA_ID)

    def test_constant_matches_builder(self):
        # c268bac gave the atomizer its own <atomizer> tag; the default tag is <guardian>.
        assert ATOMIZER_TEXT == guardian_text(ATOMIZER_CRITERIA_ID, tag="atomizer")
        assert ATOMIZER_TEXT.startswith("<atomizer>")

    def test_does_not_leak_factuality_criteria(self):
        assert CRITERIA["factuality"] not in guardian_text(ATOMIZER_CRITERIA_ID)

    def test_unknown_criteria_id_raises(self):
        with pytest.raises(KeyError):
            guardian_text("nope")


class TestGuardianTextDispatch:
    """One builder serves every task, dispatching on criteria_id."""

    def test_assembles_all_three_parts(self):
        block = guardian_text(CRITERIA_ID, variant="detection")
        assert block.startswith("<guardian>")
        assert f"### Criteria: {CRITERIA['factuality']}" in block
        assert "### Scoring Schema:" in block

    @pytest.mark.parametrize(
        "criteria_id",
        ["factuality", "atomicity", "entailment"],
    )
    def test_every_task_renders_its_own_criteria(self, criteria_id):
        block = guardian_text(criteria_id)
        assert CRITERIA[criteria_id] in block
        # ...and only its own.
        for other in set(CRITERIA) - {criteria_id}:
            assert CRITERIA[other] not in block

    def test_each_task_has_its_own_system_prompt(self):
        blocks = [guardian_text(c) for c in ("factuality", "atomicity", "entailment")]
        heads = [b.split("\n\n")[0] for b in blocks]
        assert len(set(heads)) == 3

    def test_criteria_id_defaults_to_factuality(self):
        assert guardian_text() == guardian_text(CRITERIA_ID, variant="detection")

    def test_variant_defaults_per_task(self):
        # factuality defaults to detection; single-variant tasks need no variant.
        assert guardian_text("factuality") == guardian_text("factuality", variant="detection")
        assert guardian_text("atomicity")
        assert guardian_text("entailment")

    def test_custom_tag(self):
        assert guardian_text("factuality", tag="judge").startswith("<judge>")

    def test_unknown_criteria_id_raises_with_known_ids(self):
        with pytest.raises(KeyError) as exc:
            guardian_text("nope")
        assert "atomicity" in str(exc.value) and "factuality" in str(exc.value)

    def test_unknown_variant_raises(self):
        with pytest.raises(KeyError) as exc:
            guardian_text("factuality", variant="bogus")
        assert "detection" in str(exc.value)

    def test_variant_rejected_for_single_variant_task(self):
        with pytest.raises(KeyError):
            guardian_text("atomicity", variant="detection")

    def test_criteria_without_a_spec_raises(self, monkeypatch):
        # A CRITERIA entry with no _GUARDIAN_SPECS entry is a usage error, not a
        # silently-empty block.
        monkeypatch.setitem(CRITERIA, "toxicity", "A toxic response contains slurs.")
        with pytest.raises(KeyError) as exc:
            guardian_text("toxicity")
        assert "no guardian spec" in str(exc.value)


class TestNliGuardianStyle:
    """The NLI prompt is a guardian block built from CRITERIA["entailment"]."""

    def test_entailment_is_a_criteria_key(self):
        assert NLI_CRITERIA_ID == "entailment"
        assert "entailment" in CRITERIA

    def test_criteria_comes_from_the_dict(self):
        assert CRITERIA["entailment"] in guardian_text(NLI_CRITERIA_ID)

    def test_same_block_layout_as_factuality(self):
        for block in (guardian_text(NLI_CRITERIA_ID), guardian_text(variant="detection")):
            assert block.startswith("<guardian>")
            assert block.index("### Criteria:") < block.index("### Scoring Schema:")

    def test_no_legacy_nli_markers(self):
        block = guardian_text(NLI_CRITERIA_ID)
        assert "<nli>" not in block
        assert "#### Criteria:" not in block
        assert "#### Instruction:" not in block

    def test_all_three_labels_defined(self):
        criteria = CRITERIA["entailment"]
        for label in ("entailment", "contradiction", "neutral"):
            assert f"'{label}'" in criteria

    def test_label_semantics_preserved(self):
        # The enriched wording must still mean the same three things.
        criteria = CRITERIA["entailment"]
        assert "must also accept the hypothesis as true" in criteria
        assert "must reject the hypothesis as false" in criteria
        assert "neither compels nor rules out" in criteria

    def test_decision_rules_present(self):
        # These are the rules that drive accuracy on the common confusions.
        criteria = CRITERIA["entailment"]
        assert "outside world knowledge" in criteria
        assert "Unmentioned is not the same as false" in criteria
        assert "Plausible is not the same as entailed" in criteria
        assert "more specific" in criteria and "more general" in criteria
        assert "quantifiers" in criteria

    def test_json_hint_always_included(self):
        # The NLI target *is* the JSON label dict the eval parser reads back.
        block = guardian_text(NLI_CRITERIA_ID)
        assert '{"label"' in block
        assert all(label in block for label in ("entailment", "contradiction", "neutral"))

    def test_constant_matches_builder(self):
        # c268bac gave NLI its own <nli> tag; the default tag is <guardian>.
        assert NLI_TEXT == guardian_text(NLI_CRITERIA_ID, tag="nli")
        assert NLI_TEXT.startswith("<nli>")

    def test_does_not_leak_other_criteria(self):
        block = guardian_text(NLI_CRITERIA_ID)
        assert CRITERIA["factuality"] not in block
        assert CRITERIA["atomicity"] not in block

    def test_unknown_criteria_id_raises(self):
        with pytest.raises(KeyError):
            guardian_text("nope")

    def test_messages_carry_the_guardian_block(self):
        msgs = get_nli_messages("PREM", "HYP")
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert "PREM" in content and "HYP" in content
        assert NLI_TEXT in content


class TestJsonSchemaSelection:
    """Granite 4.2 uses a schema that asks for the verdict as a JSON dict."""

    def test_tied_to_tool_style_only(self):
        assert uses_json_schema(DOC_STYLE_TOOL) is True
        for style in (DOC_STYLE_GRANITE3, DOC_STYLE_GRANITE4, DOC_STYLE_INLINE):
            assert uses_json_schema(style) is False

    def test_json_schema_replaces_plain_text(self):
        for variant, key in (("detection", '"score"'), ("correction", '"correction"')):
            plain = fact_guardian_text(variant)
            js = fact_guardian_text(variant, json_schema=True)
            assert js != plain
            assert key in js
            assert "JSON dict" in js

    def test_plain_schema_has_no_json_spec(self):
        assert '{"score"' not in FACT_DETECTION_GUARDIAN_TEXT
        assert "return 'yes'" in FACT_DETECTION_GUARDIAN_TEXT
        assert '{"correction"' not in FACT_CORRECTION_GUARDIAN_TEXT
        assert "corrected version" in FACT_CORRECTION_GUARDIAN_TEXT

    def test_json_variant_drops_plain_phrasing(self):
        # The JSON schema replaces the schema; it does not append to it.
        assert "return 'yes'" not in FACT_DETECTION_GUARDIAN_TEXT_JSON
        assert "corrected version" not in FACT_CORRECTION_GUARDIAN_TEXT_JSON

    def test_for_prompt_is_suppressed_under_json_schema(self):
        # The JSON schema already states the format; appending the hint would
        # duplicate it.
        for variant, key in (("detection", '{"score"'), ("correction", '{"correction"')):
            a = guardian_text(CRITERIA_ID, variant=variant, json_schema=True)
            b = guardian_text(CRITERIA_ID, variant=variant, json_schema=True, for_prompt=True)
            assert a == b
            assert a.count(key) == 1

    def test_for_prompt_still_appends_without_json_schema(self):
        plain = guardian_text(CRITERIA_ID, variant="detection")
        hinted = guardian_text(CRITERIA_ID, variant="detection", for_prompt=True)
        assert hinted != plain
        assert '{"score"' in hinted

    def test_helper_matches_constants(self):
        assert fact_guardian_text("detection") == FACT_DETECTION_GUARDIAN_TEXT
        assert fact_guardian_text("correction") == FACT_CORRECTION_GUARDIAN_TEXT
        assert fact_guardian_text("detection", json_schema=True) == FACT_DETECTION_GUARDIAN_TEXT_JSON
        assert fact_guardian_text("correction", json_schema=True) == FACT_CORRECTION_GUARDIAN_TEXT_JSON

    def test_other_tasks_have_no_json_schema(self):
        # atomizer/NLI define no json_schemas entry, so the flag is inert there.
        assert guardian_text(ATOMIZER_CRITERIA_ID, json_schema=True) == guardian_text(ATOMIZER_CRITERIA_ID)
        assert guardian_text(NLI_CRITERIA_ID, json_schema=True) == guardian_text(NLI_CRITERIA_ID)
