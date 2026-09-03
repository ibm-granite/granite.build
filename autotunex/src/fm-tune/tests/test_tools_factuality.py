"""Tests for autotune.tools.build_factuality_dataset — pure helpers."""

import json
from unittest.mock import MagicMock

import pytest

from autotune.tools.build_factuality_dataset import (
    DOC_STYLE_GRANITE3,
    DOC_STYLE_GRANITE4,
    DOC_STYLE_INLINE,
    _build_documents,
    _build_messages,
    _extract_contexts,
    _guardian_block,
    _record_correction,
    _record_detection,
    _resolve_formats,
    _unique_strings,
    convert_split,
    detect_document_style,
    main,
    output_name_for,
    resolve_document_style,
    template_supports_kwarg,
)

# Minimal stand-ins for the discriminating parts of each real template.
GRANITE3_TEMPLATE = "{%- if documents %}{{- document['doc_id'] }}{{- document['text'] }}{%- endif %}"
GRANITE4_TEMPLATE = "{%- if documents %}{%- set x = x + (document | tojson) %}{%- endif %}"
GRANITE42_TEMPLATE = "{%- if enable_thinking %}<|im_start|>assistant<think>{%- endif %}"
PLAIN_TEMPLATE = "{%- for message in messages %}{{ message.content }}{%- endfor %}"


class TestUniqueStrings:
    def test_preserves_order(self):
        assert _unique_strings(["b", "a", "c", "a", "b"]) == ["b", "a", "c"]

    def test_empty(self):
        assert _unique_strings([]) == []

    def test_no_duplicates(self):
        assert _unique_strings(["a", "b", "c"]) == ["a", "b", "c"]


class TestGuardianBlock:
    def test_detection_for_prompt(self):
        block = _guardian_block(is_detection=True, for_prompt=True)
        assert "yes" in block.lower()
        assert "json" in block.lower()
        assert "score" in block

    def test_detection_no_prompt(self):
        block = _guardian_block(is_detection=True, for_prompt=False)
        assert "yes" in block.lower()
        # No JSON dict format hint
        assert "json dict" not in block.lower()

    def test_correction_for_prompt(self):
        block = _guardian_block(is_detection=False, for_prompt=True)
        assert "corrected" in block.lower()
        assert "correction" in block.lower()

    def test_correction_no_prompt(self):
        block = _guardian_block(is_detection=False, for_prompt=False)
        assert "corrected" in block.lower()
        assert "json dict" not in block.lower()


class TestExtractContexts:
    def test_basic(self):
        rec = {
            "c_a1": {"text": "context 1"},
            "c_a2": {"text": "context 2"},
            "other": "ignored",
        }
        out = _extract_contexts(rec)
        assert sorted(out) == ["context 1", "context 2"]

    def test_dedup(self):
        rec = {
            "c_a1": {"text": "same"},
            "c_a2": {"text": "same"},
            "c_a3": {"text": "different"},
        }
        out = _extract_contexts(rec)
        assert len(out) == 2

    def test_skips_none_text(self):
        rec = {"c_a1": {"text": None}, "c_a2": {"text": "real"}}
        assert _extract_contexts(rec) == ["real"]

    def test_skips_non_c_a_keys(self):
        rec = {"foo": {"text": "x"}, "bar": "no dict"}
        assert _extract_contexts(rec) == []


class TestBuildMessages:
    def test_three_messages(self):
        msgs = _build_messages("query", "response", is_detection=True, for_prompt=False)
        assert len(msgs) == 3
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "query"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "response"
        assert msgs[2]["role"] == "user"
        assert "guardian" in msgs[2]["content"]


class TestRecordDetection:
    def test_chat_format(self):
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "c_a1": {"text": "context"},
        }
        out = _record_detection(dp, tokenizer=None, fmt="chat", for_prompt=False)
        # In chat format, input is a list of messages
        assert isinstance(out["input"], list)
        assert len(out["input"]) == 3
        # Output is JSON dict with score
        parsed = json.loads(out["output"])
        assert parsed == {"score": "yes"}
        assert "documents" in out

    def test_formatted_uses_tokenizer(self):
        tok = MagicMock()
        tok.apply_chat_template.return_value = "TEMPLATED"
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "No"},
            "c_a1": {"text": "context"},
        }
        out = _record_detection(dp, tokenizer=tok, fmt="formatted", for_prompt=False)
        assert out["input"] == "TEMPLATED"
        assert json.loads(out["output"]) == {"score": "no"}


class TestRecordCorrection:
    def test_chat_format_short_response(self):
        tok = MagicMock()
        # Tokenizer returns 5 tokens — under max_length
        tok.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "correction": {"text": "C"},
            "c_a1": {"text": "context"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=False)
        assert out is not None
        parsed = json.loads(out["output"])
        assert parsed == {"correction": "C"}

    def test_length_gate_filters_long_correction(self):
        tok = MagicMock()
        # Returns 1000 tokens — exceeds max_length
        tok.return_value = {"input_ids": list(range(1000))}
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "correction": {"text": "C" * 5000},
            "c_a1": {"text": "context"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=False)
        assert out is None

    def test_missing_correction_uses_none(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1, 2, 3]}
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "c_a1": {"text": "context"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=False)
        parsed = json.loads(out["output"])
        assert parsed == {"correction": "none"}

    def test_include_meta_adds_fields(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1, 2, 3]}
        dp = {
            "query": "QUERY",
            "response": {"text": "RESPONSE", "label": "Yes"},
            "correction": {"text": "C"},
            "c_a1": {"text": "ctx"},
        }
        out = _record_correction(dp, tokenizer=tok, fmt="chat", for_prompt=False, max_length=100, include_meta=True)
        assert out["query"] == "QUERY"
        assert out["response"] == "RESPONSE"
        # The gold label rides along too, so an eval file can be scored.
        assert out["label"] == "Yes"

    def test_include_meta_matches_across_formats(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1, 2, 3]}
        tok.apply_chat_template.return_value = "RENDERED"
        tok.chat_template = GRANITE4_TEMPLATE
        dp = {
            "query": "QUERY",
            "response": {"text": "RESPONSE", "label": "Yes"},
            "correction": {"text": "C"},
            "c_a1": {"text": "ctx"},
        }
        kw = dict(for_prompt=False, max_length=100, include_meta=True)
        chat = _record_correction(dp, tokenizer=tok, fmt="chat", **kw)
        fmtd = _record_correction(dp, tokenizer=tok, fmt="formatted", **kw)
        for key in ("query", "response", "label"):
            assert chat[key] == fmtd[key]


class TestDetectDocumentStyle:
    def test_granite3_template_reads_doc_id(self):
        assert detect_document_style(GRANITE3_TEMPLATE) == DOC_STYLE_GRANITE3

    def test_granite4_template_uses_tojson(self):
        assert detect_document_style(GRANITE4_TEMPLATE) == DOC_STYLE_GRANITE4

    def test_granite42_dropped_documents(self):
        # Granite 4.2 has zero occurrences of "document" -> must inline.
        assert detect_document_style(GRANITE42_TEMPLATE) == DOC_STYLE_INLINE

    def test_plain_template_ignores_documents(self):
        assert detect_document_style(PLAIN_TEMPLATE) == DOC_STYLE_INLINE

    def test_missing_template(self):
        assert detect_document_style(None) == DOC_STYLE_INLINE
        assert detect_document_style("") == DOC_STYLE_INLINE


class TestTemplateSupportsKwarg:
    def test_present(self):
        assert template_supports_kwarg(GRANITE42_TEMPLATE, "enable_thinking") is True

    def test_absent(self):
        assert template_supports_kwarg(GRANITE4_TEMPLATE, "enable_thinking") is False

    def test_no_template(self):
        assert template_supports_kwarg(None, "enable_thinking") is False


class TestBuildDocuments:
    def test_granite3_includes_doc_id(self):
        docs = _build_documents(["a", "b"], DOC_STYLE_GRANITE3)
        assert docs == [{"doc_id": "0", "text": "a"}, {"doc_id": "1", "text": "b"}]

    def test_granite4_omits_doc_id(self):
        # Keys render verbatim in 4.x, so doc_id must not leak into the prompt.
        docs = _build_documents(["a", "b"], DOC_STYLE_GRANITE4)
        assert docs == [{"text": "a"}, {"text": "b"}]
        assert all("doc_id" not in d for d in docs)

    def test_one_document_per_passage(self):
        assert len(_build_documents(["a", "b", "c"], DOC_STYLE_GRANITE4)) == 3

    def test_empty(self):
        assert _build_documents([], DOC_STYLE_GRANITE4) == []


class TestExtractContextsOrdering:
    def test_numeric_order_not_file_order(self):
        dp = {"c_a10": {"text": "ten"}, "c_a2": {"text": "two"}, "c_a1": {"text": "one"}}
        assert _extract_contexts(dp) == ["one", "two", "ten"]

    def test_non_numeric_keys_sort_last(self):
        dp = {"c_ax": {"text": "x"}, "c_a1": {"text": "one"}}
        assert _extract_contexts(dp) == ["one", "x"]


class TestInlineStyle:
    def _dp(self):
        return {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "c_a1": {"text": "PASSAGE-ONE"},
            "c_a2": {"text": "PASSAGE-TWO"},
        }

    def test_chat_inlines_and_omits_documents_column(self):
        out = _record_detection(self._dp(), tokenizer=None, fmt="chat", for_prompt=False, doc_style=DOC_STYLE_INLINE)
        # The driver would silently drop a documents column on such a template.
        assert "documents" not in out
        assert out["input"][0]["role"] == "system"
        content = out["input"][0]["content"]
        assert "PASSAGE-ONE" in content and "PASSAGE-TWO" in content
        assert len(out["input"]) == 4

    def test_native_styles_keep_documents_column(self):
        for style in (DOC_STYLE_GRANITE3, DOC_STYLE_GRANITE4):
            out = _record_detection(self._dp(), tokenizer=None, fmt="chat", for_prompt=False, doc_style=style)
            assert "documents" in out
            assert len(out["documents"]) == 2
            assert out["input"][0]["role"] == "user"

    def test_formatted_inline_does_not_pass_documents_kwarg(self):
        tok = MagicMock()
        tok.chat_template = GRANITE42_TEMPLATE
        tok.apply_chat_template.return_value = "RENDERED"
        _record_detection(self._dp(), tokenizer=tok, fmt="formatted", for_prompt=False, doc_style=DOC_STYLE_INLINE)
        kwargs = tok.apply_chat_template.call_args.kwargs
        assert "documents" not in kwargs
        # thinking is off by default so the JSON target isn't inside <think>.
        assert kwargs["enable_thinking"] is False

    def test_formatted_native_passes_documents_kwarg(self):
        tok = MagicMock()
        tok.chat_template = GRANITE4_TEMPLATE
        tok.apply_chat_template.return_value = "RENDERED"
        _record_detection(self._dp(), tokenizer=tok, fmt="formatted", for_prompt=False, doc_style=DOC_STYLE_GRANITE4)
        kwargs = tok.apply_chat_template.call_args.kwargs
        assert kwargs["documents"] == [{"text": "PASSAGE-ONE"}, {"text": "PASSAGE-TWO"}]
        # Template has no enable_thinking -> must not be forwarded.
        assert "enable_thinking" not in kwargs

    def test_correction_inline(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1, 2, 3]}
        dp = dict(self._dp(), correction={"text": "C"})
        out = _record_correction(
            dp,
            tokenizer=tok,
            fmt="chat",
            for_prompt=False,
            max_length=100,
            include_meta=False,
            doc_style=DOC_STYLE_INLINE,
        )
        assert "documents" not in out
        assert "PASSAGE-ONE" in out["input"][0]["content"]


class TestResolveDocumentStyle:
    def _tok(self, template):
        tok = MagicMock()
        tok.chat_template = template
        return tok

    def test_auto_detects(self):
        style = resolve_document_style(
            "auto", self._tok(GRANITE3_TEMPLATE), inline_documents=True, renders_template=True
        )
        assert style == DOC_STYLE_GRANITE3

    def test_explicit_overrides_detection(self):
        style = resolve_document_style(
            DOC_STYLE_GRANITE4, self._tok(GRANITE3_TEMPLATE), inline_documents=True, renders_template=True
        )
        assert style == DOC_STYLE_GRANITE4

    def test_hard_guard_when_inlining_disabled(self):
        # Refuse rather than emit records whose context never reaches the prompt.
        with pytest.raises(SystemExit) as exc:
            resolve_document_style("auto", self._tok(GRANITE42_TEMPLATE), inline_documents=False, renders_template=True)
        assert "silently dropped" in str(exc.value)

    def test_inline_allowed_by_default(self):
        style = resolve_document_style(
            "auto", self._tok(GRANITE42_TEMPLATE), inline_documents=True, renders_template=True
        )
        assert style == DOC_STYLE_INLINE

    def test_no_template_but_formatted_requested(self):
        with pytest.raises(SystemExit) as exc:
            resolve_document_style("auto", self._tok(None), inline_documents=True, renders_template=True)
        assert "no chat template" in str(exc.value)

    def test_no_template_chat_only_falls_back(self):
        style = resolve_document_style("auto", self._tok(None), inline_documents=True, renders_template=False)
        assert style == DOC_STYLE_INLINE


class TestResolveFormats:
    def test_both(self):
        assert _resolve_formats("both") == ["formatted", "chat"]

    def test_single(self):
        assert _resolve_formats("chat") == ["chat"]
        assert _resolve_formats("formatted") == ["formatted"]


class TestOutputNameFor:
    def test_derives_from_input_stem_task_and_format(self):
        assert output_name_for("/a/b/eli5_raw_train.json", "detection", "chat") == "eli5_raw_train_detection_chat.jsonl"

    def test_strips_directory_and_extension(self):
        assert output_name_for("/deep/path/bio.json", "correction", "formatted") == ("bio_correction_formatted.jsonl")

    def test_no_extension(self):
        assert output_name_for("raw", "detection", "chat") == "raw_detection_chat.jsonl"

    def test_dotted_stem_keeps_leading_parts(self):
        assert output_name_for("a.b.json", "detection", "chat") == "a.b_detection_chat.jsonl"


class TestMainWritesBothFiles:
    RAW = [
        {"query": "Q1", "response": {"text": "R1", "label": "Yes"}, "c_a1": {"text": "CTX-A"}},
        {"query": "Q2", "response": {"text": "R2", "label": "No"}, "c_a1": {"text": "CTX-B"}},
    ]

    def _tok(self):
        tok = MagicMock()
        tok.chat_template = GRANITE4_TEMPLATE
        tok.apply_chat_template.return_value = "RENDERED"
        tok.return_value = {"input_ids": [1, 2, 3]}
        return tok

    def _write_raw(self, tmp_path, name="eli5_raw_train.json"):
        p = tmp_path / name
        p.write_text(json.dumps(self.RAW), encoding="utf-8")
        return p

    def test_format_both_writes_two_files(self, tmp_path, monkeypatch):
        in_path = self._write_raw(tmp_path)
        out_dir = tmp_path / "out"
        monkeypatch.setattr("autotune.tools.build_factuality_dataset._load_tokenizer", lambda *a, **k: self._tok())

        rc = main(
            [
                "--input",
                str(in_path),
                "--output-dir",
                str(out_dir),
                "--task",
                "detection",
                "--model",
                "dummy",
            ]
        )
        assert rc == 0

        # Names come from the input stem + task + format.
        formatted = out_dir / "eli5_raw_train_detection_formatted.jsonl"
        chat = out_dir / "eli5_raw_train_detection_chat.jsonl"
        assert formatted.exists() and chat.exists()

        f_lines = formatted.read_text(encoding="utf-8").strip().split("\n")
        c_lines = chat.read_text(encoding="utf-8").strip().split("\n")
        assert len(f_lines) == len(c_lines) == 2

        # Every line round-trips as JSON, with the expected shape per format.
        for line in f_lines:
            assert isinstance(json.loads(line)["input"], str)
        for line in c_lines:
            rec = json.loads(line)
            assert isinstance(rec["input"], list)
            assert rec["documents"] == [{"text": "CTX-A"}] or rec["documents"] == [{"text": "CTX-B"}]

    def test_single_format_writes_one_file(self, tmp_path, monkeypatch):
        in_path = self._write_raw(tmp_path, "bio.json")
        out_dir = tmp_path / "out"
        monkeypatch.setattr("autotune.tools.build_factuality_dataset._load_tokenizer", lambda *a, **k: self._tok())
        rc = main(
            [
                "--input",
                str(in_path),
                "--output-dir",
                str(out_dir),
                "--task",
                "detection",
                "--format",
                "chat",
                "--model",
                "dummy",
            ]
        )
        assert rc == 0
        assert (out_dir / "bio_detection_chat.jsonl").exists()
        assert not (out_dir / "bio_detection_formatted.jsonl").exists()

    def test_missing_input_returns_1(self, tmp_path, capsys):
        rc = main(
            [
                "--input",
                str(tmp_path / "nope.json"),
                "--output-dir",
                str(tmp_path / "out"),
                "--task",
                "detection",
            ]
        )
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_include_meta_off_by_default(self, tmp_path, monkeypatch):
        raw = [dict(self.RAW[0], correction={"text": "C"})]
        in_path = tmp_path / "eli5.json"
        in_path.write_text(json.dumps(raw), encoding="utf-8")
        out_dir = tmp_path / "out"
        monkeypatch.setattr("autotune.tools.build_factuality_dataset._load_tokenizer", lambda *a, **k: self._tok())
        args = [
            "--input",
            str(in_path),
            "--output-dir",
            str(out_dir),
            "--task",
            "correction",
            "--format",
            "chat",
            "--model",
            "dummy",
        ]
        assert main(args) == 0
        rec = json.loads((out_dir / "eli5_correction_chat.jsonl").read_text(encoding="utf-8").strip())
        assert "query" not in rec

        assert main(args + ["--include-meta"]) == 0
        rec = json.loads((out_dir / "eli5_correction_chat.jsonl").read_text(encoding="utf-8").strip())
        assert rec["query"] == "Q1" and rec["response"] == "R1"

    def test_default_format_is_both(self):
        parser_default = _resolve_formats("both")
        assert parser_default == ["formatted", "chat"]


class TestContextListSchema:
    """Biographies-style records carry contexts as a flat list of strings."""

    def _dp(self, **over):
        dp = {
            "query": "Q",
            "response": {"text": "R", "label": "Yes"},
            "contexts": ["LONG-A", "LONG-B"],
            "contexts_short": ["SHORT-A", "SHORT-B"],
        }
        dp.update(over)
        return dp

    def test_prefers_contexts_short(self):
        assert _extract_contexts(self._dp()) == ["SHORT-A", "SHORT-B"]

    def test_explicit_field_wins(self):
        assert _extract_contexts(self._dp(), "contexts") == ["LONG-A", "LONG-B"]

    def test_falls_back_to_contexts_when_short_absent(self):
        dp = self._dp()
        del dp["contexts_short"]
        assert _extract_contexts(dp) == ["LONG-A", "LONG-B"]

    def test_c_a_keys_take_precedence(self):
        # An ELI5-style record keeps using its c_a* keys even if a list exists.
        dp = self._dp(c_a1={"text": "CA-ONE"})
        assert _extract_contexts(dp) == ["CA-ONE"]

    def test_strips_and_drops_empty_strings(self):
        assert _extract_contexts(self._dp(contexts_short=["  A  ", "", "   ", "B"])) == ["A", "B"]

    def test_dedups(self):
        assert _extract_contexts(self._dp(contexts_short=["A", "A", "B"])) == ["A", "B"]

    def test_ignores_non_string_elements(self):
        assert _extract_contexts(self._dp(contexts_short=["A", None, 3])) == ["A"]

    def test_non_list_field_ignored(self):
        dp = self._dp(contexts_short="not a list")
        assert _extract_contexts(dp) == ["LONG-A", "LONG-B"]

    def test_record_gets_documents_from_list(self):
        out = _record_detection(self._dp(), tokenizer=None, fmt="chat", for_prompt=False)
        assert out["documents"] == [{"text": "SHORT-A"}, {"text": "SHORT-B"}]


class TestNoContextFiltering:
    """Records with no grounding context are dropped, not emitted context-free."""

    NO_CTX = {"query": "Q", "response": {"text": "R", "label": "Yes"}}

    def test_detection_drops_record(self):
        assert _record_detection(self.NO_CTX, tokenizer=None, fmt="chat", for_prompt=False) is None

    def test_correction_drops_record(self):
        tok = MagicMock()
        tok.return_value = {"input_ids": [1, 2, 3]}
        out = _record_correction(
            dict(self.NO_CTX, correction={"text": "C"}),
            tokenizer=tok,
            fmt="chat",
            for_prompt=False,
            max_length=100,
            include_meta=False,
        )
        assert out is None

    def test_empty_context_list_also_drops(self):
        assert (
            _record_detection(
                dict(self.NO_CTX, contexts=[], contexts_short=[]),
                tokenizer=None,
                fmt="chat",
                for_prompt=False,
            )
            is None
        )

    def test_convert_split_reports_and_filters(self, tmp_path, capsys):
        raw = [
            {"query": "Q1", "response": {"text": "R1", "label": "Yes"}, "c_a1": {"text": "CTX"}},
            dict(self.NO_CTX),  # no context -> filtered
            {"query": "Q3", "response": {"text": "R3", "label": "No"}, "contexts_short": ["S"]},
        ]
        in_path = tmp_path / "mixed.json"
        in_path.write_text(json.dumps(raw), encoding="utf-8")
        out_path = tmp_path / "out.jsonl"

        convert_split(
            input_path=str(in_path),
            output_path=str(out_path),
            task="detection",
            fmt="chat",
            tokenizer=None,
        )
        lines = out_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert "skipped 1 without context" in capsys.readouterr().out
        # Both surviving records carry a documents column.
        assert all("documents" in json.loads(x) for x in lines)
