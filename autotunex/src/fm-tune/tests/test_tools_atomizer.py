"""Tests for autotune.tools.build_atomizer_dataset — pure helpers + CLI."""

import json
from unittest.mock import MagicMock

from autotune.tools.build_atomizer_dataset import _format_atoms, _make_record, main

GRANITE4_TEMPLATE = "{%- if documents %}{%- set x = x + (document | tojson) %}{%- endif %}"
GRANITE42_TEMPLATE = "{%- if enable_thinking %}<|im_start|>assistant<think>{%- endif %}"


def _tok(template=GRANITE4_TEMPLATE):
    tok = MagicMock()
    tok.chat_template = template
    tok.apply_chat_template.return_value = "RENDERED"
    tok.return_value = {"input_ids": [1, 2, 3]}
    return tok


def _dp(**over):
    dp = {"query": "Q", "response": "RESP", "atoms": ["a one", "a two"]}
    dp.update(over)
    return dp


class TestFormatAtoms:
    def test_joins_one_per_line(self):
        assert _format_atoms(["a", "b"]) == "a\nb"

    def test_strips_and_drops_empties(self):
        assert _format_atoms(["  a  ", "", "   ", "b"]) == "a\nb"

    def test_ignores_non_strings(self):
        assert _format_atoms(["a", None, 3, "b"]) == "a\nb"

    def test_all_empty(self):
        assert _format_atoms(["", "  "]) == ""


class TestMakeRecord:
    def test_chat_format_shape(self):
        out = _make_record(_dp(), tokenizer=None, fmt="chat")
        # Three turns: the query, the response to decompose, the instruction.
        assert [m["role"] for m in out["input"]] == ["user", "assistant", "user"]
        assert out["input"][1]["content"] == "RESP"
        assert out["output"] == "a one\na two"

    def test_formatted_uses_template(self):
        tok = _tok()
        out = _make_record(_dp(), tokenizer=tok, fmt="formatted")
        assert out["input"] == "RENDERED"
        assert out["output"] == "a one\na two"

    def test_thinking_disabled_by_default(self):
        tok = _tok(GRANITE42_TEMPLATE)
        _make_record(_dp(), tokenizer=tok, fmt="formatted")
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False

    def test_thinking_can_be_enabled(self):
        tok = _tok(GRANITE42_TEMPLATE)
        _make_record(_dp(), tokenizer=tok, fmt="formatted", thinking=True)
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is True

    def test_never_passes_documents(self):
        # The atomizer has no RAG context, so documents= must never be forwarded.
        tok = _tok()
        _make_record(_dp(), tokenizer=tok, fmt="formatted")
        assert "documents" not in tok.apply_chat_template.call_args.kwargs

    def test_missing_query(self):
        assert _make_record(_dp(query=""), tokenizer=None, fmt="chat") is None
        assert _make_record({"response": "R", "atoms": ["a"]}, tokenizer=None, fmt="chat") is None

    def test_missing_response(self):
        assert _make_record(_dp(response="  "), tokenizer=None, fmt="chat") is None

    def test_atoms_not_a_list(self):
        assert _make_record(_dp(atoms="a"), tokenizer=None, fmt="chat") is None

    def test_empty_atoms_dropped(self):
        assert _make_record(_dp(atoms=["", "  "]), tokenizer=None, fmt="chat") is None


class TestMainBothFormats:
    def _write_raw(self, tmp_path, records):
        p = tmp_path / "raw.json"
        p.write_text(json.dumps(records), encoding="utf-8")
        return p

    def test_writes_two_files(self, tmp_path, monkeypatch):
        raw = [_dp(), _dp(query="Q2", response="R2", atoms=["b one"])]
        in_path = self._write_raw(tmp_path, raw)
        out_dir = tmp_path / "out"

        monkeypatch.setattr("autotune.tools.build_atomizer_dataset.load_tokenizer", lambda *a, **k: _tok())
        assert main(["--input", str(in_path), "--output-dir", str(out_dir), "--model", "dummy"]) == 0

        # Names come from the input stem + format.
        formatted = out_dir / "raw_formatted.jsonl"
        chat = out_dir / "raw_chat.jsonl"
        assert formatted.exists() and chat.exists()

        f_lines = formatted.read_text(encoding="utf-8").strip().split("\n")
        c_lines = chat.read_text(encoding="utf-8").strip().split("\n")
        assert len(f_lines) == len(c_lines) == 2
        for line in f_lines:
            assert isinstance(json.loads(line)["input"], str)
        for line in c_lines:
            assert isinstance(json.loads(line)["input"], list)

    def test_single_format_writes_one_file(self, tmp_path, monkeypatch):
        in_path = self._write_raw(tmp_path, [_dp()])
        out_dir = tmp_path / "out"
        monkeypatch.setattr("autotune.tools.build_atomizer_dataset.load_tokenizer", lambda *a, **k: _tok())
        rc = main(["--input", str(in_path), "--output-dir", str(out_dir), "--format", "chat", "--model", "dummy"])
        assert rc == 0
        assert (out_dir / "raw_chat.jsonl").exists()
        assert not (out_dir / "raw_formatted.jsonl").exists()

    def test_missing_input_returns_1(self, tmp_path, capsys):
        rc = main(["--input", str(tmp_path / "nope.json"), "--output-dir", str(tmp_path / "out")])
        assert rc == 1
        assert "not found" in capsys.readouterr().err

    def test_length_gate_drops_long_output(self, tmp_path, monkeypatch):
        in_path = self._write_raw(tmp_path, [_dp()])
        out_dir = tmp_path / "out"
        tok = _tok()
        tok.return_value = {"input_ids": list(range(5000))}
        monkeypatch.setattr("autotune.tools.build_atomizer_dataset.load_tokenizer", lambda *a, **k: tok)
        rc = main(
            [
                "--input",
                str(in_path),
                "--output-dir",
                str(out_dir),
                "--format",
                "chat",
                "--model",
                "dummy",
                "--max-output-length",
                "10",
            ]
        )
        assert rc == 0
        assert (out_dir / "raw_chat.jsonl").read_text(encoding="utf-8").strip() == ""
