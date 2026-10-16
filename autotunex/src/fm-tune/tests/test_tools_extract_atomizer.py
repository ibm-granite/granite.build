"""Tests for autotune.tools.extract_atomizer_raw — pure helpers + CLI."""

import json

from autotune.tools.extract_atomizer_raw import (
    convert,
    extract_atoms,
    extract_record,
    main,
    output_name_for,
)


def _dp(**over):
    dp = {
        "query": "Q",
        "response": {"text": "RESP"},
        "a1": {"text": "atom one"},
        "a2": {"text": "atom two"},
    }
    dp.update(over)
    return dp


class TestExtractAtoms:
    def test_numeric_order_not_lexical(self):
        dp = {f"a{i}": {"text": f"atom {i}"} for i in (10, 2, 1)}
        assert extract_atoms(dp) == ["atom 1", "atom 2", "atom 10"]

    def test_handles_a0(self):
        dp = {"a0": {"text": "zero"}, "a1": {"text": "one"}}
        assert extract_atoms(dp) == ["zero", "one"]

    def test_excludes_context_keys(self):
        # c_a* are RAG context keys, not atoms — the anchored match must skip them.
        dp = {"a1": {"text": "atom"}, "c_a1": {"text": "CTX"}, "c_a1_2": {"text": "CTX2"}}
        assert extract_atoms(dp) == ["atom"]

    def test_strips_and_drops_empty(self):
        dp = {"a1": {"text": "  spaced  "}, "a2": {"text": "   "}, "a3": {"text": ""}}
        assert extract_atoms(dp) == ["spaced"]

    def test_ignores_non_dict_and_non_string(self):
        dp = {"a1": "flat", "a2": {"text": None}, "a3": {"text": 3}, "a4": {"text": "ok"}}
        assert extract_atoms(dp) == ["ok"]

    def test_no_atom_keys(self):
        assert extract_atoms({"query": "Q"}) == []

    def test_ignores_other_a_prefixed_keys(self):
        dp = {"a1": {"text": "atom"}, "answer": {"text": "no"}, "a1b": {"text": "no"}}
        assert extract_atoms(dp) == ["atom"]


class TestExtractRecord:
    def test_flattens_nested_response(self):
        out = extract_record(_dp())
        assert out == {"query": "Q", "response": "RESP", "atoms": ["atom one", "atom two"]}

    def test_accepts_flat_response_string(self):
        out = extract_record(_dp(response="FLAT"))
        assert out["response"] == "FLAT"

    def test_drops_when_no_atoms(self):
        assert extract_record({"query": "Q", "response": {"text": "R"}}) is None

    def test_drops_empty_query(self):
        assert extract_record(_dp(query="  ")) is None

    def test_drops_empty_response(self):
        assert extract_record(_dp(response={"text": "  "})) is None
        assert extract_record(_dp(response=None)) is None


class TestOutputNameFor:
    def test_renames_raw_segment(self):
        assert output_name_for("/a/eli5_raw_train.json") == "eli5_atomizer_raw_train.json"
        assert output_name_for("bio_raw_test.json") == "bio_atomizer_raw_test.json"

    def test_without_raw_segment(self):
        assert output_name_for("/a/plain.json") == "plain_atomizer_raw.json"


class TestConvert:
    def test_writes_and_reports(self, tmp_path, capsys):
        raw = [_dp(), _dp(query="Q2"), {"query": "no atoms", "response": {"text": "R"}}]
        src = tmp_path / "eli5_raw_val.json"
        src.write_text(json.dumps(raw), encoding="utf-8")
        dst = tmp_path / "out.json"

        convert(str(src), str(dst))
        out = json.loads(dst.read_text(encoding="utf-8"))
        assert len(out) == 2
        assert all(sorted(r) == ["atoms", "query", "response"] for r in out)
        assert "skipped 1 unusable" in capsys.readouterr().out

    def test_rejects_non_list(self, tmp_path):
        src = tmp_path / "bad.json"
        src.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        try:
            convert(str(src), str(tmp_path / "o.json"))
        except ValueError as e:
            assert "top-level JSON list" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestMain:
    def test_end_to_end(self, tmp_path):
        src = tmp_path / "eli5_raw_test.json"
        src.write_text(json.dumps([_dp()]), encoding="utf-8")
        out_dir = tmp_path / "atomizer"
        assert main(["--input", str(src), "--output-dir", str(out_dir)]) == 0
        assert (out_dir / "eli5_atomizer_raw_test.json").exists()

    def test_missing_input_returns_1(self, tmp_path, capsys):
        rc = main(["--input", str(tmp_path / "nope.json"), "--output-dir", str(tmp_path)])
        assert rc == 1
        assert "not found" in capsys.readouterr().err
