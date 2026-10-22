"""Held-out NLI sources must preserve label meaning and the Granite SFT schema."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from autotune.tools import build_nli_dataset as nli

HF_LABELS = ("entailment", "neutral", "contradiction")


class Rows(list):
    features = {"label": SimpleNamespace(names=list(HF_LABELS))}


def source_rows(benchmark, per_label=1):
    rows = Rows(
        {"premise": f"Premise {i}", "hypothesis": f"Hypothesis {i}", "label": i % 3} for i in range(per_label * 3)
    )
    if benchmark == "mnli":
        # Streamed splits may carry no label metadata; the HF fallback order applies.
        rows.features = {"label": SimpleNamespace(dtype="int64")}
    return rows


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Tokenizer:
    chat_template = "{% if enable_thinking %}<think>{% endif %}"

    def __init__(self):
        self.thinking = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt
        self.thinking.append(enable_thinking)
        suffix = "<think>" if enable_thinking else "<think></think>"
        return messages[0]["content"] + "<|im_start|>assistant\n" + suffix

    def __call__(self, text, add_special_tokens=False):
        # Exercise the >= gate boundary for one label while keeping the others.
        if text == '{"label": "neutral"}':
            return {"input_ids": [1] * 4}
        return {"input_ids": text.split()}


def test_default_splits_and_duplicate_selection():
    specs = nli.resolve_benchmark_splits(["snli", "anli", "mnli", "snli"])
    assert [(spec.dataset, spec.split, spec.stem) for spec in specs] == [
        ("stanfordnlp/snli", "test", "snli_test"),
        ("facebook/anli", "test_r1", "anli_test_r1"),
        ("facebook/anli", "test_r2", "anli_test_r2"),
        ("facebook/anli", "test_r3", "anli_test_r3"),
        ("nyu-mll/multi_nli", "validation_matched", "mnli_validation_matched"),
    ]


def test_validation_splits():
    assert [s.split for s in nli.resolve_benchmark_splits(["snli", "anli", "mnli"], "validation")] == [
        "validation",
        "dev_r1",
        "dev_r2",
        "dev_r3",
        "validation_matched",
    ]


@pytest.mark.parametrize("benchmark", ["snli", "anli", "mnli"])
def test_numeric_labels_and_premise_direction(benchmark):
    rows = source_rows(benchmark)
    records, stats = nli.normalize_benchmark_records(rows, benchmark, rows.features["label"])
    assert [row["label"] for row in records] == list(HF_LABELS)
    assert records[1]["premise"] == "Premise 1"
    assert records[1]["hypothesis"] == "Hypothesis 1"
    assert stats["source_rows"] == stats["valid_rows"] == 3
    assert stats["label_counts"] == dict.fromkeys(HF_LABELS, 1)


def test_metadata_controls_integer_mapping():
    feature = SimpleNamespace(names=["contradiction", "entailment", "neutral"])
    rows, _ = nli.normalize_benchmark_records(source_rows("snli"), "snli", feature)
    assert [row["label"] for row in rows] == feature.names
    with pytest.raises(ValueError, match="unsupported label metadata"):
        nli.normalize_benchmark_records([], "snli", SimpleNamespace(names=["yes", "no"]))


def test_rejected_rows_are_counted():
    valid = {"premise": "P", "hypothesis": "H", "label": " Entailment "}
    rows = [valid, {**valid, "label": -1}, {**valid, "label": "unlabeled"}, None]
    rows.extend({**valid, "label": label} for label in (-2, 3, True, 1.0, None, "unknown"))
    rows.extend([{**valid, "premise": " "}, {**valid, "hypothesis": None}])
    records, stats = nli.normalize_benchmark_records(rows, "snli")
    assert records == [{**valid, "label": "entailment"}]
    assert stats["source_rows"] == 12
    assert stats["skipped_unlabeled"] == 2
    assert stats["skipped_malformed"] == 9


def test_hub_loader_requests_only_selected_split(monkeypatch):
    loader = Mock(return_value=source_rows("anli"))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=loader))
    spec = nli.resolve_benchmark_splits(["anli"], "validation")[1]
    nli._load_benchmark_split(spec)
    loader.assert_called_once_with("facebook/anli", name="plain_text", split="dev_r2", streaming=True)


@pytest.mark.parametrize("thinking", [False, True])
def test_full_build_produces_aligned_sft_files_and_manifest(tmp_path, monkeypatch, thinking):
    loaded = []

    def load(spec):
        loaded.append(spec)
        rows = source_rows(spec.benchmark)
        rows.append({**rows[0], "label": -1})
        rows.append({"label": 0})
        return rows

    monkeypatch.setattr(nli, "_load_benchmark_split", load)
    tokenizer = Tokenizer()
    monkeypatch.setattr(nli, "load_tokenizer", lambda *args: tokenizer)
    args = ["--benchmarks", "snli", "anli", "mnli", "--output-dir", str(tmp_path), "--model", "granite"]
    assert nli.main(args + (["--thinking"] if thinking else [])) == 0
    assert len(loaded) == 5  # Each source split is loaded once for both formats.
    assert len(list(tmp_path.glob("*.jsonl"))) == 10
    assert tokenizer.thinking == [thinking] * 15
    manifest = json.loads((tmp_path / "nli_benchmarks_manifest.json").read_text())
    assert manifest["settings"]["thinking"] is thinking
    assert manifest["settings"]["model"] == "granite"
    for spec, entry in zip(loaded, manifest["benchmarks"]):
        assert entry["dataset"] == spec.dataset
        assert entry["split"] == spec.split
        assert entry["source_rows"] == 5
        assert entry["valid_rows"] == 3
        assert entry["skipped_unlabeled"] == entry["skipped_malformed"] == 1
        formatted = read_jsonl(tmp_path / entry["outputs"][0]["file"])
        chat = read_jsonl(tmp_path / entry["outputs"][1]["file"])
        for f, c, label in zip(formatted, chat, HF_LABELS):
            assert set(f) == set(c) == {"input", "output"}
            assert isinstance(f["input"], str) and isinstance(c["input"], list)
            assert f["input"].startswith(c["input"][0]["content"])
            assert f["output"] == c["output"] == json.dumps({"label": label})
        for output in entry["outputs"]:
            assert output["written"] == 3
            assert output["label_counts"] == dict.fromkeys(HF_LABELS, 1)


def test_local_input_and_benchmarks_in_one_run(tmp_path, monkeypatch):
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps([{"premise": str(i), "hypothesis": "H", "label": label} for i in range(4) for label in HF_LABELS])
    )
    monkeypatch.setattr(nli, "_load_benchmark_split", lambda spec: source_rows(spec.benchmark, per_label=4))
    out = tmp_path / "out"
    args = ["--input", str(raw), "--benchmarks", "mnli", "--output-dir", str(out), "--format", "chat"]
    assert nli.main(args + ["--max-output-length", "0", "--name-tag", "test"]) == 0
    local = read_jsonl(out / "raw_test_chat.jsonl")
    benchmark = read_jsonl(out / "mnli_validation_matched_test_chat.jsonl")
    assert len(local) == len(benchmark) == 12
    assert len(list(out.glob("*.jsonl"))) == 2
    assert (out / "nli_benchmarks_manifest_test.json").is_file()


def test_length_gate_is_aligned_and_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(nli, "_load_benchmark_split", lambda spec: source_rows(spec.benchmark))
    monkeypatch.setattr(nli, "load_tokenizer", lambda *args: Tokenizer())
    assert (
        nli.main(
            ["--benchmarks", "snli", "--output-dir", str(tmp_path), "--max-output-length", "4", "--model", "granite"]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "nli_benchmarks_manifest.json").read_text())
    for output in manifest["benchmarks"][0]["outputs"]:
        assert output["written"] == 2
        assert output["skipped_length"] == 1
        assert output["label_counts"] == {"entailment": 1, "contradiction": 1}
        assert [json.loads(row["output"])["label"] for row in read_jsonl(tmp_path / output["file"])] == [
            "entailment",
            "contradiction",
        ]


@pytest.mark.parametrize("failure", ["unlabeled", "length", "download"])
def test_failed_build_is_nonzero_and_preserves_previous_file(tmp_path, monkeypatch, capsys, failure):
    def load(spec):
        if failure == "download":
            raise OSError("offline")
        rows = source_rows(spec.benchmark)
        if failure == "unlabeled":
            for row in rows:
                row["label"] = -1
        return rows

    monkeypatch.setattr(nli, "_load_benchmark_split", load)
    monkeypatch.setattr(nli, "load_tokenizer", lambda *args: Tokenizer())
    previous = tmp_path / "snli_test_formatted.jsonl"
    previous.write_text("previous build")
    assert (
        nli.main(
            ["--benchmarks", "snli", "--output-dir", str(tmp_path), "--max-output-length", "1", "--model", "granite"]
        )
        == 1
    )
    assert previous.read_text() == "previous build"
    assert list(tmp_path.iterdir()) == [previous]
    error = capsys.readouterr().err
    assert "[error]" in error
    if failure == "download":
        assert "stanfordnlp/snli (test)" in error and "offline" in error
    else:
        assert "no valid labeled rows" in error or "no rows remain" in error


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "is required"),
        (["--benchmarks", "mnli", "--benchmark-split", "test"], "no gold labels"),
        (["--eli5-train", "t.json"], "needs all of"),
        (["--eli5-train", "t", "--eli5-val", "v", "--eli5-test", "x", "--benchmarks", "snli"], "do not apply"),
        (["--eli5-train", "t", "--eli5-val", "v", "--eli5-test", "x", "--train-mix", "eli5=10"], "divisible by 3"),
        (["--eli5-train", "t", "--eli5-val", "v", "--eli5-test", "x", "--val-mix", "anli=3"], "SOURCE in"),
        (["--eli5-train", "t", "--eli5-val", "v", "--eli5-test", "x", "--test-balanced-head", "100"], "head"),
        (["--input", "raw.json", "--benchmark-split", "validation"], "requires --benchmarks"),
    ],
)
def test_invalid_cli_fails_before_loading(tmp_path, monkeypatch, capsys, args, message):
    tokenizer_loader = Mock()
    hub_loader = Mock()
    monkeypatch.setattr(nli, "load_tokenizer", tokenizer_loader)
    monkeypatch.setattr(nli, "_load_benchmark_split", hub_loader)
    with pytest.raises(SystemExit) as exc:
        nli.main(["--output-dir", str(tmp_path), *args])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    tokenizer_loader.assert_not_called()
    hub_loader.assert_not_called()


def test_local_and_benchmark_output_collision_is_rejected(tmp_path):
    raw = tmp_path / "snli_test.json"
    raw.write_text("[]")
    with pytest.raises(SystemExit):
        nli.main(["--input", str(raw), "--benchmarks", "snli", "--output-dir", str(tmp_path)])
    assert list(tmp_path.iterdir()) == [raw]
