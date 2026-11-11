"""Mixed NLI build: ELI5 + MNLI + SNLI train/val, balanced ELI5 test, benchmark tests.

What matters beyond the counts:

- **No leakage.** Nothing in any test set (ELI5 test or the five benchmark splits)
  may reach val or train, and nothing in val may reach train.
- **Row alignment.** ``raw.json``, ``formatted`` and ``chat`` are three views of one
  split, so ``source`` in the raw file labels each formatted/chat row.
- **Balanced test head.** An eval with ``limit: 3000`` reads the first rows only, so
  those must carry an equal share of each label.
"""

import collections
import json
from types import SimpleNamespace

import pytest

from autotune.tools import build_nli_dataset as nli

LABELS = ("entailment", "contradiction", "neutral")
HF_LABELS = ("entailment", "neutral", "contradiction")


class Rows(list):
    features = {"label": SimpleNamespace(names=list(HF_LABELS))}


def hf_rows(tag, per_label):
    return Rows({"premise": f"{tag} P{i}", "hypothesis": f"{tag} H{i}", "label": i % 3} for i in range(per_label * 3))


class Tokenizer:
    chat_template = "{% if enable_thinking %}<think>{% endif %}"

    def __init__(self):
        self.thinking = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.thinking.append(enable_thinking)
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_raw(path, tag, per_label, labels=LABELS):
    rows = [
        {"premise": f"{tag} P{i}", "hypothesis": f"{tag} H{i}", "label": labels[i % len(labels)]}
        for i in range(per_label * len(labels))
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Raw ELI5 files plus a fake hub where SNLI val and the SNLI test share one pair."""
    loaded = []

    def load(spec):
        loaded.append((spec.dataset, spec.split))
        rows = hf_rows(f"{spec.benchmark}:{spec.split}", per_label=20)
        if (spec.benchmark, spec.split) == ("snli", "validation"):
            # A leaked eval pair: it must never be sampled into val.
            rows[0] = {"premise": "snli:test P0 ", "hypothesis": " snli:test H0", "label": 0}
        return rows

    monkeypatch.setattr(nli, "_load_benchmark_split", load)
    tokenizer = Tokenizer()
    monkeypatch.setattr(nli, "load_tokenizer", lambda *args: tokenizer)
    paths = {}
    for split, per_label in (("train", 30), ("val", 10), ("test", 12)):
        paths[split] = tmp_path / f"eli5_{split}.json"
        write_raw(paths[split], f"eli5:{split}", per_label)
    # The ELI5 test file is label-blocked, so a balanced head needs real reordering;
    # append a train pair to it, which must then be excluded from train.
    test_rows = json.loads(paths["test"].read_text())
    test_rows.sort(key=lambda r: r["label"])
    test_rows.append({"premise": "eli5:train P0", "hypothesis": "eli5:train H0", "label": "entailment"})
    paths["test"].write_text(json.dumps(test_rows))
    out = tmp_path / "out"
    args = [
        "--eli5-train",
        str(paths["train"]),
        "--eli5-val",
        str(paths["val"]),
        "--eli5-test",
        str(paths["test"]),
        "--output-dir",
        str(out),
        "--model",
        "granite",
        "--train-mix",
        "eli5=30,mnli=15,snli=15",
        "--val-mix",
        "eli5=9,mnli=6,snli=6",
        "--test-balanced-head",
        "9",
    ]
    return SimpleNamespace(args=args, out=out, loaded=loaded, tokenizer=tokenizer, test_rows=test_rows)


def keys(rows):
    return {(r["premise"].strip(), r["hypothesis"].strip()) for r in rows}


def test_mix_counts_and_balance(env):
    assert nli.main(env.args) == 0
    manifest = json.loads((env.out / "nli_mix_manifest.json").read_text())
    assert manifest["splits"]["train"]["counts"] == {
        "eli5": dict.fromkeys(LABELS, 10),
        "mnli": dict.fromkeys(LABELS, 5),
        "snli": dict.fromkeys(LABELS, 5),
    }
    assert manifest["splits"]["val"]["counts"] == {
        "eli5": dict.fromkeys(LABELS, 3),
        "mnli": dict.fromkeys(LABELS, 2),
        "snli": dict.fromkeys(LABELS, 2),
    }
    # Test is ELI5 only, in full.
    assert manifest["splits"]["test"]["total"] == len(env.test_rows)
    assert set(manifest["splits"]["test"]["counts"]) == {"eli5"}
    # MNLI/SNLI train and val come from their training-side splits.
    assert ("nyu-mll/multi_nli", "train") in env.loaded
    assert ("nyu-mll/multi_nli", "validation_mismatched") in env.loaded
    assert ("stanfordnlp/snli", "validation") in env.loaded


def test_balanced_test_head_and_order(env):
    assert nli.main(env.args) == 0
    test = json.loads((env.out / "nli_mix_test_raw.json").read_text())
    assert collections.Counter(r["label"] for r in test[:9]) == dict.fromkeys(LABELS, 3)
    # Everything else follows in file order, and nothing is lost or duplicated.
    head = keys(test[:9])
    rest = [r for r in env.test_rows if (r["premise"].strip(), r["hypothesis"].strip()) not in head]
    assert [(r["premise"], r["label"]) for r in test[9:]] == [(r["premise"], r["label"]) for r in rest]
    manifest = json.loads((env.out / "nli_mix_manifest.json").read_text())
    assert manifest["splits"]["test"]["head"]["label_counts"] == dict.fromkeys(LABELS, 3)


def test_no_leakage_across_splits(env):
    assert nli.main(env.args) == 0
    raw = {s: json.loads((env.out / f"nli_mix_{s}_raw.json").read_text()) for s in ("train", "val", "test")}
    bench = set()
    for stem in ("snli_test", "mnli_validation_matched", "anli_test_r1", "anli_test_r2", "anli_test_r3"):
        rows = read_jsonl(env.out / f"{stem}_chat.jsonl")
        assert len(rows) == 60  # Benchmarks are kept in full.
        for row in rows:
            content = row["input"][0]["content"]
            premise = content.split("Text A (premise): ")[1].split("\n\nText B")[0]
            hypothesis = content.split("Text B (hypothesis): ")[1].split("\n\n")[0]
            bench.add((premise.strip(), hypothesis.strip()))
    test = keys(raw["test"]) | bench
    assert not keys(raw["val"]) & test
    assert not keys(raw["train"]) & (test | keys(raw["val"]))
    for split in ("train", "val"):
        assert len(keys(raw[split])) == len(raw[split])
    manifest = json.loads((env.out / "nli_mix_manifest.json").read_text())
    assert manifest["splits"]["val"]["excluded_overlap"]["snli"] == 1
    assert manifest["splits"]["train"]["excluded_overlap"]["eli5"] == 1


def test_files_are_row_aligned_and_thinking_off(env):
    assert nli.main(env.args) == 0
    for split in ("train", "val", "test"):
        raw = json.loads((env.out / f"nli_mix_{split}_raw.json").read_text())
        formatted = read_jsonl(env.out / f"nli_mix_{split}_formatted.jsonl")
        chat = read_jsonl(env.out / f"nli_mix_{split}_chat.jsonl")
        assert len(raw) == len(formatted) == len(chat)
        for r, f, c in zip(raw, formatted, chat):
            assert r["premise"] in c["input"][0]["content"]
            assert f["input"] == "<user>" + c["input"][0]["content"] + "<assistant>"
            assert f["output"] == c["output"] == json.dumps({"label": r["label"]})
    assert env.tokenizer.thinking and not any(env.tokenizer.thinking)


def test_same_seed_reproduces_and_overwrite_is_guarded(env, capsys):
    assert nli.main(env.args) == 0
    first = {p.name: p.read_text() for p in env.out.iterdir()}
    assert nli.main(env.args) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert nli.main(env.args + ["--overwrite"]) == 0
    assert {p.name: p.read_text() for p in env.out.iterdir()} == first
    assert nli.main(env.args + ["--overwrite", "--seed", "7"]) == 0
    assert (env.out / "nli_mix_train_raw.json").read_text() != first["nli_mix_train_raw.json"]


def test_short_label_supply_fails(env, capsys):
    args = [a if a != "eli5=30,mnli=15,snli=15" else "eli5=93,mnli=15,snli=15" for a in env.args]
    assert nli.main(args) == 1
    assert "usable" in capsys.readouterr().err


def test_unbalanceable_test_head_fails(env, capsys):
    rows = [r for r in env.test_rows if r["label"] != "neutral"] + env.test_rows[-1:]
    test_path = env.args[env.args.index("--eli5-test") + 1]
    with open(test_path, "w") as f:
        json.dump(rows + [{"premise": "x", "hypothesis": "y", "label": "neutral"}], f)
    assert nli.main(env.args) == 1
    assert "balanced head" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("eli5=3,mnli=6", {"eli5": 3, "mnli": 6}),
        ("", {}),
        ("snli=0", {"snli": 0}),
    ],
)
def test_parse_mix(spec, expected):
    assert nli.parse_mix(spec) == expected


@pytest.mark.parametrize("spec", ["eli5", "eli5=4", "eli5=-3", "eli5=x", "anli=3", "eli5=3,eli5=6"])
def test_parse_mix_rejects(spec):
    with pytest.raises(ValueError):
        nli.parse_mix(spec)
