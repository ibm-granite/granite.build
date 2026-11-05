"""Unit tests for distill_probe.py.

The probes themselves need the image's torch, transformers and a GPU. What is tested
here is everything around them that can be wrong on a laptop: the scratch copy never
touching the checkpoint, the arguments, the per-byte normalisation, and -- with a stub
transformers -- that the tokenizer probe's verdict counts what it says it counts.
"""

import inspect
import json
import sys
import types
from pathlib import Path

import distill_probe
import pytest
from distill_probe import (
    PINNED_CLASS,
    SPLIT_PROBES,
    TOKENIZER_FILES,
    nll_per_byte,
    parse_args,
    read_conversations,
    stage_tokenizer_copies,
)


def _checkpoint(tmp_path, declared="GPT2Tokenizer", files=TOKENIZER_FILES):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for name in files:
        (ckpt / name).write_text("{}" if name.endswith(".json") else "x")
    (ckpt / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": declared})
    )
    (ckpt / "model.safetensors").write_text("weights")
    return ckpt


class TestStageTokenizerCopies:
    def test_only_the_pinned_copy_is_rewritten(self, tmp_path, capsys):
        ckpt = _checkpoint(tmp_path)
        before = {p.name: p.read_bytes() for p in ckpt.iterdir()}

        asis, pinned = stage_tokenizer_copies(ckpt, tmp_path / "work")

        def cls(d):
            return json.loads((d / "tokenizer_config.json").read_text())[
                "tokenizer_class"
            ]

        assert cls(asis) == "GPT2Tokenizer"
        assert cls(pinned) == PINNED_CLASS
        # Writing tokenizer_class into the real checkpoint would silently change what
        # the recorded eval row refers to.
        assert {p.name: p.read_bytes() for p in ckpt.iterdir()} == before
        assert "PROBE declared_class GPT2Tokenizer" in capsys.readouterr().out

    def test_the_weights_are_not_copied(self, tmp_path):
        asis, pinned = stage_tokenizer_copies(_checkpoint(tmp_path), tmp_path / "w")
        assert not (asis / "model.safetensors").exists()
        assert not (pinned / "model.safetensors").exists()

    def test_an_absent_file_is_reported_not_fatal(self, tmp_path, capsys):
        files = tuple(f for f in TOKENIZER_FILES if f != "merges.txt")
        stage_tokenizer_copies(_checkpoint(tmp_path, files=files), tmp_path / "w")
        assert "PROBE absent merges.txt" in capsys.readouterr().out


def test_the_split_probes_exercise_the_regex():
    """A probe set of plain ASCII words would report SAME whether or not the override
    fired. Contractions, a leading space, digits and a newline are where plain
    ByteLevel and Sequence[Split,ByteLevel] disagree."""
    for probe in ("It's 42 degrees.", " leading space", "don't 1234 tokenize"):
        assert probe in SPLIT_PROBES
    assert "a\nb" in SPLIT_PROBES
    assert "<|start_of_role|>assistant<|end_of_role|>" in SPLIT_PROBES


def test_conversations_skip_rows_without_messages(tmp_path):
    corpus = tmp_path / "c.jsonl"
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(5)]
    rows.insert(1, {"messages": []})
    rows.insert(3, {"conversations": []})
    corpus.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = read_conversations(str(corpus), 3)
    assert [m[0]["content"] for m in got] == ["0", "1", "2"]


def test_nll_is_normalised_per_byte():
    """The coarser segmentation wins per-token by construction; bytes are the
    invariant."""
    assert nll_per_byte(10.0, 4) == 2.5
    assert nll_per_byte(0.0, 0) == 0.0


def test_the_fit_probe_weights_the_mean_loss_by_position_count():
    """HF returns the MEAN nll over predicted positions, so summing raw losses across
    rows of different lengths weights a short row equally with a long one."""
    src = inspect.getsource(distill_probe.tokenizer_fit)
    assert "n_pred = ids.shape[1] - 1" in src
    assert "out.loss.float().item() * n_pred" in src
    assert 'text.encode("utf-8")' in src


def test_the_student_probe_tries_both_attention_implementations():
    """ "Loads, but FA2 does not resolve" is a different diagnosis from "does not
    load", so a failure must say which, and why."""
    src = inspect.getsource(distill_probe.load_student)
    assert '("flash_attention_2", "eager")' in src
    assert "traceback.print_exc()" in src
    assert "logits_scaling" in src and "tie_word_embeddings" in src


class TestArgs:
    @pytest.mark.parametrize("probe", ["load-student", "tokenizer"])
    def test_a_probe_needs_only_a_checkpoint(self, probe):
        args = parse_args([probe, "--checkpoint", "/c", "--corpus", ""])
        assert (args.probe, args.checkpoint) == (probe, "/c")

    @pytest.mark.parametrize(
        "argv",
        [
            ["nonsense", "--checkpoint", "/c"],
            ["tokenizer", "--checkpoint", ""],
            ["tokenizer-fit", "--checkpoint", "/c", "--corpus", ""],
            ["tokenizer-fit", "--checkpoint", "/c", "--corpus", "/x", "--rows", "0"],
        ],
    )
    def test_a_bad_invocation_is_refused(self, argv):
        with pytest.raises(SystemExit):
            parse_args(argv)


class _FakeTokenizer:
    """Pinned segments by character; as-is does too, except that it splits on
    whitespace first. So exactly the probes holding a space or a newline differ."""

    def __init__(self, path):
        cfg = json.loads((path / "tokenizer_config.json").read_text())
        self.pinned = cfg["tokenizer_class"] == PINNED_CLASS
        self.backend_tokenizer = types.SimpleNamespace(pre_tokenizer="stub")

    def __call__(self, text):
        if self.pinned or not any(c.isspace() for c in text):
            pieces = list(text)
        else:
            pieces = text.split()
        return {"input_ids": [hash(p) % 1000 for p in pieces]}


def test_the_tokenizer_probe_counts_the_probes_that_differ(
    tmp_path, monkeypatch, capsys
):
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda p: _FakeTokenizer(Path(p))
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(distill_probe.tempfile, "mkdtemp", lambda: str(tmp_path / "s"))

    ckpt = _checkpoint(tmp_path)
    assert distill_probe.main(["tokenizer", "--checkpoint", str(ckpt)]) == 0

    out = capsys.readouterr().out
    differing = sum(any(c.isspace() for c in s) for s in SPLIT_PROBES)
    assert f"PROBE VERDICT differing={differing}/{len(SPLIT_PROBES)}" in out
    assert "PROBE DONE tokenizer" in out
