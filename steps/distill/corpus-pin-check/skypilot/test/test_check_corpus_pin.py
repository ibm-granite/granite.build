"""Unit tests for check_corpus_pin.py: what is compared, and that all of it is reported."""

import json

import pytest
from check_corpus_pin import main, pin_problems

_BUILD = dict(
    teacher="/models/granite-4.1-3b-pinned/",
    max_length=4096,
    think_policy="strip",
    documents_policy="keep",
    eval_fraction=0.01,
)


def _manifest(**over):
    m = {
        "tokenizer_identity": "granite-4.1-3b-pinned",
        "policies": {
            "max_length": 4096,
            "think_policy": "strip",
            "documents_policy": "keep",
            "completion_boundary": "last_message",
        },
        "eval_fraction": 0.01,
    }
    m.update(over)
    return m


def _corpus(tmp_path, manifest):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("train.jsonl", "eval.jsonl"):
        (corpus / name).write_text("{}\n")
    (corpus / "corpus_manifest.json").write_text(json.dumps(manifest))
    return corpus


def _tok(path, template="tpl"):
    path.mkdir()
    (path / "tokenizer.json").write_text("{}")
    (path / "chat_template.jinja").write_text(template)
    return path


def test_a_matching_pin_has_no_problems(tmp_path):
    corpus = _corpus(tmp_path, _manifest())
    problems, mode = pin_problems(
        corpus, _manifest(), tokenizer_dir=tmp_path / "tok", **_BUILD
    )
    assert problems == []
    assert mode.startswith("identity-only")


def test_a_corpus_with_no_eval_split_needs_no_eval_jsonl(tmp_path):
    """prep_corpus writes eval.jsonl only when eval_fraction > 0, and forces 0 for a
    merged corpus, so its absence there is the consistent state, not a cleaned-up pin.
    """
    build = dict(_BUILD, eval_fraction=0.0)
    corpus = _corpus(tmp_path, _manifest(eval_fraction=0.0))
    (corpus / "eval.jsonl").unlink()
    problems, _ = pin_problems(
        corpus, _manifest(eval_fraction=0.0), tokenizer_dir=tmp_path / "tok", **build
    )
    assert problems == []


def test_a_corpus_with_an_eval_split_still_needs_eval_jsonl(tmp_path):
    corpus = _corpus(tmp_path, _manifest())
    (corpus / "eval.jsonl").unlink()
    problems, _ = pin_problems(
        corpus, _manifest(), tokenizer_dir=tmp_path / "tok", **_BUILD
    )
    assert problems == [f"eval.jsonl is missing from {corpus}"]


def test_every_mismatch_is_reported_not_just_the_first(tmp_path):
    corpus = tmp_path / "empty"
    corpus.mkdir()
    manifest = _manifest(
        tokenizer_identity="granite-4.0-350m",
        policies={"max_length": 2048, "think_policy": "keep"},
        eval_fraction=0.05,
    )
    problems, _ = pin_problems(corpus, manifest, tokenizer_dir=tmp_path, **_BUILD)
    joined = "\n".join(problems)
    for needle in (
        "train.jsonl is missing",
        "eval.jsonl is missing",
        "tokenizer_identity 'granite-4.0-350m'",
        "policies.max_length 2048 != 4096",
        "policies.think_policy 'keep' != 'strip'",
        "policies.documents_policy None != 'keep'",
        "policies.completion_boundary None != 'last_message'",
        "eval_fraction 0.05 != 0.01",
    ):
        assert needle in joined
    assert len(problems) == 8


def test_a_surviving_tokenizer_path_is_byte_compared(tmp_path):
    pinned = _tok(tmp_path / "pinned", template="old")
    here = _tok(tmp_path / "here", template="new")
    manifest = _manifest(tokenizer_path=str(pinned))
    corpus = _corpus(tmp_path, manifest)

    problems, mode = pin_problems(corpus, manifest, tokenizer_dir=here, **_BUILD)

    assert mode == "byte-compared against the pinned tokenizer_path"
    assert len(problems) == 1
    assert problems[0].startswith("chat_template.jinja differs")


def test_no_tokenizer_path_never_compares_against_the_cwd(tmp_path, monkeypatch):
    """Path("") is ".", which is a directory: the inline original compared against
    whatever the job's working directory held."""
    monkeypatch.chdir(tmp_path)
    corpus = _corpus(tmp_path, _manifest())
    problems, mode = pin_problems(
        corpus, _manifest(), tokenizer_dir=tmp_path / "tok", **_BUILD
    )
    assert problems == [] and mode.startswith("identity-only")


def _argv(corpus, tok, out):
    return [
        "x",
        str(corpus),
        _BUILD["teacher"],
        "4096",
        "strip",
        "keep",
        "0.01",
        str(tok),
        str(out),
    ]


def test_main_writes_the_report_and_registers_it(tmp_path, capsys):
    corpus = _corpus(tmp_path, _manifest(seed=7))
    out = tmp_path / "pin" / "corpus_pin.json"

    assert main(_argv(corpus, tmp_path, out)) == 0

    report = json.loads(out.read_text())
    assert report["accepted"] is True and report["seed"] == 7
    assert f"GB_ARTIFACT_ID:pin_check GB_ARTIFACT_PATH:{out}" in capsys.readouterr().out


def test_main_refuses_with_the_whole_list(tmp_path, capsys):
    corpus = _corpus(tmp_path, _manifest(eval_fraction=0.5))
    with pytest.raises(SystemExit, match=r"FATAL \[corpus-pin-check\]: 1 mismatch"):
        main(_argv(corpus, tmp_path, tmp_path / "o.json"))
    assert "CORPUS-PIN REJECTED" in capsys.readouterr().out


def test_main_refuses_a_corpus_with_no_manifest(tmp_path):
    with pytest.raises(SystemExit, match="no corpus_manifest.json"):
        main(_argv(tmp_path, tmp_path, tmp_path / "o.json"))
