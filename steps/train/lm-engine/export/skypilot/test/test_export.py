# SPDX-License-Identifier: Apache-2.0

"""Orchestration tests for ``src/export.py``.

The export driver's job is to sequence upstream's two stages and then refuse to
publish anything it has not verified. Those two properties are what is tested here,
with the stage functions monkeypatched: the real stages need torch, transformers and
lm-engine, which only exist inside the family image, and the sequencing is where the
bugs that matter live.

The single most load-bearing assertion in this file is that
``GB_ARTIFACT_ID:hf_model`` is *not* printed on any failure path. gbserver scrapes
that line from stdout to bind the target's ``hf_model`` output; printing it for an
export that failed a check would hand a broken model to every downstream eval, which
is the exact shape of the incidents ``export_checks.py`` exists to stop.
"""

import json
import pathlib
import subprocess
import sys

import export
import pytest

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

ARTIFACT = "GB_ARTIFACT_ID:hf_model"


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def tree(tmp_path):
    """A train_dir with one rung, a tokenizer dir, and the output path."""
    train_dir = tmp_path / "train"
    (train_dir / "global_step2000").mkdir(parents=True)
    (train_dir / "latest_checkpointed_iteration.json").write_text(
        json.dumps({"latest_checkpointed_iteration": 2000})
    )
    tokenizer = tmp_path / "tok"
    tokenizer.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tokenizer / name).write_text((_FIXTURES / "tokenizer" / name).read_text())
    return train_dir, tokenizer, tmp_path / "hf"


@pytest.fixture
def argv(tree):
    train_dir, tokenizer, hf_dir = tree
    return [
        "--train-dir",
        str(train_dir),
        "--output-dir",
        str(hf_dir),
        "--tokenizer-path",
        str(tokenizer),
        "--model-type",
        "granite",
    ]


@pytest.fixture
def stages(monkeypatch, tree):
    """Replace every heavy stage with a recorder; return the call log.

    The fakes still write the files the next stage reads, so ``main`` exercises its
    real control flow (manifest writing, cleanup, the artifact line) over a plausible
    directory, not over nothing.
    """
    _, _, hf_dir = tree
    calls: list[str] = []

    def record(name, result=None, side_effect=None):
        def fake(*a, **k):
            calls.append(name)
            if side_effect is not None:
                side_effect(*a, **k)
            return result

        return fake

    def wrote_unsharded(config_path, *a, **k):
        # unshard's own output: the dir the parity check and the cleanup act on
        unsharded = pathlib.Path(json.loads(config_path.read_text())["unsharded_path"])
        unsharded.mkdir(parents=True, exist_ok=True)
        (unsharded / "config.json").write_text("{}")

    def wrote_hf(unsharded, model_type, out, *a, **k):
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            json.dumps({"model_type": model_type, "vocab_size": 64})
        )

    monkeypatch.setattr(
        export, "run_unshard", record("unshard", side_effect=wrote_unsharded)
    )
    monkeypatch.setattr(export, "convert", record("convert", side_effect=wrote_hf))
    monkeypatch.setattr(
        export, "copy_tokenizer", record("tokenizer", result=["tokenizer.json"])
    )
    monkeypatch.setattr(export, "run_checks", record("checks", result=[]))
    monkeypatch.setattr(export, "check_parity", record("parity", result=0.001))
    monkeypatch.setattr(export, "lm_engine_sha", lambda *a, **k: "a9c6e2b8")
    return calls


def _manifest(hf_dir):
    return json.loads((hf_dir / export.MANIFEST_NAME).read_text())


def _write_manifest(hf_dir, **overrides):
    hf_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "iteration": 2000,
        "lm_engine_sha": "a9c6e2b8",
        "model_type": "granite",
        "parity_passed": True,
        "parity_max_delta": 0.001,
        "parity_tolerance": 0.05,
    }
    record.update(overrides)
    (hf_dir / export.MANIFEST_NAME).write_text(json.dumps(record))


# ------------------------------------------------------------------- ordering


def test_the_stages_run_in_order(argv, stages, tree, capsys):
    assert export.main(argv) == 0
    assert stages == ["unshard", "convert", "tokenizer", "checks", "parity"]


def test_the_tokenizer_is_copied_after_the_conversion(argv, stages):
    """upstream's exporter re-saves a tokenizer; the byte copy must overwrite it."""
    export.main(argv)
    assert stages.index("tokenizer") > stages.index("convert")


def test_the_checks_run_after_the_tokenizer_copy(argv, stages):
    export.main(argv)
    assert stages.index("checks") > stages.index("tokenizer")


def test_a_successful_export_prints_the_artifact_line_once(argv, stages, tree, capsys):
    _, _, hf_dir = tree
    assert export.main(argv) == 0
    out = capsys.readouterr().out
    assert out.count(ARTIFACT) == 1
    assert f"GB_ARTIFACT_PATH:{hf_dir}" in out


def test_a_successful_export_deletes_the_unsharded_dir(argv, stages, tree):
    _, _, hf_dir = tree
    assert export.main(argv) == 0
    assert not export.work_dir(hf_dir).exists()


def test_a_successful_export_writes_the_manifest(argv, stages, tree):
    _, _, hf_dir = tree
    export.main(argv)
    record = _manifest(hf_dir)
    assert record["iteration"] == 2000
    assert record["model_type"] == "granite"
    assert record["lm_engine_sha"] == "a9c6e2b8"
    assert record["parity_passed"] is True
    assert record["parity_max_delta"] == pytest.approx(0.001)
    assert record["parity_tolerance"] == pytest.approx(0.05)
    assert record["dtype"] == "bf16"


def test_the_iteration_is_resolved_from_the_train_dir_when_not_given(
    argv, stages, tree
):
    _, _, hf_dir = tree
    export.main(argv)
    assert _manifest(hf_dir)["iteration"] == 2000


def test_an_explicit_iteration_is_used_verbatim(argv, stages, tree):
    train_dir, _, hf_dir = tree
    (train_dir / "global_step1000").mkdir()
    export.main(argv + ["--iteration", "1000"])
    assert _manifest(hf_dir)["iteration"] == 1000


def test_an_iteration_the_train_dir_does_not_hold_is_refused(
    argv, stages, tree, capsys
):
    assert export.main(argv + ["--iteration", "999"]) == 2
    assert stages == []
    assert ARTIFACT not in capsys.readouterr().out


def test_the_unshard_config_carries_the_load_path_and_iteration(
    argv, monkeypatch, tree, stages
):
    seen = {}

    def capture(config_path):
        seen.update(json.loads(config_path.read_text()))
        unsharded = pathlib.Path(seen["unsharded_path"])
        unsharded.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(export, "run_unshard", capture)
    train_dir, _, hf_dir = tree
    export.main(argv)
    assert seen["load_args"]["load_path"] == str(train_dir)
    assert seen["load_args"]["iteration"] == 2000
    assert seen["mixed_precision_args"]["dtype"] == "bf16"
    assert seen["unsharded_path"] == str(export.unsharded_dir(hf_dir))


# ------------------------------------------------------------------- failures


@pytest.mark.parametrize(
    "stage", ["run_unshard", "convert", "copy_tokenizer", "check_parity"]
)
def test_a_stage_raising_fails_the_step_without_the_artifact_line(
    argv, stages, tree, monkeypatch, capsys, stage
):
    def boom(*a, **k):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(export, stage, boom)
    assert export.main(argv) != 0
    captured = capsys.readouterr()
    assert ARTIFACT not in captured.out
    assert "upstream exploded" in captured.out + captured.err


@pytest.mark.parametrize(
    "stage", ["run_unshard", "convert", "copy_tokenizer", "check_parity"]
)
def test_a_stage_raising_keeps_the_unsharded_dir_for_debugging(
    argv, stages, tree, monkeypatch, stage
):
    def boom(*a, **k):
        raise RuntimeError("upstream exploded")

    monkeypatch.setattr(export, stage, boom)
    _, _, hf_dir = tree
    export.main(argv)
    # run_unshard is the one stage that fails before anything is written into the
    # work dir; the work dir itself must still be there either way.
    assert export.work_dir(hf_dir).is_dir()


def test_a_failed_check_exits_non_zero_and_prints_every_error(
    argv, stages, tree, monkeypatch, capsys
):
    monkeypatch.setattr(
        export,
        "run_checks",
        lambda *a, **k: ["rope_theta is absent", "tokenizer.json differs"],
    )
    assert export.main(argv) == 1
    captured = capsys.readouterr()
    assert ARTIFACT not in captured.out
    text = captured.out + captured.err
    assert "rope_theta is absent" in text
    assert "tokenizer.json differs" in text


def test_a_failed_check_does_not_run_parity_or_delete_the_unsharded_dir(
    argv, stages, tree, monkeypatch
):
    monkeypatch.setattr(export, "run_checks", lambda *a, **k: ["nope"])
    _, _, hf_dir = tree
    export.main(argv)
    assert "parity" not in stages
    assert export.unsharded_dir(hf_dir).is_dir()


def test_a_failed_check_writes_no_manifest(argv, stages, tree, monkeypatch):
    monkeypatch.setattr(export, "run_checks", lambda *a, **k: ["nope"])
    _, _, hf_dir = tree
    export.main(argv)
    assert not (hf_dir / export.MANIFEST_NAME).exists()


def test_parity_outside_the_tolerance_fails_the_step(
    argv, stages, tree, monkeypatch, capsys
):
    monkeypatch.setattr(export, "check_parity", lambda *a, **k: 0.5)
    assert export.main(argv) == 1
    captured = capsys.readouterr()
    assert ARTIFACT not in captured.out
    text = captured.out + captured.err
    assert "5.000e-01" in text and "0.05" in text


def test_parity_outside_the_tolerance_keeps_the_unsharded_dir(
    argv, stages, tree, monkeypatch
):
    monkeypatch.setattr(export, "check_parity", lambda *a, **k: 0.5)
    _, _, hf_dir = tree
    export.main(argv)
    assert export.unsharded_dir(hf_dir).is_dir()


def test_a_failed_parity_is_recorded_in_the_manifest(argv, stages, tree, monkeypatch):
    """So a re-run sees a recorded failure and exports again rather than skipping."""
    monkeypatch.setattr(export, "check_parity", lambda *a, **k: 0.5)
    _, _, hf_dir = tree
    export.main(argv)
    record = _manifest(hf_dir)
    assert record["parity_passed"] is False
    assert record["parity_max_delta"] == pytest.approx(0.5)


def test_the_tolerance_is_configurable(argv, stages, tree, monkeypatch):
    monkeypatch.setattr(export, "check_parity", lambda *a, **k: 0.5)
    assert export.main(argv + ["--parity-tolerance", "1.0"]) == 0


# ---------------------------------------------------------------- idempotence


def test_a_matching_manifest_skips_the_work_and_prints_the_artifact_line(
    argv, stages, tree, capsys
):
    _, _, hf_dir = tree
    _write_manifest(hf_dir)
    assert export.main(argv) == 0
    assert stages == []
    assert capsys.readouterr().out.count(ARTIFACT) == 1


@pytest.mark.parametrize(
    "override",
    [
        {"iteration": 1000},
        {"lm_engine_sha": "deadbeef"},
        {"model_type": "llama"},
    ],
    ids=["iteration", "lm_engine_sha", "model_type"],
)
def test_a_mismatched_manifest_re_runs_the_export(argv, stages, tree, override):
    _, _, hf_dir = tree
    _write_manifest(hf_dir, **override)
    assert export.main(argv) == 0
    assert stages == ["unshard", "convert", "tokenizer", "checks", "parity"]


def test_a_manifest_recording_a_failed_parity_re_runs_the_export(argv, stages, tree):
    _, _, hf_dir = tree
    _write_manifest(hf_dir, parity_passed=False)
    assert export.main(argv) == 0
    assert stages == ["unshard", "convert", "tokenizer", "checks", "parity"]


def test_a_manifest_missing_the_parity_flag_re_runs_the_export(argv, stages, tree):
    _, _, hf_dir = tree
    _write_manifest(hf_dir)
    record = _manifest(hf_dir)
    del record["parity_passed"]
    (hf_dir / export.MANIFEST_NAME).write_text(json.dumps(record))
    assert export.main(argv) == 0
    assert stages == ["unshard", "convert", "tokenizer", "checks", "parity"]


@pytest.mark.parametrize("text", ["", "{", "[]", "null", '"granite"'])
def test_a_malformed_manifest_re_runs_the_export_without_crashing(
    argv, stages, tree, text
):
    _, _, hf_dir = tree
    hf_dir.mkdir(parents=True, exist_ok=True)
    (hf_dir / export.MANIFEST_NAME).write_text(text)
    assert export.main(argv) == 0
    assert stages == ["unshard", "convert", "tokenizer", "checks", "parity"]


def test_no_manifest_at_all_runs_the_export(argv, stages, tree):
    assert export.main(argv) == 0
    assert stages == ["unshard", "convert", "tokenizer", "checks", "parity"]


# -------------------------------------------------------------------- helpers


def test_the_expected_dict_is_read_from_a_json_file(
    argv, stages, tree, monkeypatch, tmp_path
):
    seen = {}
    monkeypatch.setattr(
        export, "run_checks", lambda hf, tok, expected: seen.update(expected) or []
    )
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({"rope_theta": 10000000}))
    export.main(argv + ["--expected", str(path)])
    assert seen == {"rope_theta": 10000000}


def test_no_expected_file_means_an_empty_expectation(argv, stages, tree, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        export, "run_checks", lambda hf, tok, expected: seen.update(x=expected) or []
    )
    export.main(argv)
    assert seen["x"] == {}


def test_an_unreadable_expected_file_is_refused_before_any_work(
    argv, stages, tree, tmp_path, capsys
):
    path = tmp_path / "expected.json"
    path.write_text("{not json")
    assert export.main(argv + ["--expected", str(path)]) == 2
    assert stages == []
    assert ARTIFACT not in capsys.readouterr().out


def test_a_missing_tokenizer_dir_is_refused_before_any_work(argv, stages, tree, capsys):
    bad = [a if a != str(tree[1]) else str(tree[1]) + "-gone" for a in argv]
    assert export.main(bad) == 2
    assert stages == []
    assert ARTIFACT not in capsys.readouterr().out


def test_a_missing_train_dir_is_refused_before_any_work(argv, stages, tree, capsys):
    bad = [a if a != str(tree[0]) else str(tree[0]) + "-gone" for a in argv]
    assert export.main(bad) == 2
    assert stages == []
    assert ARTIFACT not in capsys.readouterr().out


def test_an_unknown_model_type_is_refused_before_any_work(argv, stages, capsys):
    with pytest.raises(SystemExit):
        export.main(argv[:-1] + ["qwen"])
    assert stages == []


def test_the_lm_engine_sha_falls_back_to_none_when_git_is_unavailable(tmp_path):
    """The image has /app/src; a laptop or a stripped image does not."""
    assert export.lm_engine_sha(str(tmp_path / "nope")) is None


def test_the_artifact_line_names_the_output_dir(tmp_path):
    line = export.artifact_line(tmp_path / "hf")
    assert line == f"GB_ARTIFACT_ID:hf_model GB_ARTIFACT_PATH:{tmp_path / 'hf'}"


# ------------------------------------------------------- the tokenizer copy itself


def _tokenizer_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        source = _FIXTURES / "tokenizer" / name
        (path / name).write_bytes(source.read_bytes())
    return path


def test_copy_tokenizer_copies_the_files_byte_for_byte(tmp_path):
    src = _tokenizer_dir(tmp_path / "tok")
    dst = tmp_path / "hf"
    copied = export.copy_tokenizer(src, dst)
    assert copied == [
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for name in copied:
        assert (dst / name).read_bytes() == (src / name).read_bytes()


def test_copy_tokenizer_overwrites_the_file_upstream_re_saved(tmp_path):
    """``export_to_huggingface`` ends in ``tokenizer.save_pretrained``; this wins."""
    src = _tokenizer_dir(tmp_path / "tok")
    dst = tmp_path / "hf"
    dst.mkdir()
    (dst / "tokenizer.json").write_text('{"truncation": {"max_length": 466}}')
    export.copy_tokenizer(src, dst)
    assert (dst / "tokenizer.json").read_bytes() == (
        src / "tokenizer.json"
    ).read_bytes()


def test_copy_tokenizer_removes_a_tokenizer_file_the_source_does_not_have(tmp_path):
    """A synthesised file left next to the copied ones would fail check_tokenizer."""
    src = _tokenizer_dir(tmp_path / "tok")
    dst = tmp_path / "hf"
    dst.mkdir()
    (dst / "added_tokens.json").write_text("{}")
    export.copy_tokenizer(src, dst)
    assert not (dst / "added_tokens.json").exists()


def test_copy_tokenizer_leaves_the_model_files_alone(tmp_path):
    src = _tokenizer_dir(tmp_path / "tok")
    dst = tmp_path / "hf"
    dst.mkdir()
    (dst / "config.json").write_text("{}")
    (dst / "model.safetensors").write_bytes(b"weights")
    export.copy_tokenizer(src, dst)
    assert (dst / "config.json").exists()
    assert (dst / "model.safetensors").read_bytes() == b"weights"


# ------------------------------------------------------------- run_checks itself


def test_run_checks_reports_a_config_that_was_never_written(tmp_path):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    errors = export.run_checks(hf_dir, _tokenizer_dir(tmp_path / "tok"), {})
    assert len(errors) == 1
    assert "config.json" in errors[0]


def test_run_checks_reports_a_config_that_is_not_json(tmp_path):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{not json")
    errors = export.run_checks(hf_dir, _tokenizer_dir(tmp_path / "tok"), {})
    assert len(errors) == 1
    assert "config.json" in errors[0]


def test_run_checks_runs_both_halves_not_just_the_first(tmp_path):
    """One failed export should say everything that is wrong with it."""
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(
        json.dumps({"model_type": "granite", "rope_theta": 10000})
    )
    src = _tokenizer_dir(tmp_path / "tok")
    errors = export.run_checks(hf_dir, src, {"rope_theta": 10000000})
    assert any("rope_theta" in e for e in errors)
    assert any("tokenizer.json" in e for e in errors)


# --------------------------------------------------------------------- purity


def test_the_module_imports_without_torch_or_transformers():
    """Heavy imports live inside the stages, so the module loads on a CPU pod.

    ``python -I`` drops ``cwd`` and ``PYTHONPATH``, so ``src/`` is injected here.
    """
    code = (
        f"import sys; sys.path.insert(0, {str(_SRC)!r}); import export;"
        "heavy = {m for m in sys.modules if m.split('.')[0] in"
        " {'torch', 'transformers', 'lm_engine', 'tools', 'safetensors'}};"
        "assert not heavy, heavy; print(export.ARTIFACT_ID)"
    )
    out = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "hf_model"
