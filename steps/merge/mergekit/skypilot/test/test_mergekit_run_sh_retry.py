# SPDX-License-Identifier: Apache-2.0

"""Tests for run.sh: retries of a finished run, the shared Hub cache, and slow-tokenizer files."""

import shutil

import pytest
from mergekit_test_helpers import (  # pylint: disable=unused-import
    _RECIPE,
    _markers,
    _model,
    _refused,
    _run,
    _stub_argv,
    env_fixture,
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


def test_hf_cache_is_on_the_shared_workdir(env):
    """Hub downloads are cached on GB_SHARED_WORKDIR (once per cluster, like the uv
    cache); the LoRA merge cache stays in the run's own workdir."""
    shared = env["tmp"] / "shared"
    shared.mkdir()
    proc = _run(env, extra_env={"GB_SHARED_WORKDIR": str(shared)})
    assert proc.returncode == 0, proc.stderr
    argv = _stub_argv(env)
    assert argv[argv.index("--transformers-cache") + 1] == str(
        shared / ".mergekit-hf-cache"
    )
    assert argv[argv.index("--lora-merge-cache") + 1] == str(
        env["work"] / ".mergekit" / "lora-cache"
    )


@pytest.mark.parametrize(
    "vocab", ["vocab.json", "vocab.txt", "spiece.model", "tokenizer.tiktoken"]
)
def test_slow_tokenizer_files_count_as_a_tokenizer(env, vocab):
    """A slow tokenizer re-saved by MergeKit has no tokenizer.json, only its own vocab."""
    proc = _run(env, extra_env={"STUB_VOCAB": vocab})
    assert proc.returncode == 0, proc.stderr
    assert "GB_ARTIFACT_ID:model" in proc.stdout


def test_tokenizer_config_alone_is_not_a_tokenizer(env):
    _refused(
        _run(env, extra_env={"STUB_VOCAB": "special_tokens_map.json"}),
        "the merge wrote no tokenizer",
    )


def test_retry_of_the_same_run_registers_the_finished_output(env):
    """An attempt finished the merge and moved it into place, then the job was lost; the
    retry (same retry key and inputs, same output path) registers that output rather
    than fail."""
    first = _run(env, **{"retry-key": "b-1/merge/0"})
    assert first.returncode == 0, first.stderr
    env["argv_file"].unlink()
    retry = _run(env, **{"retry-key": "b-1/merge/0"})
    assert retry.returncode == 0, retry.stderr
    # The same markers, plus one saying the output was reused (what the build test
    # asserts, so it can tell a reuse from a second merge).
    assert _markers(retry) == [
        "GB_STEP_METADATA_KEY:reused_earlier_attempt GB_STEP_METADATA_VALUE:true",
        *_markers(first),
    ]
    assert "earlier attempt of this step" in retry.stdout
    assert not env["argv_file"].exists()  # no second merge
    # The record lives beside the output, not in the model dir a push would upload.
    out = env["work"] / "out" / "model"
    assert not list(out.glob("*mergekit-done*"))
    assert (out.parent / ".model.mergekit-done").is_file()


@pytest.mark.parametrize("second", ["b-2/merge/0", ""])
def test_another_run_still_refuses_an_existing_output(env, second):
    """Only a retry of the step that wrote the output may re-register it."""
    assert _run(env, **{"retry-key": "b-1/merge/0"}).returncode == 0
    _refused(
        _run(env, **{"retry-key": second}),
        "output path already exists and is not an empty directory",
    )


def test_no_retry_key_writes_no_record(env):
    """A direct run (no --retry-key) leaves nothing beside the output."""
    assert _run(env).returncode == 0
    assert not list((env["work"] / "out").glob(".*mergekit-done"))


def test_a_failed_merge_writes_no_record(env):
    _refused(
        _run(env, extra_env={"STUB_FAIL": "1"}, **{"retry-key": "b-1/merge/0"}),
        "mergekit-yaml failed",
    )
    assert not list((env["work"] / "out").glob(".*mergekit-done"))


def test_a_failed_move_removes_the_record(env):
    """If the move nests the stage inside a reappeared output dir, the step fails, and
    the record goes too, so a retry can't register whatever is at the output."""
    out = env["work"] / "out" / "model"
    out.mkdir(parents=True)
    wrapper = env["tmp"] / "bin" / "rmdir"
    # An rmdir that "succeeds" but leaves the dir (it reappeared after the rmdir).
    wrapper.write_text("#!/usr/bin/env bash\nexit 0\n")
    wrapper.chmod(0o755)
    _refused(_run(env, **{"retry-key": "b-1/merge/0"}), "did not land")
    assert not (out.parent / ".model.mergekit-done").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"recipe": {**_RECIPE, "merge_method": "task_arithmetic"}},
        {"args": {"out-shard-size": "1B"}},
        {"copy-tokenizer": "false"},
        {"mergekit": "image:ghcr.io/x/mergekit:2"},
    ],
    ids=["recipe", "args", "flag", "mergekit"],
)
def test_same_key_but_a_changed_merge_is_refused(env, change):
    """The record matches only the merge it was written for: the same key with a changed
    recipe, args, flag or MergeKit (a re-run, not a retry) is refused, not
    re-registered."""
    assert _run(env, **{"retry-key": "b-1/merge/0"}).returncode == 0
    env["argv_file"].unlink()
    _refused(
        _run(env, **{"retry-key": "b-1/merge/0", **change}),
        "output path already exists and is not an empty directory",
    )
    assert not env["argv_file"].exists()


def test_same_key_but_another_input_is_refused(env, tmp_path):
    """Another input under the same key is a re-run: refused, and nothing merges."""
    assert _run(env, **{"retry-key": "b-1/merge/0"}).returncode == 0
    other = _model(tmp_path / "staged" / "other")
    env["argv_file"].unlink()
    _refused(
        _run(env, **{"retry-key": "b-1/merge/0", "input-model_1": other}),
        "output path already exists",
    )
    assert not env["argv_file"].exists()


def test_a_record_from_the_previous_format_is_not_matched(env):
    """A record written by #629's version (`run <step run id>`, no `config` line) never
    matches, so the old refusal applies rather than a wrong re-registration."""
    assert _run(env).returncode == 0  # an output, but no record (no key)
    out = env["work"] / "out" / "model"
    (out.parent / ".model.mergekit-done").write_text(
        "run b-1/merge/0\nGB_ARTIFACT_ID:model GB_ARTIFACT_PATH:/elsewhere\n"
    )
    env["argv_file"].unlink()
    _refused(_run(env, **{"retry-key": "b-1/merge/0"}), "output path already exists")
    assert not env["argv_file"].exists()  # refused before merging, not after


def test_no_retry_key_hashes_nothing(env, tmp_path):
    """Without a key nothing is hashed, so a hashing failure can't fail a direct run."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "hashlib.py").write_text("raise ImportError('no hashlib here')\n")
    pythonpath = {"PYTHONPATH": str(broken)}
    assert _run(env, extra_env=pythonpath).returncode == 0
    # Control: with a key the same environment fails at the hash, as it should.
    shutil.rmtree(env["work"] / "out")
    _refused(
        _run(env, extra_env=pythonpath, **{"retry-key": "b-1/merge/0"}),
        "could not hash the merge's inputs",
    )
