# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for run.sh: retries of a finished run, the shared Hub cache, and slow-tokenizer files."""

import shutil

import pytest
from mergekit_test_helpers import (  # pylint: disable=unused-import
    _markers,
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
    retry (same run id, same output path) registers that output rather than fail."""
    first = _run(env, **{"run-id": "tsr-1"})
    assert first.returncode == 0, first.stderr
    env["argv_file"].unlink()
    retry = _run(env, **{"run-id": "tsr-1"})
    assert retry.returncode == 0, retry.stderr
    assert _markers(retry) == _markers(first)
    assert "earlier attempt of this run" in retry.stdout
    assert not env["argv_file"].exists()  # no second merge
    # The record lives beside the output, not in the model dir a push would upload.
    out = env["work"] / "out" / "model"
    assert not list(out.glob("*mergekit-done*"))
    assert (out.parent / ".model.mergekit-done").is_file()


@pytest.mark.parametrize("second", ["tsr-2", ""])
def test_another_run_still_refuses_an_existing_output(env, second):
    """Only the run that wrote the output may re-register it."""
    assert _run(env, **{"run-id": "tsr-1"}).returncode == 0
    _refused(
        _run(env, **{"run-id": second}),
        "output path already exists and is not an empty directory",
    )


def test_no_run_id_writes_no_record(env):
    """A direct run (no --run-id) leaves nothing beside the output."""
    assert _run(env).returncode == 0
    assert not list((env["work"] / "out").glob(".*mergekit-done"))


def test_a_failed_merge_writes_no_record(env):
    _refused(
        _run(env, extra_env={"STUB_FAIL": "1"}, **{"run-id": "tsr-1"}),
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
    _refused(_run(env, **{"run-id": "tsr-1"}), "did not land")
    assert not (out.parent / ".model.mergekit-done").exists()
