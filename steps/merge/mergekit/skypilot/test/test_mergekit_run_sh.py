# Copyright Granite.Build Authors
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

"""Tests for src/run.sh against a stub mergekit-yaml.

The stub records its argv and writes a one-tensor safetensors file plus config.json,
so these tests check run.sh's own behaviour (paths, staging, flags, markers) without
MergeKit. ``test_real_merge`` at the end runs the real mergekit-yaml on two tiny
random models when mergekit is installed.
"""

import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys

import pytest
import yaml
from mergekit_test_helpers import (  # pylint: disable=unused-import
    _RECIPE,
    _markers,
    _model,
    _real_env,
    _refused,
    _run,
    _run_argv,
    _stub_argv,
    env_fixture,
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


# --- happy path ---------------------------------------------------------------------


def test_success_registers_absolute_output(env):
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    out = env["work"] / "out" / "model"
    assert (out / "model.safetensors").is_file()
    assert _markers(proc) == [
        "GB_STEP_METADATA_KEY:merge_method GB_STEP_METADATA_VALUE:linear",
        _markers(proc)[1],  # mergekit_version: whatever is installed, or unknown
        "GB_STEP_METADATA_KEY:tensor_count GB_STEP_METADATA_VALUE:1",
        f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}",
    ]
    assert not list(out.parent.glob(".mergekit-stage.*"))


def test_output_dir_mode_follows_umask(env):
    """The moved-in output is not left 0700 by mktemp -d."""
    proc = subprocess.run(
        ["bash", "-c", 'umask 022 && exec bash "$@"', "bash", *_run_argv(env)],
        cwd=env["work"],
        env=env["env"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    mode = (env["work"] / "out" / "model").stat().st_mode & 0o777
    assert mode == 0o755


def test_recipe_has_staged_paths(env):
    assert _run(env).returncode == 0
    recipe = yaml.safe_load((env["work"] / ".mergekit" / "merge.yaml").read_text())
    assert [m["model"] for m in recipe["models"]] == [env["m1"], env["base"]]


def test_mergekit_argv(env):
    """Recipe, staging dir, the step-owned caches, then the step and args flags."""
    proc = _run(env, args={"out-shard-size": "1B", "write-model-card": False})
    assert proc.returncode == 0, proc.stderr
    argv = _stub_argv(env)
    work = env["work"] / ".mergekit"
    assert argv[0] == str(work / "merge.yaml")
    # MergeKit writes into <parent>/.mergekit-stage.XXXXXX/<output name>.
    staged = pathlib.Path(argv[1])
    assert staged.parent.name.startswith(".mergekit-stage.")
    assert staged.name == "model"
    assert argv[2:] == [
        "--transformers-cache",
        str(env["work"] / ".mergekit-hf-cache"),  # GB_SHARED_WORKDIR unset: the workdir
        "--lora-merge-cache",
        str(work / "lora-cache"),
        "--copy-tokenizer",
        "--lazy-unpickle",
        "--no-allow-crimes",
        "--no-trust-remote-code",
        "--out-shard-size",
        "1B",
        "--no-write-model-card",
    ]


def test_cuda_only_with_allocated_gpus(env):
    assert _run(env, extra_env={"SKYPILOT_NUM_GPUS_PER_NODE": "1"}).returncode == 0
    assert "--cuda" in _stub_argv(env)
    assert (
        _run(
            env, output="./out2", extra_env={"SKYPILOT_NUM_GPUS_PER_NODE": "0"}
        ).returncode
        == 0
    )
    assert "--cuda" not in _stub_argv(env)


def test_empty_output_dir_is_reused(env):
    (env["work"] / "out" / "model").mkdir(parents=True)
    assert _run(env).returncode == 0


# --- refusals: nothing registered, nothing left behind --------------------------------


def test_merge_failure(env):
    proc = _run(env, extra_env={"STUB_FAIL": "1"})
    _refused(proc, "mergekit-yaml failed (exit 3)")
    out = env["work"] / "out"
    assert not (out / "model").exists()
    assert not list(out.glob(".mergekit-stage.*"))


def test_merge_with_no_tensors(env):
    _refused(_run(env, extra_env={"STUB_EMPTY": "1"}), "the merge wrote no tensors")


def test_nonempty_output_dir(env):
    out = env["work"] / "out" / "model"
    out.mkdir(parents=True)
    (out / "keep.txt").write_text("x")
    _refused(_run(env), "already exists and is not an empty directory")
    assert (out / "keep.txt").read_text() == "x"


def test_output_inside_an_input(env):
    _refused(
        _run(env, output=env["base"] + "/merged"),
        "output path is inside input 'base_model'",
    )


def test_input_inside_the_output(env):
    _refused(_run(env, output=str(env["tmp"] / "staged")), "is inside the output path")


@pytest.mark.parametrize(
    "output, message",
    [
        ("./a/../b", "must not contain '..'"),
        ('./has"quote', "must not contain a '\"'"),
        ("./trailing ", "end in a space"),
    ],
)
def test_bad_output_paths(env, output, message):
    _refused(_run(env, output=output), message)


def test_symlinked_output_is_refused(env):
    """A symlink to an empty dir would get the model nested inside its target."""
    target = env["tmp"] / "elsewhere"
    target.mkdir()
    (env["work"] / "model").symlink_to(target)
    _refused(_run(env, output="./model"), "output path is a symlink")
    assert not list(target.iterdir())


@pytest.mark.parametrize("char", ["@", "+"])
def test_staged_path_with_mergekit_syntax_chars(env, char):
    """'@' and '+' in a staged path would be read by MergeKit as a revision or a LoRA."""
    odd = _model(env["tmp"] / f"stage{char}x" / "m1")
    _refused(_run(env, **{"input-model_1": odd}), "contains '@' or '+'")
    assert not env["argv_file"].exists()


def test_missing_input_dir(env):
    _refused(
        _run(env, **{"input-model_1": "/does/not/exist"}),
        "input model_1 is not a directory",
    )


def test_input_without_config(env):
    bare = env["tmp"] / "bare"
    bare.mkdir()
    _refused(_run(env, **{"input-model_1": str(bare)}), "has no config.json")


def test_prepare_error_stops_before_merge(env):
    _refused(_run(env, args={"cuda": True}), "args.cuda is not allowed")
    assert not env["argv_file"].exists()


def test_mv_that_nests_is_cleaned_up(env):
    """If the empty output dir survives rmdir, mv nests the stage: refuse and clean up."""
    fake_rmdir = env["tmp"] / "bin" / "rmdir"
    fake_rmdir.write_text(
        "#!/usr/bin/env bash\nexit 0\n"
    )  # 'succeeds', removes nothing
    fake_rmdir.chmod(0o755)
    out = env["work"] / "out" / "model"
    out.mkdir(parents=True)
    _refused(_run(env), "the merged model did not land")
    assert out.is_dir() and not list(out.iterdir())
    assert not list(out.parent.glob(".mergekit-stage.*"))


# --- output check: MergeKit swallows tokenizer errors, so check what landed ----------


def test_no_tokenizer_with_copy_tokenizer_is_refused(env):
    proc = _run(env, extra_env={"STUB_NO_TOKENIZER": "1"})
    _refused(proc, "the merge wrote no tokenizer, though one was expected")
    assert not (env["work"] / "out" / "model").exists()


def test_swallowed_tokenizer_error_is_refused_even_if_files_landed(env):
    """MergeKit's "Failed to copy tokenizer" line fails the step, whatever was copied."""
    proc = _run(
        env,
        extra_env={
            "STUB_STDERR": "ERROR:mergekit.merge:Failed to copy tokenizer. The merge was still successful"
        },
    )
    _refused(proc, "MergeKit failed to copy the tokenizer")
    assert not (env["work"] / "out" / "model").exists()


def test_mergekit_stderr_still_reaches_the_log_and_stdout_is_separate(env):
    """stderr reaches the job log and the captured log; stdout stays on stdout only."""
    proc = _run(
        env,
        extra_env={
            "STUB_STDERR": "stub-progress-line",
            "STUB_STDOUT": "stub-stdout-line",
        },
    )
    assert proc.returncode == 0, proc.stderr
    log = (env["work"] / ".mergekit" / "mergekit.log").read_text()
    assert "stub-progress-line" in proc.stderr and "stub-progress-line" in log
    assert "stub-progress-line" not in proc.stdout
    assert "stub-stdout-line" in proc.stdout  # fails if 1>&3 is dropped
    assert "stub-stdout-line" not in log and "stub-stdout-line" not in proc.stderr


def test_capture_fd_is_not_leaked_to_mergekit(env):
    proc = _run(env, extra_env={"STUB_FD3_PROBE": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "fd3-closed" in proc.stderr


def _seed_stale_log(env, mode=0o644):
    log = env["work"] / ".mergekit" / "mergekit.log"
    log.parent.mkdir(parents=True)
    log.write_text("ERROR:mergekit.merge:Failed to copy tokenizer. (earlier run)\n")
    log.chmod(mode)
    return log


def test_stale_log_from_an_earlier_run_is_not_matched(env):
    """A warm re-run starts with a fresh log: the earlier line is gone (fails with tee -a)."""
    log = _seed_stale_log(env)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "earlier run" not in log.read_text()


def test_stale_read_only_log_is_replaced(env):
    """A stale log tee can't open for writing is removed first, not matched (fails
    without the rm: tee can't truncate it, and grep would read the old line)."""
    if os.geteuid() == 0:
        pytest.skip("root can write a mode-0444 file")
    log = _seed_stale_log(env, 0o444)
    proc = _run(env)
    assert proc.returncode == 0, proc.stderr
    assert "earlier run" not in log.read_text()


def test_unreadable_log_fails_the_step(env):
    """If grep can't read the log (exit 2), that is a failure, not "no match"."""
    real_grep = shutil.which("grep")
    fake = env["tmp"] / "bin" / "grep"
    # Fail only the run.sh log check (a -q search of mergekit.log); pass the rest through.
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in *mergekit.log*) exit 2 ;; esac\n'
        f'exec {real_grep} "$@"\n'
    )
    fake.chmod(0o755)
    _refused(_run(env), "could not read MergeKit's log")


def test_unwritable_log_fails_the_step(env):
    """If tee can't write the log, the swallowed-error check can't run: fail."""
    stub = env["tmp"] / "bin" / "tee"
    stub.write_text(
        "#!/usr/bin/env bash\ncat >&2\nexit 1\n"
    )  # forwards, writes no file
    stub.chmod(0o755)
    _refused(_run(env), "could not write MergeKit's log")


def test_exempted_path_still_needs_the_requested_template(env):
    """tokenizer_source + chat_template + copy_tokenizer false: template still checked."""
    recipe = {**_RECIPE, "tokenizer_source": "base", "chat_template": "auto"}
    _refused(
        _run(env, recipe=recipe, **{"copy-tokenizer": "false"}),
        "has no chat template",
    )


def test_exempted_path_still_needs_a_tokenizer(env):
    recipe = {**_RECIPE, "tokenizer_source": "base"}
    _refused(
        _run(
            env,
            recipe=recipe,
            extra_env={"STUB_NO_TOKENIZER": "1"},
            **{"copy-tokenizer": "false"},
        ),
        "the merge wrote no tokenizer",
    )


def test_mergekit_exit_code_survives_the_capture(env):
    _refused(_run(env, extra_env={"STUB_FAIL": "1"}), "mergekit-yaml failed (exit 3)")


def test_sentencepiece_only_tokenizer_is_accepted(env):
    """tokenizer.model + tokenizer_config.json, no tokenizer.json: a real tokenizer."""
    proc = _run(env, extra_env={"STUB_SPM": "1"})
    assert proc.returncode == 0, proc.stderr


def test_tokenizer_without_tokenizer_config_is_refused(env):
    _refused(
        _run(env, extra_env={"STUB_NO_TOK_CONFIG": "1"}),
        "the merge wrote no tokenizer",
    )


def test_no_tokenizer_is_fine_with_copy_tokenizer_false(env):
    proc = _run(
        env, extra_env={"STUB_NO_TOKENIZER": "1"}, **{"copy-tokenizer": "false"}
    )
    assert proc.returncode == 0, proc.stderr


def test_requested_template_missing_from_output_is_refused(env):
    recipe = {**_RECIPE, "chat_template": "chatml"}
    _refused(_run(env, recipe=recipe), "has no chat template")


@pytest.mark.parametrize(
    "where", ["chat_template.jinja", "chat_template.json", "tokenizer_config.json"]
)
def test_requested_template_found_in_any_saved_form(env, where):
    """transformers saves the template in one of three places, by version."""
    recipe = {**_RECIPE, "chat_template": "chatml"}
    if where == "chat_template.jinja":
        proc = _run(env, recipe=recipe, extra_env={"STUB_TEMPLATE": "{% x %}"})
    else:
        # Write the template where this transformers version would, after the stub runs.
        wrapper = env["tmp"] / "bin" / "mergekit-yaml"
        body = wrapper.read_text().replace(
            "exit 0\n",
            f'echo \'{{"chat_template": "{{% x %}}"}}\' > "$out/{where}"\nexit 0\n',
            1,
        )
        wrapper.write_text(body)
        proc = _run(env, recipe=recipe)
    assert proc.returncode == 0, proc.stderr
    assert "GB_ARTIFACT_ID:model" in proc.stdout


def test_mergekit_sees_the_real_output_name(env):
    """MergeKit titles the model card with basename(out_path): not a staging name."""
    assert _run(env, output="./out/my-merged-model").returncode == 0
    assert pathlib.Path(_stub_argv(env)[1]).name == "my-merged-model"


def test_staging_dir_that_cannot_be_created_fails_early(env):
    """A read-only output parent fails before the merge, with a clear message."""
    if os.geteuid() == 0:
        pytest.skip("root can write a read-only dir")
    parent = env["work"] / "ro"
    parent.mkdir()
    parent.chmod(0o555)
    try:
        _refused(_run(env, output="./ro/model"), "cannot create a staging dir")
        assert not env["argv_file"].exists()  # mergekit-yaml never ran
    finally:
        parent.chmod(0o755)


def test_python3_only_image_works(env, tmp_path_factory):
    """With no `python` on PATH (many images have only python3), run.sh still works."""
    only = tmp_path_factory.mktemp("py3only")
    # A wrapper, not a symlink: a symlinked venv interpreter loses the venv's
    # site-packages (and so PyYAML).
    (only / "python3").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (only / "python3").chmod(0o755)
    for tool in (
        "bash",
        "env",
        "dirname",
        "basename",
        "mkdir",
        "mktemp",
        "rm",
        "rmdir",
        "mv",
        "ls",
        "sed",
        "grep",
        "tee",
        "cat",
        "chmod",
        "umask",
        "printf",
    ):
        found = shutil.which(tool)
        if found and not (only / tool).exists():
            (only / tool).symlink_to(found)
    bindir = env["tmp"] / "bin"
    proc = subprocess.run(
        ["/bin/bash", *_run_argv(env)],
        cwd=env["work"],
        env={**env["env"], "PATH": f"{bindir}:{only}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "GB_ARTIFACT_ID:model" in proc.stdout


def test_python_without_yaml_is_refused(env, tmp_path_factory):
    """An interpreter that can't import yaml fails up front, naming PyYAML."""
    fake = tmp_path_factory.mktemp("noyaml") / "python3"
    fake.write_text(
        "#!/bin/sh\ncase \"$*\" in *'import yaml'*) exit 1 ;; esac\nexit 0\n"
    )
    fake.chmod(0o755)
    proc = _run(env, extra_env={"PATH": f"{fake.parent}:{env['env']['PATH']}"})
    _refused(proc, "cannot import yaml")


def test_missing_venv(env):
    _refused(_run(env, venv="./nope"), "no venv at './nope'")


# --- real MergeKit --------------------------------------------------------------------


def test_real_merge(tmp_path):
    """A real linear merge of two tiny random Llamas, as MergeKit's own tests build them."""
    pytest.importorskip("mergekit")
    transformers = pytest.importorskip("transformers")
    if not shutil.which("mergekit-yaml"):
        pytest.skip("mergekit-yaml not on PATH")
    paths = {}
    for name in ("m1", "base"):
        cfg = transformers.LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_attention_heads=16,
            num_hidden_layers=2,
        )
        p = tmp_path / name
        transformers.LlamaForCausalLM(cfg).save_pretrained(p, safe_serialization=True)
        paths[name] = str(p)
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "tmp": tmp_path,
        "work": work,
        "m1": paths["m1"],
        "base": paths["base"],
        "env": _real_env(),
    }
    proc = _run(env, **{"copy-tokenizer": "false"})
    assert proc.returncode == 0, proc.stderr
    out = work / "out" / "model"
    assert f"GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:{out}" in proc.stdout
    # The model card MergeKit writes: YAML front matter, then "# <basename(out_path)>".
    titles = [
        ln for ln in (out / "README.md").read_text().splitlines() if ln.startswith("# ")
    ]
    assert titles[:1] == ["# model"], titles[:1]
    assert not list(out.parent.glob(".mergekit-stage.*"))
    with open(out / "model.safetensors", "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        assert len(json.loads(f.read(size))) > 1


def _tiny_llama_with_tokenizer(path, transformers, tokenizers, template=None) -> str:
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=16,
        num_hidden_layers=2,
    )
    transformers.LlamaForCausalLM(cfg).save_pretrained(path, safe_serialization=True)
    vocab = {f"t{i}": i for i in range(63)} | {"[UNK]": 63}
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizers.Tokenizer(
            tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
        )
    )
    if template:
        tok.chat_template = template
    tok.save_pretrained(path)
    return str(path)


def test_real_merge_with_a_lora_adapter(tmp_path):
    """`model_1+base_model` with base_model a PEFT LoRA dir: MergeKit applies the adapter.

    Exercises the `+lora` substitution and an input that has only adapter_config.json,
    through the real run.sh and mergekit-yaml.
    """
    pytest.importorskip("mergekit")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    peft = pytest.importorskip("peft")
    if not shutil.which("mergekit-yaml"):
        pytest.skip("mergekit-yaml not on PATH")
    m1 = _tiny_llama_with_tokenizer(tmp_path / "m1", transformers, tokenizers)
    lora_dir = tmp_path / "lora"
    model = transformers.AutoModelForCausalLM.from_pretrained(m1)
    # init_lora_weights=False: PEFT's default zeroes lora_B, a no-op adapter that would let
    # this pass even if MergeKit ignored `+lora`.
    cfg = peft.LoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["q_proj", "v_proj"],
        init_lora_weights=False,
    )
    peft.get_peft_model(model, cfg).save_pretrained(lora_dir)
    assert (lora_dir / "adapter_config.json").is_file()
    assert not (lora_dir / "config.json").exists()  # an adapter-only input
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "tmp": tmp_path,
        "work": work,
        "m1": m1,
        "base": str(lora_dir),
        "env": _real_env(),
    }
    recipe = {
        "merge_method": "passthrough",
        "models": [{"model": "model_1+base_model"}],
    }
    proc = _run(env, recipe=recipe)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = work / "out" / "model"
    assert "GB_ARTIFACT_ID:model" in proc.stdout
    # The recipe MergeKit ran had both parts substituted.
    ran = yaml.safe_load((work / ".mergekit" / "merge.yaml").read_text())
    assert ran["models"][0]["model"] == f"{m1}+{lora_dir}"
    assert (out / "tokenizer_config.json").is_file()
    # The adapter was applied: q_proj changed, and a module it doesn't target did not.
    safetensors = pytest.importorskip("safetensors.torch")
    torch = pytest.importorskip("torch")

    def tensors(d):
        merged = {}
        for f in pathlib.Path(d).glob("*.safetensors"):
            merged.update(safetensors.load_file(f))
        return merged

    before, after = tensors(m1), tensors(out)
    q = "model.layers.0.self_attn.q_proj.weight"
    k = "model.layers.0.self_attn.k_proj.weight"
    assert not torch.equal(before[q], after[q])
    assert torch.equal(before[k], after[k])


def test_real_merge_exempted_path_auto_without_templates_is_refused(tmp_path):
    """tokenizer_source + chat_template: auto + copy_tokenizer false, untemplated inputs.

    MergeKit builds the tokenizer, and auto finds no template, so it sets none.
    """
    pytest.importorskip("mergekit")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    if not shutil.which("mergekit-yaml"):
        pytest.skip("mergekit-yaml not on PATH")
    m1 = _tiny_llama_with_tokenizer(tmp_path / "m1", transformers, tokenizers)
    base = _tiny_llama_with_tokenizer(tmp_path / "base", transformers, tokenizers)
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "tmp": tmp_path,
        "work": work,
        "m1": m1,
        "base": base,
        "env": _real_env(),
    }
    recipe = {
        **_RECIPE,
        "base_model": "base_model",
        "tokenizer_source": "base",
        "chat_template": "auto",
    }
    _refused(
        _run(env, recipe=recipe, **{"copy-tokenizer": "false"}), "has no chat template"
    )
    assert not (work / "out" / "model").exists()


def test_real_merge_partial_tokenizer_copy_is_refused(tmp_path):
    """A copy that fails after tokenizer.json landed: only MergeKit's log line shows it.

    The donor is base_model (MergeKit's _get_donor_model picks base_model first); its
    chat_template.jinja is made unreadable, so _copy_tokenizer raises mid-copy and
    MergeKit logs "Failed to copy tokenizer ... still successful".
    """
    pytest.importorskip("mergekit")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    if not shutil.which("mergekit-yaml"):
        pytest.skip("mergekit-yaml not on PATH")
    if os.geteuid() == 0:
        pytest.skip("root can read a mode-000 file")
    tpl = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    m1 = _tiny_llama_with_tokenizer(tmp_path / "m1", transformers, tokenizers, tpl)
    base = _tiny_llama_with_tokenizer(tmp_path / "base", transformers, tokenizers, tpl)
    unreadable = pathlib.Path(base) / "chat_template.jinja"
    unreadable.chmod(0)
    if os.access(
        unreadable, os.R_OK
    ):  # CAP_DAC_OVERRIDE, or a filesystem ignoring modes
        unreadable.chmod(0o600)
        pytest.skip("a mode-000 file is still readable here")
    try:
        work = tmp_path / "work"
        work.mkdir()
        env = {
            "tmp": tmp_path,
            "work": work,
            "m1": m1,
            "base": base,
            "env": _real_env(),
        }
        proc = _run(env, recipe={**_RECIPE, "base_model": "base_model"})
        assert "Failed to copy tokenizer" in proc.stderr  # MergeKit did swallow it
        _refused(proc, "MergeKit failed to copy the tokenizer")
        assert not (work / "out" / "model").exists()
    finally:
        unreadable.chmod(0o600)


@pytest.mark.parametrize(
    "chat_template, templated_inputs, ok, message",
    [
        (None, True, True, ""),
        ("chatml", True, True, ""),
        ("auto", True, True, ""),
        # MergeKit rejects a template under 20 characters inside _copy_tokenizer's try
        # and logs "The merge was still successful": the output check catches it.
        ("{%if 1%}x{%endif%}", True, False, "MergeKit failed to copy the tokenizer"),
        # auto with no template among the inputs: MergeKit silently sets none.
        ("auto", False, False, "has no chat template"),
    ],
)
def test_real_merge_tokenizer_and_template(
    tmp_path, chat_template, templated_inputs, ok, message
):
    """Real mergekit-yaml with --copy-tokenizer: what lands is what is registered."""
    pytest.importorskip("mergekit")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    if not shutil.which("mergekit-yaml"):
        pytest.skip("mergekit-yaml not on PATH")
    tpl = (
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        if templated_inputs
        else None
    )
    m1 = _tiny_llama_with_tokenizer(tmp_path / "m1", transformers, tokenizers, tpl)
    base = _tiny_llama_with_tokenizer(tmp_path / "base", transformers, tokenizers, tpl)
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "tmp": tmp_path,
        "work": work,
        "m1": m1,
        "base": base,
        "env": _real_env(),
    }
    recipe = {**_RECIPE}
    if chat_template is not None:
        recipe["chat_template"] = chat_template
    proc = _run(env, recipe=recipe)
    if ok:
        assert proc.returncode == 0, proc.stderr
        assert (work / "out" / "model" / "tokenizer_config.json").is_file()
    else:
        _refused(proc, message)
        assert not (work / "out" / "model").exists()


# Every tokenizer path x chat_template option, through the real run.sh and real
# mergekit-yaml: the step must register exactly the merges whose output is good.
_PATHS = {
    "copy": ({}, "true"),
    "none": ({}, "false"),
    "source": ({"tokenizer_source": "base"}, "false"),
    "block": ({"tokenizer": {"source": "base"}}, "false"),
    "empty-block": ({"tokenizer": {}}, "false"),
    "copy+source": ({"tokenizer_source": "base"}, "true"),
}
_TEMPLATES = {
    "none": None,
    "chatml": "chatml",
    "auto": "auto",
    "inline": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    "short": "{%if 1%}x{%endif%}",
}


def _has_tokenizer(d: pathlib.Path) -> bool:
    return (d / "tokenizer_config.json").is_file() and (
        (d / "tokenizer.json").is_file() or (d / "tokenizer.model").is_file()
    )


def _has_template(d: pathlib.Path) -> bool:
    if (d / "chat_template.jinja").is_file() or (d / "chat_template.json").is_file():
        return True
    try:
        return bool(
            json.loads((d / "tokenizer_config.json").read_text()).get("chat_template")
        )
    except (OSError, ValueError):
        return False


@pytest.fixture(name="tiny_pairs", scope="module")
def tiny_pairs_fixture(tmp_path_factory):
    """Two tiny Llama pairs with tokenizers: with and without a chat template."""
    pytest.importorskip("mergekit")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    if not shutil.which("mergekit-yaml"):
        pytest.skip("mergekit-yaml not on PATH")
    root = tmp_path_factory.mktemp("pairs")
    tpl = _TEMPLATES["inline"]
    return {
        "templated": tuple(
            _tiny_llama_with_tokenizer(root / n, transformers, tokenizers, tpl)
            for n in ("m_t", "b_t")
        ),
        "untemplated": tuple(
            _tiny_llama_with_tokenizer(root / n, transformers, tokenizers)
            for n in ("m_n", "b_n")
        ),
    }


@pytest.mark.parametrize("inputs", ["templated", "untemplated"])
@pytest.mark.parametrize("template", list(_TEMPLATES))
@pytest.mark.parametrize("path", list(_PATHS))
def test_real_merge_matrix(tmp_path, tiny_pairs, path, template, inputs):
    """The step's verdict equals an independent check of plain mergekit-yaml's output."""
    extra, copy = _PATHS[path]
    tpl = _TEMPLATES[template]
    m1, base = tiny_pairs[inputs]
    recipe = {**_RECIPE, "base_model": "base_model", **extra}
    if tpl is not None:
        recipe["chat_template"] = tpl
    # Ground truth: plain mergekit-yaml on the same recipe, staged paths substituted.
    truth = {**recipe, "base_model": base}
    truth["models"] = [
        {"model": m1, "parameters": {"weight": 0.5}},
        {"model": base, "parameters": {"weight": 0.5}},
    ]
    (tmp_path / "truth.yaml").write_text(json.dumps(truth))
    out_truth = tmp_path / "truth-out"
    plain = subprocess.run(
        [
            "mergekit-yaml",
            str(tmp_path / "truth.yaml"),
            str(out_truth),
            "--copy-tokenizer" if copy == "true" else "--no-copy-tokenizer",
            "--quiet",
            "-v",  # INFO: MergeKit's own "Saving/Copying/Reserializing tokenizer" lines
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # Whether MergeKit tried to put a tokenizer in the output, from MergeKit itself (its
    # INFO log), not from the step's rule: so a wrong rule in the step can't agree with
    # a wrong rule here.
    wants_tokenizer = (
        any(
            line in plain.stderr
            for line in (
                "Saving tokenizer",
                "Copying tokenizer",
                "Reserializing tokenizer",
            )
        )
        or "Failed to copy tokenizer" in plain.stderr
    )
    good = (
        plain.returncode == 0
        and "Failed to copy tokenizer" not in plain.stderr
        and (not wants_tokenizer or _has_tokenizer(out_truth))
        and (tpl is None or _has_template(out_truth))
    )
    work = tmp_path / "work"
    work.mkdir()
    env = {
        "tmp": tmp_path,
        "work": work,
        "m1": m1,
        "base": base,
        "env": _real_env(),
    }
    proc = _run(env, recipe=recipe, **{"copy-tokenizer": copy})
    registered = proc.returncode == 0 and "GB_ARTIFACT_ID:model" in proc.stdout
    assert (
        registered == good
    ), f"registered={registered} but MergeKit's output good={good}\n{proc.stderr[-2000:]}"
