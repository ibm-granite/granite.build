#!/usr/bin/env python3

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

"""Unit tests for the granite-switch answerability-sr recipe on AWS.

The recipe is rendered with its parameters the way `gb build start` renders
it. The train and eval commands are then run in bash with the GPU tools
(sr-train, accelerate, python, nvidia-smi) replaced by stubs that record their
arguments, so the shell the byoc step runs is checked end to end: the EFS
paths, the single-GPU and DDP launches, and that the eval reads the adapter
the train wrote.
"""

import json
import os
import pathlib
import subprocess

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RECIPE = _REPO / "recipes" / "granite-switch" / "skypilot" / "aws" / "answerability-sr"
# setup-local-space.sh fills in its EFS id and region for the user's own space.
_ENV = (
    _REPO
    / "recipes"
    / "granite-switch"
    / "skypilot"
    / "aws"
    / "environment-template.yaml"
)
_EFS = "/mnt/gb-shared/"

# A stub for each GPU tool: log its argv (NUL-separated) and, for the trainers,
# write an adapter_config.json where the generated config's output_dir points,
# as sr-train does; the eval stub writes eval_summary.md to --output-path.
_STUB = r"""#!/bin/bash
name="$(basename "$0")"
{ printf '%s' "$name"; for a in "$@"; do printf '\0%s' "$a"; done; printf '\n'; } >> "$STUB_LOG"
case "$name" in
  sr-train|accelerate)
    out="$(sed -n 's/^  output_dir: //p' "$SR_CFG")"
    mkdir -p "$out" && echo '{"base_model_name_or_path": "/cache/base"}' > "$out/adapter_config.json" ;;
  python)
    if [ "${1:-}" = "-c" ]; then exec /usr/bin/env python3 "$@"; fi
    prev=""; for a in "$@"; do [ "$prev" = "--output-path" ] && echo "| ok |" > "$a/eval_summary.md"; prev="$a"; done ;;
esac
exit 0
"""


# The recipe ships without data; a test supplies a stand-in, as the user would.
# The default environment is the one setup-local-space.sh writes for us-east-1.
_LOCAL_ENV = "space://environments/skypilot-local/aws-gb-shared"
_DATA = "DATA_URI=hf:///datasets/org/repo"


def _render(tmp_path, *params):
    """build.yaml rendered as `gb build start` does: --param KEY=VALUE over
    parameters.yaml."""
    file_params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    return apply_parameters(
        (_RECIPE / "build.yaml").read_text(),
        list(params),
        file_params,
        str(tmp_path),
    )


def _build(tmp_path, *params):
    """The rendered `granite.build` mapping, with stand-in data."""
    rendered = _render(tmp_path, _DATA, *params)
    return yaml.safe_load(rendered)["granite.build"]


def _step(target):
    """A target's byoc step config; it must be its only step."""
    (step,) = target["steps"]
    assert step["step_uri"] == "space://steps/misc/byoc"
    return step["config"]


def _run(tmp_path, target_name, *params):
    """Run a target's rendered command; return (calls, efs_root)."""
    build = _build(tmp_path, *params)
    cmd = _step(build["targets"][target_name])["byoc_config"]["command"]
    efs = tmp_path / "efs"
    repo = tmp_path / "repo"
    jsonl = repo / "train"
    jsonl.mkdir(parents=True)
    for f in ("train.jsonl", "test.jsonl"):
        (jsonl / f).write_text("")
    (repo / "eval").mkdir(parents=True)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    adapter = efs / "granite-switch/answerability-sr/adapters/t1"
    if target_name == "eval":
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
    cmd = (
        cmd.replace("{{ bindings.train_data.binding.path }}", str(repo))
        .replace("{{ bindings.eval_data.binding.path }}", str(repo))
        .replace("{{ bindings.base_model.binding.path }}", str(base))
        .replace("{{ bindings.adapter.binding.path }}", str(adapter))
        .replace(_EFS, f"{efs}/")
    )
    work = tmp_path / "work"
    (work / "code").mkdir(parents=True)
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for tool in ("sr-train", "accelerate", "python", "nvidia-smi"):
        (stubs / tool).write_text(_STUB)
        (stubs / tool).chmod(0o755)
    log = tmp_path / "calls.log"
    env = {
        "PATH": f"{stubs}:/usr/bin:/bin",
        "STUB_LOG": str(log),
        "SR_CFG": str(work / "sr_answerability.yaml"),
    }
    proc = subprocess.run(
        ["bash", "-c", cmd],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    calls = [line.split("\0") for line in log.read_text().splitlines()]
    return calls, efs, proc.stdout


def test_the_data_has_no_default(tmp_path):
    with pytest.raises(Exception, match="DATA_URI"):
        _render(tmp_path)
    build = _build(tmp_path)
    data = _DATA.split("=", 1)[1]
    assert build["targets"]["train"]["inputs"]["train_data"]["uri"] == data
    assert build["targets"]["eval"]["inputs"]["eval_data"]["uri"] == data


def test_both_targets_run_in_the_local_environment_by_default(tmp_path):
    build = _build(tmp_path)
    assert list(build["targets"]) == ["train", "eval"]
    for target in build["targets"].values():
        assert target["environment_uri"] == _LOCAL_ENV


def test_both_targets_run_in_the_given_environment(tmp_path):
    uri = "space://environments/skypilot-local/aws-gb-shared-usw2"
    build = _build(tmp_path, f"ENVIRONMENT_URI={uri}")
    for target in build["targets"].values():
        assert target["environment_uri"] == uri


def test_no_lsf_only_resources_remain(tmp_path):
    build = _build(tmp_path)
    for target in build["targets"].values():
        config = _step(target)
        assert config["launcher_config"]["resources"] == {"accelerators": "H100:1"}
        assert config["compute_config"] == {"num_nodes": 1}


def test_outputs_are_on_the_efs_under_run_id(tmp_path):
    build = _build(tmp_path, "RUN_ID=t1")
    adapter = build["targets"]["train"]["outputs"]["adapter"]["uri"]
    report = build["targets"]["eval"]["outputs"]["report"]["uri"]
    assert adapter == f"env://{_EFS}granite-switch/answerability-sr/adapters/t1"
    assert report == f"env://{_EFS}granite-switch/answerability-sr/reports/t1"
    assert build["name"] == "answerability-sr-t1"


def test_eval_binds_the_train_adapter(tmp_path):
    build = _build(tmp_path)
    assert build["targets"]["eval"]["inputs"]["adapter"] == {"binding": "train.adapter"}


def test_instance_types_only_when_set(tmp_path):
    build = _build(
        tmp_path,
        "ACCELERATORS=H200:8",
        "INSTANCE_TYPE=p5en.48xlarge",
        "EVAL_INSTANCE_TYPE=p5.4xlarge",
    )
    train = _step(build["targets"]["train"])["launcher_config"]["resources"]
    evals = _step(build["targets"]["eval"])["launcher_config"]["resources"]
    assert train == {"accelerators": "H200:8", "instance_type": "p5en.48xlarge"}
    assert evals == {"accelerators": "H100:1", "instance_type": "p5.4xlarge"}


def test_single_gpu_train_runs_sr_train_into_the_efs(tmp_path):
    calls, efs, _ = _run(tmp_path, "train", "RUN_ID=t1", "MAX_STEPS=2")
    trainers = [c for c in calls if c[0] in ("sr-train", "accelerate")]
    assert [c[0] for c in trainers] == ["sr-train"]
    assert trainers[0][-2:] == ["--max-steps", "2"]
    cfg = (tmp_path / "work" / "sr_answerability.yaml").read_text()
    out = efs / "granite-switch/answerability-sr/adapters/t1"
    assert f"  output_dir: {out}" in cfg.splitlines()
    assert "    cross_stream: 16" in cfg.splitlines()
    saved = json.loads((out / "adapter_config.json").read_text())
    assert saved["base_model_name_or_path"] == "ibm-granite/granite-4.2-8b"


def test_ddp_train_runs_accelerate_with_one_process_per_gpu(tmp_path):
    calls, _, _ = _run(
        tmp_path,
        "train",
        "RUN_ID=t1",
        "SR_NUM_PROCESSES=8",
        "GRADIENT_ACCUMULATION=1",
    )
    (launch,) = [c for c in calls if c[0] in ("sr-train", "accelerate")]
    assert launch[:8] == [
        "accelerate",
        "launch",
        "--num_processes",
        "8",
        "--multi_gpu",
        "--mixed_precision",
        "bf16",
        "-m",
    ]
    assert launch[8] == "shadow_residual.training.train"
    assert "--skip-post-train-generate" in launch
    cfg = (tmp_path / "work" / "sr_answerability.yaml").read_text()
    assert "  gradient_accumulation: 1" in cfg.splitlines()


def test_eval_scores_the_adapter_into_the_efs_report(tmp_path):
    calls, efs, out = _run(tmp_path, "eval", "RUN_ID=t1")
    (ev,) = [c for c in calls if c[0] == "python" and "-m" in c]
    args = dict(zip(ev[1::2], ev[2::2]))
    assert args["-m"] == "shadow_residual.eval.answerability_eval"
    assert args["--adapter-path"] == str(
        efs / "granite-switch/answerability-sr/adapters/t1"
    )
    report = efs / "granite-switch/answerability-sr/reports/t1"
    assert args["--output-path"] == str(report)
    assert (report / "eval_summary.md").is_file()
    assert f"GB_ARTIFACT_ID:report GB_ARTIFACT_PATH:{report}" in out


def test_a_rerun_replaces_the_previous_adapter(tmp_path):
    stale = tmp_path / "efs/granite-switch/answerability-sr/adapters/t1/stale.bin"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    _run(tmp_path, "train", "RUN_ID=t1")
    assert not stale.exists()


@pytest.mark.parametrize("target", ["train", "eval"])
@pytest.mark.parametrize(
    "run_id,message",
    [
        ("", "RUN_ID is empty"),
        ("../..", "unsafe RUN_ID"),
        ("x/../../..", "unsafe RUN_ID"),
        ("..", "unsafe RUN_ID"),
        ("a/b", "unsafe RUN_ID"),
    ],
)
def test_the_shell_guard_never_clears_outside_the_run_dir(
    tmp_path, target, run_id, message
):
    """The render-time check rejects these first; the guard before `rm -rf` is
    the second line of defence, so it is run on its own with the RUN_ID
    swapped back in after rendering."""
    build = _build(tmp_path, "RUN_ID=rid")
    cmd = _step(build["targets"][target])["byoc_config"]["command"]
    out = cmd[cmd.index('OUT="') :].split(";", 1)[0].replace("/rid", f"/{run_id}")
    guard = cmd[cmd.index('case "$OUT"') :].split("esac;", 1)[0] + "esac"
    proc = subprocess.run(
        ["bash", "-c", f"set -e; {out}; {guard}; echo cleared"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert message in proc.stdout
    assert "cleared" not in proc.stdout


@pytest.mark.parametrize("target", ["train", "eval"])
def test_the_shell_guard_passes_a_plain_run_id(tmp_path, target):
    build = _build(tmp_path, "RUN_ID=rid")
    cmd = _step(build["targets"][target])["byoc_config"]["command"]
    out = cmd[cmd.index('OUT="') :].split(";", 1)[0]
    guard = cmd[cmd.index('case "$OUT"') :].split("esac;", 1)[0] + "esac"
    proc = subprocess.run(
        ["bash", "-c", f"set -e; {out}; {guard}; echo cleared"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cleared" in proc.stdout


def test_environment_pins_byoc_to_the_efs_region():
    config = yaml.safe_load(_ENV.read_text())["config"]
    region = config["shared_filesystem"]["efs"]["region"]
    infra = config["steps"]["byoc"]["launcher_config"]["resources"]["infra"]
    assert infra == f"aws/{region}"


def _readme_examples():
    """The two JSON rows in README.md, "Your data": train first, then eval."""
    text = (_RECIPE / "README.md").read_text()
    section = text[text.index("## Your data") : text.index("### The base model")]
    blocks = [b.split("```", 1)[0] for b in section.split("```json\n")[1:]]
    return [json.loads(b) for b in blocks]


def test_the_readme_training_row_has_the_shape_sr_train_reads():
    train, _ = _readme_examples()
    roles = [m["role"] for m in train["messages"]]
    # The label is the assistant turn; the loss mask starts after the marker.
    assert roles[-2:] == ["user", "assistant"]
    assert set(train) <= {"messages", "documents", "tools"}
    assert all(isinstance(d, (str, dict)) for d in train["documents"])


def test_the_readme_eval_row_has_the_shape_answerability_eval_reads():
    _, row = _readme_examples()
    by_role = {m["role"]: m["content"] for m in row["messages"]}
    assert by_role["answerability"] in ("answerable", "unanswerable")
    assert isinstance(by_role["documents"], list)
    assert all(isinstance(d, str) for d in by_role["documents"])
    assert row["messages"][0]["role"] == "user"


@pytest.mark.parametrize(
    "run_id", ["", ".", "..", "../..", "x/../../../..", "a/b", "a b", "-rf", "r1;rm"]
)
def test_an_unsafe_run_id_fails_the_render(tmp_path, run_id):
    """RUN_ID names a dir the targets `rm -rf`: only a plain name may pass."""
    with pytest.raises(Exception, match="RUN_ID_must_be"):
        _build(tmp_path, f"RUN_ID={run_id}")


@pytest.mark.parametrize("run_id", ["r1", "sr-full-1", "smoke_h200.c", "123"])
def test_a_plain_run_id_renders(tmp_path, run_id):
    build = _build(tmp_path, f"RUN_ID={run_id}")
    assert build["name"] == f"answerability-sr-{run_id}"


def test_train_creates_its_dirs_world_writable_for_other_uids(tmp_path):
    """The lora recipe's steps run as root in a container and SR's as a
    non-root user on the bare instance; the EFS does not remap root. So each
    level a recipe creates on the EFS is 1777 (sticky), as gbserver's per-run
    tree is, and the other recipe can still add its own dirs beside it."""
    _, efs, _ = _run(tmp_path, "train", "RUN_ID=t1")
    for rel in (
        "granite-switch",
        "granite-switch/answerability-sr",
        "granite-switch/answerability-sr/adapters",
    ):
        assert (efs / rel).stat().st_mode & 0o7777 == 0o1777, rel


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_train_fails_early_when_a_parent_is_not_writable(tmp_path):
    """A parent left read-only for this uid by an earlier (root) run fails
    before training, naming the fix."""
    locked = tmp_path / "efs" / "granite-switch"
    locked.mkdir(parents=True)
    locked.chmod(0o555)
    try:
        build = _build(tmp_path, "RUN_ID=t1")
        cmd = _step(build["targets"]["train"])["byoc_config"]["command"]
        # OUT and ROOT, then the RUN_ID guard, the 1777 walk and the mkdir —
        # without the input checks in between, which need real inputs.
        assigns = cmd[cmd.index('OUT="') : cmd.index("CFG=")]
        walk = cmd[cmd.index('case "$OUT"') : cmd.index("printf '%s\\n'")]
        snippet = (assigns + walk).replace(_EFS, f"{tmp_path / 'efs'}/")
        proc = subprocess.run(
            ["bash", "-c", "set -e; " + snippet + " echo GOT_PAST_THE_WALK"],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        locked.chmod(0o755)
    assert proc.returncode != 0
    assert "sudo chmod 1777" in proc.stdout
    assert "GOT_PAST_THE_WALK" not in proc.stdout
