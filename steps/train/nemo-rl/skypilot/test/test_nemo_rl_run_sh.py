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

"""Unit tests for src/nemo_rl_run.sh.

Everything the node actually runs — `uv`, the driver, the converter — is
stubbed, so these tests pin the SHELL the step emits: the Hydra argv assembled
for the driver (injected model/dataset, forced tensorboard/checkpoint overrides,
passthrough order), checkpoint discovery, the converter invocation, and the
artifact marker. The real training is Mode-2's job.
"""

import pathlib
import stat
import subprocess

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _STEP_DIR / "src" / "nemo_rl_run.sh"


def _write_uv_stub(bin_dir: pathlib.Path, uvlog: pathlib.Path) -> None:
    """A fake `uv` that logs its argv and fakes the converter's output.

    For a training run it only records argv. For a convert run (argv mentions
    convert_dcp_to_hf.py) it mirrors the real convert_dcp_to_hf.py: it REFUSES a
    --hf-ckpt-path that already exists (the real converter raises FileExistsError
    there), and otherwise creates the dir and drops a file into it so the step's
    HF-dir non-empty check passes.
    """
    stub = bin_dir / "uv"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" >> "$UVLOG"\n'
        'printf -- "---\\n" >> "$UVLOG"\n'
        'hf=""\n'
        'prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "--hf-ckpt-path" ]; then hf="$a"; fi\n'
        '  prev="$a"\n'
        "done\n"
        'case "$*" in\n'
        "  *convert_dcp_to_hf.py*)\n"
        '    if [ -e "$hf" ]; then\n'
        '      echo "FileExistsError: HF checkpoint already exists at $hf" >&2\n'
        "      exit 1\n"
        "    fi\n"
        '    mkdir -p "$hf"; : > "$hf/model.safetensors" ;;\n'
        "esac\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


def _fake_root(
    tmp_path: pathlib.Path,
    algo: str,
    config_name: str,
    config_body: str = "policy: {}\n",
) -> pathlib.Path:
    root = tmp_path / "nemo-rl"
    (root / "examples" / "configs").mkdir(parents=True)
    (root / "examples" / "converters").mkdir(parents=True)
    (root / f"examples/run_{algo}.py").write_text("# driver\n")
    (root / f"examples/configs/{config_name}").write_text(config_body)
    (root / "examples/converters/convert_dcp_to_hf.py").write_text("# conv\n")
    return root


def _run(
    tmp_path,
    extra_argv,
    make_ckpt=True,
    make_driver=True,
    algo="sft",
    config_name="sft.yaml",
    clearml_mode="off",
    env_extra=None,
    config_body="policy: {}\n",
):
    root = _fake_root(tmp_path, algo, config_name, config_body)
    if not make_driver:
        # Simulate an algorithm whose driver is absent from the image so the
        # run script's driver-resolution guard fires.
        (root / f"examples/run_{algo}.py").unlink()
    workdir = tmp_path / "work"
    (workdir / "src").mkdir(parents=True)
    # The run script invokes <its own dir>/clearml_launch.py via uv; a stub file
    # is enough since uv is faked.
    (workdir / "src" / "clearml_launch.py").write_text("# stub\n")
    src_run = workdir / "src" / "nemo_rl_run.sh"
    src_run.write_text(_SCRIPT.read_text())
    src_run.chmod(src_run.stat().st_mode | stat.S_IEXEC)

    if make_ckpt:
        step = workdir / "results" / algo / "step_10"
        (step / "policy" / "weights").mkdir(parents=True)
        (step / "config.yaml").write_text("policy: {}\n")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uvlog = tmp_path / "uv.log"
    _write_uv_stub(bin_dir, uvlog)

    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "UVLOG": str(uvlog),
    }
    if env_extra:
        env.update(env_extra)
    argv = [
        "bash",
        "src/nemo_rl_run.sh",
        "--algorithm",
        algo,
        "--config-name",
        config_name,
        "--nemo-rl-root",
        str(root),
        "--artifact-id",
        "model",
        "--clearml-mode",
        clearml_mode,
    ] + extra_argv
    proc = subprocess.run(
        argv, capture_output=True, text=True, cwd=str(workdir), env=env
    )
    runs = []
    if uvlog.exists():
        for block in uvlog.read_text().split("---\n"):
            words = [w for w in block.splitlines() if w != ""]
            if words:
                runs.append(words)
    return proc, runs, workdir


def _train_argv(runs):
    """The driver argv the training `uv run` passed after `clearml_launch.py`."""
    for words in runs:
        if any(w.endswith("clearml_launch.py") for w in words):
            return words
    raise AssertionError(f"no training run found in {runs}")


def _hydra_overrides(train_argv):
    """Everything after the driver's `--config <path>` on the training command."""
    i = train_argv.index("--config")
    return train_argv[i + 2 :]


def _marker(stdout):
    for ln in stdout.splitlines():
        if ln.startswith("LLMB_ARTIFACT_ID:"):
            return ln
    return None


class TestArgAssembly:
    def test_injects_base_model_and_forced_overrides(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base", "--", "sft.max_num_steps=5"]
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert "policy.model_name=/staged/base" in ov
        assert "sft.max_num_steps=5" in ov
        assert "logger.tensorboard_enabled=true" in ov
        # Forced overrides come LAST so they win over any user override.
        assert ov.index("sft.max_num_steps=5") < ov.index(
            "logger.tensorboard_enabled=true"
        )
        assert any(o.startswith("checkpointing.checkpoint_dir=") for o in ov)

    def test_disables_wandb_by_default_before_user_overrides(self, tmp_path):
        # The shipped configs (e.g. sft.yaml) default logger.wandb_enabled=True,
        # which makes NeMo-RL call wandb.init() and abort with "No API key
        # configured" — the step threads no WANDB_API_KEY, it tracks via
        # tensorboard+ClearML. So wandb must be off by default. It is an
        # OVERRIDABLE default (injected before "$@"), so a user who brings a key
        # can re-enable it.
        proc, runs, _ = _run(
            tmp_path,
            ["--base-model", "/staged/base", "--", "logger.wandb_enabled=true"],
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert "logger.wandb_enabled=false" in ov
        # Injected before the user override, so the user's `=true` wins.
        assert ov.index("logger.wandb_enabled=false") < ov.index(
            "logger.wandb_enabled=true"
        )

    def test_injects_cluster_gpus_per_node_when_given(self, tmp_path):
        # sft.yaml defaults cluster.gpus_per_node=1, so without this the step
        # would train on a single GPU even on a multi-GPU node. It is a FORCED
        # override (after "$@"), so it wins over any user override, keeping the
        # node's GPU count and NeMo-RL's cluster view in lockstep.
        proc, runs, _ = _run(
            tmp_path,
            ["--base-model", "/staged/base", "--num-gpus-per-node", "4"],
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert "cluster.gpus_per_node=4" in ov

    def test_omits_cluster_gpus_per_node_when_absent(self, tmp_path):
        proc, runs, _ = _run(tmp_path, ["--base-model", "/staged/base"])
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert not any(o.startswith("cluster.gpus_per_node=") for o in ov)

    def test_injects_dataset_under_configured_key(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path,
            [
                "--base-model",
                "/staged/base",
                "--dataset",
                "/staged/ds",
                "--dataset-key",
                "data.train.input_file",
            ],
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert "data.train.input_file=/staged/ds" in ov

    def test_omits_dataset_when_absent(self, tmp_path):
        proc, runs, _ = _run(tmp_path, ["--base-model", "/staged/base"])
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert not any(o.startswith("data.train.input_file=") for o in ov)

    def test_resolves_driver_and_config_paths(self, tmp_path):
        proc, runs, _ = _run(tmp_path, ["--base-model", "/staged/base"])
        train = _train_argv(runs)
        assert any(a.endswith("examples/run_sft.py") for a in train)
        cfg = train[train.index("--config") + 1]
        assert cfg.endswith("examples/configs/sft.yaml")


class TestModelSaveFormat:
    # NeMo-RL's V2 DTensor worker (policy.dtensor_cfg._v2: true, the default for
    # sft/grpo_math_1B/ppo_math_1B) defaults model_save_format=safetensors, which
    # writes sharded HF safetensors under weights/model/ with NO DCP .metadata.
    # But the step always runs convert_dcp_to_hf.py, which REQUIRES a .metadata.
    # So for a V2 config the step must force checkpointing.model_save_format=
    # torch_save (the DCP writer that emits .metadata). For a V1 worker the same
    # override is FORBIDDEN — lm_policy raises "model_save_format must be None or
    # omitted if using DTensorPolicyWorker (_v2=False)" — so it must be gated on
    # the config declaring _v2: true. The key is absent from the resolved config,
    # so it is force-added with Hydra's `++` prefix (plain `key=value` raises in
    # struct mode).
    _V2 = "policy:\n  dtensor_cfg:\n    _v2: true\n"
    _V1_EXPLICIT = "policy:\n  dtensor_cfg:\n    _v2: false\n"
    _V1_ABSENT = "policy:\n  dtensor_cfg: {}\n"
    _V2_COMMENTED = "policy:\n  dtensor_cfg:\n    # _v2: true\n    enabled: true\n"

    def test_v2_config_forces_torch_save(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base"], config_body=self._V2
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert "++checkpointing.model_save_format=torch_save" in ov
        # Forced tail, so it lands after any user override.
        assert ov.index("++checkpointing.model_save_format=torch_save") > ov.index(
            "logger.wandb_enabled=false"
        )

    def test_v1_explicit_false_omits_model_save_format(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base"], config_body=self._V1_EXPLICIT
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert not any("model_save_format" in o for o in ov)

    def test_v1_absent_omits_model_save_format(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base"], config_body=self._V1_ABSENT
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert not any("model_save_format" in o for o in ov)

    def test_commented_v2_is_not_treated_as_v2(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base"], config_body=self._V2_COMMENTED
        )
        assert proc.returncode == 0, proc.stderr
        ov = _hydra_overrides(_train_argv(runs))
        assert not any("model_save_format" in o for o in ov)


class TestClearmlWith:
    def test_off_mode_adds_no_with_clearml(self, tmp_path):
        _, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base"], clearml_mode="off"
        )
        train = _train_argv(runs)
        # No ephemeral clearml dep on the disabled path.
        assert "--with" not in train

    def test_on_mode_adds_with_clearml(self, tmp_path):
        _, runs, _ = _run(tmp_path, ["--base-model", "/staged/base"], clearml_mode="on")
        train = _train_argv(runs)
        i = train.index("--with")
        assert train[i + 1] == "clearml"

    def test_auto_mode_with_creds_adds_with_clearml(self, tmp_path):
        _, runs, _ = _run(
            tmp_path,
            ["--base-model", "/staged/base"],
            clearml_mode="auto",
            env_extra={
                "CLEARML_API_HOST": "h",
                "CLEARML_API_ACCESS_KEY": "a",
                "CLEARML_API_SECRET_KEY": "s",
            },
        )
        assert "--with" in _train_argv(runs)

    def test_auto_mode_without_creds_adds_no_with_clearml(self, tmp_path):
        _, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base"], clearml_mode="auto"
        )
        assert "--with" not in _train_argv(runs)


class TestConversionAndMarker:
    def test_converts_latest_checkpoint(self, tmp_path):
        proc, runs, work = _run(tmp_path, ["--base-model", "/staged/base"])
        assert proc.returncode == 0, proc.stderr
        conv = next(r for r in runs if any("convert_dcp_to_hf.py" in w for w in r))
        assert any(w.endswith("step_10/config.yaml") for w in conv)
        assert any(w.endswith("step_10/policy/weights") for w in conv)

    def test_does_not_hand_converter_a_preexisting_hf_dir(self, tmp_path):
        # convert_dcp_to_hf.py raises FileExistsError if --hf-ckpt-path already
        # exists, so the step must NOT pre-create the HF output dir before the
        # converter runs. A green run proves the converter saw a fresh path.
        proc, runs, work = _run(tmp_path, ["--base-model", "/staged/base"])
        assert proc.returncode == 0, proc.stderr
        assert "FileExistsError" not in proc.stderr
        marker = _marker(proc.stdout)
        assert marker is not None, proc.stdout

    def test_emits_marker_with_absolute_hf_path(self, tmp_path):
        proc, runs, work = _run(tmp_path, ["--base-model", "/staged/base"])
        marker = _marker(proc.stdout)
        assert marker is not None, proc.stdout
        assert marker.startswith("LLMB_ARTIFACT_ID:model LLMB_ARTIFACT_PATH:/")

    def test_marker_matches_shipped_skypilot_monitor(self, tmp_path):
        # Pin the marker to the regex the step's monitor actually uses
        # (ref: space://monitors/skypilot). A prefix the monitor does not match
        # registers ZERO artifacts while the build still reports SUCCESS.
        import re

        import yaml

        repo = _STEP_DIR.parents[3]  # steps/train/nemo-rl/skypilot -> repo root
        mon = yaml.safe_load(
            (repo / "src/gbserver/builtins/monitors/skypilot/monitor.yaml").read_text()
        )
        cfg = next(
            c
            for c in mon["config"]["event_configs"]
            if "ARTIFACT_PATH" in c["line_regex"]
        )
        proc, _, work = _run(tmp_path, ["--base-model", "/staged/base"])
        line = next(
            ln for ln in proc.stdout.splitlines() if re.search(cfg["line_regex"], ln)
        )
        fields = {f["field_name"]: f for f in cfg["event_fields"]}
        assert re.search(fields["binding_id"]["field_regex"], line).group(0) == "model"
        assert re.search(fields["path"]["field_regex"], line).group(0) == str(
            work / "hf_output"
        )

    def test_pushes_hf_dir_with_hf_cli(self, tmp_path):
        # The image's huggingface_hub removed `huggingface-cli` ("deprecated and
        # no longer works. Use `hf` instead."), so the push must use `hf upload`,
        # with an explicit "." path_in_repo so the HF dir's contents land at the
        # repo root.
        proc, runs, work = _run(
            tmp_path,
            ["--base-model", "/staged/base", "--hf-output-repo", "org/model"],
        )
        assert proc.returncode == 0, proc.stderr
        assert not any("huggingface-cli" in w for r in runs for w in r)
        up = next(r for r in runs if "upload" in r)
        i = up.index("hf")
        assert up[i : i + 5] == [
            "hf",
            "upload",
            "org/model",
            str(work / "hf_output"),
            ".",
        ]

    def test_no_push_without_hf_output_repo(self, tmp_path):
        _, runs, _ = _run(tmp_path, ["--base-model", "/staged/base"])
        assert not any("upload" in r for r in runs)

    def test_errors_when_no_checkpoint(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path, ["--base-model", "/staged/base"], make_ckpt=False
        )
        assert proc.returncode != 0
        assert any("no checkpoint" in ln.lower() for ln in proc.stderr.splitlines())
        assert _marker(proc.stdout) is None

    def test_errors_when_driver_missing(self, tmp_path):
        proc, runs, _ = _run(
            tmp_path,
            ["--base-model", "/staged/base"],
            make_driver=False,
            algo="nonesuch",
        )
        assert proc.returncode != 0
        assert any("run_nonesuch.py" in ln for ln in proc.stderr.splitlines())


class TestScriptIsValidShell:
    def test_bash_parses_the_script(self):
        assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0

    def test_shellcheck_is_clean(self):
        import shutil

        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck not installed")
        assert subprocess.run(["shellcheck", str(_SCRIPT)]).returncode == 0
