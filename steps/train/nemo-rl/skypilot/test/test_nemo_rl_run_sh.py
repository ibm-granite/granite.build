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
    convert_dcp_to_hf.py) it also drops a file into --hf-ckpt-path so the HF-dir
    non-empty check passes.
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
        '  *convert_dcp_to_hf.py*) mkdir -p "$hf"; : > "$hf/model.safetensors" ;;\n'
        "esac\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


def _fake_root(tmp_path: pathlib.Path, algo: str, config_name: str) -> pathlib.Path:
    root = tmp_path / "nemo-rl"
    (root / "examples" / "configs").mkdir(parents=True)
    (root / "examples" / "converters").mkdir(parents=True)
    (root / f"examples/run_{algo}.py").write_text("# driver\n")
    (root / f"examples/configs/{config_name}").write_text("policy: {}\n")
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
):
    root = _fake_root(tmp_path, algo, config_name)
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
        if ln.startswith("GB_ARTIFACT_ID:"):
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

    def test_emits_marker_with_absolute_hf_path(self, tmp_path):
        proc, runs, work = _run(tmp_path, ["--base-model", "/staged/base"])
        marker = _marker(proc.stdout)
        assert marker is not None, proc.stdout
        assert marker.startswith("GB_ARTIFACT_ID:model GB_ARTIFACT_PATH:/")

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
