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

"""Render tests for the `nemo-rl` step's step-template.yaml.

Cluster-agnostic (Mode 1), so this sits at the root of the step's test/ dir and
is not copied by `make publish-step`. The template computes VALUES and hands
them as argv to bundled scripts, so the meaningful assertions are "what argv
does each script receive?" (asked via bash, which is what splits/unquotes on the
node) and "is the rendered block valid bash?".
"""

import pathlib
import shutil
import subprocess

import pytest
import yaml

jinja2 = pytest.importorskip("jinja2", reason="jinja2 renders the step template")

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = _STEP_DIR / "step-template.yaml"

_TMPDIR = "."


@pytest.fixture(scope="module")
def template() -> dict:
    return yaml.safe_load(_TEMPLATE.read_text())


@pytest.fixture(scope="module")
def nrl_defaults(template) -> dict:
    return dict(template["config"]["nemo_rl_config"])


@pytest.fixture(scope="module")
def compute_defaults(template) -> dict:
    return dict(template["config"]["compute_config"])


@pytest.fixture(scope="module")
def launcher(template) -> dict:
    return template["environment_configs"]["Skypilot"]["launchers"]["nemo_rl"]["config"]


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "_TMPDIR", str(tmp_path))


_BASE_BINDING = {"base_model": {"binding": {"path": "/staged/base"}}}


def _cfg(nrl_defaults: dict, **over) -> dict:
    base = dict(nrl_defaults, algorithm="sft", config_name="sft.yaml", output="model")
    base.update(over)
    return base


def _raise_error(msg):
    """Mirror of gbserver's strict_env `raise_error` global (template.py).

    The launcher-config fill uses the strict Jinja env, which exposes
    `raise_error` so a template can fail the build at render time. The step's
    accelerators value calls it when no GPU type is given.
    """
    raise ValueError(msg)


# ComputeConfig always populates every field (model defaults), so a caller that
# overrides one key still sees the rest — merge the override over this floor.
_DEFAULT_COMPUTE = {"num_nodes": 1, "num_gpus_per_node": 8, "gpu_model": ""}


def _render(source, nrl, compute=None, bindings=None) -> str:
    return jinja2.Template(source, undefined=jinja2.StrictUndefined).render(
        config={
            "nemo_rl_config": nrl,
            "compute_config": {**_DEFAULT_COMPUTE, **(compute or {})},
        },
        bindings=_BASE_BINDING if bindings is None else bindings,
        raise_error=_raise_error,
    )


def _bash_ok(script: str) -> bool:
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    return subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def _argv_from(rendered: str, measure: str) -> list[str]:
    """argv a rendered block hands to one bundled script.

    `measure` is the script to instrument (nemo_rl_guard.sh / nemo_rl_setup.sh /
    nemo_rl_run.sh): it echoes one ARG: line per argv word, while the OTHER
    scripts are stubbed no-ops so their own argv is not mistaken for the measured
    one.
    """
    if shutil.which("bash") is None:  # pragma: no cover
        pytest.skip("bash not available")
    scripts = ["nemo_rl_guard.sh", "nemo_rl_setup.sh", "nemo_rl_run.sh"]
    lines = ["set -e", "mkdir -p ./src"]
    for s in scripts:
        if s == measure:
            lines.append(
                f"printf \"%s\\n\" '#!/usr/bin/env bash' "
                f'\'for a in "$@"; do echo "ARG:$a"; done\' > ./src/{s}'
            )
        else:
            lines.append(f"printf '#!/usr/bin/env bash\\ntrue\\n' > ./src/{s}")
        lines.append(f"chmod +x ./src/{s}")
    lines.append(rendered)
    proc = subprocess.run(
        ["bash", "-c", "\n".join(lines)],
        capture_output=True,
        text=True,
        cwd=_TMPDIR,
    )
    assert proc.returncode == 0, f"rendered block failed: {proc.stderr}"
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("ARG:")]


def _opt(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _overrides(argv):
    return argv[argv.index("--") + 1 :] if "--" in argv else []


class TestStepContract:
    def test_is_an_exec_step_named_nemo_rl(self, template):
        assert template["name"] == "nemo-rl"
        assert template["type"] == "exec"

    def test_no_image_ref_token(self):
        assert "${IMAGE_REF}" not in _TEMPLATE.read_text()

    def test_serves_every_skypilot_endpoint(self, template):
        skypilot = template["environment_configs"]["Skypilot"]
        assert "subtypes" not in skypilot
        assert skypilot["default_launcher"] == "nemo_rl"

    def test_uses_the_shared_skypilot_monitor(self, template):
        monitors = template["environment_configs"]["Skypilot"]["monitors"]
        assert monitors["skypilot_monitor"]["ref"] == "space://monitors/skypilot"

    def test_bundles_src_via_file_mounts(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_single_node_default(self, compute_defaults):
        assert compute_defaults["num_nodes"] == 1


class TestImageSelection:
    def test_default_image_is_the_release_image(self, launcher, nrl_defaults):
        rendered = _render(launcher["image_id"], _cfg(nrl_defaults))
        assert rendered.startswith("docker:nvcr.io/nvidia/nemo-rl")

    def test_custom_image_renders_docker_ref(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, image_id="quay.io/org/nemo-rl:1.2.3")
        assert _render(launcher["image_id"], cfg) == "docker:quay.io/org/nemo-rl:1.2.3"

    def test_image_id_is_not_shell_escaped(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, image_id="quay.io/org/nemo-rl@sha256:ab'cd")
        assert _render(launcher["image_id"], cfg) == (
            "docker:quay.io/org/nemo-rl@sha256:ab'cd"
        )

    def test_sqsh_path_is_docker_prefixed_too(self, launcher, nrl_defaults):
        # On SLURM the image may be a prebuilt squashfs path (slurm-sft does this).
        # It still gets `docker:`: SkyPilot's SLURM cloud only reads an image_id
        # through extract_docker_image(), which needs the prefix, then hands a
        # `.sqsh` path to Pyxis as-is. A bare path would silently run WITHOUT the
        # container. See test_skypilot_extracts_a_docker_prefixed_sqsh_path.
        sqsh = "/proj/x/images/nemo-rl-v0.7.0.sqsh"
        cfg = _cfg(nrl_defaults, image_id=sqsh)
        assert _render(launcher["image_id"], cfg) == f"docker:{sqsh}"

    def test_skypilot_extracts_a_docker_prefixed_sqsh_path(self):
        # Pins the SkyPilot half of the contract above, against the installed fork.
        sky = pytest.importorskip("sky", reason="skypilot not installed")
        sqsh = "/proj/x/images/nemo-rl-v0.7.0.sqsh"
        assert sky.Resources(image_id=f"docker:{sqsh}").extract_docker_image() == sqsh
        assert sky.Resources(image_id=sqsh).extract_docker_image() is None


class TestRenderTimeValidation:
    """Config-only checks fail the strict launcher-config fill, i.e. at step
    creation, before anything is provisioned. The launcher prepends the inline
    `hf download` to `setup:`, so a check in a script there would only run after
    the node is up and the base model is downloaded. Both blocks validate, since
    the launcher fills them independently.
    """

    @pytest.mark.parametrize("block", ["setup", "run"])
    @pytest.mark.parametrize(
        "over,compute,needle",
        [
            ({"algorithm": ""}, None, "algorithm is required"),
            ({"config_name": ""}, None, "config_name is required"),
            ({"output": ""}, None, "output is required"),
            ({"output": "my model"}, None, "whitespace or a double quote"),
            ({"output": 'my"model'}, None, "whitespace or a double quote"),
            ({}, {"num_nodes": 2}, "num_nodes must be 1"),
        ],
    )
    def test_bad_config_fails_the_render(
        self, launcher, nrl_defaults, block, over, compute, needle
    ):
        with pytest.raises(ValueError, match=needle):
            _render(launcher[block], _cfg(nrl_defaults, **over), compute=compute)

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_valid_config_renders(self, launcher, nrl_defaults, block):
        assert _bash_ok(_render(launcher[block], _cfg(nrl_defaults)))


class TestGuardArgv:
    def test_both_phases_pass_identical_guard_argv(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults)
        setup = _argv_from(_render(launcher["setup"], cfg), "nemo_rl_guard.sh")
        run = _argv_from(_render(launcher["run"], cfg), "nemo_rl_guard.sh")
        assert setup == run

    def test_guard_sees_the_binding_paths(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, dataset_override_key="data.train.input_file")
        bound = dict(_BASE_BINDING, dataset={"binding": {"path": "/staged/ds"}})
        argv = _argv_from(
            _render(launcher["run"], cfg, bindings=bound), "nemo_rl_guard.sh"
        )
        assert argv == [
            "--base-model",
            "/staged/base",
            "--dataset",
            "/staged/ds",
            "--dataset-key",
            "data.train.input_file",
        ]


class TestAccelerators:
    """The step must translate compute_config into a SkyPilot `accelerators`
    request. gbserver's launcher does NOT source GPUs from compute_config (they
    flow via launcher_config.resources), so the step renders them here.
    """

    def test_accelerators_renders_type_and_count(self, launcher):
        value = _render(
            launcher["resources"]["accelerators"],
            {},
            compute={"num_nodes": 1, "num_gpus_per_node": 4, "gpu_model": "A10G"},
        )
        assert value == "A10G:4"

    def test_accelerators_requires_gpu_model(self, launcher):
        # Empty gpu_model would render accelerators="" which SkyPilot reads as
        # {'': 1} — a request for a blank-named accelerator that silently
        # provisions a CPU-only node. Fail at render time instead.
        with pytest.raises(Exception):
            _render(
                launcher["resources"]["accelerators"],
                {},
                compute={"num_nodes": 1, "num_gpus_per_node": 2, "gpu_model": ""},
            )


class TestRunArgv:
    def test_run_sees_core_options(self, launcher, nrl_defaults):
        argv = _argv_from(
            _render(launcher["run"], _cfg(nrl_defaults)), "nemo_rl_run.sh"
        )
        assert _opt(argv, "--algorithm") == "sft"
        assert _opt(argv, "--config-name") == "sft.yaml"
        assert _opt(argv, "--artifact-id") == "model"
        assert _opt(argv, "--base-model") == "/staged/base"
        assert _opt(argv, "--nemo-rl-root").endswith("nemo-rl")

    def test_run_sees_num_gpus_per_node_from_compute_config(
        self, launcher, nrl_defaults
    ):
        argv = _argv_from(
            _render(
                launcher["run"],
                _cfg(nrl_defaults),
                compute={"num_nodes": 1, "num_gpus_per_node": 4, "gpu_model": "A10G"},
            ),
            "nemo_rl_run.sh",
        )
        assert _opt(argv, "--num-gpus-per-node") == "4"

    def test_output_path_defaults_to_workdir_hf_output(self, launcher, nrl_defaults):
        argv = _argv_from(
            _render(launcher["run"], _cfg(nrl_defaults)), "nemo_rl_run.sh"
        )
        assert _opt(argv, "--output-path") == "./hf_output"

    def test_output_path_passes_through(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, output_path="/shared/models/it's-sft")
        argv = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert _opt(argv, "--output-path") == "/shared/models/it's-sft"

    def test_overrides_render_as_key_equals_value(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, overrides={"sft.max_num_steps": 5})
        argv = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert "sft.max_num_steps=5" in _overrides(argv)

    def test_boolean_override_renders_lowercase(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, overrides={"logger.wandb_enabled": False})
        argv = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert "logger.wandb_enabled=false" in _overrides(argv)

    def test_dataset_passed_only_when_bound(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, dataset_override_key="data.train.input_file")
        bound = dict(_BASE_BINDING, dataset={"binding": {"path": "/staged/ds"}})
        argv_bound = _argv_from(
            _render(launcher["run"], cfg, bindings=bound), "nemo_rl_run.sh"
        )
        assert _opt(argv_bound, "--dataset") == "/staged/ds"
        assert _opt(argv_bound, "--dataset-key") == "data.train.input_file"

        argv_unbound = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert "--dataset" not in argv_unbound

    def test_optional_clearml_and_hf_opts(self, launcher, nrl_defaults):
        cfg = _cfg(
            nrl_defaults,
            clearml_project="proj",
            clearml_task_name="t",
            hf_output_repo="org/repo",
        )
        argv = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert _opt(argv, "--clearml-project") == "proj"
        assert _opt(argv, "--clearml-task") == "t"
        assert _opt(argv, "--hf-output-repo") == "org/repo"


class TestHfOutputPrivate:
    @pytest.mark.parametrize(
        "value,want",
        [
            (None, "true"),
            (True, "true"),
            ("true", "true"),
            (False, "false"),
            ("false", "false"),
        ],
    )
    def test_maps_to_run_flag(self, launcher, nrl_defaults, value, want):
        over = {"hf_output_repo": "org/repo"}
        if value is not None:
            over["hf_output_private"] = value
        argv = _argv_from(
            _render(launcher["run"], _cfg(nrl_defaults, **over)), "nemo_rl_run.sh"
        )
        assert _opt(argv, "--hf-output-private") == want

    def test_omitted_without_a_push(self, launcher, nrl_defaults):
        argv = _argv_from(
            _render(launcher["run"], _cfg(nrl_defaults)), "nemo_rl_run.sh"
        )
        assert "--hf-output-private" not in argv


class TestClearmlMode:
    @pytest.mark.parametrize(
        "value,mode",
        [
            ("auto", "auto"),
            (True, "on"),
            ("true", "on"),
            (False, "off"),
            ("false", "off"),
        ],
    )
    def test_clearml_enabled_maps_to_mode(self, launcher, nrl_defaults, value, mode):
        cfg = _cfg(nrl_defaults, clearml_enabled=value)
        argv = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert _opt(argv, "--clearml-mode") == mode


class TestNonScalarOverrides:
    """`q(v)` renders `v | string`, a Python repr: {'a': 1} / [True, 1], which
    Hydra's grammar rejects (quoted dict keys) or misreads (True). Verified
    against hydra-core 1.3.2: even JSON {"a": 1} is rejected. So list/dict
    values fail the render; a structured value is written as a Hydra-syntax
    string instead (e.g. "{a: 1}" or "[1,2]")."""

    @pytest.mark.parametrize("value", [{"a": 1}, [1, 2], [True], []])
    def test_list_or_dict_value_fails_the_render(self, launcher, nrl_defaults, value):
        cfg = _cfg(nrl_defaults, overrides={"policy.x": value})
        with pytest.raises(ValueError, match="policy.x"):
            _render(launcher["run"], cfg)

    def test_hydra_syntax_string_passes_through(self, launcher, nrl_defaults):
        cfg = _cfg(nrl_defaults, overrides={"policy.x": "{a: 1}", "data.y": "[1,2]"})
        argv = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert _overrides(argv) == ["policy.x={a: 1}", "data.y=[1,2]"]

    @pytest.mark.parametrize("value,want", [(3, "3"), (0.5, "0.5"), ("s", "s")])
    def test_scalars_still_render(self, launcher, nrl_defaults, value, want):
        cfg = _cfg(nrl_defaults, overrides={"k": value})
        argv = _argv_from(_render(launcher["run"], cfg), "nemo_rl_run.sh")
        assert _overrides(argv) == [f"k={want}"]


class TestRenderedBlocksAreValidBash:
    def test_setup_block_parses(self, launcher, nrl_defaults):
        assert _bash_ok(_render(launcher["setup"], _cfg(nrl_defaults)))

    def test_run_block_parses_with_tricky_override(self, launcher, nrl_defaults):
        # A quote in an override value must not break the rendered shell.
        cfg = _cfg(nrl_defaults, overrides={"data.prompt": "it's fine"})
        assert _bash_ok(_render(launcher["run"], cfg))
