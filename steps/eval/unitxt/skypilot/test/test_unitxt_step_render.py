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

"""Render tests for the `eval/unitxt` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``.

The step treats ``unitxt-evaluate`` as a black box, so these tests pin what the
rendered ``run`` block hands it, by executing the block with a stub CLI on PATH,
plus the declared ``model`` input, the artifact marker and the bare-node/image
switch.
"""

import os
import pathlib
import shutil
import subprocess

import pytest
import yaml

jinja2 = pytest.importorskip("jinja2", reason="jinja2 renders the step template")

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = _STEP_DIR / "step-template.yaml"


@pytest.fixture(scope="module")
def template() -> dict:
    """The step template, parsed as YAML (its Jinja lives inside string scalars)."""
    return yaml.safe_load(_TEMPLATE.read_text())


@pytest.fixture(scope="module")
def defaults(template) -> dict:
    return dict(template["config"]["unitxt_config"])


@pytest.fixture(scope="module")
def launcher(template) -> dict:
    return template["environment_configs"]["Skypilot"]["launchers"]["unitxt"]["config"]


# Where the launcher puts the bound `model` input (an hf:// URI is downloaded here).
_MODEL_PATH = "/workdir/inputs/model"


def _cfg(defaults: dict, **over) -> dict:
    """The USAGE.md example config, with overrides."""
    base = dict(
        defaults,
        tasks="card=cards.mmlu_pro.engineering",
        model_args="torch_dtype=bfloat16,device=cuda",
        limit=10,
    )
    base.update(over)
    return base


def _render(source: str, unitxt_config: dict, model_path: str = _MODEL_PATH) -> str:
    return jinja2.Template(source, undefined=jinja2.StrictUndefined).render(
        config={"unitxt_config": unitxt_config},
        bindings={"model": {"binding": {"path": model_path}}},
    )


def _bash_ok(script: str) -> bool:
    """True if bash can parse the script (catches templating slips)."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    return subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def _run(rendered: str, cwd: pathlib.Path) -> subprocess.CompletedProcess:
    """Execute a rendered `run` block with a stub unitxt-evaluate on PATH.

    The stub prints one ``ARG:<word>`` line per argument, so assertions are on the
    argv bash actually built rather than on the rendered text.
    """
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    bin_dir = cwd / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "unitxt-evaluate"
    stub.write_text('#!/usr/bin/env bash\nfor a in "$@"; do echo "ARG:$a"; done\n')
    stub.chmod(0o755)
    (cwd / "venv" / "bin").mkdir(parents=True)
    (cwd / "venv" / "bin" / "activate").write_text("")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", "-c", rendered], capture_output=True, text=True, cwd=cwd, env=env
    )


def _argv(proc: subprocess.CompletedProcess) -> list[str]:
    assert proc.returncode == 0, f"rendered block failed: {proc.stderr}"
    return [l[len("ARG:") :] for l in proc.stdout.splitlines() if l.startswith("ARG:")]


def _opt(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


class TestStepContract:
    def test_is_a_custom_step_named_unitxt(self, template):
        assert template["name"] == "unitxt"
        assert template["type"] == "custom"

    def test_no_image_ref_token(self):
        """Public-image step: nothing for publish-step's ${IMAGE_REF} to substitute."""
        assert "${IMAGE_REF}" not in _TEMPLATE.read_text()

    def test_serves_every_skypilot_endpoint(self, template):
        skypilot = template["environment_configs"]["Skypilot"]
        assert "subtypes" not in skypilot
        assert skypilot["default_launcher"] == "unitxt"

    def test_declares_a_required_model_input_and_a_results_output(self, template):
        """The model is a declared input read from bindings (the #457 convention)."""
        assert template["inputs"]["allow_unknown"] is True
        assert template["inputs"]["required"] == {
            "model": {"type": "model", "accept": ["uri", "binding"]}
        }
        assert "optional" not in template["inputs"]
        assert template["outputs"]["optional"]["results"]["type"] == "dataset"

    def test_the_model_is_not_a_config_key(self, defaults):
        assert "model" not in defaults

    def test_uses_the_shared_skypilot_monitor(self, template):
        monitors = template["environment_configs"]["Skypilot"]["monitors"]
        assert monitors["skypilot_monitor"]["ref"] == "space://monitors/skypilot"


class TestCliInvocation:
    def test_usage_example_reaches_the_cli(self, launcher, defaults, tmp_path):
        argv = _argv(_run(_render(launcher["run"], _cfg(defaults)), tmp_path))
        assert argv == [
            "--tasks",
            "card=cards.mmlu_pro.engineering",
            "--model",
            "hf",
            "--model_args",
            f"pretrained={_MODEL_PATH},torch_dtype=bfloat16,device=cuda",
            "--batch_size",
            "1",
            "--trust_remote_code",
            "--limit",
            "10",
            "--output_path",
            str((tmp_path / "output").resolve()),
        ]

    def test_no_extra_model_args_passes_only_pretrained(
        self, launcher, defaults, tmp_path
    ):
        argv = _argv(
            _run(_render(launcher["run"], _cfg(defaults, model_args="")), tmp_path)
        )
        assert _opt(argv, "--model_args") == f"pretrained={_MODEL_PATH}"

    def test_the_bound_model_path_is_passed(self, launcher, defaults, tmp_path):
        rendered = _render(launcher["run"], _cfg(defaults), model_path="/m/granite")
        argv = _argv(_run(rendered, tmp_path))
        assert _opt(argv, "--model_args").startswith("pretrained=/m/granite,")

    def test_zero_limit_passes_no_limit(self, launcher, defaults, tmp_path):
        argv = _argv(_run(_render(launcher["run"], _cfg(defaults, limit=0)), tmp_path))
        assert "--limit" not in argv

    def test_trust_remote_code_can_be_turned_off(self, launcher, defaults, tmp_path):
        cfg = _cfg(defaults, trust_remote_code=False)
        argv = _argv(_run(_render(launcher["run"], cfg), tmp_path))
        assert "--trust_remote_code" not in argv

    def test_batch_size_is_passed(self, launcher, defaults, tmp_path):
        cfg = _cfg(defaults, batch_size=8)
        argv = _argv(_run(_render(launcher["run"], cfg), tmp_path))
        assert _opt(argv, "--batch_size") == "8"

    def test_multi_task_string_stays_one_word(self, launcher, defaults, tmp_path):
        tasks = "card=cards.text2sql.bird+card=cards.mmlu_pro.engineering"
        argv = _argv(
            _run(_render(launcher["run"], _cfg(defaults, tasks=tasks)), tmp_path)
        )
        assert _opt(argv, "--tasks") == tasks


class TestOutput:
    def test_marker_registers_the_absolute_output_dir(
        self, launcher, defaults, tmp_path
    ):
        proc = _run(_render(launcher["run"], _cfg(defaults)), tmp_path)
        assert proc.returncode == 0, proc.stderr
        expected = (
            f"GB_ARTIFACT_ID:results GB_ARTIFACT_PATH:{(tmp_path / 'output').resolve()}"
        )
        assert proc.stdout.splitlines()[-1] == expected

    def test_custom_output_path_is_created(self, launcher, defaults, tmp_path):
        cfg = _cfg(defaults, output_path="res/mmlu")
        argv = _argv(_run(_render(launcher["run"], cfg), tmp_path))
        assert _opt(argv, "--output_path") == str((tmp_path / "res/mmlu").resolve())
        assert (tmp_path / "res" / "mmlu").is_dir()


class TestRefusedConfig:
    def _refused(self, launcher, cfg, tmp_path, message):
        proc = _run(_render(launcher["run"], cfg), tmp_path)
        assert proc.returncode != 0
        assert message in proc.stderr
        assert "ARG:" not in proc.stdout

    def test_empty_tasks_is_refused(self, launcher, defaults, tmp_path):
        self._refused(
            launcher,
            _cfg(defaults, tasks=""),
            tmp_path,
            "unitxt_config.tasks is required",
        )

    @pytest.mark.parametrize("model", ["hf", "cross_provider"])
    def test_the_old_model_key_is_refused(self, launcher, defaults, model, tmp_path):
        self._refused(
            launcher,
            _cfg(defaults, model=model),
            tmp_path,
            "unitxt_config.model is no longer read",
        )

    @pytest.mark.parametrize(
        "model_args",
        ["pretrained=ibm-granite/granite-4.2-3b", "device=cuda,pretrained=x"],
    )
    def test_pretrained_in_model_args_is_refused(
        self, launcher, defaults, model_args, tmp_path
    ):
        self._refused(
            launcher,
            _cfg(defaults, model_args=model_args),
            tmp_path,
            "must not set pretrained",
        )

    def test_json_model_args_is_refused(self, launcher, defaults, tmp_path):
        self._refused(
            launcher,
            _cfg(defaults, model_args='{"torch_dtype": "bfloat16"}'),
            tmp_path,
            "not JSON",
        )


class TestEscaping:
    @pytest.mark.parametrize(
        "field", ["tasks", "model_args", "output_path", "batch_size"]
    )
    def test_a_quote_and_backtick_cannot_execute(
        self, launcher, defaults, field, tmp_path
    ):
        canary = tmp_path / "canary"
        cfg = _cfg(defaults, **{field: f"x'`touch {canary}`'"})
        rendered = _render(launcher["run"], cfg)
        assert _bash_ok(rendered)
        _run(rendered, tmp_path)
        assert not canary.exists(), f"{field} executed an embedded command"

    def test_the_model_path_cannot_execute(self, launcher, defaults, tmp_path):
        canary = tmp_path / "canary"
        rendered = _render(
            launcher["run"], _cfg(defaults), model_path=f"x'`touch {canary}`'"
        )
        assert _bash_ok(rendered)
        _run(rendered, tmp_path)
        assert not canary.exists(), "the model path executed an embedded command"

    def test_a_quote_survives_as_data(self, launcher, defaults, tmp_path):
        cfg = _cfg(defaults, model_args="revision=o'brien")
        argv = _argv(_run(_render(launcher["run"], cfg), tmp_path))
        assert (
            _opt(argv, "--model_args") == f"pretrained={_MODEL_PATH},revision=o'brien"
        )

    @pytest.mark.parametrize(
        "field", ["pip_index_url", "unitxt_version", "torch_package", "torch_index_url"]
    )
    def test_setup_values_are_escaped(self, launcher, defaults, field):
        rendered = _render(launcher["setup"], _cfg(defaults, **{field: "a'b"}))
        assert _bash_ok(rendered)
        assert "'\"'\"'" in rendered

    def test_hf_packages_are_escaped(self, launcher, defaults):
        rendered = _render(launcher["setup"], _cfg(defaults, hf_packages=["a'b"]))
        assert _bash_ok(rendered)
        assert "'a'\"'\"'b'" in rendered


class TestImageSelection:
    def test_empty_image_renders_bare_node(self, launcher, defaults):
        assert _render(launcher["image_id"], _cfg(defaults)) == ""

    def test_image_renders_docker_ref(self, launcher, defaults):
        cfg = _cfg(defaults, unitxt_image="quay.io/org/unitxt:1.26.10")
        assert _render(launcher["image_id"], cfg) == "docker:quay.io/org/unitxt:1.26.10"

    def test_bare_node_installs_torch_then_unitxt_and_the_hf_packages(
        self, launcher, defaults
    ):
        setup = _render(launcher["setup"], _cfg(defaults))
        assert "uv venv ./venv" in setup
        torch = "uv pip install --index-url 'https://pypi.org/simple' \\\n  'torch'"
        rest = (
            "uv pip install --index-url 'https://pypi.org/simple' \\\n"
            "  'unitxt==1.26.10' 'transformers' 'accelerate' 'tabulate'"
        )
        assert torch in setup and rest in setup
        assert setup.index(torch) < setup.index(rest)
        assert ". ./venv/bin/activate" in _render(launcher["run"], _cfg(defaults))

    def test_torch_index_url_applies_only_to_torch(self, launcher, defaults):
        cpu = "https://download.pytorch.org/whl/cpu"
        setup = _render(launcher["setup"], _cfg(defaults, torch_index_url=cpu))
        assert f"uv pip install --index-url '{cpu}' \\\n  'torch'" in setup
        assert (
            "uv pip install --index-url 'https://pypi.org/simple' \\\n"
            "  'unitxt==1.26.10'" in setup
        )

    def test_torch_and_hf_packages_can_be_pinned(self, launcher, defaults):
        cfg = _cfg(
            defaults, torch_package="torch==2.8.0", hf_packages=["transformers==5.0"]
        )
        setup = _render(launcher["setup"], cfg)
        assert "'torch==2.8.0'" in setup
        assert "'unitxt==1.26.10' 'transformers==5.0'" in setup
        assert "accelerate" not in setup

    def test_uv_cache_stays_in_the_build_workdir(self, launcher, defaults):
        """The launcher cds into $GB_BUILD_WORKDIR first, so $PWD is inside it."""
        setup = _render(launcher["setup"], _cfg(defaults))
        assert 'export UV_CACHE_DIR="$PWD/.uv-cache"' in setup
        code = [l for l in setup.splitlines() if not l.lstrip().startswith("#")]
        assert not any("GB_SHARED_WORKDIR" in l for l in code)

    def test_image_mode_skips_venv_and_install(self, launcher, defaults):
        cfg = _cfg(defaults, unitxt_image="quay.io/org/unitxt:1")
        assert "venv" not in _render(launcher["setup"], cfg)
        assert "venv" not in _render(launcher["run"], cfg)


class TestRenderedShellIsValid:
    @pytest.mark.parametrize(
        "over",
        [
            {},
            {"limit": 0},
            {"model_args": ""},
            {"trust_remote_code": False},
            {"hf_packages": []},
            {"torch_index_url": "https://download.pytorch.org/whl/cpu"},
            {"unitxt_image": "quay.io/org/unitxt:1"},
        ],
    )
    def test_both_blocks_parse(self, launcher, defaults, over):
        cfg = _cfg(defaults, **over)
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg))
