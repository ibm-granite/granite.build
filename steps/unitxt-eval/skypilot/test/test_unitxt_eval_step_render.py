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

"""Render tests for the `unitxt-eval` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``.

The step treats ``unitxt-evaluate`` as a black box, so these tests pin what the
rendered ``run`` block hands it, by executing the block with a stub CLI on PATH,
plus the artifact marker and the bare-node/image switch.
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
    return template["environment_configs"]["Skypilot"]["launchers"]["unitxt-eval"][
        "config"
    ]


def _cfg(defaults: dict, **over) -> dict:
    """The README's example config, with overrides."""
    base = dict(
        defaults,
        tasks="card=cards.mmlu_pro.engineering",
        model="cross_provider",
        model_args="model_name=llama-3-1-8b-instruct",
        limit=10,
    )
    base.update(over)
    return base


def _render(source: str, unitxt_config: dict) -> str:
    return jinja2.Template(source, undefined=jinja2.StrictUndefined).render(
        config={"unitxt_config": unitxt_config}
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
    def test_is_a_custom_step_named_unitxt_eval(self, template):
        assert template["name"] == "unitxt-eval"
        assert template["type"] == "custom"

    def test_no_image_ref_token(self):
        """Public-image step: nothing for publish-step's ${IMAGE_REF} to substitute."""
        assert "${IMAGE_REF}" not in _TEMPLATE.read_text()

    def test_serves_every_skypilot_endpoint(self, template):
        skypilot = template["environment_configs"]["Skypilot"]
        assert "subtypes" not in skypilot
        assert skypilot["default_launcher"] == "unitxt-eval"

    def test_declares_no_fixed_inputs_and_a_results_output(self, template):
        """The model is named in model_args, so no input name is fixed."""
        assert template["inputs"] == {"allow_unknown": True}
        assert template["outputs"]["optional"]["results"]["type"] == "dataset"

    def test_uses_the_shared_skypilot_monitor(self, template):
        monitors = template["environment_configs"]["Skypilot"]["monitors"]
        assert monitors["skypilot_monitor"]["ref"] == "space://monitors/skypilot"


class TestCliInvocation:
    def test_readme_example_reaches_the_cli(self, launcher, defaults, tmp_path):
        argv = _argv(_run(_render(launcher["run"], _cfg(defaults)), tmp_path))
        assert argv == [
            "--tasks",
            "card=cards.mmlu_pro.engineering",
            "--model",
            "cross_provider",
            "--model_args",
            "model_name=llama-3-1-8b-instruct",
            "--limit",
            "10",
            "--output_path",
            str((tmp_path / "output").resolve()),
        ]

    def test_zero_limit_passes_no_limit(self, launcher, defaults, tmp_path):
        argv = _argv(_run(_render(launcher["run"], _cfg(defaults, limit=0)), tmp_path))
        assert "--limit" not in argv

    def test_multi_task_string_stays_one_word(self, launcher, defaults, tmp_path):
        tasks = "card=cards.text2sql.bird+card=cards.mmlu_pro.engineering"
        argv = _argv(
            _run(_render(launcher["run"], _cfg(defaults, tasks=tasks)), tmp_path)
        )
        assert _opt(argv, "--tasks") == tasks

    def test_hf_model_is_passed_through(self, launcher, defaults, tmp_path):
        # unitxt-evaluate --help: `hf` requires pretrained=..., cross_provider model_name=...
        cfg = _cfg(
            defaults, model="hf", model_args="pretrained=ibm-granite/granite-4.2-3b"
        )
        argv = _argv(_run(_render(launcher["run"], cfg), tmp_path))
        assert _opt(argv, "--model") == "hf"
        assert _opt(argv, "--model_args") == "pretrained=ibm-granite/granite-4.2-3b"


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


class TestRequiredConfig:
    @pytest.mark.parametrize("field", ["tasks", "model_args"])
    def test_empty_required_field_is_refused(self, launcher, defaults, field, tmp_path):
        proc = _run(_render(launcher["run"], _cfg(defaults, **{field: ""})), tmp_path)
        assert proc.returncode != 0
        assert f"unitxt_config.{field} is required" in proc.stderr
        assert "ARG:" not in proc.stdout


class TestEscaping:
    @pytest.mark.parametrize("field", ["tasks", "model", "model_args", "output_path"])
    def test_a_quote_and_backtick_cannot_execute(
        self, launcher, defaults, field, tmp_path
    ):
        canary = tmp_path / "canary"
        cfg = _cfg(defaults, **{field: f"x'`touch {canary}`'"})
        rendered = _render(launcher["run"], cfg)
        assert _bash_ok(rendered)
        _run(rendered, tmp_path)
        assert not canary.exists(), f"{field} executed an embedded command"

    def test_a_quote_survives_as_data(self, launcher, defaults, tmp_path):
        cfg = _cfg(defaults, model_args="model_name=o'brien")
        argv = _argv(_run(_render(launcher["run"], cfg), tmp_path))
        assert _opt(argv, "--model_args") == "model_name=o'brien"

    @pytest.mark.parametrize("field", ["pip_index_url", "unitxt_version"])
    def test_setup_values_are_escaped(self, launcher, defaults, field):
        rendered = _render(launcher["setup"], _cfg(defaults, **{field: "a'b"}))
        assert _bash_ok(rendered)
        assert "'\"'\"'" in rendered


class TestImageSelection:
    def test_empty_image_renders_bare_node(self, launcher, defaults):
        assert _render(launcher["image_id"], _cfg(defaults)) == ""

    def test_image_renders_docker_ref(self, launcher, defaults):
        cfg = _cfg(defaults, unitxt_image="quay.io/org/unitxt:1.26.10")
        assert _render(launcher["image_id"], cfg) == "docker:quay.io/org/unitxt:1.26.10"

    def test_bare_node_installs_the_pinned_unitxt(self, launcher, defaults):
        setup = _render(launcher["setup"], _cfg(defaults))
        assert "uv venv ./venv" in setup
        assert "'unitxt==1.26.10'" in setup
        assert ". ./venv/bin/activate" in _render(launcher["run"], _cfg(defaults))

    def test_image_mode_skips_venv_and_install(self, launcher, defaults):
        cfg = _cfg(defaults, unitxt_image="quay.io/org/unitxt:1")
        assert "venv" not in _render(launcher["setup"], cfg)
        assert "venv" not in _render(launcher["run"], cfg)


class TestRenderedShellIsValid:
    @pytest.mark.parametrize(
        "over", [{}, {"limit": 0}, {"unitxt_image": "quay.io/org/unitxt:1"}]
    )
    def test_both_blocks_parse(self, launcher, defaults, over):
        cfg = _cfg(defaults, **over)
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg))
