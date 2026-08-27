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

"""Render tests for the `dpk` step's step-template.yaml.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only, like ``eval/skypilot/test/test_eval.py``) and is not copied by
``make publish-step``.

The step's whole value is that a build names a DPK transform once and the step
derives the rest, so these tests pin the derivations and the shape of the shell
they render into:

* ``transform:`` → the python module *and* the pip extra, for any transform,
  with no per-transform table in the step (that is what keeps it general).
* ``args:`` → ``--flag 'value'``. Keys are full flag names because DPK's own
  prefix is an arbitrary abbreviation for ~40% of transforms (``tkn_`` for
  ``dpk_tokenization``, ``gra_`` for ``gopher_repetition_annotator``), so the
  step must not try to infer them.
* the rendered ``setup``/``run`` are valid bash — the step emits shell, and a
  templating slip (a stray line continuation swallowing the artifact marker) is
  invisible until a cluster run fails.
"""

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
    return dict(template["config"]["dpk_config"])


@pytest.fixture(scope="module")
def launcher(template) -> dict:
    return template["environment_configs"]["Skypilot"]["launchers"]["dpk"]["config"]


def _render(source: str, dpk_config: dict, bindings: dict | None = None) -> str:
    """Render one of the launcher's shell blocks the way gbserver would."""
    return jinja2.Template(source, undefined=jinja2.StrictUndefined).render(
        config={"dpk_config": dpk_config}, bindings=bindings or {}
    )


def _bash_ok(script: str) -> bool:
    """True if bash can parse the script (catches templating slips)."""
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    return subprocess.run(["bash", "-n"], input=script, text=True).returncode == 0


def _transform_cfg(defaults: dict, **over) -> dict:
    base = dict(
        defaults,
        transform="tokenization2arrow",
        input="docs",
        output="tokens",
        output_path="/shared/tokens",
    )
    base.update(over)
    return base


_BINDINGS = {"docs": {"binding": {"path": "/staged/docs"}}}


class TestStepContract:
    """The step declares what the framework and USAGE.md promise."""

    def test_is_an_exec_step_named_dpk(self, template):
        assert template["name"] == "dpk"
        assert template["type"] == "exec"

    def test_no_image_ref_token(self):
        """Public-image step: nothing for publish-step's ${IMAGE_REF} to substitute."""
        assert "${IMAGE_REF}" not in _TEMPLATE.read_text()

    def test_serves_every_skypilot_endpoint(self, template):
        """No `subtypes:` restriction => resolves on slurm/kubernetes/aws/lsf alike."""
        skypilot = template["environment_configs"]["Skypilot"]
        assert "subtypes" not in skypilot
        assert skypilot["default_launcher"] == "dpk"

    def test_uses_the_shared_skypilot_monitor(self, template):
        monitors = template["environment_configs"]["Skypilot"]["monitors"]
        assert monitors["skypilot_monitor"]["ref"] == "space://monitors/skypilot"

    def test_bundles_src_via_file_mounts(self, launcher):
        """The supported relative-source mechanism — how src/ reaches the node."""
        assert launcher["file_mounts"] == {"src": "src"}

    def test_transform_and_command_both_default_empty(self, defaults):
        """`transform` XOR `command`: neither is implied, the build picks one."""
        assert defaults["transform"] == ""
        assert defaults["command"] == ""


class TestImageSelection:
    def test_empty_image_renders_bare_node(self, launcher, defaults):
        rendered = _render(launcher["image_id"], defaults)
        assert rendered == ""

    def test_image_renders_docker_ref(self, launcher, defaults):
        cfg = dict(defaults, image="quay.io/org/img:1.2.3")
        assert _render(launcher["image_id"], cfg) == "docker:quay.io/org/img:1.2.3"


class TestDerivations:
    """One `transform:` value drives both the module and the pip extra."""

    @pytest.mark.parametrize(
        "transform,module,extra",
        [
            (
                "tokenization2arrow",
                "dpk_tokenization2arrow.runtime",
                "tokenization2arrow",
            ),
            ("pii_redactor", "dpk_pii_redactor.runtime", "pii-redactor"),
            ("doc_id", "dpk_doc_id.runtime", "doc-id"),
            ("text_encoder", "dpk_text_encoder.runtime", "text-encoder"),
            ("ededup", "dpk_ededup.runtime", "ededup"),
        ],
    )
    def test_module_and_extra_derive_from_transform(
        self, launcher, defaults, transform, module, extra
    ):
        """Adding a transform is a build.yaml change, never a step change."""
        cfg = _transform_cfg(defaults, transform=transform)
        assert f"python -m {module}" in _render(launcher["run"], cfg, _BINDINGS)
        assert f"'data-prep-toolkit-transforms[{extra}]==1.1.8'" in _render(
            launcher["setup"], cfg
        )

    def test_dpk_version_is_honored(self, launcher, defaults):
        cfg = _transform_cfg(defaults, dpk_version="1.1.7")
        assert "==1.1.7'" in _render(launcher["setup"], cfg)

    def test_module_override_wins(self, launcher, defaults):
        """Needed for the *.ray.runtime variants."""
        cfg = _transform_cfg(defaults, module="dpk_tokenization2arrow.ray.runtime")
        assert "python -m dpk_tokenization2arrow.ray.runtime" in _render(
            launcher["run"], cfg, _BINDINGS
        )

    def test_extras_override_replaces_the_derived_extra(self, launcher, defaults):
        cfg = _transform_cfg(defaults, extras=["ray"])
        setup = _render(launcher["setup"], cfg)
        assert "[ray]==" in setup
        assert "tokenization2arrow]" not in setup

    def test_no_extras_installs_the_bare_package(self, launcher, defaults):
        """noop / c4_annotator declare no extra of their own."""
        cfg = _transform_cfg(defaults, transform="noop", no_extras=True)
        setup = _render(launcher["setup"], cfg)
        assert "'data-prep-toolkit-transforms==1.1.8'" in setup
        assert "[" not in setup.split("data-prep-toolkit-transforms")[1][:2]

    def test_extra_packages_are_appended(self, launcher, defaults):
        cfg = _transform_cfg(defaults, packages=["pyarrow"])
        setup = _render(launcher["setup"], cfg)
        assert "'pyarrow'" in setup
        assert "data-prep-toolkit-transforms[tokenization2arrow]" in setup


class TestTransformArgs:
    def test_args_render_as_full_flag_names(self, launcher, defaults):
        """Keys are DPK's own spelling — the step never adds a prefix."""
        cfg = _transform_cfg(
            defaults, args={"tkn_tokenizer": "hf-internal-testing/llama-tokenizer"}
        )
        run = _render(launcher["run"], cfg, _BINDINGS)
        assert "--tkn_tokenizer 'hf-internal-testing/llama-tokenizer'" in run

    def test_arg_order_is_preserved(self, launcher, defaults):
        cfg = _transform_cfg(defaults, args={"a_one": 1, "b_two": 2, "c_three": 3})
        run = _render(launcher["run"], cfg, _BINDINGS)
        assert run.index("--a_one") < run.index("--b_two") < run.index("--c_three")

    def test_zero_is_passed_not_dropped(self, launcher, defaults):
        """tkn_chunk_size: 0 is meaningful — falsy values must survive."""
        cfg = _transform_cfg(defaults, args={"tkn_chunk_size": 0})
        assert "--tkn_chunk_size '0'" in _render(launcher["run"], cfg, _BINDINGS)

    def test_true_renders_a_bare_flag(self, launcher, defaults):
        cfg = _transform_cfg(defaults, args={"run_locally": True})
        run = _render(launcher["run"], cfg, _BINDINGS)
        assert "--run_locally" in run
        assert "--run_locally '" not in run

    def test_value_with_single_quotes_survives_the_shell(self, launcher, defaults):
        """Regression guard for a real bug found by the pii_redactor fixture.

        Several DPK transforms take python-literal values: pii_redactor's
        --pii_redactor_entities is ``ast.literal_eval``'d, so it must reach python
        as ``['PERSON','EMAIL_ADDRESS']``. Naive single-quoting emitted
        ``'['PERSON',...]'``, which bash collapses to ``[PERSON,...]`` — bare names
        that literal_eval rejects with ValueError. The value must be re-quoted.
        """
        value = "['PERSON','EMAIL_ADDRESS']"
        cfg = _transform_cfg(defaults, args={"pii_redactor_entities": value})
        run = _render(launcher["run"], cfg, _BINDINGS)

        # Ask bash what it would actually pass, rather than eyeballing the quoting.
        line = next(
            l
            for l in run.splitlines()
            if "pii_redactor_entities" in l and l.strip().startswith("DPK_ARGS+=")
        )
        probe = f'DPK_ARGS=()\n{line.strip()}\nprintf "%s" "${{DPK_ARGS[1]}}"'
        out = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
        assert out.stdout == value, f"bash mangled it to {out.stdout!r}"

    def test_false_and_none_are_omitted(self, launcher, defaults):
        cfg = _transform_cfg(defaults, args={"off": False, "unset": None})
        run = _render(launcher["run"], cfg, _BINDINGS)
        assert "--off" not in run
        assert "--unset" not in run


class TestIoWiring:
    def test_each_binding_is_exported(self, launcher, defaults):
        cfg = _transform_cfg(defaults)
        bindings = {
            "docs": {"binding": {"path": "/staged/docs"}},
            "extra": {"binding": {"path": "/staged/extra"}},
        }
        run = _render(launcher["run"], cfg, bindings)
        assert "export LLMB_INPUT_docs='/staged/docs'" in run
        assert "export LLMB_INPUT_extra='/staged/extra'" in run

    def test_input_feeds_data_local_config(self, launcher, defaults):
        run = _render(launcher["run"], _transform_cfg(defaults), _BINDINGS)
        assert 'IN="$LLMB_INPUT_docs"' in run
        assert "'input_folder': '$IN'" in run
        assert "'output_folder': '$OUT'" in run

    def test_output_path_is_created_and_marked(self, launcher, defaults):
        run = _render(launcher["run"], _transform_cfg(defaults), _BINDINGS)
        assert "OUT='/shared/tokens'" in run
        assert 'mkdir -p "$OUT"' in run
        assert 'echo "LLMB_ARTIFACT_ID:tokens LLMB_ARTIFACT_PATH:$OUT"' in run


class TestCommandMode:
    def test_command_is_injected_verbatim(self, launcher, defaults):
        cmd = 'python src/validate_tokens.py "$LLMB_INPUT_tokens" out --input in'
        cfg = dict(defaults, command=cmd, packages=["pyarrow"])
        run = _render(launcher["run"], cfg, {"tokens": {"binding": {"path": "/tok"}}})
        assert cmd in run

    def test_command_mode_skips_the_transform_invocation(self, launcher, defaults):
        cfg = dict(defaults, command="echo hi", packages=["pyarrow"])
        run = _render(launcher["run"], cfg, {})
        assert "python -m dpk_" not in run
        assert "--data_local_config" not in run

    def test_command_mode_still_exports_bindings(self, launcher, defaults):
        cfg = dict(defaults, command="echo hi")
        run = _render(launcher["run"], cfg, {"tokens": {"binding": {"path": "/tok"}}})
        assert "export LLMB_INPUT_tokens='/tok'" in run

    def test_command_mode_installs_only_its_packages(self, launcher, defaults):
        """No transform => no DPK requirement is synthesized."""
        cfg = dict(defaults, command="echo hi", packages=["pyarrow"])
        setup = _render(launcher["setup"], cfg)
        assert "'pyarrow'" in setup
        assert "data-prep-toolkit-transforms" not in setup


class TestVenvHandling:
    def test_bare_node_builds_a_venv(self, launcher, defaults):
        cfg = _transform_cfg(defaults)
        assert "python -m venv ./venv" in _render(launcher["setup"], cfg)
        assert ". ./venv/bin/activate" in _render(launcher["run"], cfg, _BINDINGS)

    def test_image_mode_skips_venv_and_pip(self, launcher, defaults):
        """An image already provides DPK, so nothing is installed at run time."""
        cfg = _transform_cfg(defaults, image="quay.io/org/dpk:1")
        setup = _render(launcher["setup"], cfg)
        run = _render(launcher["run"], cfg, _BINDINGS)
        assert "venv" not in setup
        assert "pip install" not in setup
        assert "venv" not in run
        # the transform still runs
        assert "python -m dpk_tokenization2arrow.runtime" in run


class TestRenderedShellIsValid:
    """The step emits bash; a templating slip is invisible until a cluster run."""

    @pytest.mark.parametrize(
        "cfg_kwargs,bindings",
        [
            ({}, _BINDINGS),
            ({"args": {"tkn_chunk_size": 0, "flag": True}}, _BINDINGS),
            ({"image": "quay.io/org/dpk:1"}, _BINDINGS),
            ({"packages": ["pyarrow"]}, _BINDINGS),
        ],
    )
    def test_transform_mode_parses(self, launcher, defaults, cfg_kwargs, bindings):
        cfg = _transform_cfg(defaults, **cfg_kwargs)
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg, bindings))

    def test_command_mode_parses(self, launcher, defaults):
        cfg = dict(defaults, command="echo hi; echo there", packages=["pyarrow"])
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg, {}))

    def test_artifact_marker_is_not_swallowed_by_a_continuation(
        self, launcher, defaults
    ):
        """Regression guard.

        An earlier draft emitted args as backslash-continued lines, so the final
        flag's trailing "\\" spliced the next line into the python invocation. The
        marker must be its own command.
        """
        cfg = _transform_cfg(defaults, args={"tkn_chunk_size": 0})
        run = _render(launcher["run"], cfg, _BINDINGS)
        marker_line = next(
            line for line in run.splitlines() if "LLMB_ARTIFACT_ID" in line
        )
        assert marker_line.lstrip().startswith("echo ")
        preceding = run.split(marker_line)[0].rstrip().splitlines()[-1]
        assert not preceding.rstrip().endswith("\\")
