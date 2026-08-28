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

# The rendered blocks now invoke the bundled scripts rather than inlining the
# shell, so the meaningful assertion is "what argv does the script receive?".
# Ask bash, rather than parsing the rendered text with a regex: bash is the thing
# that actually splits and unquotes these words on the node, so a quoting slip
# shows up here exactly as it would in production.
_SCRIPTS = {"run": "dpk_run.sh", "setup": "dpk_setup.sh"}


def _script_argv(rendered: str, which: str) -> list[str]:
    """Return the argv the rendered block passes to the bundled script.

    Replaces the `bash ./src/<script>` invocation with a stub that prints one
    argument per line, then executes the rendered block. Everything else in the
    block (the venv activation, the LLMB_INPUT_ exports) is stubbed or harmless.
    """
    if shutil.which("bash") is None:  # pragma: no cover - bash is present in CI
        pytest.skip("bash not available")
    script = _SCRIPTS[which]
    harness = "\n".join(
        [
            "set -e",
            # Stub the venv activation the run block performs in bare-node mode.
            "mkdir -p ./venv/bin && : > ./venv/bin/activate",
            # Stand in for the real script: emit argv, one per line, NUL-free.
            "mkdir -p ./src",
            f'printf "%s\\n" \'#!/usr/bin/env bash\' \'for a in "$@"; do echo "ARG:$a"; done\''
            f" > ./src/{script}",
            f"chmod +x ./src/{script}",
            rendered,
        ]
    )
    proc = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, cwd=_TMPDIR
    )
    assert proc.returncode == 0, f"rendered block failed: {proc.stderr}"
    return [
        line[len("ARG:") :]
        for line in proc.stdout.splitlines()
        if line.startswith("ARG:")
    ]


@pytest.fixture(autouse=True)
def _tmp_cwd(tmp_path, monkeypatch):
    """Give _script_argv a scratch dir, so stubs never touch the step tree."""
    monkeypatch.setitem(globals(), "_TMPDIR", str(tmp_path))


_TMPDIR = "."


def _opt(argv: list[str], name: str) -> str | None:
    """Return the value following ``name`` in an argv list, or None."""
    return argv[argv.index(name) + 1] if name in argv else None


def _passthrough(argv: list[str]) -> list[str]:
    """Return the transform flags: everything after the ``--`` separator."""
    return argv[argv.index("--") + 1 :] if "--" in argv else []


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
        run_argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(run_argv, "--module") == module
        setup_argv = _script_argv(_render(launcher["setup"], cfg), "setup")
        assert _passthrough(setup_argv) == [
            f"data-prep-toolkit-transforms[{extra}]==1.1.8"
        ]

    def test_dpk_version_is_honored(self, launcher, defaults):
        cfg = _transform_cfg(defaults, dpk_version="1.1.7")
        assert "==1.1.7'" in _render(launcher["setup"], cfg)

    def test_module_override_wins(self, launcher, defaults):
        """Needed for the *.ray.runtime variants."""
        cfg = _transform_cfg(defaults, module="dpk_tokenization2arrow.ray.runtime")
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--module") == "dpk_tokenization2arrow.ray.runtime"

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
    """`args` become real argv words handed to dpk_run.sh after the `--`."""

    def test_args_render_as_full_flag_names(self, launcher, defaults):
        """Keys are DPK's own spelling — the step never adds a prefix."""
        cfg = _transform_cfg(
            defaults, args={"tkn_tokenizer": "hf-internal-testing/llama-tokenizer"}
        )
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == [
            "--tkn_tokenizer",
            "hf-internal-testing/llama-tokenizer",
        ]

    def test_arg_order_is_preserved(self, launcher, defaults):
        cfg = _transform_cfg(defaults, args={"a_one": 1, "b_two": 2, "c_three": 3})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == [
            "--a_one", "1", "--b_two", "2", "--c_three", "3",
        ]

    def test_zero_is_passed_not_dropped(self, launcher, defaults):
        """tkn_chunk_size: 0 is meaningful — falsy values must survive."""
        cfg = _transform_cfg(defaults, args={"tkn_chunk_size": 0})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--tkn_chunk_size", "0"]

    def test_true_renders_a_bare_flag(self, launcher, defaults):
        cfg = _transform_cfg(defaults, args={"run_locally": True})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--run_locally"]

    def test_value_with_single_quotes_survives_the_shell(self, launcher, defaults):
        """Regression guard for a real bug found by the pii_redactor fixture.

        Several DPK transforms take python-literal values: pii_redactor's
        --pii_redactor_entities is ``ast.literal_eval``'d, so it must reach python
        as ``['PERSON','EMAIL_ADDRESS']``. Naive single-quoting emitted
        ``'['PERSON',...]'``, which bash collapses to ``[PERSON,...]`` — bare names
        that literal_eval rejects with ValueError.

        Asserting on the argv bash actually built is strictly stronger than
        matching the rendered text: this is the value the transform receives.
        """
        value = "['PERSON','EMAIL_ADDRESS']"
        cfg = _transform_cfg(defaults, args={"pii_redactor_entities": value})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--pii_redactor_entities", value]

    def test_false_and_none_are_omitted(self, launcher, defaults):
        cfg = _transform_cfg(defaults, args={"off": False, "unset": None})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == []


class TestOutputPathDefault:
    """`output_path` is optional and defaults to ./output in the step's workdir.

    The template's job is to pass the right --output-path; making it ABSOLUTE is
    dpk_run.sh's job, covered by test_dpk_run_sh.py.
    """

    def test_omitted_output_path_defaults_to_output(self, launcher, defaults):
        cfg = _transform_cfg(defaults, output_path="")
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--output-path") == "./output"

    def test_explicit_output_path_is_passed_through(self, launcher, defaults):
        rendered = _render(launcher["run"], _transform_cfg(defaults), _BINDINGS)
        assert _opt(_script_argv(rendered, "run"), "--output-path") == "/shared/tokens"

    def test_default_output_still_parses(self, launcher, defaults):
        cfg = _transform_cfg(defaults, output_path="")
        assert _bash_ok(_render(launcher["run"], cfg, _BINDINGS))


class TestExtraArgs:
    """The verbatim escape hatch, for flags the `args` map cannot express."""

    def test_default_adds_nothing(self, launcher, defaults):
        """Empty extra_args contributes no argv beyond what `args` rendered."""
        argv = _script_argv(_render(launcher["run"], _transform_cfg(defaults), _BINDINGS), "run")
        assert _passthrough(argv) == []

    def test_extra_args_are_appended_after_args(self, launcher, defaults):
        cfg = _transform_cfg(
            defaults, args={"tkn_chunk_size": 0}, extra_args="--tkn_text_lang en"
        )
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == [
            "--tkn_chunk_size", "0", "--tkn_text_lang", "en",
        ]

    def test_extra_args_are_word_split_by_the_shell(self, launcher, defaults):
        """The contract that distinguishes extra_args from args.

        `args` values are quoted to reach the transform byte-for-byte; extra_args
        is expanded by the remote shell instead, so it becomes several words.
        """
        cfg = _transform_cfg(defaults, extra_args="--flag one --other two")
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--flag", "one", "--other", "two"]

    def test_extra_args_render_valid_shell(self, launcher, defaults):
        cfg = _transform_cfg(defaults, extra_args="--flag 'a value' --bare")
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        # Single-quoting inside extra_args is honoured by the shell, so the
        # spaced value arrives as ONE word.
        assert _passthrough(argv) == ["--flag", "a value", "--bare"]

    def test_command_mode_ignores_extra_args(self, launcher, defaults):
        """extra_args belongs to the derived invocation, which command mode skips."""
        cfg = dict(defaults, command="echo hi", extra_args="--should-not-appear")
        run = _render(launcher["run"], cfg, _BINDINGS)
        assert "--should-not-appear" not in run


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

    def test_input_is_passed_as_the_bindings_staged_path(self, launcher, defaults):
        """--input-path resolves through $LLMB_INPUT_<input>, not a hardcoded path.

        Assembling DPK's --data_local_config from it is dpk_run.sh's job, covered
        by test_dpk_run_sh.py.
        """
        rendered = _render(launcher["run"], _transform_cfg(defaults), _BINDINGS)
        argv = _script_argv(rendered, "run")
        assert _opt(argv, "--input-path") == "/staged/docs"

    def test_artifact_id_is_the_declared_output(self, launcher, defaults):
        rendered = _render(launcher["run"], _transform_cfg(defaults), _BINDINGS)
        assert _opt(_script_argv(rendered, "run"), "--artifact-id") == "tokens"


class TestCommandMode:
    def test_command_is_injected_verbatim(self, launcher, defaults):
        cmd = 'python src/validate_tokens.py "$LLMB_INPUT_tokens" out --input in'
        cfg = dict(defaults, command=cmd, packages=["pyarrow"])
        run = _render(launcher["run"], cfg, {"tokens": {"binding": {"path": "/tok"}}})
        assert cmd in run

    def test_command_mode_skips_the_transform_invocation(self, launcher, defaults):
        cfg = dict(defaults, command="echo hi", packages=["pyarrow"])
        run = _render(launcher["run"], cfg, {})
        assert "dpk_run.sh" not in run
        assert "--artifact-id" not in run

    def test_command_mode_still_exports_bindings(self, launcher, defaults):
        cfg = dict(defaults, command="echo hi")
        run = _render(launcher["run"], cfg, {"tokens": {"binding": {"path": "/tok"}}})
        assert "export LLMB_INPUT_tokens='/tok'" in run

    def test_command_mode_installs_only_its_packages(self, launcher, defaults):
        """No transform => no DPK requirement is synthesized."""
        cfg = dict(defaults, command="echo hi", packages=["pyarrow"])
        argv = _script_argv(_render(launcher["setup"], cfg), "setup")
        assert _passthrough(argv) == ["pyarrow"]


class TestVenvHandling:
    def test_bare_node_builds_a_venv(self, launcher, defaults):
        """setup delegates the venv to dpk_setup.sh; run activates it."""
        cfg = _transform_cfg(defaults)
        argv = _script_argv(_render(launcher["setup"], cfg), "setup")
        assert _opt(argv, "--venv") == "./venv"
        assert ". ./venv/bin/activate" in _render(launcher["run"], cfg, _BINDINGS)

    def test_requirements_are_passed_as_argv(self, launcher, defaults):
        """The derived DPK requirement reaches the script as ONE argv word.

        The "[extra]" in data-prep-toolkit-transforms[tokenization2arrow] would be
        a glob candidate if it were not quoted; asserting on bash-split argv proves
        it arrives intact. The uv/UV_CACHE_DIR mechanics are the script's own
        contract, covered by test_dpk_setup_sh.py.
        """
        argv = _script_argv(_render(launcher["setup"], _transform_cfg(defaults)), "setup")
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[tokenization2arrow]==1.1.8"
        ]
        assert _opt(argv, "--index-url") == "https://pypi.org/simple"

    def test_image_mode_skips_venv_and_pip(self, launcher, defaults):
        """An image already provides DPK, so nothing is installed at run time."""
        cfg = _transform_cfg(defaults, image="quay.io/org/dpk:1")
        setup = _render(launcher["setup"], cfg)
        run = _render(launcher["run"], cfg, _BINDINGS)
        assert "dpk_setup.sh" not in setup
        assert "venv" not in setup
        assert "venv" not in run
        # the transform still runs
        assert "--module 'dpk_tokenization2arrow.runtime'" in run


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

    def test_no_trailing_continuation_swallows_what_follows(
        self, launcher, defaults
    ):
        """Regression guard, retargeted to the new seam.

        An earlier draft emitted args as backslash-continued lines, so the final
        flag's trailing "\\" spliced the next line into the python invocation. The
        args now render on ONE line as argv to dpk_run.sh, so the equivalent risk
        is that line ending in a stray "\\" and swallowing whatever follows.

        The marker itself moved into dpk_run.sh (see test_dpk_run_sh.py, which
        asserts it is emitted as its own command).
        """
        cfg = _transform_cfg(defaults, args={"tkn_chunk_size": 0})
        run = _render(launcher["run"], cfg, _BINDINGS)
        invocation = next(
            line
            for line in run.splitlines()
            if "dpk_run.sh" in line and not line.lstrip().startswith("#")
        )
        # The invocation spans continuations by design; the LAST line of it (the
        # argv line) must not continue into anything.
        argv_line = next(
            line for line in run.splitlines() if "--artifact-id" in line
        )
        tail_idx = run.splitlines().index(argv_line)
        last = [l for l in run.splitlines()[tail_idx:] if l.strip()][-1]
        assert not last.rstrip().endswith("\\")
        assert invocation.strip().startswith("bash ./src/dpk_run.sh")
