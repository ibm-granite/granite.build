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
    block (the venv activation, the GB_INPUT_ exports) is stubbed or harmless.
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

    def test_transform_is_the_only_mode(self, defaults):
        """The step runs exactly one DPK transform, with no arbitrary-command mode.

        The transform's own pip extra is its whole dependency set, so a build never
        names packages either.
        """
        assert defaults["transform"] == ""
        assert "command" not in defaults
        assert "packages" not in defaults


class TestImageSelection:
    def test_empty_image_renders_bare_node(self, launcher, defaults):
        rendered = _render(launcher["image_id"], defaults)
        assert rendered == ""

    def test_image_renders_docker_ref(self, launcher, defaults):
        cfg = dict(defaults, dpk_image="quay.io/org/img:1.2.3")
        assert _render(launcher["image_id"], cfg) == "docker:quay.io/org/img:1.2.3"

    def test_image_id_is_deliberately_not_shell_escaped(self, launcher, defaults):
        """The one config value that must NOT get the '"'"' treatment.

        Every other author-supplied value in this step is shell-escaped, so this is
        the documented exception rather than an oversight: image_id never enters a
        shell. skypilot.py hands it to sky.Resources(image_id=...) as a python value,
        so escaping would corrupt a legitimate reference instead of protecting
        anything. Pinned so a future sweep for "unescaped interpolations" does not
        helpfully break it.
        """
        cfg = dict(defaults, dpk_image="quay.io/org/img@sha256:abc'def")
        assert _render(launcher["image_id"], cfg) == (
            "docker:quay.io/org/img@sha256:abc'def"
        )


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

    def test_ray_is_not_installed_by_default(self, launcher, defaults):
        assert defaults["ray_enabled"] is False
        argv = _script_argv(
            _render(launcher["setup"], _transform_cfg(defaults)), "setup"
        )
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[tokenization2arrow]==1.1.8"
        ]

    def test_ray_enabled_adds_the_ray_extra_alongside_the_derived_one(
        self, launcher, defaults
    ):
        """Additive: the transform's own extra carries its real dependencies."""
        cfg = _transform_cfg(defaults, transform="pii_redactor", ray_enabled=True)
        argv = _script_argv(_render(launcher["setup"], cfg), "setup")
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[pii-redactor,ray]==1.1.8"
        ]

    def test_ray_enabled_also_switches_the_module(self, launcher, defaults):
        """Ray needs the extra AND the .ray.runtime module — one flag sets both."""
        cfg = _transform_cfg(defaults, ray_enabled=True)
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--module") == "dpk_tokenization2arrow.ray.runtime"

    def test_ray_disabled_uses_the_pure_python_runtime(self, launcher, defaults):
        argv = _script_argv(
            _render(launcher["run"], _transform_cfg(defaults), _BINDINGS), "run"
        )
        assert _opt(argv, "--module") == "dpk_tokenization2arrow.runtime"

    def test_ray_enabled_also_passes_run_locally_true(self, launcher, defaults):
        """The THIRD thing Ray needs, and the one easiest to miss.

        DPK's Ray launcher defaults --run_locally to FALSE, which means
        ray.init("ray://localhost:10001") — connect to an EXISTING cluster. This step
        provisions none, so without the flag the transform waits on a cluster nobody
        started. Local is the only mode that can work here.
        """
        cfg = _transform_cfg(defaults, ray_enabled=True)
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--run_locally", "true"]

    def test_run_locally_is_not_passed_without_ray(self, launcher, defaults):
        """It is a Ray-launcher flag; the pure-python launcher does not accept it."""
        argv = _script_argv(
            _render(launcher["run"], _transform_cfg(defaults), _BINDINGS), "run"
        )
        assert "--run_locally" not in argv

    def test_run_locally_precedes_the_user_args(self, launcher, defaults):
        """Ordering is the contract: argparse takes the LAST occurrence."""
        cfg = _transform_cfg(defaults, ray_enabled=True, args={"tkn_chunk_size": 0})
        flags = _passthrough(
            _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        )
        assert flags == ["--run_locally", "true", "--tkn_chunk_size", "0"]

    def test_module_override_wins_over_ray_enabled(self, launcher, defaults):
        """The escape hatch for the transforms DPK has not normalised yet.

        In DPK 1.1.8, 32 of 43 data transforms expose dpk_<t>.ray.runtime; the rest
        use ray.transform (doc_quality, fdedup, lang_id, ...) or ship no Ray
        package. The step derives by rule and does not carry an exception table —
        that list is being normalised upstream, so it would go stale — so those
        builds set `module` explicitly.
        """
        cfg = _transform_cfg(
            defaults,
            transform="doc_quality",
            ray_enabled=True,
            module="dpk_doc_quality.ray.transform",
        )
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--module") == "dpk_doc_quality.ray.transform"
        # The ray EXTRA is still installed — only the module was overridden.
        setup_argv = _script_argv(_render(launcher["setup"], cfg), "setup")
        assert _passthrough(setup_argv) == [
            "data-prep-toolkit-transforms[doc-quality,ray]==1.1.8"
        ]


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
            "--a_one",
            "1",
            "--b_two",
            "2",
            "--c_three",
            "3",
        ]

    def test_zero_is_passed_not_dropped(self, launcher, defaults):
        """tkn_chunk_size: 0 is meaningful — falsy values must survive."""
        cfg = _transform_cfg(defaults, args={"tkn_chunk_size": 0})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--tkn_chunk_size", "0"]

    @pytest.mark.parametrize("value,rendered", [(True, "true"), (False, "false")])
    def test_booleans_render_with_a_VALUE_not_as_a_bare_flag(
        self, launcher, defaults, value, rendered
    ):
        """DPK has no store_true flags — every boolean takes a value.

        All 33 of DPK 1.1.8's boolean arguments are declared
        `type=lambda x: bool(str2bool(x))`, so a bare `--flag` makes argparse consume
        the NEXT token as its value: it would swallow the following flag name or die
        with "expected one argument". Lowercased because that is what str2bool reads.
        """
        cfg = _transform_cfg(defaults, args={"ededup_use_snapshot": value})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--ededup_use_snapshot", rendered]

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

    def test_only_none_omits_a_flag(self, launcher, defaults):
        """`false` is a SETTING and must be sent; only `null` means "do not pass it".

        Regression guard: `false` used to be dropped alongside `null`, so a build that
        explicitly disabled a DPK boolean silently got DPK's own default instead —
        with no error and no way to tell from the rendered command.
        """
        cfg = _transform_cfg(defaults, args={"off": False, "unset": None})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--off", "false"]


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


class TestArgsIsTheOnlyFlagChannel:
    """`args` is the single way to pass transform flags, so one quoting model.

    Every value is quoted for the shell, so it reaches the transform byte-for-byte
    and nothing is word-split by accident. A value that must vary per run is a
    $${PARAM} build parameter or {{ run_metadata.* }}, both resolved before the step
    renders — so they land in the persisted config and in lineage rather than being
    decided on a node.
    """

    def test_no_raw_flag_string_channel(self, defaults):
        """A second, unquoted flag channel would mean two quoting models."""
        assert "extra_args" not in defaults

    def test_no_args_yields_no_flags(self, launcher, defaults):
        argv = _script_argv(
            _render(launcher["run"], _transform_cfg(defaults), _BINDINGS), "run"
        )
        assert _passthrough(argv) == []

    def test_every_arg_value_is_quoted_so_nothing_is_word_split(
        self, launcher, defaults
    ):
        """The property `extra_args` did NOT have: a spaced value stays one argv word.

        This is why `args` is the safe default — the build author never owns the
        quoting.
        """
        cfg = _transform_cfg(defaults, args={"tkn_tokenizer": "two words here"})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--tkn_tokenizer", "two words here"]

    def test_a_dollar_value_is_not_expanded_by_the_shell(self, launcher, defaults):
        """`args` values reach the transform literally, `$`-signs included.

        A value that must genuinely vary per run is written with build.yaml Jinja
        (resolved before this renders), not with shell expansion here.
        """
        cfg = _transform_cfg(defaults, args={"tkn_tokenizer": "$NOT_EXPANDED"})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--tkn_tokenizer", "$NOT_EXPANDED"]

    def test_a_build_yaml_jinja_value_passes_through_verbatim(self, launcher, defaults):
        """The supported route for a dynamic value.

        gbserver fills the build.yaml's Jinja before the step template renders, so
        by this point the value is already a concrete string. Simulate that: an
        already-resolved value is quoted and forwarded like any other.
        """
        cfg = _transform_cfg(defaults, args={"tkn_doc_id_column": "run-a1b2c3"})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _passthrough(argv) == ["--tkn_doc_id_column", "run-a1b2c3"]


class TestInputNamesBecomeShellIdentifiers:
    """A declared input name is an arbitrary dict key; $GB_INPUT_<name> is not.

    Input names are unvalidated (the framework's own name checks are about SQL
    safety), so `raw-docs` used to render `export GB_INPUT_raw-docs=...` — which
    bash rejects as "not a valid identifier", aborting the ENTIRE run block under
    `set -euo pipefail` before the transform starts, with an error that names bash
    rather than the input. samples/templates/local_multi_stage/build.yaml ships
    inputs named `tuning-data` and `wait-for-eval`, so this is reachable.
    """

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("raw-docs", "raw_docs"),
            ("docs.v2", "docs_v2"),
            ("a b", "a_b"),
            ("dôcs", "d_cs"),  # non-ASCII is not a shell identifier either
            ("2docs", "2docs"),  # a leading digit is fine: it is a SUFFIX
            ("UPPER_ok", "UPPER_ok"),  # already valid, must pass through untouched
        ],
    )
    def test_name_is_sanitized_in_both_places(self, launcher, defaults, name, expected):
        """The export and the --input-path reference must agree, or the transform
        reads an unset variable. One Jinja macro feeds both for that reason."""
        cfg = _transform_cfg(defaults, input=name)
        rendered = _render(launcher["run"], cfg, {name: {"binding": {"path": "/p"}}})
        assert f"export GB_INPUT_{expected}='/p'" in rendered
        assert _bash_ok(rendered)
        # The argv assertion is what proves the two agree end to end.
        assert _opt(_script_argv(rendered, "run"), "--input-path") == "/p"

    def test_colliding_names_fail_loudly_before_the_transform(self, launcher, defaults):
        """Sanitizing is many-to-one, so a collision must not resolve silently.

        `raw-docs` and `raw.docs` both map to GB_INPUT_raw_docs; the second export
        would win and the transform would read the WRONG directory while still
        exiting 0. Since `raise_error` is only on the strict Jinja environment (this
        renders with strict=False), the guard is shell that exits non-zero.
        """
        cfg = _transform_cfg(defaults, input="raw-docs")
        bindings = {
            "raw-docs": {"binding": {"path": "/a"}},
            "raw.docs": {"binding": {"path": "/b"}},
        }
        rendered = _render(launcher["run"], cfg, bindings)
        assert _bash_ok(rendered)
        assert "exit 1" in rendered
        assert "map to the same shell variable" in rendered
        # And it must abort BEFORE the transform is invoked.
        assert rendered.index("exit 1") < rendered.index("dpk_run.sh")

    def test_the_collision_message_cannot_execute_a_name(
        self, launcher, defaults, tmp_path
    ):
        """The diagnostic must not interpolate raw names into a double-quoted echo.

        An input name is author-controlled text; a backtick or $( ) inside one would
        run as a command on the node when the guard fired. So the message names only
        the SANITIZED variable, which is [A-Za-z0-9_] by construction and inert.
        """
        canary = tmp_path / "canary"
        bad = f"d`touch {canary}`"
        bindings = {
            f"{bad}-x": {"binding": {"path": "/a"}},
            f"{bad}.x": {"binding": {"path": "/b"}},
        }
        rendered = _render(
            launcher["run"], _transform_cfg(defaults, input=f"{bad}-x"), bindings
        )
        assert "exit 1" in rendered  # the guard did fire
        assert _bash_ok(rendered)
        # The raw name must not reach the shell as code. (Backticks DO appear in the
        # block's explanatory comments, so the test is execution, not their absence.)
        assert f"touch {canary}" not in rendered
        subprocess.run(["bash", "-c", rendered], capture_output=True, cwd=_TMPDIR)
        assert not canary.exists()

    def test_distinct_names_do_not_trip_the_collision_guard(self, launcher, defaults):
        """Over-correction guard: two names that merely both need sanitizing are
        fine as long as they stay distinct."""
        cfg = _transform_cfg(defaults, input="raw-docs")
        bindings = {
            "raw-docs": {"binding": {"path": "/a"}},
            "extra-docs": {"binding": {"path": "/b"}},
        }
        rendered = _render(launcher["run"], cfg, bindings)
        assert "exit 1" not in rendered
        assert "export GB_INPUT_raw_docs='/a'" in rendered
        assert "export GB_INPUT_extra_docs='/b'" in rendered


class TestEveryConfigValueIsEscaped:
    """Quoting is applied to ALL author-controlled config, not just paths.

    The review flagged the input/output PATHS. The same hazard applies to `module`,
    `output` and `transform`: each is interpolated into a single-quoted word, so a
    quote breaks out of that context and a backtick after it runs on the node. One
    Jinja macro escapes them all, rather than a per-field judgement about which
    strings are "trusted".
    """

    @pytest.mark.parametrize("field", ["transform", "module", "output"])
    def test_a_quote_and_backtick_cannot_execute(
        self, launcher, defaults, field, tmp_path
    ):
        """The test is EXECUTION, not the absence of a backtick.

        A backtick inside a correctly escaped '"'"' sequence is inert data, so
        asserting it never appears in the text would be both wrong and untestable.
        What matters is that running the block does not execute it.
        """
        canary = tmp_path / "canary"
        payload = f"x'`touch {canary}`'"
        cfg = _transform_cfg(defaults, validate=True, **{field: payload})
        rendered = _render(launcher["run"], cfg, _BINDINGS)
        assert _bash_ok(rendered), f"{field} broke the rendered shell"
        _script_argv(rendered, "run")  # executes the block with the script stubbed
        assert not canary.exists(), f"{field} executed an embedded command"

    @pytest.mark.parametrize("field", ["transform", "module", "output"])
    def test_a_plain_quote_survives_as_data(self, launcher, defaults, field):
        """Escaped, not stripped: the value must still reach the script intact."""
        cfg = _transform_cfg(defaults, validate=True, **{field: "a'b"})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        opt = {
            "transform": "--validate",
            "module": "--module",
            "output": "--artifact-id",
        }
        assert _opt(argv, opt[field]) == "a'b"

    @pytest.mark.parametrize("field", ["pip_index_url", "dpk_version"])
    def test_the_setup_block_is_escaped_too(self, launcher, defaults, field, tmp_path):
        """`setup` had the same gap, in the phase that runs FIRST — before the venv.

        These two fields cover BOTH interpolations in the block: `pip_index_url` goes
        straight into --index-url, and `dpk_version` is folded into the derived
        requirement specifier. Macros are block-scoped in Jinja, so `setup` carries
        its own copy of the escaping rule; this is what keeps the two from drifting.

        `transform` is deliberately NOT included, though it also reaches this block:
        it arrives through the SAME q(dpk_req) call as `dpk_version`, so it is the
        same code path and adds no coverage — verified by reverting that call and
        confirming the dpk_version case alone goes red. Including it also required a
        canary path with no underscore ANYWHERE, because `transform` is legitimately
        rewritten with replace("_", "-") for the pip extra and that rewrite lands
        inside the payload too, silently retargeting the `touch` so the assertion
        checks a file the command never wrote. Chasing an underscore-free path is
        what made this test flaky; `transform`'s escaping is covered by the run-block
        tests above, which have no such rewrite.
        """
        canary = tmp_path / "canary"
        cfg = _transform_cfg(defaults, **{field: f"x'`touch {canary}`'"})
        rendered = _render(launcher["setup"], cfg)
        assert _bash_ok(rendered), f"{field} broke the rendered setup shell"
        _script_argv(rendered, "setup")  # executes it with dpk_setup.sh stubbed
        assert not canary.exists(), f"setup executed a command from {field}"

    def test_setup_values_survive_as_data(self, launcher, defaults):
        """Escaped, not mangled: a quoted index URL still arrives verbatim."""
        cfg = _transform_cfg(defaults, pip_index_url="https://ex.com/a'b/simple")
        argv = _script_argv(_render(launcher["setup"], cfg), "setup")
        assert _opt(argv, "--index-url") == "https://ex.com/a'b/simple"

    def test_the_normal_requirement_specifier_is_unchanged(self, launcher, defaults):
        """Regression fence: escaping must not perturb the ordinary case.

        The `[extra]` brackets and `==` are why this is passed as real argv in the
        first place; the escaping filter must leave them untouched.
        """
        cfg = _transform_cfg(defaults, transform="pii_redactor", ray_enabled=True)
        argv = _script_argv(_render(launcher["setup"], cfg), "setup")
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[pii-redactor,ray]==1.1.8"
        ]


class TestArgsKeysMustBeFlagNames:
    """An args KEY renders as a bare `--<key>` word, unquoted — as a flag must.

    So it cannot be quote-escaped like a value: escaping would silently build a
    wrong flag name and leave DPK to reject it with a confusing argparse error. A
    real DPK flag name is always [A-Za-z0-9_], so anything else is refused by name
    before the transform runs.
    """

    @pytest.mark.parametrize(
        "key", ["k`touch /tmp/x`", "k;rm -rf /", "k$(id)", "k v", "k-dash", "k'q"]
    )
    def test_a_non_identifier_key_is_refused(self, launcher, defaults, key):
        rendered = _render(
            launcher["run"], _transform_cfg(defaults, args={key: "v"}), _BINDINGS
        )
        assert "exit 1" in rendered
        assert "not a valid DPK flag name" in rendered
        # And it must refuse BEFORE the transform is invoked.
        assert rendered.index("exit 1") < rendered.index("dpk_run.sh")
        assert _bash_ok(rendered)

    @pytest.mark.parametrize(
        "key", ["tkn_chunk_size", "runtime_num_workers", "UPPER_2", "a1"]
    )
    def test_real_flag_names_pass(self, launcher, defaults, key):
        """Over-correction guard: legitimate DPK flag names must not be refused."""
        rendered = _render(
            launcher["run"], _transform_cfg(defaults, args={key: "v"}), _BINDINGS
        )
        assert "not a valid DPK flag name" not in rendered
        assert _passthrough(_script_argv(rendered, "run")) == [f"--{key}", "v"]


class TestRayAndImageAreRefusedTogether:
    """ray_enabled + dpk_image silently delivered 2 of its 3 documented jobs.

    dpk_image skips the install entirely, so the `ray` pip extra never lands — while
    `run` still switches to .ray.runtime and passes --run_locally true. That is the
    exact half-applied subset the one-flag design exists to prevent, surfacing as an
    import error mid-run on a node. Refused at bring-up instead.
    """

    def test_the_combination_is_refused_in_setup(self, launcher, defaults):
        cfg = _transform_cfg(defaults, ray_enabled=True, dpk_image="quay.io/o/i:1")
        rendered = _render(launcher["setup"], cfg)
        assert "exit 1" in rendered
        assert "ray_enabled cannot be combined with dpk_image" in rendered
        assert _bash_ok(rendered)

    @pytest.mark.parametrize(
        "kw",
        [
            {"dpk_image": "quay.io/o/i:1"},
            {"ray_enabled": True},
            {},
        ],
    )
    def test_each_alone_is_still_allowed(self, launcher, defaults, kw):
        """Over-correction guard: only the broken COMBINATION is refused."""
        rendered = _render(launcher["setup"], _transform_cfg(defaults, **kw))
        assert "exit 1" not in rendered
        assert _bash_ok(rendered)

    def test_ray_alone_still_installs_the_ray_extra(self, launcher, defaults):
        argv = _script_argv(
            _render(launcher["setup"], _transform_cfg(defaults, ray_enabled=True)),
            "setup",
        )
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[tokenization2arrow,ray]==1.1.8"
        ]


class TestIoWiring:
    def test_each_binding_is_exported(self, launcher, defaults):
        cfg = _transform_cfg(defaults)
        bindings = {
            "docs": {"binding": {"path": "/staged/docs"}},
            "extra": {"binding": {"path": "/staged/extra"}},
        }
        run = _render(launcher["run"], cfg, bindings)
        assert "export GB_INPUT_docs='/staged/docs'" in run
        assert "export GB_INPUT_extra='/staged/extra'" in run

    @pytest.mark.parametrize(
        "path", ["/staged/o'brien", "/staged/it's/docs", "/staged/a'b'c"]
    )
    def test_a_quote_in_a_binding_path_does_not_break_the_run_block(
        self, launcher, defaults, path
    ):
        """Regression: the GB_INPUT_ export interpolated a path unescaped.

        `export GB_INPUT_docs='/staged/o'brien'` closes the quote early, which is a
        SYNTAX error — it takes down the whole run block, not just this one line, so
        the transform never runs and the failure names no cause. args already got
        this escaping; paths did not.

        Reachable: an hf:// path is hash-derived, but an env:/// path is the build
        author's verbatim URI text and EnvURI only checks that it is absolute.

        The value is asserted through the argv the script receives, so this pins
        that the path arrives INTACT rather than merely that bash accepted it.
        """
        cfg = _transform_cfg(defaults)
        bindings = {"docs": {"binding": {"path": path}}}
        argv = _script_argv(_render(launcher["run"], cfg, bindings), "run")
        assert _opt(argv, "--input-path") == path

    @pytest.mark.parametrize("out", ["/shared/o'ut", "/shared/it's/tokens"])
    def test_a_quote_in_output_path_does_not_break_the_run_block(
        self, launcher, defaults, out
    ):
        """Same hazard on output_path, which is author-supplied config directly."""
        cfg = _transform_cfg(defaults, output_path=out)
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--output-path") == out

    def test_input_is_passed_as_the_bindings_staged_path(self, launcher, defaults):
        """--input-path resolves through $GB_INPUT_<input>, not a hardcoded path.

        Assembling DPK's --data_local_config from it is dpk_run.sh's job, covered
        by test_dpk_run_sh.py.
        """
        rendered = _render(launcher["run"], _transform_cfg(defaults), _BINDINGS)
        argv = _script_argv(rendered, "run")
        assert _opt(argv, "--input-path") == "/staged/docs"

    def test_artifact_id_is_the_declared_output(self, launcher, defaults):
        rendered = _render(launcher["run"], _transform_cfg(defaults), _BINDINGS)
        assert _opt(_script_argv(rendered, "run"), "--artifact-id") == "tokens"


class TestValidateFlag:
    """`validate: true` passes --validate <transform> to dpk_run.sh.

    The step.yaml's whole job here is to forward the transform NAME; finding and
    running src/validate_<name>.py is dpk_run.sh's (see test_dpk_run_sh.py).
    """

    def test_default_is_off(self, defaults):
        assert defaults["validate"] is False

    def test_off_passes_no_validate_flag(self, launcher, defaults):
        argv = _script_argv(
            _render(launcher["run"], _transform_cfg(defaults), _BINDINGS), "run"
        )
        assert "--validate" not in argv

    def test_on_passes_the_transform_name(self, launcher, defaults):
        cfg = _transform_cfg(defaults, validate=True)
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--validate") == "tokenization2arrow"

    def test_the_name_follows_the_transform(self, launcher, defaults):
        """Forwarded verbatim, so a validator added later needs no step change."""
        cfg = _transform_cfg(defaults, transform="pii_redactor", validate=True)
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--validate") == "pii_redactor"

    def test_validate_coexists_with_args(self, launcher, defaults):
        """--validate is an option, so it must land BEFORE the `--` separator."""
        cfg = _transform_cfg(defaults, validate=True, args={"tkn_chunk_size": 0})
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert argv.index("--validate") < argv.index("--")
        assert _passthrough(argv) == ["--tkn_chunk_size", "0"]

    def test_validate_renders_valid_shell(self, launcher, defaults):
        cfg = _transform_cfg(defaults, validate=True, args={"a": 1})
        assert _bash_ok(_render(launcher["run"], cfg, _BINDINGS))


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
        argv = _script_argv(
            _render(launcher["setup"], _transform_cfg(defaults)), "setup"
        )
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[tokenization2arrow]==1.1.8"
        ]
        assert _opt(argv, "--index-url") == "https://pypi.org/simple"

    def test_image_mode_skips_venv_and_pip(self, launcher, defaults):
        """An image already provides DPK, so nothing is installed at run time."""
        cfg = _transform_cfg(defaults, dpk_image="quay.io/org/dpk:1")
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
            ({"args": {"off": False, "unset": None}}, _BINDINGS),
            ({"dpk_image": "quay.io/org/dpk:1"}, _BINDINGS),
            ({"ray_enabled": True}, _BINDINGS),
        ],
    )
    def test_transform_mode_parses(self, launcher, defaults, cfg_kwargs, bindings):
        cfg = _transform_cfg(defaults, **cfg_kwargs)
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg, bindings))

    def test_ray_enabled_parses(self, launcher, defaults):
        """The ray branch changes both blocks, so parse both."""
        cfg = _transform_cfg(defaults, ray_enabled=True, args={"tkn_chunk_size": 0})
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg, _BINDINGS))

    def test_image_mode_parses(self, launcher, defaults):
        """dpk_image skips setup's install entirely — the empty block must still parse."""
        cfg = _transform_cfg(defaults, dpk_image="quay.io/org/dpk:1.1.8")
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg, _BINDINGS))

    def test_no_trailing_continuation_swallows_what_follows(self, launcher, defaults):
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
        argv_line = next(line for line in run.splitlines() if "--artifact-id" in line)
        tail_idx = run.splitlines().index(argv_line)
        last = [l for l in run.splitlines()[tail_idx:] if l.strip()][-1]
        assert not last.rstrip().endswith("\\")
        assert invocation.strip().startswith("bash ./src/dpk_run.sh")
