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
    """Render one of the launcher's shell blocks the way gbserver would.

    BOTH blocks need `bindings` now, not just `run`: the required-config guards are
    duplicated into `setup` so an invalid build is refused before the install, and the
    input guard reads the declared inputs. A setup render without them refuses with
    "declares NO inputs at all" — which is correct behaviour, and was the cause when
    eleven pre-existing tests went red on that duplication.
    """
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
        input_path="/staged/docs",
        output="tokens",
        output_path="/shared/tokens",
    )
    base.update(over)
    return base


# The template no longer reads `bindings` at all: a build resolves its own input path
# and passes it as `input_path`, the byoc pattern. This is kept only because the render
# signature still accepts it, and passing it proves the template ignores it.
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
            # Both blocks now call src/dpk_guard.sh before the script under test, so
            # stub it as a no-op: it must neither refuse nor emit ARG: lines that would
            # be mistaken for the argv being measured. Its own behaviour is covered by
            # test_dpk_guard_sh.py, which executes the real script.
            "printf '#!/usr/bin/env bash\\ntrue\\n' > ./src/dpk_guard.sh",
            "chmod +x ./src/dpk_guard.sh",
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
        setup_argv = _script_argv(_render(launcher["setup"], cfg, _BINDINGS), "setup")
        assert _passthrough(setup_argv) == [
            f"data-prep-toolkit-transforms[{extra}]==1.1.8"
        ]

    def test_dpk_version_is_honored(self, launcher, defaults):
        cfg = _transform_cfg(defaults, dpk_version="1.1.7")
        assert "==1.1.7'" in _render(launcher["setup"], cfg, _BINDINGS)

    def test_module_override_wins(self, launcher, defaults):
        """The escape hatch, for a transform DPK has not kept on the rule."""
        cfg = _transform_cfg(defaults, module="dpk_doc_quality.something_else")
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--module") == "dpk_doc_quality.something_else"


class TestPurePythonIsTheOnlyRuntime:
    """The step runs DPK's pure-python runtime, always. There is no Ray mode.

    Ray was evaluated and removed: DPK 1.1.8's Ray launcher only ever calls
    ray.init("ray://localhost:10001") — a hardcoded address with no host/port
    argument — so it cannot reach a real cluster, and provisioning one per step is
    out of this step's scope. Throughput comes from the pure-python runtime's own
    multiprocessing pool instead, reached through `args`.

    Deliberately only TWO tests. Ray needed three coupled changes (the module, the
    pip extra, and --run_locally), and the bug it caused was applying a subset of
    them — so what is worth guarding is each leg, once. Asserting the module and the
    extra again here would only restate
    TestDerivations::test_module_and_extra_derive_from_transform, which already pins
    both across five transforms and therefore fails first; a third test asserting
    `"ray_enabled" not in defaults` would only catch someone re-adding the field on
    purpose. Five guards that are really two reads as more coverage than it is.
    """

    def test_nothing_renders_a_ray_module(self, launcher, defaults):
        """Leg 1, wider than the derivation test: greps the WHOLE rendered block
        across several config shapes, not just the --module argv on the default
        path."""
        for kw in ({}, {"dpk_image": "quay.io/o/i:1"}, {"validate": True}):
            rendered = _render(
                launcher["run"], _transform_cfg(defaults, **kw), _BINDINGS
            )
            assert ".ray.runtime" not in rendered

    def test_run_locally_is_never_injected(self, launcher, defaults):
        """Leg 3, and the only test that covers it.

        The template injected --run_locally independently of the module, which is
        precisely how the half-application happened: module switched, extra missing.
        A Ray-launcher flag the pure-python launcher does not accept.
        """
        argv = _script_argv(
            _render(launcher["run"], _transform_cfg(defaults), _BINDINGS), "run"
        )
        assert "--run_locally" not in argv


class TestParallelismIsATransformFlag:
    """Overriding the pool size is `args: {runtime_num_processors: N}`.

    The DEFAULT is not set here. dpk_run.sh sizes the pool from the job's CPU
    allocation at run time, because this template renders on the server while the
    pool runs on the node — see test_dpk_run_sh.py's TestPoolSizing. So the template
    injects nothing, and an `args` value simply lands after the script's own flag,
    where argparse's last-occurrence rule makes it win.
    """

    def test_the_template_injects_no_pool_flag(self, launcher, defaults):
        """The default belongs to the script, not the template: nothing here."""
        argv = _script_argv(
            _render(launcher["run"], _transform_cfg(defaults), _BINDINGS), "run"
        )
        assert not any("runtime_num_processors" in a for a in argv)

    def test_the_pool_size_passes_through_args(self, launcher, defaults):
        cfg = _transform_cfg(defaults, args={"runtime_num_processors": 8})
        flags = _passthrough(
            _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        )
        assert flags == ["--runtime_num_processors", "8"]

    def test_it_is_ordered_with_the_other_flags(self, launcher, defaults):
        """No special-casing: it renders in `args` order like any other flag."""
        cfg = _transform_cfg(
            defaults, args={"runtime_num_processors": 4, "tkn_chunk_size": 0}
        )
        flags = _passthrough(
            _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        )
        assert flags == [
            "--runtime_num_processors",
            "4",
            "--tkn_chunk_size",
            "0",
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


class TestBothBlocksCallTheGuard:
    """The template's remaining responsibility: WIRING, not the guard logic itself.

    The four config guards now live in src/dpk_guard.sh, executed directly by
    test_dpk_guard_sh.py. What only the template can get wrong is calling it — in both
    phases, with every value it needs, and before any expensive work. `setup` is the
    expensive phase: it carries the launcher's `hf download` and the whole dependency
    install, so a guard that ran only in `run` cost a full install per invalid build.

    (This replaces a class that compared the two blocks' duplicated guard TEXT for
    drift. There is one script now, called twice, so there is nothing to drift.)
    """

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_the_block_calls_the_guard(self, launcher, defaults, block):
        rendered = _render(launcher[block], _transform_cfg(defaults), _BINDINGS)
        assert "bash ./src/dpk_guard.sh" in rendered

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_the_guard_precedes_all_the_work(self, launcher, defaults, block):
        """Refusing after the install is what this arrangement exists to avoid."""
        rendered = _render(launcher[block], _transform_cfg(defaults), _BINDINGS)
        work = "dpk_setup.sh" if block == "setup" else "dpk_run.sh"
        assert rendered.index("dpk_guard.sh") < rendered.index(work)

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_every_value_the_guard_checks_is_passed(self, launcher, defaults, block):
        """A missing option would shift argv and could make a bad config look valid."""
        cfg = _transform_cfg(defaults, module="dpk_x.runtime", dpk_image="q.io/i:1")
        rendered = _render(launcher[block], cfg, _BINDINGS)
        for opt in (
            "--transform",
            "--module",
            "--dpk-image",
            "--output",
            "--input-path",
        ):
            assert opt in rendered, f"{opt} not passed in the {block} block"

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_the_resolved_input_path_is_passed(self, launcher, defaults, block):
        """The guard checks the PATH now, not a name against a list of bindings."""
        cfg = _transform_cfg(defaults, input_path="/staged/elsewhere")
        rendered = _render(launcher[block], cfg, _BINDINGS)
        assert "--input-path '/staged/elsewhere'" in rendered

    @pytest.mark.parametrize("block", ["setup", "run"])
    def test_a_quote_in_a_declared_name_cannot_break_the_call(
        self, launcher, defaults, block
    ):
        """Names are author text interpolated into the call, so q() applies here too."""
        bindings = {"o'brien": {"binding": {"path": "/a"}}}
        cfg = _transform_cfg(defaults, input="o'brien")
        assert _bash_ok(_render(launcher[block], cfg, bindings))


class TestTheStepNeverLearnsBindingNames:
    """Regression fence for the byoc switch: the template must not read `bindings`.

    It used to take `input: <name>` and resolve the name itself, which required exporting
    $GB_INPUT_<name> for every declared input and reading exactly one of them back —
    variables no bundled script, no other step, and no build ever read. That indirection
    cost a name sanitizer, a collision guard for the sanitizer being many-to-one, two
    guards validating the name against the bindings, and a `set -u` abort of the whole
    run block when a name was mistyped. All of it is deleted, so these assert it stays
    deleted rather than being reintroduced by a well-meaning "the step should resolve
    this" change.
    """

    def test_no_gb_input_variable_is_exported(self, launcher, defaults):
        for block in ("setup", "run"):
            rendered = _render(launcher[block], _transform_cfg(defaults), _BINDINGS)
            assert "GB_INPUT_" not in rendered

    def test_the_blocks_render_identically_with_no_bindings_at_all(
        self, launcher, defaults
    ):
        """The sharpest form: bindings are not an input to rendering any more."""
        cfg = _transform_cfg(defaults)
        for block in ("setup", "run"):
            assert _render(launcher[block], cfg, {}) == _render(
                launcher[block], cfg, _BINDINGS
            )

    def test_the_input_path_reaches_the_script_verbatim(self, launcher, defaults):
        cfg = _transform_cfg(defaults, input_path="/staged/some where/docs")
        argv = _script_argv(_render(launcher["run"], cfg, _BINDINGS), "run")
        assert _opt(argv, "--input-path") == "/staged/some where/docs"

    @pytest.mark.parametrize(
        "path", ["/staged/o'brien", "/staged/it's/docs", "/staged/a'b'c"]
    )
    def test_a_quote_in_the_path_survives_and_cannot_break_the_block(
        self, launcher, defaults, path
    ):
        """A path is author-controlled config text, so q() still applies to it."""
        cfg = _transform_cfg(defaults, input_path=path)
        rendered = _render(launcher["run"], cfg, _BINDINGS)
        assert _bash_ok(rendered)
        assert _opt(_script_argv(rendered, "run"), "--input-path") == path


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
        rendered = _render(launcher["setup"], cfg, _BINDINGS)
        assert _bash_ok(rendered), f"{field} broke the rendered setup shell"
        _script_argv(rendered, "setup")  # executes it with dpk_setup.sh stubbed
        assert not canary.exists(), f"setup executed a command from {field}"

    def test_setup_values_survive_as_data(self, launcher, defaults):
        """Escaped, not mangled: a quoted index URL still arrives verbatim."""
        cfg = _transform_cfg(defaults, pip_index_url="https://ex.com/a'b/simple")
        argv = _script_argv(_render(launcher["setup"], cfg, _BINDINGS), "setup")
        assert _opt(argv, "--index-url") == "https://ex.com/a'b/simple"

    def test_the_normal_requirement_specifier_is_unchanged(self, launcher, defaults):
        """Regression fence: escaping must not perturb the ordinary case.

        The `[extra]` brackets and `==` are why this is passed as real argv in the
        first place; the escaping filter must leave them untouched.
        """
        cfg = _transform_cfg(defaults, transform="pii_redactor")
        argv = _script_argv(_render(launcher["setup"], cfg, _BINDINGS), "setup")
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[pii-redactor]==1.1.8"
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


class TestIoWiring:
    """What the step passes to dpk_run.sh for input and output.

    The input half used to be indirect: exports of $GB_INPUT_<name> for every declared
    input, then one read back. A build now resolves the path itself and passes it as
    `input_path`, so there is nothing between config and argv — which is what
    TestTheStepNeverLearnsBindingNames fences.
    """

    def test_the_input_path_is_passed_straight_through(self, launcher, defaults):
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
        argv = _script_argv(_render(launcher["setup"], cfg, _BINDINGS), "setup")
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
            _render(launcher["setup"], _transform_cfg(defaults), _BINDINGS), "setup"
        )
        assert _passthrough(argv) == [
            "data-prep-toolkit-transforms[tokenization2arrow]==1.1.8"
        ]
        assert _opt(argv, "--index-url") == "https://pypi.org/simple"

    def test_image_mode_skips_venv_and_pip(self, launcher, defaults):
        """An image already provides DPK, so nothing is installed at run time."""
        cfg = _transform_cfg(defaults, dpk_image="quay.io/org/dpk:1")
        setup = _render(launcher["setup"], cfg, _BINDINGS)
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
            ({"args": {"runtime_num_processors": 8}}, _BINDINGS),
        ],
    )
    def test_transform_mode_parses(self, launcher, defaults, cfg_kwargs, bindings):
        cfg = _transform_cfg(defaults, **cfg_kwargs)
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg, bindings))

    def test_image_mode_parses(self, launcher, defaults):
        """dpk_image skips setup's install entirely — the empty block must still parse."""
        cfg = _transform_cfg(defaults, dpk_image="quay.io/org/dpk:1.1.8")
        assert _bash_ok(_render(launcher["setup"], cfg))
        assert _bash_ok(_render(launcher["run"], cfg, _BINDINGS))

    def test_no_continuation_swallows_what_follows(self, launcher, defaults):
        """Regression guard for the bug class that cost this step a cluster run.

        An earlier draft emitted args as backslash-continued lines, so a stray
        trailing "\\" spliced the following line into the invocation and swallowed the
        artifact marker.

        Asserted on the ARGV BASH BUILT, not on the text. The previous version scanned
        forward to the last non-empty line and checked only that one, so an INTERIOR
        stray continuation passed it — proved by patching one into the `--` separator
        line, which the old assertion missed and this one catches. Comparing argv is
        strictly stronger: any splice changes the words the script receives.
        """
        cfg = _transform_cfg(defaults, args={"tkn_chunk_size": 0})
        rendered = _render(launcher["run"], cfg, _BINDINGS)
        argv = _script_argv(rendered, "run")
        assert argv == [
            "--module",
            "dpk_tokenization2arrow.runtime",
            "--input-path",
            "/staged/docs",
            "--output-path",
            "/shared/tokens",
            "--artifact-id",
            "tokens",
            "--",
            "--tkn_chunk_size",
            "0",
        ]
        invocation = next(
            line
            for line in rendered.splitlines()
            if "dpk_run.sh" in line and not line.lstrip().startswith("#")
        )
        assert invocation.strip().startswith("bash ./src/dpk_run.sh")
