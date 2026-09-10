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

"""Behaviour tests for the bundled ``src/dpk_guard.sh``.

These guards used to be ~130 lines of Jinja duplicated across the ``setup`` and ``run``
blocks, which meant they could only be *rendered and pattern matched*, and needed their
own test just to police the two copies for drift. Moving them into a script makes them
executable, ``shellcheck``-able and testable directly — which is the same reason the
rest of this step's shell lives in ``src/*.sh``.

Each test runs the real script. What matters about a guard is the pair (does it refuse
the configs it should, does it stay silent on the ones it should), so both directions
are asserted throughout: a guard that refuses everything is as useless as one that
refuses nothing.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1 only) and
is not copied by ``make publish-step``.
"""

import pathlib
import shutil
import subprocess

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _STEP_DIR / "src" / "dpk_guard.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)

# A valid invocation. Tests override one field at a time, so any refusal is
# attributable to that field rather than to the fixture drifting.
_OK = {
    "transform": "tokenization2arrow",
    "module": "",
    "dpk_image": "",
    "input": "docs",
    "output": "tokens",
}


def _run(declared=("docs",), **over):
    """Run dpk_guard.sh with _OK overridden, returning (rc, dpk: lines)."""
    cfg = dict(_OK, **over)
    argv = [
        "--transform",
        cfg["transform"],
        "--module",
        cfg["module"],
        "--dpk-image",
        cfg["dpk_image"],
        "--input",
        cfg["input"],
        "--output",
        cfg["output"],
        "--",
        *declared,
    ]
    proc = subprocess.run(
        ["bash", str(_SCRIPT), *argv],
        capture_output=True,
        text=True,
        # A clean env: the script's verdict must not depend on the caller's shell.
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    return proc.returncode, [
        l for l in proc.stderr.splitlines() if l.startswith("dpk:")
    ]


class TestScriptIsValidShell:
    def test_parses_under_bash(self):
        assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0

    @pytest.mark.skipif(
        shutil.which("shellcheck") is None, reason="shellcheck not installed"
    )
    def test_shellcheck_is_clean(self):
        """Possible at all only because these guards are a file, not YAML."""
        proc = subprocess.run(
            ["shellcheck", str(_SCRIPT)], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stdout


class TestValidConfigPasses:
    """The other half of every guard: silence on a config that is fine."""

    def test_transform_only(self):
        assert _run() == (0, [])

    def test_transform_with_an_explicit_module(self):
        assert _run(module="dpk_custom.runtime") == (0, [])

    def test_image_and_module_without_a_transform(self):
        """The one legitimate exemption — see TestTransformExemption."""
        rc, msgs = _run(transform="", dpk_image="quay.io/o/i:1", module="dpk_x.runtime")
        assert (rc, msgs) == (0, [])

    def test_several_declared_inputs(self):
        rc, msgs = _run(declared=("docs", "extra", "third"), input="extra")
        assert (rc, msgs) == (0, [])


class TestTransformExemption:
    """`transform` supplies TWO things, so an exemption must supply both.

    It gives a module name AND a pip extra. `module` replaces only the first;
    `dpk_image` removes the need for the second but supplies no module. This condition
    was wrong in BOTH single-override directions before landing on the conjunction, and
    each wrong version shipped with a test asserting its own wrong side.
    """

    def test_no_transform_and_no_override_is_refused(self):
        rc, msgs = _run(transform="")
        assert rc == 1
        assert any("dpk_config.transform is required" in m for m in msgs)

    def test_module_alone_is_refused(self):
        """`module` gives a module but no dependencies: the venv would be empty."""
        rc, msgs = _run(transform="", module="dpk_custom.runtime")
        assert rc == 1
        assert any("dpk_config.transform is required" in m for m in msgs)

    def test_image_alone_is_refused(self):
        """`dpk_image` skips the install but leaves the module as `dpk_.runtime`.

        Verified against the real interpreter: `python -m dpk_.runtime` raises
        ModuleNotFoundError: No module named 'dpk_'.
        """
        rc, msgs = _run(transform="", dpk_image="quay.io/o/i:1")
        assert rc == 1
        assert any("dpk_config.transform is required" in m for m in msgs)

    def test_the_message_says_why_neither_override_suffices(self):
        """ "transform is required" alone invites setting an override again."""
        _, msgs = _run(transform="")
        joined = "\n".join(msgs)
        assert "neither override replaces it alone" in joined
        assert "'dpk_image' and 'module'" in joined


class TestOutputGuard:
    def test_empty_output_is_refused_by_name(self):
        rc, msgs = _run(output="")
        assert rc == 1
        assert any("dpk_config.output is required" in m for m in msgs)

    def test_a_mistyped_output_is_NOT_caught_here(self):
        """Documenting a real limit rather than pretending coverage.

        Declared OUTPUTS never reach the node, so only emptiness is checkable. A
        mistyped one emits GB_ARTIFACT_ID:<typo>, which buildrun.py logs and ignores —
        leaving a green target that registered nothing. Recorded in Known gaps.
        """
        assert _run(output="tokns") == (0, [])


class TestInputGuard:
    def test_empty_input_is_refused_by_name(self):
        rc, msgs = _run(input="")
        assert rc == 1
        assert any("dpk_config.input is required" in m for m in msgs)

    def test_a_mistyped_input_is_refused_and_the_valid_names_listed(self):
        """The point of the guard: say what is wrong AND what the choices are.

        Unguarded this rendered $GB_INPUT_dcos and died at `set -u` with
        "GB_INPUT_dcos: unbound variable" — naming bash rather than the mistake.
        """
        rc, msgs = _run(declared=("docs", "extra"), input="dcos")
        assert rc == 1
        assert any("names no declared input" in m for m in msgs)
        assert any(m.strip().endswith("docs") for m in msgs)
        assert any(m.strip().endswith("extra") for m in msgs)

    def test_a_target_with_no_declared_inputs_says_so(self):
        rc, msgs = _run(declared=())
        assert rc == 1
        assert any("declares NO inputs at all" in m for m in msgs)

    def test_the_listing_shows_the_name_the_author_wrote(self):
        """Not the sanitized $GB_INPUT_ form.

        A build sets `input: raw-docs`, so reporting "raw_docs" would send them chasing
        a name they never typed.
        """
        rc, msgs = _run(declared=("raw-docs",), input="")
        assert rc == 1
        assert any(m.strip().endswith("raw-docs") for m in msgs)

    def test_a_name_matching_only_after_sanitizing_is_still_a_typo(self):
        """`raw_docs` is not `raw-docs`: the comparison is on the raw names.

        Accepting the sanitized form would let a build name a variable rather than an
        input, and then read a path the step never staged under that name.
        """
        rc, _ = _run(declared=("raw-docs",), input="raw_docs")
        assert rc == 1

    @pytest.mark.parametrize(
        "name", ["d`touch /tmp/dpk_guard_pwn`", "d$(touch /tmp/dpk_guard_pwn)"]
    )
    def test_a_declared_name_cannot_execute(self, name, tmp_path):
        """Raw names are author text and are printed, so printf, never echo "...".

        The template's collision guard reports only SANITIZED names for this reason;
        this one has to show raw ones to be useful, so it prints them as data.
        """
        canary = pathlib.Path("/tmp/dpk_guard_pwn")
        canary.unlink(missing_ok=True)
        rc, msgs = _run(declared=(name,), input="")
        assert rc == 1
        assert not canary.exists()
        # The name still reaches the operator verbatim, which is the useful part.
        assert any(name in m for m in msgs)


class TestOptionParsing:
    def test_options_may_be_empty_but_must_be_present(self):
        """Every option is required-but-possibly-empty: emptiness is the thing checked.

        A missing option would silently shift the argv and could make a bad config look
        valid, so the wiring passes all five unconditionally.
        """
        proc = subprocess.run(
            ["bash", str(_SCRIPT), "--transform", "t", "--", "docs"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        )
        # output is empty because it was never passed -> refused by the output guard.
        assert proc.returncode == 1
        assert "dpk_config.output is required" in proc.stderr

    def test_the_separator_is_what_ends_option_parsing(self):
        """Declared names come after `--`, so a name that looks like an option is safe."""
        rc, msgs = _run(declared=("--transform",), input="--transform")
        assert (rc, msgs) == (0, [])
