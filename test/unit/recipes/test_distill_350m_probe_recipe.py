#!/usr/bin/env python3

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

"""Unit tests for the distill-probe recipe.

This recipe is two inline shell scripts with Python heredocs in them, and the
first version of it failed on the cluster for a reason no YAML validator catches:
``command: >-`` is a FOLDED block scalar, so YAML joins its lines with SPACES.
The heredoc collapsed onto one line, Python died of a syntax error before printing
anything, and both targets reached RUNNING and then FAILED with an empty job log.
That was build ``77aef958``.

``command: |`` — a LITERAL block scalar — is the fix, and the tests below are the
guard. They do to the rendered command exactly what the cluster does: hand the
shell to ``bash -n``, and hand every heredoc body to ``compile()``. A recipe whose
inline script cannot parse should fail here, in under a second, rather than after
a queue wait.

The recipe carries no parameters.yaml: it has no ``$${...}`` markers, because
nothing about it varies per run.
"""

import pathlib
import re
import subprocess
import textwrap

import pytest
import yaml

_BUILD = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-350m"
    / "lsf"
    / "distill-probe"
    / "build.yaml"
)

_TARGETS = ["load-student", "tokenizer", "tokenizer-fit"]
_HEREDOC = re.compile(r"<<'PY'\n(.*?)\n\s*PY(?:\n|$)", re.S)


@pytest.fixture(scope="module")
def build():
    return yaml.safe_load(_BUILD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def commands(build):
    targets = build["granite.build"]["targets"]
    return {
        name: targets[name]["steps"][0]["config"]["command_config"]["command"]
        for name in targets
    }


def test_the_recipe_declares_both_probes(build):
    assert list(build["granite.build"]["targets"]) == _TARGETS


def test_it_takes_no_parameters():
    """Nothing about a probe varies per run, so there is no parameters.yaml and
    there must be no markers left behind expecting one."""
    assert "$${" not in _BUILD.read_text(encoding="utf-8")


def test_no_retries(build):
    """A probe that needed a retry has already told you something."""
    assert build["granite.build"]["retries"]["max_retries"] == 0


def test_neither_probe_declares_an_output(build):
    """The answers are decisions for a human, not artifacts for a step."""
    for target in build["granite.build"]["targets"].values():
        assert "outputs" not in target


class TestTheScriptsSurviveYaml:
    """The bug that cost build 77aef958, asserted in both directions."""

    @pytest.mark.parametrize("name", _TARGETS)
    def test_the_command_is_a_literal_not_a_folded_scalar(self, name, commands):
        """A folded scalar (>-) joins lines with spaces. The tell is a command with
        far fewer newlines than it has statements — and a heredoc that has become
        one unreadable line."""
        cmd = commands[name]
        assert cmd.count("\n") > 10, "looks folded: the heredoc has been flattened"

    @pytest.mark.parametrize("name", _TARGETS)
    def test_every_heredoc_is_opened_and_closed(self, name, commands):
        cmd = commands[name]
        assert cmd.count("<<'PY'") == len(
            _HEREDOC.findall(cmd)
        ), "a heredoc is opened but its terminator was not found on its own line"

    @pytest.mark.parametrize("name", _TARGETS)
    def test_the_shell_parses(self, name, commands):
        result = subprocess.run(
            ["bash", "-n"], input=commands[name], text=True, capture_output=True
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("name", _TARGETS)
    def test_every_embedded_python_body_compiles(self, name, commands):
        """What the cluster does, done here for free. YAML's literal block already
        strips the block indent, so the body needs only a textwrap dedent."""
        bodies = _HEREDOC.findall(commands[name])
        assert bodies, f"{name} has no Python heredoc; has the probe changed shape?"
        for i, body in enumerate(bodies):
            compile(textwrap.dedent(body), f"{name}-heredoc-{i}", "exec")

    @pytest.mark.parametrize("name", _TARGETS)
    def test_the_python_is_invoked_from_the_venv(self, name, commands):
        """/stage/.venv is not on PATH in this image; a bare `python` is rc=127."""
        assert "/stage/.venv/bin/python" in commands[name]
        assert not re.search(r"^\s*python ", commands[name], re.M)


class TestTheProbesAnswerTheirQuestions:
    def test_the_fit_probe_normalises_per_byte_not_per_token(self, commands):
        """Two tokenizations of the same text yield different token counts, so
        perplexity-per-token is not comparable between them — the coarser
        segmentation wins by construction. Bytes are the invariant, and the VERDICT
        line must be the per-byte one."""
        cmd = commands["tokenizer-fit"]
        assert "nll_per_byte" in cmd
        assert 'encode("utf-8")' in cmd
        verdict = [l for l in cmd.split("\n") if "PROBE VERDICT" in l]
        assert verdict and "nll_per_byte" in verdict[0]

    def test_the_fit_probe_weights_the_mean_loss_by_position_count(self, commands):
        """HF returns the MEAN nll over predicted positions, so summing raw losses
        across rows of different lengths silently weights short rows equally with
        long ones. The multiply by n_pred is what makes the total a real total."""
        assert "n_pred = ids.shape[1] - 1" in commands["tokenizer-fit"]
        assert "out.loss.float().item() * n_pred" in commands["tokenizer-fit"]

    def test_the_verdicts_are_greppable(self, commands):
        """Both probes are read by grepping for PROBE, so each must print a single
        summary line rather than leaving the reader to interpret a traceback."""
        for name in _TARGETS:
            assert "PROBE VERDICT" in commands[name]
            assert f"PROBE DONE {name}" in commands[name]

    def test_the_student_probe_tries_both_attention_implementations(self, commands):
        """ "Loads, but FA2 does not resolve" is a different diagnosis from "does not
        load": the first breaks only the SFT-control arm, because sft.py selects FA2
        for granitemoehybrid and distill-gold never sets attn_implementation."""
        cmd = commands["load-student"]
        assert "flash_attention_2" in cmd and "eager" in cmd
        assert "traceback.print_exc()" in cmd, "a failure must print WHY"

    def test_the_student_probe_reports_the_config_values_used_downstream(
        self, commands
    ):
        """logits_scaling gates the fused JSD kernel and tie_word_embeddings gates a
        ZeRO-3 checkpoint hazard, so read them once here rather than guessing later."""
        cmd = commands["load-student"]
        assert "logits_scaling" in cmd
        assert "tie_word_embeddings" in cmd

    def test_the_tokenizer_probe_never_writes_to_the_checkpoint(self, commands):
        """It compares a rewritten COPY. Writing tokenizer_class into the real
        checkpoint would silently change what your recorded eval row refers to."""
        cmd = commands["tokenizer"]
        assert "mktemp -d" in cmd
        assert 'cp "$SRC/$f"' in cmd
        # the rewrite targets the scratch copy, never $SRC
        rewrite = cmd.split("rewrote tokenizer_class")[0]
        assert "p.write_text" in rewrite
        assert '"pinned"' in rewrite

    def test_the_tokenizer_probe_exercises_the_split_regex(self, commands):
        """A probe set that only used plain ASCII words would report SAME whether or
        not the override fired. Contractions, a leading space, digits and a newline
        are where plain ByteLevel and Sequence[Split,ByteLevel] disagree."""
        cmd = commands["tokenizer"]
        for probe in ("It's 42 degrees.", " leading space", "don't 1234 tokenize"):
            assert probe in cmd
        assert r'"a\nb"' in cmd, "the newline probe must survive YAML as an escape"
        assert "<|start_of_role|>assistant<|end_of_role|>" in cmd
