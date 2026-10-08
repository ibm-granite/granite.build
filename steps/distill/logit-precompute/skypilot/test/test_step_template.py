"""Contract tests for step-template.yaml.

Scope note: the source-delivery half of this template is spliced VERBATIM from
distill-tokenizer-align, and that step's test_source_contract.py asserts every ported
step's copy is byte-identical to it. So this file asserts only what is specific to THIS
step — including the four launcher facts the distill-sft port paid an allocation
each to learn, since this script has the same shape.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEP = _HERE / "step-template.yaml"
_RUN_PC = _HERE / "src" / "run-precompute.sh"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["precompute"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def pc_sh():
    return _RUN_PC.read_text()


@pytest.fixture(scope="module")
def monitor(step):
    return step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check."""
    script = re.sub(r"\{%.*?%\}", " ", script, flags=re.S)
    return re.sub(r"\{\{.*?\}\}", "X", script, flags=re.S)


class TestRunScriptIsValidShell:
    def test_bash_accepts_the_rendered_script(self, run_script):
        result = subprocess.run(
            ["bash", "-n"],
            input=_as_shell(run_script),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_the_ported_launcher_script_is_valid_shell(self):
        result = subprocess.run(
            ["bash", "-n", str(_RUN_PC)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_no_login_shell_anywhere(self, run_script):
        assert "bash -lc" not in run_script

    def test_no_jinja_comment_sequence(self, run_script):
        assert "${#" not in run_script


class TestLauncherFactsLearnedElsewhere:
    """The four things distill-sft cost an allocation each to discover.

    run-precompute.sh has the same shape as run-sft.sh — same discovery helper, same
    unguarded CHECKOUT_ROOT, same bare `accelerate`, same kernel preflight — so these are
    asserted here rather than rediscovered.
    """

    def test_the_guarded_knobs_are_exported_not_checkout_root(self, run_script):
        assert (
            'export PRECOMPUTE_SRC="$CODE_DIR/src/gb_steps_post_training/distillation"'
            in run_script
        )
        assert (
            'export LIB_DIR="$CODE_DIR/steps/distill-logit-precompute/src/lib"'
            in run_script
        )
        assert "export CHECKOUT_ROOT=" not in run_script

    def test_the_script_still_overwrites_checkout_root(self, pc_sh):
        """If upstream ever guards it, exporting CHECKOUT_ROOT becomes the simpler fix and
        this test should be revisited rather than the workaround silently kept."""
        assert re.search(r'^CHECKOUT_ROOT="\$\(cd', pc_sh, re.M)

    def test_the_knobs_this_relies_on_are_still_guarded(self, pc_sh):
        assert 'if [[ -z "${PRECOMPUTE_SRC:-}" ]]' in pc_sh
        assert 'if [[ -z "${LIB_DIR:-}" ]]' in pc_sh

    def test_the_venv_is_on_path_for_the_bare_accelerate(self, run_script, pc_sh):
        assert "export PATH=/stage/.venv/bin:$PATH" in run_script
        assert "export ACCELERATE=" in run_script
        assert 'ACCELERATE="${ACCELERATE:-accelerate}"' in pc_sh

    def test_missing_hub_kernels_are_fetched_in_job(self, launcher):
        assert launcher["envs"]["GOLD_PREFETCH_KERNELS"] == "1"

    def test_the_hub_is_reachable_because_this_step_loads_the_teacher(self, launcher):
        assert "HF_HUB_OFFLINE" not in launcher["envs"]
        assert launcher["envs"]["HF_HOME"] == "/opt/hf-cache"


class TestResponseTemplateNewline:
    """Same transport as distill-sft: the trailing newline is data."""

    def test_the_default_carries_an_escape_not_a_real_newline(self, step):
        rt = step["config"]["precompute_config"]["response_template"]
        assert rt.endswith("\\n")
        assert "\n" not in rt

    def test_the_decode_uses_the_sentinel(self, run_script):
        assert "printf '%b.'" in run_script
        assert "%.}" in run_script

    def test_the_decode_actually_round_trips(self, step):
        """Executed, not pattern-matched. $(...) strips trailing newlines, so the naive form
        loses exactly the character this carries."""
        rt = step["config"]["precompute_config"]["response_template"]
        script = (
            f"RT_RAW='{rt}'\n"
            'GOOD="$(printf \'%b.\' "$RT_RAW")"; GOOD="${GOOD%.}"\n'
            'NAIVE="$(printf \'%b\' "$RT_RAW")"\n'
            'printf "%s|%s|" "$(($(printf %s "$GOOD" | wc -c)))" "$(($(printf %s "$NAIVE" | wc -c)))"\n'
        )
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        ).stdout
        good, naive = re.search(r"(\d+)\|(\d+)\|", out).groups()
        assert int(good) == int(naive) + 1

    def test_the_decoded_variable_is_what_is_passed(self, run_script):
        assert '--response-template "$RESPONSE_TEMPLATE"' in run_script


class TestArtifactContract:
    def test_the_only_declared_output_is_the_logits(self, step):
        assert list(step["outputs"]["required"]) == ["teacher_logits"]

    def test_it_is_a_fileset(self, step):
        """shards/ + index.jsonl + meta.json is a derived artifact read back by a trainer."""
        assert step["outputs"]["required"]["teacher_logits"]["type"] == "fileset"

    def test_the_script_prints_the_marker(self, step, pc_sh):
        declared = set(step["outputs"]["required"])
        printed = set(re.findall(r"GB_ARTIFACT_ID:(\w+)", pc_sh))
        assert declared == printed, f"declared={declared} printed={printed}"

    def test_the_template_does_not_print_it_too(self, run_script):
        """Upstream's did, which would register two NEWARTIFACT events for one id — the same
        duplication the distill-eval port dropped."""
        assert "GB_ARTIFACT_ID" not in run_script

    def test_no_legacy_marker_prefix(self, pc_sh):
        assert "LLMB_ARTIFACT_ID:" not in pc_sh

    def test_the_out_dir_is_absolutised_before_the_script_runs(self, run_script):
        assert 'OUT_DIR="$WORK/$OUT_DIR"' in run_script
        assert run_script.index('OUT_DIR="$WORK/$OUT_DIR"') < run_script.index(
            "--output-dir"
        )


class TestFlagSurface:
    BOOLEAN_KEYS = ("ignore_documents", "allow_tokenizer_mismatch")
    # Consumed by the run block's own Jinja guard and deliberately never reaching
    # run-precompute.sh: the residency preflight runs BEFORE the teacher is loaded, in the
    # template, so a `--check-weight-residency` flag on run-precompute.sh would be one
    # nothing reads -- and test_script_parses_every_flag_the_template_passes below would
    # then fail on it, correctly. Named after distill-gold's set of the same name. That both
    # keys are actually wired, switchable and overridable is asserted in distill-gold's
    # test_weight_residency_contract.py, which owns them across the three steps that carry
    # the preflight.
    STEP_ONLY = ("check_weight_residency", "allow_offline_weights")

    def test_every_config_key_reaches_the_script(self, step, run_script):
        for block in ("precompute_config", "workload"):
            for key in step["config"][block]:
                if key in self.BOOLEAN_KEYS or key in self.STEP_ONLY:
                    continue
                assert (
                    "--" + key.replace("_", "-")
                ) in run_script, f"{block}.{key} never reaches run-precompute.sh"

    def test_booleans_are_flag_pairs_not_values(self, step, run_script):
        for key in self.BOOLEAN_KEYS:
            flag = "--" + key.replace("_", "-")
            assert f"{{% if config.precompute_config.{key} %}}{flag}" in run_script
            assert f"--no-{key.replace('_', '-')}" in run_script
            assert f"{flag} {{{{" not in run_script

    @staticmethod
    def _preflight_flags():
        """Flags of the residency preflight, read out of its own argparse.

        The preflight is a different program invoked on its own line, so its flags are not
        run-precompute.sh's business. Deriving them here rather than writing a literal means
        the exemption cannot outlive the flag it excuses: drop --allow-offline from the module
        and this test starts demanding run-precompute.sh parse it again.
        """
        module = _HERE / "src" / "check_weight_residency.py"
        if not module.exists():
            return set()
        return set(
            re.findall(
                r'ap\.add_argument\(\s*"(--[a-z][a-z0-9-]*)"', module.read_text()
            )
        )

    def test_script_parses_every_flag_the_template_passes(self, run_script, pc_sh):
        handled = set(re.findall(r"^\s*(--[a-z-]+)\)", pc_sh, re.M))
        elsewhere = {
            "--quiet",
            "--porcelain",
            "--all",
            "--verify-only",
        } | self._preflight_flags()
        body = "\n".join(
            line
            for line in _as_shell(run_script).splitlines()
            if not line.lstrip().startswith("#")
        )
        for flag in set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]*)", body)):
            if flag in elsewhere:
                continue
            assert flag in handled, f"run-precompute.sh does not parse {flag}"


class TestConfigDefaults:
    def test_the_inputs_are_not_defaulted(self, step):
        cfg = step["config"]["precompute_config"]
        assert cfg["corpus_path"] == ""
        assert cfg["teacher_model_path"] == ""

    def test_the_teacher_tokenizer_is_a_separate_key(self, step):
        """As in the trainer: the tokenizer that defines the index's token ids need not be
        the model directory's own."""
        assert "teacher_tokenizer_path" in step["config"]["precompute_config"]

    def test_a_partial_precompute_cannot_pass_silently(self, step):
        """The output of a run that skipped most of the corpus still loads, memmaps and
        trains — the symptom is a slightly worse student found weeks later. max_skip_fraction
        is the refusal threshold that makes that loud."""
        frac = step["config"]["precompute_config"]["max_skip_fraction"]
        assert 0 < frac < 1.0

    def test_tokenizer_mismatch_is_refused_by_default(self, step):
        assert step["config"]["precompute_config"]["allow_tokenizer_mismatch"] is False

    def test_single_node_by_default(self, step):
        """Sharding is `i % num_nodes == node_id`, so a single node started under a
        num_nodes it does not have silently precomputes ONE RESIDUE CLASS of the corpus.
        """
        assert step["config"]["workload"]["nodes"] == 1

    def test_the_node_refusal_is_still_in_the_script(self, pc_sh):
        assert re.search(r"\(\(\s*NODES\s*>\s*1\s*\)\)|NODES\s*-gt\s*1", pc_sh)

    def test_max_length_is_at_least_the_trainers(self, step):
        """A logit file can serve a longer training budget than the one it was made for,
        never a shorter one."""
        assert step["config"]["precompute_config"]["max_length"] >= 4096

    def test_step_type_is_a_real_enum_member(self, step):
        from gbcommon.types.stepconfig import StepType

        assert step["type"] in {m.value for m in StepType}


class TestMonitor:
    def test_log_retrieval_is_periodic(self, monitor):
        """One forward pass of a 30B teacher over a real corpus is an hours-long job."""
        assert monitor["config"]["log_retrieval"]["mode"].endswith("'periodic') }}")
