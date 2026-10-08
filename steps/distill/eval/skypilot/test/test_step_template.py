"""Contract tests for step-template.yaml.

Scope note: the source-delivery half of this template is spliced VERBATIM from
distill-tokenizer-align, and that step's test_source_contract.py asserts every ported
step's copy is byte-identical to it. So this file asserts only what is specific to THIS
step — including the one thing that differs structurally from its siblings: the artifact
marker is printed by the SCRIPT, not by the template.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEP = _HERE / "step-template.yaml"
_RUN_EVAL = _HERE / "src" / "run-eval.sh"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["eval"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def eval_sh():
    return _RUN_EVAL.read_text()


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check.

    Block tags become a SPACE, so a ``{% if %}--flag{% else %}--no-flag{% endif %}`` pair
    does not collapse into ``--flag--no-flag``.
    """
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
            ["bash", "-n", str(_RUN_EVAL)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_no_login_shell_anywhere(self, run_script):
        assert "bash -lc" not in run_script

    def test_no_jinja_comment_sequence(self, run_script):
        """``${#VAR}`` opens a Jinja comment — comment_start_string stays the default."""
        assert "${#" not in run_script


class TestLauncher:
    def test_lsf_only(self, step):
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

    def test_runs_in_a_prebuilt_registry_image(self, launcher):
        assert launcher["image_id"].startswith("docker:")
        assert "IMAGE_REF" not in launcher["image_id"]

    def test_no_dockerfile_so_common_mk_treats_it_as_non_image(self):
        assert not (_HERE / "Dockerfile").exists()

    def test_accelerators_come_from_the_build(self, launcher):
        """This step DOES need a GPU — two models' forward passes — but the size belongs
        to the recipe: one step serves a 3B-teacher smoke run and a 30B reference run.
        """
        assert launcher["resources"] == {}

    def test_the_launcher_script_is_shipped(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_the_hub_is_reachable_because_this_step_loads_models(self, launcher):
        """NOT symmetric with the tokenizer-only ported steps, on purpose.

        granite 4.x is hybrid Mamba, so loading a MODEL makes the `kernels` package
        resolve kernels-community/causal-conv1d through the Hub API. With
        HF_HUB_OFFLINE=1 that raises OfflineModeIsEnabled *after* the load has begun —
        measured on build 15267e81. distill-gold, which loads a granite 4.x student and
        a 30B teacher on this same image, sets HF_HOME and leaves the Hub reachable.
        """
        assert "HF_HUB_OFFLINE" not in launcher["envs"]
        assert launcher["envs"]["HF_HOME"] == "/opt/hf-cache"


@pytest.fixture(scope="module")
def monitor(step):
    return step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]


class TestMonitor:
    """A teacher forward pass over 256 samples is not a seconds-long job."""

    def test_uses_the_shipped_monitor_library(self, monitor):
        assert monitor["ref"] == "space://monitors/skypilot"

    def test_log_retrieval_is_periodic_not_on_completion(self, monitor):
        """on_completion surfaces nothing until the end, so a stalled run looks identical
        to a slow one — the same reason distill-gold overrides this."""
        assert monitor["config"]["log_retrieval"]["mode"].endswith("'periodic') }}")

    def test_the_intervals_are_recipe_overridable(self, monitor):
        """A smoke run wants completion noticed promptly; a reference run does not want
        the log pulled every five minutes for hours."""
        assert (
            "config.poll_interval_seconds" in monitor["config"]["poll_interval_seconds"]
        )
        assert (
            "config.log_retrieval_interval_seconds"
            in monitor["config"]["log_retrieval"]["interval_seconds"]
        )


class TestArtifactContract:
    """The marker lives in the SCRIPT here, unlike corpus-prep and hf-export.

    run-eval.sh calls publish_artifacts() on both the success path and the
    already-measured SKIP path, because a resumed recipe whose eval says "already done"
    and then publishes nothing has broken whatever reads eval_metrics. Upstream's template
    echoed the same marker AGAIN after the script returned, which would register two
    NEWARTIFACT events for one id; the port drops that duplicate.
    """

    def test_the_only_declared_output_is_the_metrics(self, step):
        assert list(step["outputs"]["required"]) == ["eval_metrics"]

    def test_the_output_is_a_fileset_not_a_dataset(self, step):
        """It is a measurement OF a model, not data anything trains on."""
        assert step["outputs"]["required"]["eval_metrics"]["type"] == "fileset"

    def test_the_script_prints_the_marker(self, step, eval_sh):
        declared = set(step["outputs"]["required"])
        printed = set(re.findall(r"GB_ARTIFACT_ID:(\w+)", eval_sh))
        assert declared == printed, f"declared={declared} printed={printed}"

    def test_the_template_does_not_print_it_too(self, run_script):
        """Two markers for one id would register the artifact twice."""
        assert "GB_ARTIFACT_ID" not in run_script

    def test_the_marker_is_published_on_the_skip_path_as_well(self, eval_sh):
        """Counted rather than eyeballed: once in the function, once on the SKIP branch,
        once on the success path."""
        assert eval_sh.count("publish_artifacts") >= 3

    def test_the_out_dir_is_absolutised_before_the_script_runs(self, run_script):
        """The script prints the marker from its own OUT_DIR, so it must be handed an
        absolute path — a relative env: URI is rejected at config load."""
        assert 'OUT_DIR="$WORK/$OUT_DIR"' in run_script
        assert run_script.index('OUT_DIR="$WORK/$OUT_DIR"') < run_script.index(
            "--out-dir"
        )

    def test_no_legacy_marker_prefix(self, eval_sh):
        assert "LLMB_ARTIFACT_ID:" not in eval_sh


class TestFlagSurface:
    """template -> run-eval.sh drift, asserted in both directions."""

    def test_every_eval_config_key_reaches_the_script(self, step, run_script):
        for key in step["config"]["eval_config"]:
            if key == "allow_tokenizer_mismatch":
                continue  # a --flag/--no-flag pair, asserted below
            flag = "--" + key.replace("_", "-")
            if key == "output_dir":
                flag = "--out-dir"  # the script's own spelling
            assert flag in run_script, f"{key} never reaches run-eval.sh"

    def test_booleans_are_flag_pairs_not_values(self, step, run_script):
        """Jinja renders a YAML boolean with PYTHON casing, so ``--flag {{ x }}`` arrives
        as the literal string ``False``: nothing errors and the setting is silently
        inverted — here that would mean measuring across mismatched tokenizers, where the
        arithmetic succeeds and means nothing."""
        assert "allow_tokenizer_mismatch" in step["config"]["eval_config"]
        assert (
            "{% if config.eval_config.allow_tokenizer_mismatch %}"
            "--allow-tokenizer-mismatch" in run_script
        )
        assert "--no-allow-tokenizer-mismatch" in run_script
        assert "--allow-tokenizer-mismatch {{" not in run_script

    def test_script_parses_every_flag_the_template_passes(self, run_script, eval_sh):
        handled = set(re.findall(r"^\s*(--[a-z-]+)\)", eval_sh, re.M))
        body = "\n".join(
            line
            for line in _as_shell(run_script).splitlines()
            if not line.lstrip().startswith("#")
        )
        for flag in set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]*)", body)):
            if flag in ("--quiet", "--porcelain", "--all"):  # git's own flags
                continue
            assert flag in handled, f"run-eval.sh does not parse {flag}"


class TestConfigDefaults:
    def test_the_student_and_corpus_are_not_defaulted(self, step):
        cfg = step["config"]["eval_config"]
        assert cfg["student_model"] == ""
        assert cfg["corpus"] == ""

    def test_the_teacher_is_optional_and_empty(self, step):
        """With no teacher only entropy is computable, and asking for jsd without one is
        refused rather than defaulted. The emptiness is meaningful."""
        assert step["config"]["eval_config"]["teacher_model"] == ""

    def test_all_four_metrics_are_requested_by_default(self, step):
        """They are reductions over the SAME pair of logit tensors, so four cost one
        forward pass. kld and rkld together say WHICH WAY the student is wrong, which a
        single symmetric number cannot."""
        assert set(step["config"]["eval_config"]["metrics"].split(",")) == {
            "jsd",
            "kld",
            "rkld",
            "entropy",
        }

    def test_samples_are_sampled_with_a_seed_not_taken_as_a_prefix(self, step):
        """Corpora are often sorted by source or length, so a prefix measures one slice."""
        assert step["config"]["eval_config"]["seed"] == 42
        assert step["config"]["eval_config"]["max_samples"] > 0

    def test_the_incomplete_fraction_has_a_refusal_threshold(self, step):
        """A record whose answer exceeds max_length is scored on a PREFIX of that answer
        and leaves no trace in n_samples, so a clean-looking jsd can be a mean over
        half-read completions. 1.0 would disable the refusal."""
        frac = step["config"]["eval_config"]["max_incomplete_fraction"]
        assert 0 < frac < 1.0

    def test_dtype_matches_training(self, step):
        assert step["config"]["eval_config"]["dtype"] == "bfloat16"

    def test_tokenizer_mismatch_is_refused_by_default(self, step):
        """Index i denotes a different token to each model, so the arithmetic succeeds and
        measures nothing."""
        assert step["config"]["eval_config"]["allow_tokenizer_mismatch"] is False

    def test_step_type_is_a_real_enum_member(self, step):
        from gbcommon.types.stepconfig import StepType

        assert step["type"] in {m.value for m in StepType}
