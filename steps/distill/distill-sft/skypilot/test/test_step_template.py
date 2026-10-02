"""Contract tests for step-template.yaml.

Scope note: the source-delivery half of this template is spliced VERBATIM from
distill-tokenizer-align, and that step's test_source_contract.py asserts every ported
step's copy is byte-identical to it. So this file asserts only what is specific to THIS
step — most importantly the response template's trailing newline, which is data and which
cannot cross gbserver's config fill as a real character.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent.parent
_STEP = _HERE / "step-template.yaml"
_RUN_SFT = _HERE / "src" / "run-sft.sh"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["train"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def sft_sh():
    return _RUN_SFT.read_text()


@pytest.fixture(scope="module")
def monitor(step):
    return step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check.

    Block tags become a SPACE so a ``{% if %}--flag{% else %}--no-flag{% endif %}`` pair
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
            ["bash", "-n", str(_RUN_SFT)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_no_login_shell_anywhere(self, run_script):
        assert "bash -lc" not in run_script

    def test_no_jinja_comment_sequence(self, run_script):
        """``${#VAR}`` opens a Jinja comment — comment_start_string stays the default."""
        assert "${#" not in run_script


class TestResponseTemplateNewline:
    """The trailing newline is DATA: it is where loss masking begins.

    gbserver fills every config string through Jinja, and its SandboxedEnvironments are
    built without keep_trailing_newline, so Jinja strips exactly one trailing newline from
    each VALUE. A real newline in the config would therefore never reach the container, and
    the run would mask a span one token off, train, and report success — measured on
    distill-gold (build d8470f14). So it travels as a two-character escape and is decoded
    inside the container.
    """

    def test_the_default_carries_an_escape_not_a_real_newline(self, step):
        rt = step["config"]["sft_config"]["response_template"]
        assert rt.endswith("\\n"), f"expected a literal backslash-n, got {rt!r}"
        assert "\n" not in rt, "a real newline here is stripped in transit"

    def test_the_run_block_decodes_the_escape(self, run_script):
        assert "printf '%b" in run_script

    def test_the_decode_defeats_command_substitution_stripping(self, run_script):
        """``$(...)`` strips trailing newlines too, so a bare ``$(printf '%b' "$V")`` loses
        exactly the character this whole mechanism exists to carry. The sentinel is what
        makes it survive."""
        assert "printf '%b.'" in run_script
        assert '%.}"' in run_script or "%.}" in run_script

    def test_the_decode_actually_round_trips(self, step, run_script):
        """Executed, not pattern-matched: the point is the byte that arrives.

        Runs the template's own two decode lines against the template's own default and
        asserts the result ends with a real newline — and, for contrast, that the naive
        form without the sentinel does not.
        """
        rt = step["config"]["sft_config"]["response_template"]
        script = (
            f"RT_RAW='{rt}'\n"
            'GOOD="$(printf \'%b.\' "$RT_RAW")"; GOOD="${GOOD%.}"\n'
            'NAIVE="$(printf \'%b\' "$RT_RAW")"\n'
            'printf "good=%s|naive=%s|" "$(($(printf %s "$GOOD" | wc -c)))" '
            '"$(($(printf %s "$NAIVE" | wc -c)))"\n'
        )
        out = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        ).stdout
        good, naive = re.search(r"good=(\d+)\|naive=(\d+)\|", out).groups()
        assert int(good) == int(naive) + 1, (
            "the sentinel form must preserve exactly one more byte (the newline) "
            f"than the naive form; got good={good} naive={naive}"
        )

    def test_the_template_is_passed_as_the_decoded_variable(self, run_script):
        """Not re-rendered from config at the call site, which would undo the decode."""
        assert '--response-template "$RESPONSE_TEMPLATE"' in run_script


class TestLauncher:
    def test_lsf_only(self, step):
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

    def test_runs_in_a_prebuilt_registry_image(self, launcher):
        assert launcher["image_id"].startswith("docker:")
        assert "IMAGE_REF" not in launcher["image_id"]

    def test_no_dockerfile_so_common_mk_treats_it_as_non_image(self):
        assert not (_HERE / "Dockerfile").exists()

    def test_resources_come_from_the_build(self, launcher):
        assert launcher["resources"] == {}

    def test_the_entrypoints_are_shipped(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_both_entrypoints_are_present(self):
        """run-sft.sh renders the config with render_sft_config.py before launching."""
        assert (_HERE / "src" / "run-sft.sh").exists()
        assert (_HERE / "src" / "render_sft_config.py").exists()

    def test_the_venv_is_on_path_for_the_bare_accelerate_call(self, run_script, sft_sh):
        """run-sft.sh launches the trainer through a BARE `accelerate`, and this image's
        venv is not on PATH — so without this the run dies with `accelerate: command not
        found` (rc=127) after the model is resolved and the kernels are fetched, which is
        the most expensive place to discover it (build 75fa47d2).
        """
        assert "export PATH=/stage/.venv/bin:$PATH" in run_script
        assert "export ACCELERATE=" in run_script
        # The bare call is what makes both necessary; if upstream ever qualifies it, this
        # test should be revisited rather than silently kept.
        assert 'ACCELERATE="${ACCELERATE:-accelerate}"' in sft_sh

    def test_no_login_shell_would_drop_the_path_export(self, run_script):
        """A login shell re-runs /etc/profile and drops the venv from PATH."""
        assert "bash -lc" not in run_script

    def test_missing_hub_kernels_are_fetched_in_job(self, launcher):
        """run-sft.sh preflights the required kernel list and REFUSES on a cold cache,
        because those kernels are resolved at import time — so without this the failure
        lands after the allocation is up. Measured (build 0e60920a): this image caches
        mamba-ssm and causal-conv1d but not kernels-community/flash-attn2, which is the
        FA2 fallback the renderer selects for a granite student.
        """
        assert launcher["envs"]["GOLD_PREFETCH_KERNELS"] == "1"

    def test_the_hub_is_reachable_because_this_step_loads_models(self, launcher):
        """Same reason as distill-eval: granite loading resolves
        kernels-community/causal-conv1d through the Hub API, and HF_HUB_OFFLINE turns that
        into OfflineModeIsEnabled after the load has begun. distill-gold, which trains on
        this same image, sets HF_HOME and leaves the Hub reachable."""
        assert "HF_HUB_OFFLINE" not in launcher["envs"]
        assert launcher["envs"]["HF_HOME"] == "/opt/hf-cache"


class TestMonitor:
    def test_uses_the_shipped_monitor_library(self, monitor):
        assert monitor["ref"] == "space://monitors/skypilot"

    def test_log_retrieval_is_periodic(self, monitor):
        """A training run lasts far longer than on_completion surfaces anything for."""
        assert monitor["config"]["log_retrieval"]["mode"].endswith("'periodic') }}")

    def test_progress_is_reported_from_the_trainer_metrics_dict(self, monitor):
        """TRL logs a metrics dict rather than a step banner."""
        events = monitor["config"]["extra_event_configs"]
        assert any(e["line_regex"] == "'loss':" for e in events)


class TestArtifactContract:
    def test_the_only_declared_output_is_the_checkpoint(self, step):
        assert list(step["outputs"]["required"]) == ["checkpoint"]
        assert step["outputs"]["required"]["checkpoint"]["type"] == "model"

    def test_declared_and_printed_ids_match(self, step, run_script):
        declared = set(step["outputs"]["required"])
        printed = set(re.findall(r"GB_ARTIFACT_ID:(\w+)", run_script))
        assert declared == printed, f"declared={declared} printed={printed}"

    def test_the_path_is_absolutised_before_it_is_printed(self, run_script):
        assert 'OUT_DIR="$WORK/$OUT_DIR"' in run_script
        assert run_script.index('OUT_DIR="$WORK/$OUT_DIR"') < run_script.index(
            "GB_ARTIFACT_ID:checkpoint"
        )


class TestDeepspeedConfig:
    def test_the_default_is_relative_to_the_checkout(self, step):
        """gb-steps-distillation ships exactly one deepspeed config, under
        distill-gold-train's own directory — confirmed missing on a real run
        (build 30e46247) when this pointed at the pre-port configs/distillation/
        deepspeed/ path, an in-image /opt copy that does not exist in this image."""
        ds = step["config"]["sft_config"]["deepspeed_config"]
        assert not ds.startswith("/")
        assert ds.startswith("steps/distill-gold-train/configs/deepspeed/")

    def test_it_resolves_against_the_delivered_checkout(self, run_script):
        assert 'DS_CONFIG="$CODE_DIR/$DS_CONFIG"' in run_script

    def test_the_package_and_lib_are_pointed_at_the_checkout(self, run_script):
        """A non-image step has no vendor/ tree, so without this the run dies on
        `cd: .../vendor/gb_steps_post_training/distillation` after the allocation is already
        held — measured on build 739898ab, with `pkg root : <unresolved>`.
        """
        assert (
            'export SFT_SRC="$CODE_DIR/src/gb_steps_post_training/distillation"'
            in run_script
        )
        assert 'export LIB_DIR="$CODE_DIR/steps/distill-sft/src/lib"' in run_script

    def test_checkout_root_is_not_used_because_the_script_overwrites_it(
        self, run_script, sft_sh
    ):
        """Exporting CHECKOUT_ROOT was tried and does not work (build 07b9e09a).

        The script guards STEP_HOME, SFT_SRC and LIB_DIR against the environment, but
        computes CHECKOUT_ROOT unconditionally as three levels above STEP_HOME — which
        assumes the step directory sits inside the checkout. Asserted so nobody
        "simplifies" these two exports back into the one that looks more natural.
        """
        assert "export CHECKOUT_ROOT=" not in run_script
        assert re.search(r'^CHECKOUT_ROOT="\$\(cd', sft_sh, re.M), (
            "the script no longer overwrites CHECKOUT_ROOT; re-check whether exporting it "
            "is now the simpler fix"
        )

    def test_the_guarded_knobs_are_still_guarded(self, sft_sh):
        """SFT_SRC and LIB_DIR must remain `if [[ -z ... ]]`-guarded, or the exports above
        are silently discarded the way CHECKOUT_ROOT's was."""
        assert 'if [[ -z "${SFT_SRC:-}" ]]' in sft_sh
        assert 'if [[ -z "${LIB_DIR:-}" ]]' in sft_sh

    def test_a_missing_config_fails_before_the_trainer_starts(self, run_script):
        """It is an accelerate config and it is load-bearing; a missing one should not be
        discovered by accelerate after the allocation is held."""
        assert 'if [ ! -f "$DS_CONFIG" ]' in run_script


class TestFlagSurface:
    """template -> run-sft.sh drift, asserted in both directions."""

    BOOLEAN_KEYS = ("use_liger_memory_opt", "use_liger_swiglu_mlp")
    # Consumed by the run block's own Jinja guard and deliberately never reaching
    # run-sft.sh: the residency preflight runs BEFORE the trainer is launched, in the
    # template, so a `--check-weight-residency` flag on run-sft.sh would be one nothing
    # reads. Named after distill-gold's set of the same name. That both keys are actually
    # wired, switchable and overridable is asserted in distill-gold's
    # test_weight_residency_contract.py, which owns them across the three steps that carry
    # the preflight -- so the exemption cannot hide a key that goes nowhere.
    STEP_ONLY = ("check_weight_residency", "allow_offline_weights")

    def test_every_config_key_reaches_the_script(self, step, run_script):
        cfg = step["config"]
        for block in ("sft_config", "workload", "tracking"):
            for key in cfg[block]:
                if key in self.BOOLEAN_KEYS or key in self.STEP_ONLY:
                    continue
                assert (
                    "--" + key.replace("_", "-")
                ) in run_script, f"{block}.{key} never reaches run-sft.sh"

    def test_booleans_are_flag_pairs_not_values(self, step, run_script):
        """THIS STEP is where that rule was learned: jinja renders a YAML boolean with
        PYTHON casing, so ``--use-liger-memory-opt {{ ... }}`` reached the script as the
        literal string ``False``. Nothing errors, and a shell test against "true" is then
        false forever — silently, in both directions."""
        for key in self.BOOLEAN_KEYS:
            assert key in step["config"]["sft_config"]
            flag = "--" + key.replace("_", "-")
            assert f"{{% if config.sft_config.{key} %}}{flag}" in run_script
            assert f"--no-{key.replace('_', '-')}" in run_script
            assert f"{flag} {{{{" not in run_script

    @staticmethod
    def _preflight_flags():
        """Flags of the residency preflight, read out of its own argparse.

        The preflight is a different program invoked on its own line, so its flags are not
        run-sft.sh's business. Deriving them here rather than writing a literal means the
        exemption cannot outlive the flag it excuses: drop --allow-offline from the module
        and this test starts demanding run-sft.sh parse it again.
        """
        module = _HERE / "src" / "check_weight_residency.py"
        if not module.exists():
            return set()
        return set(
            re.findall(
                r'ap\.add_argument\(\s*"(--[a-z][a-z0-9-]*)"', module.read_text()
            )
        )

    def test_script_parses_every_flag_the_template_passes(self, run_script, sft_sh):
        handled = set(re.findall(r"^\s*(--[a-z-]+)\)", sft_sh, re.M))
        # git's own three, plus whatever the residency preflight parses for itself.
        elsewhere = {"--quiet", "--porcelain", "--all"} | self._preflight_flags()
        body = "\n".join(
            line
            for line in _as_shell(run_script).splitlines()
            if not line.lstrip().startswith("#")
        )
        for flag in set(re.findall(r"(?<![-\w])(--[a-z][a-z0-9-]*)", body)):
            if flag in elsewhere:
                continue
            assert flag in handled, f"run-sft.sh does not parse {flag}"


class TestConfigDefaults:
    def test_the_inputs_are_not_defaulted(self, step):
        cfg = step["config"]["sft_config"]
        assert cfg["student_model_path"] == ""
        assert cfg["corpus_path"] == ""

    def test_this_is_a_control_by_default(self, step):
        """Setting precomputed_logits_dir turns this step from the SFT control into
        forward-KL distillation against a frozen teacher — a different arm wearing this
        step's name."""
        assert step["config"]["sft_config"]["precomputed_logits_dir"] == ""

    def test_the_optimization_defaults_are_the_treatment_s(self, step):
        """A control that trains at a different batch size or learning rate than the
        treatment is not a control."""
        cfg = step["config"]["sft_config"]
        assert cfg["per_device_train_batch_size"] == 1
        assert cfg["gradient_accumulation_steps"] == 8
        assert cfg["learning_rate"] == "1e-6"
        assert cfg["seed"] == 42

    def test_the_whole_corpus_is_trained_by_default(self, step):
        """max_dataset_size must default to -1, and lowering it is not how a run is made
        small.

        sft.py compares its train manifest's kept-row count against PREP'S and raises
        ManifestDrift on any shortfall, because the only shortfall it can attribute is a row
        whose tokenization produced no prompt — i.e. prep and the trainer rendering the
        corpus differently. So a smaller max_dataset_size reads as a tokenizer mismatch:
        "off-policy, but the trainer kept 16 of prep's 64 training rows" (build 9d95bf7e).
        Shrink corpus_config.max_examples instead, so the two counts agree by construction.
        """
        assert step["config"]["sft_config"]["max_dataset_size"] == -1

    def test_resume_is_validated_rather_than_trusted(self, step):
        """The trainer resumes on the mere PRESENCE of a checkpoint directory."""
        assert step["config"]["sft_config"]["resume"] == "auto"

    def test_the_liger_options_are_off(self, step):
        for key in ("use_liger_memory_opt", "use_liger_swiglu_mlp"):
            assert step["config"]["sft_config"][key] is False

    def test_single_node_by_default(self, step):
        """run-sft.sh REFUSES nodes > 1: it gets no host list, so a multi-node accelerate
        launch would have no main_process_ip, port or machine_rank per node."""
        assert step["config"]["workload"]["nodes"] == 1

    def test_the_node_refusal_is_still_in_the_script(self, sft_sh):
        """Asserted so that raising `nodes` in a recipe cannot silently produce a run that
        trains on one node while reporting the allocation it asked for."""
        # The script writes it as an arithmetic test: `if (( NODES > 1 )); then`.
        assert re.search(r"\(\(\s*NODES\s*>\s*1\s*\)\)", sft_sh)
        assert "FATAL: nodes=" in sft_sh

    def test_tracking_is_entirely_off(self, step):
        """All five empty means off. MEASURED: clearml and wandb are both ABSENT from this
        image, so setting any of them would fail at import; tracking.py imports them lazily
        and only when a project is configured."""
        assert set(step["config"]["tracking"].values()) == {""}

    def test_step_type_is_a_real_enum_member(self, step):
        from gbcommon.types.stepconfig import StepType

        assert step["type"] in {m.value for m in StepType}
