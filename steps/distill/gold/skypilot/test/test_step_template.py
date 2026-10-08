"""Contract tests for step-template.yaml.

The launcher's ``run:`` block is a ~100-line shell script carrying runtime Jinja.
Nothing else validates it before a cluster does, and its failures are expensive:
a shell syntax error costs a queue slot, and a missing rank guard costs N
duplicate artifact registrations on a multi-node run.

These tests read the template directly (not the rendered Space), so they hold
whether or not ``make space`` has been run.
"""

import os
import re
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

_STEP = Path(__file__).resolve().parent.parent / "step-template.yaml"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check."""
    script = re.sub(r"\{%.*?%\}", "", script, flags=re.S)
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

    def test_container_venv_is_put_on_path(self, run_script):
        """The image's venv is not on PATH by default."""
        assert "export PATH=/stage/.venv/bin:$PATH" in run_script

    def test_no_login_shell_anywhere(self, run_script):
        """A login shell re-runs /etc/profile and drops the venv from PATH."""
        assert "bash -lc" not in run_script
        assert "#!/bin/bash -l" not in run_script


class TestRankHandling:
    """accelerate owns per-process rank; the provisioner's vars are node-level."""

    def test_inherited_rank_vars_are_unset_before_launch(self, run_script):
        assert "unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT" in run_script
        assert run_script.index("unset RANK") < run_script.index("accelerate launch")

    def test_topology_is_snapshotted_before_being_unset(self, run_script):
        for var in ("RANK", "TOTAL_NODES", "NUM_GPUS_PER_NODE", "MASTER_ADDR"):
            assert f"${{{var}" in run_script

    def test_master_addr_is_required_not_defaulted(self, run_script):
        """A silently-wrong master address hangs NCCL init until the timeout, so
        fail loudly instead of substituting a default."""
        assert "${MASTER_ADDR:?" in run_script

    def test_accelerate_receives_the_snapshotted_topology(self, run_script):
        for flag in (
            "--machine_rank",
            "--main_process_ip",
            "--main_process_port",
            "--num_machines",
            "--num_processes",
        ):
            assert flag in run_script


class TestRankZeroGuards:
    """The executor streams every node into one driver log.

    So anything that must happen once per RUN, rather than once per NODE, has to
    be guarded — an unguarded artifact marker registers N checkpoints.
    """

    def test_artifact_marker_is_guarded(self, run_script):
        # The FINAL marker, pinned by its trailing space: "GB_ARTIFACT_ID:checkpoint"
        # is a prefix of the watcher's "checkpoint_${step}" id, whose own guarding is
        # covered by TestPerCheckpointArtifacts (it sits in a function that only
        # rank-0-guarded code calls, so a position test would not show it).
        marker = 'echo "GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:'
        assert marker in run_script
        guard = run_script.rindex('if [ "$NODE_RANK" = "0" ]; then')
        assert guard < run_script.index(marker)

    def test_commit_metadata_is_guarded(self, run_script):
        # Anchored on the FULL key rather than the bare GB_STEP_METADATA_KEY prefix, in
        # case a future addition reuses it. The trainer's commit is now recorded by the
        # shared source-delivery region's own distill_code_commit echo (kd_sandbox_commit
        # no longer exists as a separate concept: there is only one checkout now), and
        # that region's own guard is asserted directly by
        # TestDistillSourceDelivery.test_the_region_runs_above_the_rank_split.
        marker = "GB_STEP_METADATA_KEY:distill_code_commit"
        assert marker in run_script

    def test_config_echo_is_guarded(self, run_script):
        """Printing the config N times would bury the run's real output."""
        before = run_script[: run_script.index('cat "$CFG"')]
        assert '[ "$NODE_RANK" = "0" ]' in before


class TestIdentityComesFromTheAllocation:
    def test_config_name_uses_the_runtime_node_count(self, run_script):
        """Not a build parameter: the checkpoint path must not be able to claim a
        topology the run did not have."""
        assert (
            'CONFIG_NAME="{{ config.gold_config.run_name }}_node${NODES}"' in run_script
        )

    def test_checkpoint_dir_is_under_the_build_workdir(self, run_script):
        """Per-build and publishable, unlike the launcher's hardcoded path."""
        assert (
            'CKPT_DIR="${GB_BUILD_WORKDIR:-$PWD}/checkpoints/${CONFIG_NAME}"'
            in run_script
        )


class TestOnPolicyForwardCompatibility:
    """The vLLM/trainer split is a step-level branch, needing no fork change."""

    def test_last_nodes_serve(self, run_script):
        assert "TRAINER_NODES=$(( NODES - VLLM_SERVERS ))" in run_script
        assert '[ "$NODE_RANK" -ge "$TRAINER_NODES" ]' in run_script

    def test_trainer_process_count_excludes_server_nodes(self, run_script):
        assert "--num_processes $(( TRAINER_NODES * GPUS ))" in run_script
        assert '--num_machines "$TRAINER_NODES"' in run_script

    def test_use_vllm_is_stated_in_both_directions(self, run_script):
        """Driven by the same vllm_num_servers as every other on-policy key.

        gold.py defaults --use_vllm to False and custom_gold_trainer.py branches on
        it, so a template emitting the flag only negatively leaves an on-policy run
        generating locally while its allocated server sits idle — build d77546a9 did
        exactly that. Asserted here as the unrendered expression, because this suite
        reads the template rather than a render;
        test/unit/builtins/steps/test_distill_gold.py renders both branches.
        """
        assert (
            "--use_vllm={{ 'True' if config.gold_config.vllm_num_servers "
            "| int > 0 else 'False' }}" in run_script
        )


class TestStepDeclaration:
    def test_is_a_training_step_publishing_a_model(self, step):
        assert step["type"] == "training"
        assert step["outputs"]["optional"]["checkpoint"]["type"] == "model"

    def test_restricted_to_the_lsf_subtype(self, step):
        """enroot and the LSF topology contract are LSF-specific, and the image
        is an SM90 build that cannot run on A100."""
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

    def test_no_setup_phase(self, launcher):
        """The trainer comes from /proj, so there is nothing to clone or install —
        and a setup phase would be one more thing to fail per node."""
        assert "setup" not in launcher

    def test_resources_are_left_to_the_build(self, launcher):
        """One step serves the smoke and reference runs."""
        assert launcher["resources"] == {}

    def test_renderer_is_shipped_as_a_file_mount(self, launcher):
        assert launcher["file_mounts"] == {"src": "src"}

    def test_nccl_timeout_env_is_present(self, launcher):
        """The 30B teacher forward plus ZeRO-3 collectives outlast the default
        watchdog on a healthy run, so the timeout must be set explicitly. Its
        value is templated — see TestDistributedDiagnostics."""
        for key in (
            "TORCH_NCCL_TIMEOUT_MS",
            "NCCL_TIMEOUT",
            "TORCH_NCCL_ENABLE_MONITORING",
        ):
            assert key in launcher["envs"]

    def test_monitor_uses_periodic_retrieval(self, step):
        """The default on_completion surfaces nothing until a multi-hour run ends."""
        monitor = step["environment_configs"]["Skypilot"]["monitors"][
            "skypilot_monitor"
        ]
        assert monitor["ref"] == "space://monitors/skypilot"
        assert "periodic" in monitor["config"]["log_retrieval"]["mode"]

    def test_defaults_that_break_granite_are_correct(self, step):
        gold = step["config"]["gold_config"]
        assert gold["use_liger_fused_jsd"] is False
        assert gold["response_template"] == "<|im_start|>assistant"
        assert gold["lmbda"] == 0.0
        assert gold["vllm_num_servers"] == 0

    def test_model_and_data_have_no_defaults(self, step):
        """Silently distilling the wrong model is worse than failing to start."""
        gold = step["config"]["gold_config"]
        for key in ("model_name_or_path", "teacher_model_name_or_path", "dataset_name"):
            assert gold[key] == ""


class TestRendererInvocation:
    """Every gold_config field must actually reach the renderer."""

    # Every gold_config key goes to exactly one of three places, and a key that
    # reaches NONE of them is configuration that does nothing — which reads as a
    # working knob to the next person who tunes it. The partition is asserted
    # rather than described so that adding a key forces a decision about which
    # kind it is.
    #
    # 1. STEP_ONLY — consumed by the run block or the launcher env. The nccl_*
    #    knobs configure NCCL through the environment and have no place in the
    #    trainer's config file; the rest name paths the step itself resolves.
    STEP_ONLY = {
        # Same shape of key: the residency preflight runs in the run block,
        # before the renderer is even called, so `--check-weight-residency` would be a flag
        # render_gold_config.py has no reason to know about and CustomGOLDConfig would
        # reject. That these two are wired, switchable and overridable is asserted by
        # test_weight_residency_contract.py, which owns them across all three steps that
        # carry the preflight -- so exempting them here hides nothing.
        "check_weight_residency",
        "allow_offline_weights",
        "ds_config",
        "run_name",
        "nccl_debug",
        "nccl_debug_subsys",
        "nccl_timeout_ms",
        "nccl_enable_monitoring",
        "nccl_ib_hca",
    }
    # 2. TRAINER_CLI_ONLY — handed to gold.py on its command line, NOT written
    #    into the rendered config. That mirrors the one launcher that has actually
    #    run on-policy: TRL's parse_args_and_config rejects unknown top-level keys,
    #    so putting these in the config file would bet on them being accepted
    #    config-file keys rather than merely accepted CLI flags.
    TRAINER_CLI_ONLY = {
        "vllm_mode": "--vllm_mode",
        "vllm_sync_frequency": "--vllm_sync_frequency",
    }

    # 3. Everything else reaches render_gold_config.py as --kebab-case.

    def test_all_renderer_flags_are_passed(self, run_script, step):
        for key in step["config"]["gold_config"]:
            if key in self.STEP_ONLY or key in self.TRAINER_CLI_ONLY:
                continue
            flag = "--" + key.replace("_", "-")
            assert flag in run_script, f"{key} never reaches the renderer"

    def test_trainer_cli_only_keys_reach_gold_py(self, run_script, step):
        """They bypass the renderer, so nothing else would catch them going
        nowhere — and a silently dropped vllm_mode is an on-policy run that
        cannot find its server."""
        for key, flag in self.TRAINER_CLI_ONLY.items():
            assert key in step["config"]["gold_config"], f"{key} is not a config key"
            assert flag in run_script, f"{key} never reaches gold.py"
            # And they must NOT be sent to the renderer, which would emit them
            # into the config file and risk the rejection described above.
            assert (
                "--" + key.replace("_", "-") not in run_script
            ), f"{key} is also passed to the renderer"

    def test_renderer_is_run_with_the_container_interpreter(self, run_script):
        """So the config is dumped by the same PyYAML the trainer parses with."""
        assert "/stage/.venv/bin/python ./src/render_gold_config.py" in run_script

    def test_total_nodes_is_passed_from_the_allocation(self, run_script):
        assert '--total-nodes "$NODES"' in run_script


class TestDistributedDiagnostics:
    """A hang must produce an error, not silence.

    The first 2-node run reached the training loop and then stalled on step 0 for
    an hour with no output, because the step copied the reference launcher's
    TORCH_NCCL_ENABLE_MONITORING=0 — which disables the thread that aborts a
    stalled collective. The allocation was held the whole time and nothing was
    learned from it.
    """

    def test_monitoring_defaults_on(self, step):
        """Deliberately diverging from the reference launcher: an abort with a
        named collective beats an indefinite hang."""
        assert step["config"]["gold_config"]["nccl_enable_monitoring"] is True

    def test_monitoring_is_templated_not_hardcoded(self, launcher):
        env = launcher["envs"]["TORCH_NCCL_ENABLE_MONITORING"]
        assert "{{" in env and "nccl_enable_monitoring" in env
        assert '"1"' in env and '"0"' in env, "must render 1/0, not True/False"

    def test_timeout_is_templated(self, launcher):
        for key in ("TORCH_NCCL_TIMEOUT_MS", "NCCL_TIMEOUT"):
            assert "nccl_timeout_ms" in launcher["envs"][key]

    def test_nccl_debug_is_available_and_off_by_default(self, step, launcher):
        """Off by default (very verbose), but reachable without editing the step —
        it is the only way to distinguish an IB path from a silent TCP fallback."""
        assert step["config"]["gold_config"]["nccl_debug"] == ""
        assert "nccl_debug" in launcher["envs"]["NCCL_DEBUG"]

    def test_reference_timeout_default_is_preserved(self, step):
        """A healthy 30B teacher forward is slow; the production default must stay
        generous even though debug builds lower it."""
        assert step["config"]["gold_config"]["nccl_timeout_ms"] == 3600000


class TestMasterAddressIsAnIp:
    """accelerate is given an IP, matching the reference launcher.

    The provisioner exports MASTER_ADDR as a short hostname. Rendezvous works with
    either form, but NCCL's bootstrap selects its interface from this value, so
    the validated path's choice is not something to assume equivalent.
    """

    def test_master_is_resolved_before_use(self, run_script):
        assert "/etc/hosts" in run_script
        assert run_script.index("MIP=") < run_script.index("--main_process_ip")

    def test_accelerate_receives_the_resolved_address(self, run_script):
        assert '--main_process_ip "$MIP"' in run_script
        assert '--main_process_ip "$MADDR"' not in run_script

    def test_resolution_falls_back_rather_than_failing(self, run_script):
        """A missing /etc/hosts entry must not abort the run: fall through to
        getent, then to the hostname, which is what worked before."""
        assert "getent ahostsv4" in run_script
        assert '[ -z "$MIP" ] && MIP="$MADDR"' in run_script


class TestExternalVllmServer:
    """The on-policy path where the server is a separate target, reached by URL.

    NOT YET RUN ON A CLUSTER. Everything here is a contract test; the open
    question is whether the trainer's NCCL weight-sync group can span two LSF
    allocations, which no test can answer.
    """

    def test_the_url_is_a_config_key(self, step):
        gold = step["config"]["gold_config"]
        assert gold["vllm_server_url"] == ""
        assert gold["vllm_mode"] == "server"
        assert gold["vllm_sync_frequency"] == 1

    def test_all_nodes_train_when_the_server_is_external(self, run_script):
        """No node is taken away from the trainer, because the server is not in
        this allocation. Getting this wrong wastes a node silently."""
        assert 'TRAINER_NODES="$NODES"' in run_script

    def test_the_in_allocation_split_is_still_present(self, run_script):
        """The external path is additive; the reference launcher's role split must
        remain reachable when no URL is given."""
        assert "TRAINER_NODES=$(( NODES - VLLM_SERVERS ))" in run_script
        assert "gb_steps_post_training.distillation.run_vllm_serve" in run_script

    def test_the_address_reaches_gold_py(self, run_script):
        for flag in ("--vllm_server_host", "--vllm_server_port", "--vllm_mode"):
            assert flag in run_script
        assert "$VLLM_ARGS" in run_script

    def test_the_url_is_parsed_at_run_time_not_templated(self, run_script):
        """The address is only known at run time — it arrives through a mem://
        binding — so it must be split in shell, not by Jinja."""
        assert 'VLLM_URL="{{ config.gold_config.vllm_server_url }}"' in run_script
        assert "_hostport" in run_script

    @pytest.mark.parametrize(
        "url,host,port",
        [
            ("http://host-42:8001", "host-42", "8001"),
            ("http://10.0.0.7:8001", "10.0.0.7", "8001"),
            # A trailing path is what an OpenAI-compatible base URL looks like.
            ("http://host-42:8001/v1", "host-42", "8001"),
            # No port: fall back to the reference launcher's 8001 rather than
            # handing the trainer an empty port that fails at connect time.
            ("http://host-42", "host-42", "8001"),
            ("https://host-42:9000", "host-42", "9000"),
        ],
    )
    def test_url_parsing_actually_works(self, run_script, url, host, port):
        """Runs the real parsing lines rather than pattern-matching them, because
        a wrong host here is a connection error minutes into an allocated run.
        """
        lines = _as_shell(run_script).splitlines()
        start = next(i for i, l in enumerate(lines) if "_hostport=" in l)
        # The first `esac` AFTER the function, not the first in the file. The shared
        # source-delivery region carries a case/esac inside its GIT_ASKPASS heredoc and
        # sits above this function, so scanning from zero selected that one -- making
        # lines[start:end + 1] EMPTY and every parametrization below compare an empty
        # stdout against the expected host. It failed loudly here, but the same slice
        # bug in a test that asserted something weaker would have gone on passing while
        # testing nothing at all.
        end = next(i for i, l in enumerate(lines) if i > start and "esac" in l)
        snippet = "\n".join(lines[start : end + 1]).replace(
            "${VLLM_URL#*://}", "${URL#*://}"
        )

        result = subprocess.run(
            ["bash", "-c", f'URL="{url}"\n{snippet}\necho "$VLLM_HOST $VLLM_PORT"'],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == f"{host} {port}"

    def test_an_unparseable_url_fails_loudly(self, run_script):
        """Rather than launching a trainer that cannot reach anything."""
        assert "could not parse a host out of vllm_server_url" in run_script


class TestDistillSourceDelivery:
    """The shared-checkout block, unconditional here like the other six ported steps.

    distill-gold used to be the one ported step whose trainer came from a separate,
    unpinned kd_code_dir checkout rather than this block's clone. That field is gone:
    the trainer (gold.py, custom_gold_trainer.py, and friends) is now part of the same
    public source-of-truth repo every other ported step clones, so delivery is
    unconditional rather than opt-in.
    """

    def test_the_contract_block_is_present(self, step):
        """test_source_contract.py asserts it is byte-identical to the reference; this only
        asserts distill-gold has one at all, so a failure here reads as "missing" rather
        than as an obscure ValueError from that file's .index()."""
        assert "code_config" in step["config"]
        cc = step["config"]["code_config"]
        # The public source repo, not a /proj checkout -- the value itself is owned by
        # test_source_contract.py's _PINNED_REPO/_PINNED_REF, which assert it for
        # distill-gold too now that the contract glob is `*distill*`.
        assert cc["code_dir"] == ""
        assert cc["python"] == "/stage/.venv/bin/python"
        # Empty on purpose: gb-steps-distillation is public, so the clone needs no
        # credential in the container.
        assert cc["repo"] != "" and cc["token_secret"] == ""

    def test_no_separate_trainer_checkout_field_remains(self, step):
        """kd_code_dir and deliver_distill_source are gone. Their reappearance would mean
        the trainer split back into a second, unpinned source of code."""
        assert "kd_code_dir" not in step["config"]["gold_config"]
        assert "deliver_distill_source" not in step["config"]["gold_config"]

    def test_the_region_is_unconditional(self, run_script):
        """No Jinja guard around the shared block: every ported step, including this one,
        now delivers the source unconditionally."""
        begin = run_script.index("# --- distill source delivery: BEGIN")
        end = run_script.index("# --- distill source delivery: END")
        region = run_script[begin:end]
        assert "deliver_distill_source" not in region
        assert "{% if" not in region.split("\n")[0]

    def test_the_region_runs_above_the_rank_split(self, run_script):
        """Asserted because it is the reason two tests in this file had to name their
        full anchor, and because it is a property worth being explicit about: the region
        runs on EVERY node, so its three metadata echoes are emitted N times on an
        N-node run. That is the shared region's behaviour, byte-identical in all seven
        ported steps and untested in the reference step (which asserts only that the
        keys exist, not that they are guarded) -- so it is recorded here rather than
        diverged from, since guarding it in one step would break the byte-identity
        contract for the other six.
        """
        assert run_script.index(
            "# --- distill source delivery: BEGIN"
        ) < run_script.index('if [ "$NODE_RANK" = "0" ]; then')
        assert "GB_STEP_METADATA_KEY:distill_code_dirty" in run_script

    def test_pythonpath_reaches_the_renderer(self, run_script):
        """The point of delivering the source at all: the renderer is what imports it, so
        the export must precede the invocation rather than merely existing."""
        assert run_script.index('export PYTHONPATH="$CODE_DIR/src') < run_script.index(
            "/stage/.venv/bin/python ./src/render_gold_config.py"
        )

    def test_trainer_runs_as_a_module_from_the_delivered_source(self, run_script):
        """gold.py's imports are package-qualified now (gb_steps_post_training.distillation),
        so it must be launched with accelerate's -m/--module flag from $CODE_DIR, not as a
        bare script path into a separate checkout."""
        assert "accelerate launch" in run_script
        launch_idx = run_script.index("accelerate launch")
        module_idx = run_script.index(
            "-m gb_steps_post_training.distillation.gold", launch_idx
        )
        assert module_idx > launch_idx
        assert "KD_CODE_DIR" not in run_script


class TestCeAnchorFlagsReachTheRenderer:
    """The keys are useless if the run block does not pass them, and
    test_all_renderer_flags_are_passed only checks the flag STRING exists. These
    check each one is wired to its own config key rather than to a neighbour's —
    a copy-paste that sends --ce-coef the entropy threshold would pass that test."""

    @pytest.mark.parametrize(
        "flag,key",
        [
            ("--ce-coef", "ce_coef"),
            ("--log-student-entropy", "log_student_entropy"),
            ("--entropy-guard-drop-frac", "entropy_guard_drop_frac"),
            ("--entropy-guard-baseline-steps", "entropy_guard_baseline_steps"),
            ("--entropy-guard-patience", "entropy_guard_patience"),
            ("--lmbda-schedule", "lmbda_schedule"),
            ("--lmbda-init", "lmbda_init"),
            ("--min-completion-length", "min_completion_length"),
        ],
    )
    def test_each_flag_reads_its_own_key(self, run_script, flag, key):
        pattern = re.compile(
            re.escape(flag)
            + r'\s+"?\{\{\s*config\.gold_config\.'
            + re.escape(key)
            + r"\b"
        )
        assert pattern.search(run_script), f"{flag} is not wired to {key}"

    @pytest.mark.parametrize(
        "key",
        [
            "ce_coef",
            "log_student_entropy",
            "entropy_guard_drop_frac",
            "entropy_guard_baseline_steps",
            "entropy_guard_patience",
            "lmbda_schedule",
            "lmbda_init",
            "min_completion_length",
        ],
    )
    def test_each_new_flag_carries_a_default(self, run_script, key):
        """These keys postdate the granite4-gold recipes' test-data build.yamls, which
        do not set them. Without `| default(...)` the template renders an empty
        argument and the renderer fails on a build that used to work."""
        assert re.search(
            r"config\.gold_config\." + re.escape(key) + r"\s*\|\s*default\(",
            run_script,
        ), f"{key} is templated without a default"


class TestPerCheckpointArtifacts:
    """The opt-in mid-run checkpoint watcher.

    Added for recipes/granite4-350m/lsf/distill-checkpoint-eval, which needs to export
    and evaluate checkpoint N while the run is still training rather than after it, and
    modelled on the watcher openinstruct-rl already carried.

    Two contracts, and both are the kind that fail silently:

    * OFF BY DEFAULT. Every other recipe using this step binds exactly one artifact,
      `checkpoint`, emitted when the run ends. If the watcher rendered unconditionally
      those builds would start seeing artifact ids they declare no outputs for.
    * RANK 0 ONLY. The executor streams every node's stdout into one driver log, so an
      unguarded marker registers each checkpoint once per node -- two registrations of
      one URI on the 2-node reference topology.
    """

    def test_it_is_off_by_default(self, step):
        assert step["config"]["emit_checkpoint_artifacts"] is False

    def test_the_watch_interval_has_a_default_and_a_template_default(
        self, step, run_script
    ):
        """A recipe that turns emission on without naming an interval must still
        render: `| default(60)` is what stops an empty `sleep` argument."""
        assert int(step["config"]["checkpoint_watch_interval_seconds"]) > 0
        assert "config.checkpoint_watch_interval_seconds | default(" in run_script

    def test_the_whole_mechanism_is_behind_the_flag(self, run_script):
        """Both halves -- the watcher setup before the trainer and the final sweep
        after it -- must be gated, or the default render changes."""
        blocks = re.findall(
            r"\{%\s*if config\.emit_checkpoint_artifacts[^%]*%\}", run_script
        )
        assert len(blocks) == 2, blocks
        for marker in ("watch_checkpoints &", "emit_new_checkpoints", "EMITTED_DIR"):
            assert marker in run_script

    def test_nothing_is_emitted_when_the_flag_is_off(self):
        """Renders the run block as the server would, with the flag off, and checks the
        per-checkpoint id is absent while the final one survives. The _as_shell helper
        STRIPS {% %} blocks rather than evaluating them, so it cannot see this."""
        from jinja2 import Template

        step = yaml.safe_load(_STEP.read_text())
        script = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
            "run"
        ]
        # The step's OWN declared defaults, so this renders the config a recipe that
        # sets nothing would actually get.
        config = dict(step["config"])

        off = Template(script).render(config=config)
        assert "GB_ARTIFACT_ID:checkpoint_" not in off
        assert "GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:" in off
        assert "watch_checkpoints" not in off
        assert "EMITTED_DIR" not in off

        config["emit_checkpoint_artifacts"] = True
        on = Template(script).render(config=config)
        assert "GB_ARTIFACT_ID:checkpoint_${step}" in on
        # The final marker survives alongside it.
        assert "GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:" in on
        # And the interval reached the sleep rather than rendering empty.
        assert (
            f'CKPT_WATCH_INTERVAL="{config["checkpoint_watch_interval_seconds"]}"' in on
        )

    def test_emission_is_rank_guarded(self, run_script):
        """Both the watcher start and the final sweep sit inside a NODE_RANK test."""
        start = run_script.index("watch_checkpoints &")
        guard = run_script.rindex('if [ "$NODE_RANK" = "0" ]', 0, start)
        assert start - guard < 500, "watcher start is not inside a rank-0 guard"
        sweep = run_script.index("cleanup_watcher\n  emit_new_checkpoints")
        guard = run_script.rindex('if [ "$NODE_RANK" = "0" ]', 0, sweep)
        assert sweep - guard < 500, "final sweep is not inside a rank-0 guard"

    def test_it_globs_the_hf_checkpoint_naming(self, run_script):
        """HF Trainer writes checkpoint-<N>. The id it emits is checkpoint_<N>, which
        is what a recipe's outputs must be named -- keep the two in lock-step."""
        assert 'CKPT_GLOB="checkpoint-"' in run_script
        assert (
            "GB_ARTIFACT_ID:checkpoint_${step} GB_ARTIFACT_PATH:${ckpt}" in run_script
        )

    def test_a_checkpoint_is_only_emitted_once_it_is_complete(self, run_script):
        """Trainer._save_checkpoint writes save_model -> optimizer/scheduler ->
        rng_state -> trainer_state.json, so requiring the weights, the last tokenizer
        file AND trainer_state.json brackets the whole write sequence. Without
        trainer_state.json the sentinels only bracket save_model, and an export that
        has already allocated a node can open a checkpoint still being written."""
        sentinels = re.search(r'CKPT_SENTINELS="([^"]+)"', run_script)
        assert sentinels
        assert set(sentinels.group(1).split()) == {
            "model.safetensors",
            "tokenizer.json",
            "trainer_state.json",
        }

    def test_each_checkpoint_is_emitted_exactly_once(self, run_script):
        """Markers on disk, not a shell counter: emit_new_checkpoints also runs inside
        the backgrounded subshell, whose variables never reach the parent."""
        assert 'marker="$EMITTED_DIR/$base"' in run_script
        assert '[ -e "$marker" ] && continue' in run_script
        assert 'touch "$marker"' in run_script

    def test_the_watcher_cannot_outlive_the_script(self, run_script):
        """An orphaned loop keeps emitting after the trainer has gone. The trap covers
        every exit path -- success, trainer failure, signal."""
        assert "trap 'cleanup_watcher; rm -rf \"$EMITTED_DIR\"' EXIT" in run_script
        assert 'kill "$WATCH_PID"' in run_script
        assert 'wait "$WATCH_PID"' in run_script

    def test_the_watcher_starts_before_the_trainer_and_sweeps_after_it(
        self, run_script
    ):
        start = run_script.index("watch_checkpoints &")
        launch = run_script.index("accelerate launch")
        sweep = run_script.index("cleanup_watcher\n  emit_new_checkpoints")
        assert start < launch < sweep

    def test_zero_emissions_fails_the_build_loudly(self, run_script):
        """The alternative failure is silent and expensive: a recipe's checkpoint_<N>
        outputs are never produced, every export/eval target bound to them stays
        unready, and the build hangs with nothing to read. Counted from the markers,
        not from a variable the subshell incremented."""
        assert 'EMITTED_COUNT="$(find "$EMITTED_DIR"' in run_script
        assert 'if [ "$EMITTED_COUNT" -eq 0 ]; then' in run_script
        guard = run_script.index('if [ "$EMITTED_COUNT" -eq 0 ]')
        assert "exit 1" in run_script[guard : guard + 900]

    def test_the_final_single_checkpoint_artifact_is_untouched(self, run_script):
        """Unconditional, and after the sweep: it is what every single-checkpoint
        recipe binds, and it is the one line of the output contract that does not
        depend on the watcher having worked."""
        final = run_script.index("GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:")
        sweep = run_script.index("cleanup_watcher\n  emit_new_checkpoints")
        assert sweep < final


class TestResumeFromCheckpointDir:
    """Seeding this run's checkpoint dir from an earlier run's, so gold.py's own
    auto-resume finds something. GB_BUILD_WORKDIR is new per build and per retry,
    so without this a relaunch -- or a retry -- always restarts at step 0.

    The block is EXECUTED against temporary directories rather than matched as text:
    what matters is which checkpoints land, how, and when the step refuses.
    """

    _WORLD = 16  # 2 nodes x 8 GPUs, the recipe's topology

    @staticmethod
    def _render(**overrides):
        from jinja2 import Template

        step = yaml.safe_load(_STEP.read_text())
        script = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
            "run"
        ]
        config = dict(step["config"])
        config.update(overrides)
        return Template(script).render(config=config)

    @classmethod
    def _seed_block(cls, src, vllm_server_url=""):
        step = yaml.safe_load(_STEP.read_text())
        gold_config = dict(
            step["config"]["gold_config"], vllm_server_url=vllm_server_url
        )
        rendered = cls._render(
            resume_from_checkpoint_dir=str(src), gold_config=gold_config
        )
        start = rendered.index('RESUME_SRC="')
        end = rendered.index("# Rendered per node", start)
        return rendered[start:end]

    @classmethod
    def _checkpoint(cls, root, step, complete=True, shards=None):
        ckpt = root / f"checkpoint-{step}"
        (ckpt / f"global_step{step}").mkdir(parents=True)
        names = ["model.safetensors", "tokenizer.json"]
        if complete:
            names.append("trainer_state.json")
        for name in names:
            (ckpt / name).write_text(name)
        for rank in range(cls._WORLD if shards is None else shards):
            shard = f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
            (ckpt / f"global_step{step}" / shard).write_text("x")
        return ckpt

    def _run(
        self, tmp_path, src, rank="0", nodes="2", vllm_servers="0", vllm_server_url=""
    ):
        dst = tmp_path / "run" / "checkpoints" / "gold_node2"
        script = "set -eu\n" + self._seed_block(src, vllm_server_url)
        result = subprocess.run(
            ["bash", "-c", script],
            env={
                "PATH": "/usr/bin:/bin",
                "NODE_RANK": rank,
                "NODES": nodes,
                "GPUS": "8",
                "VLLM_SERVERS": vllm_servers,
                "CKPT_DIR": str(dst),
            },
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        return result, dst

    def test_it_is_off_by_default(self, step):
        assert step["config"]["resume_from_checkpoint_dir"] == ""
        assert "RESUME_SRC" not in self._render()

    def test_it_is_not_a_trainer_key(self, step):
        """A step key, not gold_config: it names a path the step resolves, and the
        renderer's flag partition would otherwise demand a --resume-... flag."""
        assert "resume_from_checkpoint_dir" not in step["config"]["gold_config"]

    def test_seeding_happens_before_the_watcher_and_the_trainer(self):
        rendered = self._render(
            resume_from_checkpoint_dir="/src", emit_checkpoint_artifacts=True
        )
        seeded = rendered.index('touch "$RESUME_SEEDED"')
        assert seeded < rendered.index("watch_checkpoints &")
        assert seeded < rendered.index("accelerate launch")

    def test_the_rendered_block_is_valid_shell(self):
        result = subprocess.run(
            ["bash", "-n"],
            input=self._render(resume_from_checkpoint_dir="/src"),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    def test_complete_checkpoints_are_hardlinked_and_incomplete_skipped(self, tmp_path):
        src = tmp_path / "src"
        for step in (500, 1000, 1500):
            self._checkpoint(src, step)
        self._checkpoint(src, 2000, complete=False)
        (src / "train_manifest.json").write_text("{}")

        result, dst = self._run(tmp_path, src)

        assert result.returncode == 0, result.stderr
        assert sorted(p.name for p in dst.iterdir() if not p.name.startswith(".")) == [
            "checkpoint-1000",
            "checkpoint-1500",
            "checkpoint-500",
        ]
        weights = "checkpoint-1500/model.safetensors"
        assert (dst / weights).stat().st_ino == (src / weights).stat().st_ino
        # Numeric, not lexical: "checkpoint-500" sorts after "checkpoint-1500".
        assert "resumes from checkpoint-1500" in result.stdout
        assert "skipping incomplete checkpoint-2000" in result.stdout
        assert (dst / ".resume-seeded").exists()
        assert "GB_STEP_METADATA_KEY:resumed_from" in result.stdout

    def test_the_source_is_left_untouched_when_the_seed_is_deleted(self, tmp_path):
        """save_total_limit rotation deletes whole checkpoint dirs from this run."""
        src = tmp_path / "src"
        self._checkpoint(src, 500)
        result, dst = self._run(tmp_path, src)
        assert result.returncode == 0, result.stderr
        subprocess.run(["rm", "-rf", str(dst / "checkpoint-500")], check=True)
        assert (src / "checkpoint-500" / "model.safetensors").read_text() == (
            "model.safetensors"
        )

    def test_a_missing_source_fails_loudly(self, tmp_path):
        result, _ = self._run(tmp_path, tmp_path / "nope")
        assert result.returncode == 1
        assert "is not a directory" in result.stderr

    def test_no_complete_checkpoint_fails_rather_than_restarting(self, tmp_path):
        src = tmp_path / "src"
        self._checkpoint(src, 500, complete=False)
        result, dst = self._run(tmp_path, src)
        assert result.returncode == 1
        assert "no complete checkpoint-*" in result.stderr
        assert not (dst / ".resume-seeded").exists()

    def test_a_different_world_size_is_refused(self, tmp_path):
        """16 ZeRO shards cannot be loaded by 8 ranks."""
        src = tmp_path / "src"
        self._checkpoint(src, 1500)
        result, dst = self._run(tmp_path, src, nodes="1")
        assert result.returncode == 1
        assert "holds 16 ZeRO optimizer shards" in result.stderr
        assert "trains on 8 ranks" in result.stderr
        assert not (dst / ".resume-seeded").exists()

    def test_an_external_server_takes_no_trainer_node(self, tmp_path):
        """vllm_num_servers 1 with vllm_server_url set is the on-policy recipes'
        topology: the server is another target's allocation, so all 2 x 8 ranks
        train and a 16-shard checkpoint resumes. Counting the server against this
        allocation refused exactly that resume, as 8 ranks (build 9c265991's
        relaunch)."""
        src = tmp_path / "src"
        self._checkpoint(src, 500)
        result, dst = self._run(
            tmp_path,
            src,
            vllm_servers="1",
            vllm_server_url="http://10.0.0.1:8001",
        )
        assert result.returncode == 0, result.stderr
        assert "resumes from checkpoint-500" in result.stdout
        assert (dst / ".resume-seeded").exists()

    def test_an_in_allocation_server_node_is_not_a_trainer_rank(self, tmp_path):
        """Without a URL, the last vllm_num_servers nodes serve, so 3 nodes with 1
        server train on 16 ranks and 2 nodes with 1 server on only 8."""
        src = tmp_path / "src"
        self._checkpoint(src, 500)
        result, _ = self._run(tmp_path, src, nodes="3", vllm_servers="1")
        assert result.returncode == 0, result.stderr

        result, _ = self._run(tmp_path / "split", src, nodes="2", vllm_servers="1")
        assert result.returncode == 1
        assert "trains on 8 ranks" in result.stderr

    def test_a_worker_waits_for_rank_zero_and_does_not_seed(self, tmp_path):
        src = tmp_path / "src"
        self._checkpoint(src, 500)
        dst = tmp_path / "run" / "checkpoints" / "gold_node2"
        dst.mkdir(parents=True)
        (dst / ".resume-seeded").touch()
        result, _ = self._run(tmp_path, src, rank="1")
        assert result.returncode == 0, result.stderr
        assert "resume seed present" in result.stdout
        assert not (dst / "checkpoint-500").exists()


class TestNcclIbHcaOverride:
    """An optional per-run NCCL_IB_HCA, for dropping a flaky IB rail."""

    @staticmethod
    def _render(value):
        from jinja2 import Template

        step = yaml.safe_load(_STEP.read_text())
        script = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
            "run"
        ]
        config = dict(step["config"])
        config["gold_config"] = dict(config["gold_config"], nccl_ib_hca=value)
        return Template(script).render(config=config)

    def test_empty_leaves_the_tuning_file_in_charge(self, step, launcher):
        assert step["config"]["gold_config"]["nccl_ib_hca"] == ""
        assert "NCCL_IB_HCA" not in self._render("")
        # Never a launcher env: an empty value there would enable every HCA.
        assert "NCCL_IB_HCA" not in launcher["envs"]

    def test_set_is_exported_before_the_trainer(self):
        rendered = self._render("^=mlx5_1,mlx5_6,mlx5_8")
        export = rendered.index('export NCCL_IB_HCA="^=mlx5_1,mlx5_6,mlx5_8"')
        assert export < rendered.index("accelerate launch")


class TestResumeEmitSeeded:
    """resume_emit_seeded=false: seeded rungs are pre-marked, so only checkpoints
    this run writes are announced. Executed, not matched."""

    @staticmethod
    def _render(**overrides):
        from jinja2 import Template

        step = yaml.safe_load(_STEP.read_text())
        script = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
            "run"
        ]
        config = dict(step["config"], emit_checkpoint_artifacts=True)
        config.update(overrides)
        return Template(script).render(config=config)

    def _emitted(self, tmp_path, emit_seeded):
        rendered = self._render(
            resume_from_checkpoint_dir="/src", resume_emit_seeded=emit_seeded
        )
        # The watcher setup through the function definitions, then one sweep after
        # a checkpoint the "trainer" wrote.
        start = rendered.index('CKPT_GLOB="checkpoint-"')
        end = rendered.index("watch_checkpoints() {")
        sentinels = ("model.safetensors", "tokenizer.json", "trainer_state.json")
        ckpt_dir = tmp_path / "ckpts"
        for step in (500, 1500):
            d = ckpt_dir / f"checkpoint-{step}"
            d.mkdir(parents=True)
            for name in sentinels:
                (d / name).write_text("x")
        new = ckpt_dir / "checkpoint-2000"
        script = (
            "set -eu\n"
            + rendered[start:end]
            + f"mkdir -p {new}\n"
            + "".join(f"touch {new}/{n}\n" for n in sentinels)
            + "emit_new_checkpoints\n"
        )
        result = subprocess.run(
            ["bash", "-c", script],
            env={"PATH": "/usr/bin:/bin", "CKPT_DIR": str(ckpt_dir)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return re.findall(r"GB_ARTIFACT_ID:(checkpoint_\d+)", result.stdout)

    def test_default_re_emits_the_seeded_rungs(self, step, tmp_path):
        assert step["config"]["resume_emit_seeded"] is True
        assert sorted(self._emitted(tmp_path, True)) == [
            "checkpoint_1500",
            "checkpoint_2000",
            "checkpoint_500",
        ]

    def test_off_emits_only_what_this_run_wrote(self, tmp_path):
        assert self._emitted(tmp_path, False) == ["checkpoint_2000"]

    def test_off_renders_nothing_without_a_resume_source(self):
        assert "not re-emitting seeded" not in self._render(resume_emit_seeded=False)


class TestCloneHandshake:
    """Ranks other than 0 wait for rank 0's clone, and only for THIS attempt's.

    The marker used to be checked for existence alone. A relaunch into the same per-run
    workdir found the previous attempt's marker still there, so the other ranks went
    straight to training while rank 0 was deleting and re-cloning the tree.
    """

    @staticmethod
    def _wait_snippet(run_script):
        """From the marker definition through the end of the waiting branch, with the
        600s timeout cut to one 2s poll so a refusal is quick to observe."""
        anchor = run_script.index('DONE_MARKER="$CODE_DIR.gb-clone-done"')
        begin = run_script.rindex("\n", 0, anchor) + 1
        indent = run_script[begin:anchor]
        end = run_script.index(f"\n{indent}else\n", begin)
        snippet = textwrap.dedent(run_script[begin:end]) + "\nfi\n"
        return snippet.replace("-ge 600 ]", "-ge 2 ]")

    @staticmethod
    def _write_snippet(run_script):
        lines = [
            line.strip()
            for line in run_script.splitlines()
            if "$DONE_MARKER.tmp.$$" in line
        ]
        assert len(lines) == 2, lines  # the printf and the mv
        return "\n".join(lines)

    @staticmethod
    def _run(script, tmp_path, rank, launch_id):
        env = {
            "PATH": os.environ["PATH"],
            "CODE_DIR": str(tmp_path / "code"),
            "RANK": rank,
            "GB_SKYPILOT_LAUNCH_ID": launch_id,
            "SKYPILOT_INTERNAL_JOB_ID": "1",
            "LSB_JOBID": "4242",
        }
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + script],
            env=env,
            capture_output=True,
            text=True,
        )

    def test_a_previous_attempts_marker_is_not_accepted(self, run_script, tmp_path):
        (tmp_path / "code.gb-clone-done").write_text("launch-a:1:4242\n")

        result = self._run(self._wait_snippet(run_script), tmp_path, "1", "launch-b")

        assert result.returncode == 1
        assert "never appeared" in result.stderr

    def test_this_attempts_marker_is_accepted(self, run_script, tmp_path):
        wait = self._wait_snippet(run_script)
        # Rank 0's side: the same token definition, then the atomic write.
        token_lines = "\n".join(wait.splitlines()[:2])
        assert "CLONE_TOKEN=" in token_lines
        rank0 = self._run(
            token_lines + "\n" + self._write_snippet(run_script),
            tmp_path,
            "0",
            "launch-b",
        )
        assert rank0.returncode == 0, rank0.stderr
        assert (tmp_path / "code.gb-clone-done").read_text() == "launch-b:1:4242\n"

        result = self._run(wait, tmp_path, "1", "launch-b")

        assert result.returncode == 0, result.stderr
