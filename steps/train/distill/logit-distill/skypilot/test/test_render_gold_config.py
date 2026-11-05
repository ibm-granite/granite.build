"""Tests for the gold config renderer (../src/render_gold_config.py).

Run from the step directory with `make test`.

These pin the trainer requirements that fail *silently* — a run that starts, then
either crashes hours in or trains on the wrong objective. Each of the three rules
below cost real debugging time on the ansible path before it was understood, which
is why the renderer exists as a testable script rather than a shell heredoc.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_RENDER = Path(__file__).resolve().parent.parent / "src" / "render_gold_config.py"

_REQUIRED = {
    "--model-name-or-path": "/proj/kd/student_overlays/granite-4.1-3b-base-hub",
    "--teacher-model-name-or-path": "/proj/kd/teacher_overlays/granite-4.2-30b",
    "--dataset-name": "/proj/kd/data/subsampled_0.4_shuffled_nothink.jsonl",
}


def _render(tmp_path, total_nodes=2, extra=None, expect_rc=0):
    """Invoke the renderer the way the step's run block does; return the config."""
    out = tmp_path / "gold_config.yaml"
    cmd = [
        sys.executable,
        str(_RENDER),
        "--output",
        str(out),
        "--total-nodes",
        str(total_nodes),
    ]
    for flag, value in _REQUIRED.items():
        cmd += [flag, value]
    cmd += list(extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert (
        result.returncode == expect_rc
    ), f"rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    if expect_rc != 0:
        return result
    return yaml.safe_load(out.read_text())


def _dir(tmp_path, name):
    """A made output directory, for the tests that render twice and compare.

    `_render` hardcodes the output basename, so two renders need two directories, and
    the renderer does not create the parent of its `--output` (nor should it -- in the
    step's run block that parent is GB_BUILD_WORKDIR, which always exists). Passing an
    unmade `tmp_path / "a"` therefore failed on FileNotFoundError, which reads like a
    renderer bug and is not one.
    """
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestLearningRateIsAFloat:
    """The most expensive failure the renderer prevents.

    PyYAML parses a bare ``1e-05`` as a *string*. The trainer's min_lr handling
    then raises a str/float TypeError — not at startup, but once the scheduler is
    first consulted, so a multi-node job burns its queue time before failing.
    """

    def test_learning_rate_round_trips_as_a_float(self, tmp_path):
        config = _render(tmp_path, extra=["--learning-rate", "1e-05"])
        assert isinstance(config["learning_rate"], float)

    def test_min_lr_round_trips_as_a_float(self, tmp_path):
        """Nested under the scheduler, where the trainer expects it."""
        config = _render(tmp_path, extra=["--min-lr", "1e-06"])
        assert isinstance(config["lr_scheduler_kwargs"]["min_lr"], float)

    def test_rendered_text_carries_a_decimal_exponent(self, tmp_path):
        """Matches the validated reference configs (1.00e-05), not 1e-05."""
        out = tmp_path / "gold_config.yaml"
        cmd = [
            sys.executable,
            str(_RENDER),
            "--output",
            str(out),
            "--total-nodes",
            "2",
            "--learning-rate",
            "1e-05",
        ]
        for flag, value in _REQUIRED.items():
            cmd += [flag, value]
        subprocess.run(cmd, check=True, capture_output=True)
        assert "1.0e-05" in out.read_text()

    @pytest.mark.parametrize("value", ["1e-05", "1.0e-05", "0.00001"])
    def test_every_input_form_yields_the_same_float(self, tmp_path, value):
        config = _render(tmp_path, extra=["--learning-rate", value])
        assert config["learning_rate"] == pytest.approx(1e-05)


class TestBooleansAreLowerCaseYaml:
    """A Python ``True`` in YAML is a string, not a boolean."""

    def test_gradient_checkpointing_emits_yaml_true(self, tmp_path):
        out = tmp_path / "gold_config.yaml"
        cmd = [
            sys.executable,
            str(_RENDER),
            "--output",
            str(out),
            "--total-nodes",
            "2",
            "--gradient-checkpointing",
            "true",
        ]
        for flag, value in _REQUIRED.items():
            cmd += [flag, value]
        subprocess.run(cmd, check=True, capture_output=True)
        text = out.read_text()
        assert "gradient_checkpointing: true" in text
        assert "True" not in text

    def test_parsed_back_as_a_real_boolean(self, tmp_path):
        config = _render(tmp_path, extra=["--gradient-checkpointing", "true"])
        assert config["gradient_checkpointing"] is True

    @pytest.mark.parametrize(
        "given,expected",
        [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_accepts_the_forms_a_shell_renders(self, tmp_path, given, expected):
        config = _render(tmp_path, extra=["--use-liger-fused-jsd", given])
        assert config["use_liger_fused_jsd"] is expected


class TestOnPolicyBlockGating:
    """The six online keys must appear only on the online path.

    Their presence is what the trainer reads to decide whether to expect a vLLM
    server, so leaking them into an off-policy config makes it wait for a server
    nobody started.
    """

    ONLINE_KEYS = (
        "vllm_num_servers",
        "top_p",
        "use_sampled_opd_loss",
        "last_message_only",
        "clip_alpha",
        "opd_importance_sampling",
    )

    def test_off_policy_omits_all_of_them(self, tmp_path):
        config = _render(tmp_path)
        assert not [k for k in self.ONLINE_KEYS if k in config]

    def test_on_policy_emits_all_of_them(self, tmp_path):
        config = _render(
            tmp_path, total_nodes=2, extra=["--vllm-num-servers", "1", "--lmbda", "1.0"]
        )
        assert all(k in config for k in self.ONLINE_KEYS)

    def test_off_policy_still_carries_lmbda_and_beta(self, tmp_path):
        """These are loss knobs, not online-only ones; lmbda=0 IS off-policy."""
        config = _render(tmp_path)
        assert config["lmbda"] == 0.0
        assert config["beta"] == 0.0


class TestNodeSplitValidation:
    """Reject splits that cannot train, at render time rather than on the cluster."""

    def test_rejects_servers_equal_to_node_count(self, tmp_path):
        result = _render(
            tmp_path, total_nodes=2, extra=["--vllm-num-servers", "2"], expect_rc=2
        )
        assert "must be < total nodes" in result.stderr

    def test_rejects_servers_exceeding_node_count(self, tmp_path):
        _render(tmp_path, total_nodes=2, extra=["--vllm-num-servers", "3"], expect_rc=2)

    def test_rejects_on_policy_on_a_single_node(self, tmp_path):
        """The floor is 2: one server plus one trainer."""
        result = _render(
            tmp_path, total_nodes=1, extra=["--vllm-num-servers", "1"], expect_rc=2
        )
        assert "at least 2 nodes" in result.stderr

    def test_allows_off_policy_on_a_single_node(self, tmp_path):
        config = _render(tmp_path, total_nodes=1)
        assert "vllm_num_servers" not in config

    def test_allows_one_server_and_three_trainers(self, tmp_path):
        config = _render(tmp_path, total_nodes=4, extra=["--vllm-num-servers", "1"])
        assert config["vllm_num_servers"] == 1


class TestExternalVllmServer:
    """An external server (its URL from another target's mem:// binding) is not
    part of this allocation, so the node arithmetic that protects the in-allocation
    split must not fire — and the two mistakes it cannot protect against must.
    """

    _URL = ["--vllm-server-url", "http://host-42:8001"]
    _ON = ["--vllm-num-servers", "1", "--lmbda", "1.0"]

    def test_single_node_is_allowed_with_an_external_server(self, tmp_path):
        """The whole point: one node can train on-policy when nothing local
        serves. The in-allocation path rejects exactly this."""
        config = _render(tmp_path, total_nodes=1, extra=self._URL + self._ON)
        assert config["lmbda"] == pytest.approx(1.0)

    def test_server_count_may_equal_the_node_count(self, tmp_path):
        """vllm_num_servers counts EXTERNAL servers here, so it no longer competes
        with the trainer for nodes and 1-of-1 is not the pathological case the
        in-allocation check refuses."""
        _render(tmp_path, total_nodes=1, extra=self._URL + self._ON)

    def test_on_policy_keys_are_still_emitted(self, tmp_path):
        """The server being external changes WHERE it runs, not whether the run is
        on-policy, so the trainer still needs the online block."""
        config = _render(tmp_path, total_nodes=1, extra=self._URL + self._ON)
        for key in ("vllm_num_servers", "top_p", "clip_alpha"):
            assert key in config

    def test_the_url_is_not_emitted_into_the_config(self, tmp_path):
        """It is read for validation only; the trainer is given the address on
        gold.py's command line, because TRL's parse_args_and_config rejects
        unknown top-level config keys."""
        config = _render(tmp_path, total_nodes=1, extra=self._URL + self._ON)
        assert "vllm_server_url" not in config

    def test_rejects_a_url_with_no_server_count(self, tmp_path):
        """vllm_num_servers 0 leaves `online` false, so the on-policy keys would be
        omitted and the run would train off-policy while a server sat idle in
        another allocation — a full-price run producing the wrong arm."""
        _render(
            tmp_path,
            total_nodes=1,
            extra=self._URL + ["--vllm-num-servers", "0", "--lmbda", "1.0"],
            expect_rc=2,
        )

    def test_rejects_a_url_at_lmbda_zero(self, tmp_path):
        """At lmbda 0 the student never generates, so the server would be
        allocated, held, and never asked for anything."""
        _render(
            tmp_path,
            total_nodes=1,
            extra=self._URL + ["--vllm-num-servers", "1", "--lmbda", "0.0"],
            expect_rc=2,
        )

    def test_the_in_allocation_split_is_still_validated(self, tmp_path):
        """Relaxing the checks for an external server must not relax them for the
        path that still carves nodes out of its own allocation."""
        _render(tmp_path, total_nodes=1, extra=["--vllm-num-servers", "1"], expect_rc=2)


class TestTrainerContract:
    """Values whose default matters, and which must survive rendering verbatim."""

    def test_response_template_is_preserved_exactly(self, tmp_path):
        """Required for completion masking; the nothink chat template has no
        {% generation %} tag, so this string is what locates the span."""
        config = _render(tmp_path)
        assert config["response_template"] == "<|im_start|>assistant"

    def test_fused_jsd_defaults_off(self, tmp_path):
        """granite's logits_scaling=10 overflows the fused bf16 kernel -> NaN."""
        config = _render(tmp_path)
        assert config["use_liger_fused_jsd"] is False

    def test_lmbda_defaults_to_off_policy(self, tmp_path):
        config = _render(tmp_path)
        assert config["lmbda"] == 0.0

    def test_reference_hyperparameters_are_the_defaults(self, tmp_path):
        config = _render(tmp_path)
        assert config["max_length"] == 16384
        assert config["max_completion_length"] == 4096
        assert config["per_device_train_batch_size"] == 1
        assert config["save_total_limit"] == 20

    def test_model_and_data_paths_pass_through(self, tmp_path):
        config = _render(tmp_path)
        assert config["model_name_or_path"] == _REQUIRED["--model-name-or-path"]
        assert config["dataset_name"] == _REQUIRED["--dataset-name"]

    def test_output_is_parseable_by_the_trainers_own_loader(self, tmp_path):
        """safe_load is what the trainer uses, so rendering implies readability."""
        config = _render(tmp_path)
        assert isinstance(config, dict) and config


class TestSchemaMatchesTheValidatedConfigs:
    """The emitted key set must match kd-sandbox/configs/gold/, exactly.

    The trainer parses its config with TRL's ``parse_args_and_config``, which
    rejects unknown top-level keys rather than ignoring them. So an extra or
    misplaced key is not a harmless difference — it aborts the run, and only after
    the student and the 30B teacher have loaded and the allocation has been held
    for the better part of an hour.

    That is exactly what happened on the first real run: a flat ``min_lr`` gave

        ValueError: Unknown arguments from config file: ['--min_lr', '1e-06']

    after 52 minutes. These lists are transcribed from the validated configs on
    /proj, so a drift in either direction fails here in milliseconds instead.
    """

    # Off-policy keys, from
    # kd-sandbox/configs/gold/granite-4.1-3b_from-4.2-30b_*_node2.yaml
    # minus its on-policy block.
    EXPECTED_OFF_POLICY = {
        "model_name_or_path",
        "teacher_model_name_or_path",
        "dataset_name",
        "num_train_epochs",
        "dataset_num_proc",
        "learning_rate",
        "warmup_ratio",
        "lr_scheduler_type",
        "lr_scheduler_kwargs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "max_completion_length",
        "max_length",
        "gradient_checkpointing",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "logging_steps",
        "temperature",
        "lmbda",
        "beta",
        "use_liger_fused_jsd",
        "response_template",
    }
    ONLINE_ONLY = {
        "vllm_num_servers",
        "top_p",
        "use_sampled_opd_loss",
        "last_message_only",
        "clip_alpha",
        "opd_importance_sampling",
    }

    def test_off_policy_key_set_is_exact(self, tmp_path):
        assert set(_render(tmp_path)) == self.EXPECTED_OFF_POLICY

    def test_on_policy_key_set_is_exact(self, tmp_path):
        config = _render(
            tmp_path, total_nodes=2, extra=["--vllm-num-servers", "1", "--lmbda", "1.0"]
        )
        assert set(config) == self.EXPECTED_OFF_POLICY | self.ONLINE_ONLY

    def test_min_lr_is_nested_under_the_scheduler(self, tmp_path):
        """The specific failure. min_lr configures the scheduler, so a top-level
        key is not merely redundant — it is rejected."""
        config = _render(tmp_path)
        assert (
            "min_lr" not in config
        ), "a flat min_lr is rejected by TRL's parse_args_and_config"
        assert config["lr_scheduler_kwargs"]["min_lr"] == pytest.approx(1e-06)

    def test_nested_min_lr_is_a_float(self, tmp_path):
        """Nesting must not lose the float typing the trainer needs."""
        config = _render(tmp_path, extra=["--min-lr", "1e-06"])
        assert isinstance(config["lr_scheduler_kwargs"]["min_lr"], float)

    def test_save_strategy_is_emitted(self, tmp_path):
        """Every validated config sets it explicitly rather than relying on the
        HF default."""
        assert _render(tmp_path)["save_strategy"] == "steps"

    def test_max_steps_is_absent_by_default(self, tmp_path):
        """An epoch-bounded run must render exactly the key set the validated
        references carry. This is also what keeps gold-smoke's rendered config
        byte-identical now that a step-bounded sibling recipe exists — a key that
        appeared unconditionally would silently change every existing run."""
        assert "max_steps" not in _render(tmp_path)

    def test_max_steps_is_emitted_as_an_int_when_set(self, tmp_path):
        """The trainer compares it against a step counter, so a string would
        raise partway into a run rather than at parse time."""
        config = _render(tmp_path, extra=["--max-steps", "100"])
        assert config["max_steps"] == 100
        assert isinstance(config["max_steps"], int)

    def test_max_steps_zero_is_omitted_rather_than_emitted(self, tmp_path):
        """0 is the step default and means "bound by epochs". Emitting a literal
        0 would depend on the trainer reading it as unbounded rather than as no
        steps at all; omitting it does not."""
        assert "max_steps" not in _render(tmp_path, extra=["--max-steps", "0"])

    def test_max_steps_does_not_displace_the_epoch_count(self, tmp_path):
        """Both keys travel together: the trainer needs num_train_epochs present
        to build the LR schedule even when max_steps truncates the run."""
        config = _render(tmp_path, extra=["--max-steps", "100"])
        assert config["num_train_epochs"] == pytest.approx(1.0)

    def test_save_strategy_is_settable(self, tmp_path):
        """A run that wants no checkpoints should not have to edit the step."""
        assert (
            _render(tmp_path, extra=["--save-strategy", "no"])["save_strategy"] == "no"
        )

    def test_scheduler_type_pairs_with_the_kwargs(self, tmp_path):
        """cosine_with_min_lr is the scheduler that consumes min_lr; the two must
        travel together or the schedule silently is not what was asked for."""
        config = _render(tmp_path)
        assert config["lr_scheduler_type"] == "cosine_with_min_lr"
        assert "min_lr" in config["lr_scheduler_kwargs"]


class TestLossArms:
    """The arm table, and the fact that NO arm is the default and changes nothing.

    Nine of CustomGOLDConfig's loss switches are not orthogonal, and the dangerous
    combinations are the ones that SUCCEED while training something other than what the
    config names. These tests cover both halves of the contract: naming an arm enforces it,
    and naming nothing leaves every existing recipe exactly as it was.
    """

    def test_no_arm_is_the_default(self, tmp_path):
        """The whole no-behaviour-change claim for five existing recipes rests on this."""
        base = _render(_dir(tmp_path, "a"))
        with_empty = _render(_dir(tmp_path, "b"), extra=["--loss-arm", ""])
        assert base == with_empty

    def test_no_arm_adds_no_keys(self, tmp_path):
        """Stated separately from the equality above because it is the property
        test_off_policy_key_set_is_exact depends on, and a future arm default would break
        that test in a way whose cause was not obvious from its name."""
        config = _render(tmp_path)
        for key in (
            "use_kl_interpolation",
            "use_adaptive_kld",
            "use_distillm2",
            "use_uld_loss",
            "use_ce_loss",
        ):
            assert key not in config

    def test_jsd_is_an_explicit_arm_that_adds_nothing(self, tmp_path):
        """`jsd` names the default objective rather than changing it, so its rendered
        config must equal the unnamed one. It exists as an arm so that a build can STATE
        the objective and get the contradiction checks -- which is the only difference
        between it and passing no arm at all."""
        assert _render(_dir(tmp_path, "a"), extra=["--loss-arm", "jsd"]) == _render(
            _dir(tmp_path, "b")
        )

    def test_an_unknown_arm_is_refused(self, tmp_path):
        result = _render(tmp_path, extra=["--loss-arm", "kd"], expect_rc=2)
        assert "is not one of" in result.stderr

    @pytest.mark.parametrize(
        "arm,key",
        [
            ("uld", "use_uld_loss"),
            ("adaptive_kld", "use_adaptive_kld"),
        ],
    )
    def test_an_arm_emits_the_switch_the_renderer_had_no_flag_for(
        self, tmp_path, arm, key
    ):
        """The point of the table: these seven switches are CustomGOLDConfig fields that
        no CLI flag on this renderer could previously set at all."""
        config = _render(tmp_path, extra=["--loss-arm", arm])
        assert config[key] is True

    def test_ce_requires_off_policy_and_says_why(self, tmp_path):
        """A control that trains on its own output is not a control."""
        result = _render(
            tmp_path, extra=["--loss-arm", "ce", "--lmbda", "0.5"], expect_rc=2
        )
        assert "requires lmbda=0.0" in result.stderr
        assert "not a control" in result.stderr

    def test_ce_renders_at_lmbda_zero(self, tmp_path):
        assert _render(tmp_path, extra=["--loss-arm", "ce"])["use_ce_loss"] is True

    def test_a_degenerate_distillm2_is_refused_rather_than_silently_equivalent(
        self, tmp_path
    ):
        """At lmbda 0.0 the comparative loss IS forward KL. It does not crash, which is
        why this is checked here: nothing downstream could tell the difference between
        this run and a `jsd` run at beta 0.0, and the config would claim DistiLLM-2."""
        result = _render(tmp_path, extra=["--loss-arm", "distillm2"], expect_rc=2)
        assert "0.0 < lmbda < 1.0" in result.stderr
        assert "degenerates to forward KL" in result.stderr

    def test_kl_interpolation_is_refused_at_a_pure_divergence(self, tmp_path):
        """beta 0.0 short-circuits before the interpolation branch, so the flag is dead
        and the run is exactly `jsd` while reporting an interpolated objective."""
        result = _render(
            tmp_path, extra=["--loss-arm", "kl_interpolation"], expect_rc=2
        )
        assert "0.0 < beta < 1.0" in result.stderr

    def test_kl_interpolation_renders_between_the_divergences(self, tmp_path):
        config = _render(
            tmp_path, extra=["--loss-arm", "kl_interpolation", "--beta", "0.5"]
        )
        assert config["use_kl_interpolation"] is True
        assert config["beta"] == pytest.approx(0.5)

    def test_an_arm_contradicting_its_standalone_flag_is_refused(self, tmp_path):
        """The arm says liger, the flag says no liger. Refused rather than resolved
        quietly in favour of either one."""
        result = _render(tmp_path, extra=["--loss-arm", "liger_fused_jsd"], expect_rc=2)
        assert "--use-liger-fused-jsd" in result.stderr

    def test_the_contradiction_is_refused_in_the_other_direction_too(self, tmp_path):
        """THE DANGEROUS DIRECTION. use_liger_fused_jsd selects a fused branch that drops
        every other loss switch without a word, so this pair would train JSD while the
        rendered config said ULD."""
        result = _render(
            tmp_path,
            extra=["--loss-arm", "uld", "--use-liger-fused-jsd", "true"],
            expect_rc=2,
        )
        assert "implies use_liger_fused_jsd=False" in result.stderr

    def test_the_liger_arm_renders_when_the_flag_agrees(self, tmp_path):
        config = _render(
            tmp_path,
            extra=["--loss-arm", "liger_fused_jsd", "--use-liger-fused-jsd", "true"],
        )
        assert config["use_liger_fused_jsd"] is True

    def test_the_documented_escape_hatch_still_works_without_an_arm(self, tmp_path):
        """Three recipe READMEs instruct `--param USE_LIGER_FUSED_JSD=true` to answer an
        open question. With no arm named there is nothing to contradict, so that keeps
        working -- which is why no arm, rather than `jsd`, is the default."""
        config = _render(tmp_path, extra=["--use-liger-fused-jsd", "true"])
        assert config["use_liger_fused_jsd"] is True

    def test_sampled_opd_offline_is_refused_rather_than_dropped(self, tmp_path):
        """use_sampled_opd_loss is emitted only on the on-policy path, so an off-policy
        render would drop the arm's only switch and train plain JSD under a config that
        named sampled_opd."""
        result = _render(
            tmp_path,
            extra=[
                "--loss-arm",
                "sampled_opd",
                "--lmbda",
                "1.0",
                "--last-message-only",
                "true",
                "--use-sampled-opd-loss",
                "true",
            ],
            expect_rc=2,
        )
        assert "on-policy path" in result.stderr

    def test_sampled_opd_renders_on_policy(self, tmp_path):
        config = _render(
            tmp_path,
            total_nodes=2,
            extra=[
                "--loss-arm",
                "sampled_opd",
                "--lmbda",
                "1.0",
                "--last-message-only",
                "true",
                "--use-sampled-opd-loss",
                "true",
                "--vllm-num-servers",
                "1",
            ],
        )
        assert config["use_sampled_opd_loss"] is True
        assert config["last_message_only"] is True

    def test_sampled_opd_requirements_are_reported_not_coerced(self, tmp_path):
        """Rewriting a requested lmbda to satisfy an arm would change the experiment
        without saying so."""
        result = _render(
            tmp_path,
            extra=["--loss-arm", "sampled_opd", "--vllm-num-servers", "1"],
            expect_rc=2,
        )
        assert "requires lmbda=1.0" in result.stderr


class TestTheArmTableCannotDeclareAnUnenforcedRequirement:
    """A meta-assertion over the table itself.

    The enforcement loop is generic so that an arm cannot declare a requirement no branch
    reads. This asserts the converse: no arm names a requirement KIND the loop does not
    implement -- which would be a requirement that exists in the table, reads as enforced,
    and is not.
    """

    @staticmethod
    def _module():
        import importlib.util

        spec = importlib.util.spec_from_file_location("_rgc", _RENDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_every_requirement_kind_is_implemented(self):
        module = self._module()
        for name, arm in module.LOSS_ARMS.items():
            for kind in arm.get("requires", {}):
                assert kind in module._REQUIREMENT_KINDS, (
                    f"arm {name!r} requires {kind!r}, which the enforcement loop does "
                    "not implement"
                )

    def test_every_arm_documents_itself(self):
        """The help text lists the arms by name; a nameless objective in a rendered
        config is not something the next person can act on."""
        for name, arm in self._module().LOSS_ARMS.items():
            assert arm["doc"], f"arm {name!r} has no doc"

    def test_every_requirement_explains_why(self):
        """Half of these requirements are not the dataclass's rules -- they exist because
        outside them the arm silently becomes a different one. A refusal without that
        reason reads as an arbitrary restriction to work around."""
        for name, arm in self._module().LOSS_ARMS.items():
            if arm.get("requires"):
                assert arm.get("why"), f"arm {name!r} constrains without saying why"

    def test_the_standalone_flag_map_names_real_flags(self):
        """A key here that is not actually a CLI flag would make the contradiction check
        compare against an attribute that does not exist, i.e. crash on an arm rather
        than refuse a contradiction."""
        module = self._module()
        args = module._parse_args(
            [
                "--output",
                "/dev/null",
                "--total-nodes",
                "1",
                "--model-name-or-path",
                "x",
                "--teacher-model-name-or-path",
                "y",
                "--dataset-name",
                "z",
            ]
        )
        for key in module._STANDALONE_ARM_FLAGS:
            assert hasattr(args, key), f"{key} is not a parsed argument"


class TestCorpusTokenizerCheck:
    """Off by default, and LOUD when asked for without the package it needs."""

    def test_it_is_off_by_default(self, tmp_path):
        """Otherwise every existing recipe would start failing on an import."""
        assert "check_corpus_tokenizer" not in _render(tmp_path)

    def test_asking_for_it_without_the_source_is_an_error(self, tmp_path):
        """NOT a skip. A validator that quietly does not run is worse than one that is
        absent, because the operator believes they checked. The message has to name the
        config key that would deliver the package."""
        out = tmp_path / "c.yaml"
        cmd = [
            sys.executable,
            str(_RENDER),
            "--output",
            str(out),
            "--total-nodes",
            "1",
            "--check-corpus-tokenizer",
            "true",
        ]
        for flag, value in _REQUIRED.items():
            cmd += [flag, value]
        # An empty PYTHONPATH, so the outcome does not depend on what happens to be on the
        # path of whoever runs the suite.
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = ""
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, env=env
        )
        assert result.returncode == 2, result.stdout + result.stderr
        assert "gb_steps_post_training is not importable" in result.stderr
        assert not out.exists(), "a refused render must not leave a config behind"


class TestCeAnchorAndCollapseGuard:
    """The keys added after build df8512e0 collapsed a student into repetition loops.

    Two properties matter and pull against each other. The keys have to REACH the
    trainer when a recipe asks for them, and they have to be ABSENT when it does not
    — they exist only in the clone this step pins, and TRL's parse_args_and_config
    rejects unknown top-level keys outright, so an unconditional emit would make
    every config here unreadable by any other kd-sandbox checkout.
    """

    ANCHOR_KEYS = {
        "ce_coef",
        "log_student_entropy",
        "entropy_guard_drop_frac",
        "entropy_guard_baseline_steps",
        "entropy_guard_patience",
        "entropy_guard_action",
    }

    def test_none_of_them_are_emitted_by_default(self, tmp_path):
        assert self.ANCHOR_KEYS.isdisjoint(_render(tmp_path))

    def test_a_default_render_is_unchanged_by_the_new_flags_existing(self, tmp_path):
        """The regression that would break every existing recipe at once: passing the
        flags at their defaults must produce the same config as not passing them."""
        implicit = _render(tmp_path)
        explicit = _render(
            tmp_path,
            extra=[
                "--ce-coef",
                "0.0",
                "--log-student-entropy",
                "false",
                "--entropy-guard-drop-frac",
                "0.0",
                "--lmbda-schedule",
                "constant",
                "--min-completion-length",
                "0",
            ],
        )
        assert implicit == explicit

    def test_the_anchor_is_emitted_as_a_float_when_asked_for(self, tmp_path):
        config = _render(tmp_path, extra=["--ce-coef", "0.05"])
        assert isinstance(config["ce_coef"], float)
        assert config["ce_coef"] == pytest.approx(0.05)
        # Requested alone, it must not drag the guard in with it.
        assert "entropy_guard_drop_frac" not in config

    def test_the_guard_travels_with_its_two_settings(self, tmp_path):
        """Emitting the threshold without the baseline window and the patience would
        leave the trainer applying defaults the recipe never stated."""
        config = _render(
            tmp_path,
            extra=[
                "--log-student-entropy",
                "true",
                "--entropy-guard-drop-frac",
                "0.15",
                "--entropy-guard-baseline-steps",
                "30",
                "--entropy-guard-patience",
                "2",
            ],
        )
        assert config["entropy_guard_drop_frac"] == pytest.approx(0.15)
        assert config["entropy_guard_baseline_steps"] == 30
        assert config["entropy_guard_patience"] == 2
        # Defaulted, but still emitted: the trainer needs to be told, and "stop" is the
        # behaviour every recipe pinned before the knob existed.
        assert config["entropy_guard_action"] == "stop"
        assert config["log_student_entropy"] is True

    def test_warn_mode_reaches_the_trainer(self, tmp_path):
        """Build d1acf1c0's failure in one key. The guard tripped at step 77 of 2,000
        and stopped, which took every rung of a fixed 500/1000/1500/2000 export ladder
        with it. warn keeps the run going so the rungs exist; the trip is still printed
        and still checkpointed."""
        config = _render(
            tmp_path,
            extra=[
                "--log-student-entropy",
                "true",
                "--entropy-guard-drop-frac",
                "0.15",
                "--entropy-guard-action",
                "warn",
            ],
        )
        assert config["entropy_guard_action"] == "warn"

    def test_an_unknown_action_is_refused_here(self, tmp_path):
        """Not left to the trainer's __post_init__: a bad value there surfaces after
        accelerate has launched, the teacher has loaded and the allocation is held."""
        _render(
            tmp_path,
            extra=[
                "--log-student-entropy",
                "true",
                "--entropy-guard-drop-frac",
                "0.15",
                "--entropy-guard-action",
                "continue",
            ],
            expect_rc=2,
        )

    def test_a_guard_without_its_metric_is_refused(self, tmp_path):
        """The silent failure this check exists for: the guard reads what
        log_student_entropy computes, so without it the guard never arms — which is
        indistinguishable from a run that never collapsed."""
        result = _render(
            tmp_path, extra=["--entropy-guard-drop-frac", "0.15"], expect_rc=2
        )
        assert "log_student_entropy" in result.stderr

    @pytest.mark.parametrize(
        "extra",
        [
            ["--ce-coef", "-0.1"],
            ["--log-student-entropy", "true", "--entropy-guard-drop-frac", "1.0"],
            [
                "--log-student-entropy",
                "true",
                "--entropy-guard-drop-frac",
                "0.1",
                "--entropy-guard-patience",
                "0",
            ],
            [
                "--log-student-entropy",
                "true",
                "--entropy-guard-drop-frac",
                "0.1",
                "--entropy-guard-baseline-steps",
                "0",
            ],
        ],
        ids=["negative-ce", "drop-frac-1.0", "patience-0", "baseline-0"],
    )
    def test_out_of_range_values_fail_here_not_on_the_cluster(self, tmp_path, extra):
        """Validated in the renderer rather than left to the trainer's __post_init__,
        because a raise there surfaces after accelerate has launched and the teacher
        has loaded on every node of a held allocation."""
        _render(tmp_path, extra=extra, expect_rc=2)

    def test_the_logged_entropy_is_the_signal_the_loss_was_blind_to(self, tmp_path):
        """Documents why this exists, in the one place that cannot rot: df8512e0's
        train loss moved 2.7% over the 7,640 steps in which the student lost 42% of
        its entropy, so `loss` alone cannot gate a run of this shape."""
        assert "student_entropy" not in _render(tmp_path)
        assert (
            _render(tmp_path, extra=["--log-student-entropy", "true"])[
                "log_student_entropy"
            ]
            is True
        )


class TestOnPolicyShapingKeys:
    """lmbda_schedule and min_completion_length do nothing at lmbda 0, so the
    renderer refuses them there rather than letting a recipe believe it asked for
    something. Inert-but-accepted is the failure mode this file exists to remove."""

    def test_neither_is_emitted_by_default(self, tmp_path):
        config = _render(tmp_path)
        for key in ("lmbda_schedule", "lmbda_init", "min_completion_length"):
            assert key not in config

    def test_a_linear_ramp_emits_both_halves(self, tmp_path):
        config = _render(
            tmp_path,
            total_nodes=2,
            extra=[
                "--vllm-num-servers",
                "1",
                "--lmbda",
                "0.25",
                "--lmbda-schedule",
                "linear",
                "--lmbda-init",
                "0.0",
            ],
        )
        assert config["lmbda_schedule"] == "linear"
        assert config["lmbda_init"] == pytest.approx(0.0)

    def test_a_ramp_without_a_server_is_refused(self, tmp_path):
        result = _render(
            tmp_path,
            extra=["--lmbda-schedule", "linear", "--lmbda-init", "0.0"],
            expect_rc=2,
        )
        assert "vllm" in result.stderr.lower()

    def test_a_generation_floor_without_a_server_is_refused(self, tmp_path):
        result = _render(tmp_path, extra=["--min-completion-length", "32"], expect_rc=2)
        assert "min_completion_length" in result.stderr

    def test_an_unknown_schedule_is_refused(self, tmp_path):
        _render(tmp_path, extra=["--lmbda-schedule", "cosine"], expect_rc=2)
