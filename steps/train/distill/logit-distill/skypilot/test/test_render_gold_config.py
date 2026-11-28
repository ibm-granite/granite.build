# SPDX-License-Identifier: Apache-2.0

"""Tests for the gold config renderer (../src/render_gold_config.py).

Run from the step directory with `make test`.

These pin the trainer requirements that fail *silently* — a run that starts, then
either crashes hours in or trains on the wrong objective. Each of the three rules
below cost real debugging time on the ansible path before it was understood, which
is why the renderer exists as a testable script rather than a shell heredoc.
"""

import os
import re
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
        "vllm_mode",
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

    @pytest.mark.parametrize("lmo", ["false", "true"])
    def test_off_policy_refuses_the_sampled_opd_switch(self, tmp_path, lmo):
        """use_sampled_opd_loss is one of these keys, so off-policy it was dropped with
        exit 0 and the run trained plain JSD. It selects the objective, so it is refused
        rather than dropped, as the distill recipe refuses it off-policy."""
        result = _render(
            tmp_path,
            total_nodes=1,
            extra=[
                "--lmbda",
                "0.0",
                "--use-sampled-opd-loss",
                "true",
                "--last-message-only",
                lmo,
            ],
            expect_rc=2,
        )
        assert "use_sampled_opd_loss" in result.stderr
        assert "vllm_num_servers" in result.stderr

    def test_on_policy_refuses_the_sampled_opd_switch_below_lmbda_1(self, tmp_path):
        """CustomGOLDConfig requires lmbda 1.0 for it; the arm already refused other
        values, the switch alone rendered them for the trainer to refuse at startup."""
        on_policy = ["--vllm-num-servers", "1", "--last-message-only", "true"]
        sopd = ["--use-sampled-opd-loss", "true"]
        result = _render(
            tmp_path, extra=[*on_policy, *sopd, "--lmbda", "0.5"], expect_rc=2
        )
        assert "use_sampled_opd_loss" in result.stderr and "1.0" in result.stderr
        config = _render(tmp_path, extra=[*on_policy, *sopd, "--lmbda", "1.0"])
        assert config["use_sampled_opd_loss"] is True and config["lmbda"] == 1.0

    @pytest.mark.parametrize(
        "extra, needle",
        [
            (["--last-message-only", "false"], "last_message_only"),
            (
                ["--last-message-only", "true", "--use-liger-fused-jsd", "true"],
                "use_liger_fused_jsd",
            ),
        ],
        ids=["last-message-only-false", "liger"],
    )
    def test_on_policy_refuses_the_sampled_opd_switch_as_custom_gold_config_does(
        self, tmp_path, extra, needle
    ):
        """CustomGOLDConfig refuses use_sampled_opd_loss without last_message_only and
        with use_liger_fused_jsd; LOSS_ARMS refused both for the arm, the switch alone
        rendered them for the trainer to refuse at startup."""
        on_policy = ["--vllm-num-servers", "1", "--lmbda", "1.0"]
        sopd = ["--use-sampled-opd-loss", "true"]
        result = _render(tmp_path, extra=[*on_policy, *sopd, *extra], expect_rc=2)
        assert "use_sampled_opd_loss" in result.stderr and needle in result.stderr

    def test_off_policy_takes_the_sampled_opd_switch_off(self, tmp_path):
        """False is the default, and off-policy it is still not emitted."""
        config = _render(tmp_path, extra=["--use-sampled-opd-loss", "false"])
        assert "use_sampled_opd_loss" not in config

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
        # warmup_ratio until transformers 5.16 removed it; see TestTrl1Defaults.
        "warmup_steps",
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
        # Not in the validated trl 0.26.2 configs: trl 1.x defaults it to float32.
        "dtype",
    }
    ONLINE_ONLY = {
        "vllm_num_servers",
        "vllm_mode",
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

    def test_data_seed_is_absent_by_default(self, tmp_path):
        """Every recipe that predates the key must render exactly what it rendered
        before; the sampler then follows the trainer's `seed`, as it always has."""
        assert "data_seed" not in _render(_dir(tmp_path, "a"))
        assert "data_seed" not in _render(
            _dir(tmp_path, "b"), extra=["--data-seed", "-1"]
        )

    def test_data_seed_is_emitted_as_an_int_when_set(self, tmp_path):
        """The on-policy-from-a-checkpoint recipe sets it so the continuation does not
        replay the batches its init already trained on. 0 is a valid seed, so the
        sentinel for "unset" is -1, not 0."""
        for value in (0, 43):
            config = _render(
                _dir(tmp_path, str(value)), extra=["--data-seed", str(value)]
            )
            assert config["data_seed"] == value
            assert isinstance(config["data_seed"], int)

    def test_data_seed_is_not_a_trainer_extension(self):
        """It is a stock TrainingArguments field, so it must not trip the delivered-
        trainer field check that guards the trainer-extension keys."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("_rgc", _RENDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert "data_seed" not in module._TRAINER_EXTENSION_KEYS

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
        ],
    )
    def test_an_arm_emits_the_switch_the_renderer_had_no_flag_for(
        self, tmp_path, arm, key
    ):
        """The point of the table: these switches are CustomGOLDConfig fields that
        no CLI flag on this renderer could previously set at all."""

        config = _render(
            tmp_path, extra=["--loss-arm", arm, "--last-message-only", "true"]
        )
        assert config[key] is True

    @pytest.mark.parametrize("given", [None, "false"])
    def test_uld_without_last_message_only_is_refused_at_render(self, tmp_path, given):
        """Unset (the default, false) or false: the student's span and ULD's last-turn
        teacher never align on a multi-turn row, and the run would train at zero gradient.
        The message names both keys and the recipe parameter that sets the second."""
        extra = ["--loss-arm", "uld"]
        if given is not None:
            extra += ["--last-message-only", given]
        result = _render(tmp_path, extra=extra, expect_rc=2)
        assert "loss_arm='uld' requires last_message_only=true; got false" in (
            result.stderr
        )
        assert "--param LAST_MESSAGE_ONLY=true" in result.stderr

    def test_uld_emits_last_message_only_off_policy(self, tmp_path):
        """Off-policy, last_message_only is otherwise never emitted (_ONLINE_ONLY), so the
        requirement above would be checked and then dropped, and the trainer would run
        with its default, false. For uld alone it is emitted."""
        config = _render(
            tmp_path, extra=["--loss-arm", "uld", "--last-message-only", "true"]
        )
        assert config["last_message_only"] is True
        assert "vllm_num_servers" not in config

    def test_no_other_arm_emits_last_message_only_off_policy(self, tmp_path):
        """The exception is the uld arm's alone; every other arm off-policy is pinned
        byte for byte in test_render_gold_config_arms_golden.py."""
        config = _render(tmp_path, extra=["--last-message-only", "true"])
        assert "last_message_only" not in config

    def test_adaptive_kld_is_refused_with_the_reason(self, tmp_path):
        """The arm's trainer implementation does not compute the paper's objective
        (arXiv 2404.02657 weights whole distributions; the implementation switches
        per vocabulary entry, the teacher is not a stationary point, and top-1 > 0.5
        empties the head mask and silently yields plain reverse KL), so the arm is
        refused rather than rendered."""
        result = _render(tmp_path, extra=["--loss-arm", "adaptive_kld"], expect_rc=2)
        assert "is refused" in result.stderr
        assert "stationary point" in result.stderr
        assert "reverse KL" in result.stderr

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
            extra=[
                "--loss-arm",
                "uld",
                "--last-message-only",
                "true",
                "--use-liger-fused-jsd",
                "true",
            ],
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

    @pytest.mark.parametrize("lmbda", ["0.25", "1.0"])
    def test_lmbda_without_a_server_is_refused(self, tmp_path, lmbda):
        """lmbda > 0 at vllm_num_servers 0 routes generation through TRL's in-process
        generate_on_policy_outputs inside the training ranks, a path nothing here has run.
        The renderer keeps rollouts on a vLLM server rather than adopt it unvalidated.
        """
        result = _render(tmp_path, extra=["--lmbda", lmbda], expect_rc=2)
        assert "vllm_num_servers is 0" in result.stderr
        assert "generate_on_policy_outputs" in result.stderr

    def test_lmbda_zero_without_a_server_still_renders(self, tmp_path):
        assert _render(tmp_path, extra=["--lmbda", "0.0"])["lmbda"] == 0.0

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


class TestRetiredCeAnchorAndEntropyGuard:
    """The CE anchor and the entropy-collapse guard were removed in #664: a CE sweep at
    four coefficients was monotone-bad on capability, and the entropy drop the guard
    watched for was harmless (the real defect was the rope_theta export bug). Their
    flags are gone, so a stale caller fails at argparse instead of being ignored."""

    RETIRED_KEYS = {
        "ce_coef",
        "log_student_entropy",
        "entropy_guard_drop_frac",
        "entropy_guard_baseline_steps",
        "entropy_guard_patience",
        "entropy_guard_action",
    }

    def test_none_of_them_are_emitted(self, tmp_path):
        assert self.RETIRED_KEYS.isdisjoint(_render(tmp_path))

    @pytest.mark.parametrize(
        "extra",
        [
            ["--ce-coef", "0.05"],
            ["--log-student-entropy", "true"],
            ["--entropy-guard-drop-frac", "0.15"],
            ["--entropy-guard-baseline-steps", "20"],
            ["--entropy-guard-patience", "3"],
            ["--entropy-guard-action", "stop"],
        ],
    )
    def test_their_flags_are_refused(self, tmp_path, extra):
        _render(tmp_path, extra=extra, expect_rc=2)

    def test_a_default_render_is_unchanged_by_the_remaining_flags(self, tmp_path):
        """Passing the on-policy shaping flags at their defaults must produce the same
        config as not passing them."""
        implicit = _render(tmp_path)
        explicit = _render(
            tmp_path,
            extra=[
                "--lmbda-schedule",
                "constant",
                "--min-completion-length",
                "0",
            ],
        )
        assert implicit == explicit


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


class TestTrainerKeyCheck:
    """--verify-trainer-accepts-keys imports the delivered CustomGOLDConfig, and through it
    TRL. Faked here so the suite needs neither. The fake TRL stores a plain bool guard, as
    trl 1.x does: trl 0.26's (bool, version) tuple for _vllm_ascend_available, which made
    the import fail and read as a missing package (the ck2500 lmbda-ramp arms, fe650626,
    ae922fd7), is gone upstream, and the renderer no longer unwraps it (#676)."""

    _RAMP = [
        "--vllm-num-servers",
        "1",
        "--lmbda",
        "0.75",
        "--lmbda-schedule",
        "linear",
        "--lmbda-init",
        "0.25",
        "--verify-trainer-accepts-keys",
        "true",
    ]

    @staticmethod
    def _fake_packages(root, ascend_flag="False", fields=None):
        fields = fields or {"lmbda_schedule": '"constant"', "lmbda_init": "0.0"}
        (root / "trl").mkdir(parents=True)
        (root / "trl" / "__init__.py").write_text("")
        (root / "trl" / "import_utils.py").write_text(
            f"_vllm_ascend_available = {ascend_flag}\n"
            '_mergekit_available = (False, "N/A")\n'
        )
        dist = root / "gb_steps_post_training" / "distillation"
        dist.mkdir(parents=True)
        (root / "gb_steps_post_training" / "__init__.py").write_text("")
        (dist / "__init__.py").write_text("")
        body = "".join(
            f"    {name}: object = {value}\n" for name, value in fields.items()
        )
        (dist / "custom_gold_config.py").write_text(
            "from dataclasses import dataclass\n"
            "from trl import import_utils\n"
            "if import_utils._vllm_ascend_available:\n"
            "    import vllm_ascend  # noqa: F401\n"
            "@dataclass\n"
            "class CustomGOLDConfig:\n" + body
        )
        return root

    @staticmethod
    def _render_with(tmp_path, pythonpath, extra):
        out = tmp_path / "gold_config.yaml"
        cmd = [sys.executable, str(_RENDER), "--output", str(out), "--total-nodes", "2"]
        for flag, value in _REQUIRED.items():
            cmd += [flag, value]
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(pythonpath)
        result = subprocess.run(
            cmd + extra, capture_output=True, text=True, check=False, env=env
        )
        return result, out

    def test_a_trainer_with_the_fields_passes(self, tmp_path):
        pkgs = self._fake_packages(tmp_path / "pkgs")
        result, out = self._render_with(tmp_path, pkgs, self._RAMP)
        assert result.returncode == 0, result.stderr
        config = yaml.safe_load(out.read_text())
        assert config["lmbda_schedule"] == "linear"
        assert config["lmbda_init"] == pytest.approx(0.25)

    def test_a_trainer_without_the_fields_is_still_refused(self, tmp_path):
        """The check exists to refuse this."""
        pkgs = self._fake_packages(
            tmp_path / "pkgs", fields={"lmbda_schedule": '"constant"'}
        )
        result, out = self._render_with(tmp_path, pkgs, self._RAMP)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "has no field(s) lmbda_init" in result.stderr
        assert not out.exists()

    def test_no_trl_at_all_is_still_reported_as_not_importable(self, tmp_path):
        result, out = self._render_with(tmp_path, tmp_path / "empty", self._RAMP)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "gb_steps_post_training is not importable" in result.stderr
        assert not out.exists()


class TestOnPolicyStudyKeys:
    """The keys the on-policy study adds: two stock TrainingArguments fields and three that
    need the vendored trainer's on-policy study fields (see steps/train/distill/src/VENDORED.md). All five are absent unless asked for, so every existing
    recipe renders the config it rendered before."""

    _ONLINE = ["--vllm-num-servers", "1", "--lmbda", "1.0"]
    _RAMP = _ONLINE + ["--lmbda-schedule", "linear", "--lmbda-init", "0.0"]
    _NEW = (
        "adam_beta2",
        "train_sampling_strategy",
        "lmbda_ramp_start_step",
        "lmbda_ramp_steps",
        "stop_at_step",
    )

    def test_none_of_them_are_emitted_by_default(self, tmp_path):
        assert not [k for k in self._NEW if k in _render(_dir(tmp_path, "a"))]
        on = _render(_dir(tmp_path, "b"), extra=self._RAMP)
        assert not [k for k in self._NEW if k in on]

    def test_adam_beta2_is_emitted_as_a_float(self, tmp_path):
        config = _render(tmp_path, extra=["--adam-beta2", "0.95"])
        assert config["adam_beta2"] == pytest.approx(0.95)
        assert isinstance(config["adam_beta2"], float)

    @pytest.mark.parametrize("value", ["0", "1", "1.5"])
    def test_adam_beta2_outside_the_open_interval_is_refused(self, tmp_path, value):
        result = _render(tmp_path, extra=["--adam-beta2", value], expect_rc=2)
        assert "adam_beta2 must be in (0, 1)" in result.stderr

    def test_sequential_sampling_is_emitted(self, tmp_path):
        config = _render(tmp_path, extra=["--train-sampling-strategy", "sequential"])
        assert config["train_sampling_strategy"] == "sequential"

    def test_an_unknown_sampling_strategy_is_refused(self, tmp_path):
        result = _render(
            tmp_path, extra=["--train-sampling-strategy", "ordered"], expect_rc=2
        )
        assert "train_sampling_strategy must be" in result.stderr

    def test_the_stock_fields_are_not_trainer_extensions(self):
        sys.path.insert(0, str(_RENDER.parent))
        try:
            import render_gold_config as rgc  # noqa: PLC0415
        finally:
            sys.path.pop(0)
        assert (
            not {"adam_beta2", "train_sampling_strategy"} & rgc._TRAINER_EXTENSION_KEYS
        )
        assert {"lmbda_ramp_start_step", "lmbda_ramp_steps", "stop_at_step"} <= (
            rgc._TRAINER_EXTENSION_KEYS
        )

    def test_the_ramp_window_travels_with_a_linear_schedule(self, tmp_path):
        config = _render(
            tmp_path,
            extra=self._RAMP
            + ["--lmbda-ramp-start-step", "2000", "--lmbda-ramp-steps", "2500"],
        )
        assert config["lmbda_ramp_start_step"] == 2000
        assert config["lmbda_ramp_steps"] == 2500

    def test_the_ramp_window_is_refused_on_a_constant_schedule(self, tmp_path):
        result = _render(
            tmp_path, extra=self._ONLINE + ["--lmbda-ramp-steps", "1250"], expect_rc=2
        )
        assert "silently inert" in result.stderr

    def test_stop_at_step_is_emitted_and_leaves_the_horizon_alone(self, tmp_path):
        config = _render(tmp_path, extra=["--stop-at-step", "4500"])
        assert config["stop_at_step"] == 4500
        assert "max_steps" not in config

    def test_stop_at_step_past_max_steps_is_refused(self, tmp_path):
        result = _render(
            tmp_path,
            extra=["--max-steps", "2500", "--stop-at-step", "3000"],
            expect_rc=2,
        )
        assert "would never fire" in result.stderr


class TestLossArmRequirementsUnderARamp:
    """lmbda_strictly_between asks that both policy classes occur. A linear ramp from 0 to 1
    has both for every step but its two ends, so it must render, while a constant lmbda at
    an end, or a ramp that never leaves one, is still refused."""

    _ONLINE = ["--vllm-num-servers", "1", "--loss-arm", "distillm2_like"]

    def test_a_ramp_from_zero_to_one_renders(self, tmp_path):
        config = _render(
            tmp_path,
            extra=self._ONLINE
            + ["--lmbda", "1.0", "--lmbda-schedule", "linear", "--lmbda-init", "0.0"],
        )
        assert config["use_distillm2_like"] is True

    def test_a_constant_lmbda_of_one_is_still_refused(self, tmp_path):
        result = _render(tmp_path, extra=self._ONLINE + ["--lmbda", "1.0"], expect_rc=2)
        assert "0.0 < lmbda < 1.0" in result.stderr

    def test_a_ramp_that_never_moves_is_judged_by_its_one_value(self, tmp_path):
        result = _render(
            tmp_path,
            extra=self._ONLINE
            + ["--lmbda", "1.0", "--lmbda-schedule", "linear", "--lmbda-init", "1.0"],
            expect_rc=2,
        )
        assert "linear from lmbda_init=1.0" in result.stderr


# ------------------------------------------- the template's defaults are the renderer's
#
# student-sft's parity tests compare this renderer's argparse defaults with student-sft's,
# but a recipe never reaches those: it gets step-template.yaml's values, which the run block
# passes to this renderer as flags. These tests read the flags from the run block rather
# than restating them, and check that each template default parses to the renderer's own --
# so the comparison of the two renderers' defaults is also a comparison of what recipes run.

_STEP_TEMPLATE = _RENDER.parent.parent / "step-template.yaml"

# `--flag {{ config.<section>.<key> [| default(...)] }}`, optionally quoted.
_CONFIG_FLAG = re.compile(
    r"""(--[a-z][a-z0-9-]*)\s+(["']?)(\{\{\s*config\.(\w+)\.(\w+)[^}]*\}\})\2(?=\s|$)"""
)

_GOLD_REQUIRED_ARGS = [
    "--output",
    "/o",
    "--total-nodes",
    "1",
    "--model-name-or-path",
    "/m",
    "--teacher-model-name-or-path",
    "/t",
    "--dataset-name",
    "/d",
]

# Renderer flags with a default that the template does not pass from its config, and why.
_RENDERER_DEFAULTS_THE_TEMPLATE_DOES_NOT_PASS = {
    # Passed, but from the `vllm` input binding (empty when unbound), not from config.
    "--vllm-server-url",
    # Passed only when some `tracking:` key is set, so a run that tracks nothing renders the
    # run script it rendered before (golden_run_scripts.json). test_step_template.py's
    # TestTracking renders the block both ways.
    "--tracking-tool",
    "--clearml-project",
    "--clearml-run-name",
    "--wandb-entity",
    "--wandb-project",
    "--wandb-run-name",
}

# Template defaults that deliberately differ from the renderer's, as (template, renderer).
# render_gold_config.check_trainer_accepts's docstring: "Off in this script's own
# argparse and ON in the step's config ... the step's image carries the package on every
# run, while a direct invocation (a unit test rendering a key in isolation) has no package
# to ask."
_DOCUMENTED_TEMPLATE_DIVERGENCES = {
    "gold_config.verify_trainer_accepts_keys": (True, False),
}


def _gold_module():
    import render_gold_config  # noqa: PLC0415  (conftest puts ../src on sys.path)

    return render_gold_config


def _jinja(source, step):
    """Render a template fragment the way gbserver fills a step (StrictUndefined)."""
    from jinja2 import StrictUndefined  # noqa: PLC0415
    from jinja2.sandbox import SandboxedEnvironment  # noqa: PLC0415

    env = SandboxedEnvironment(undefined=StrictUndefined)
    return env.from_string(source).render(config=step["config"])


def _template_flags():
    """[(template key, [flag, value])] for each config value the run block passes the
    renderer, in the renderer invocation only (the run block passes some of the same keys
    to other scripts too)."""
    step = yaml.safe_load(_STEP_TEMPLATE.read_text())
    run_block = step["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"][
        "run"
    ]
    joined = re.sub(r"\\\n", " ", run_block)
    [command] = [
        line for line in joined.splitlines() if "/render_gold_config.py" in line
    ]
    flags = []
    for match in _CONFIG_FLAG.finditer(command):
        flag, quote, expr, section, key = match.groups()
        value = _jinja(expr, step)
        # An unquoted value reaches the renderer as one word only if it is one word.
        assert quote or (value and not value.split()[1:]), (flag, value)
        flags.append((f"{section}.{key}", [flag, value]))
    # Every config reference in the command has the form above, so none is skipped.
    assert len(re.findall(r"\bconfig\.\w+\.\w+", command)) == len(flags), command
    return flags


def _dest(flag):
    """argparse's dest for a long option declared without an explicit dest=."""
    return flag[2:].replace("-", "_")


def test_the_template_defaults_are_the_renderers_defaults():
    """Every config value the template passes parses, in the renderer, to the renderer's own
    default -- except the documented divergences above."""
    module = _gold_module()
    defaults = vars(module._parse_args(_GOLD_REQUIRED_ARGS))
    differ = {}
    for key, tokens in _template_flags():
        dest = _dest(tokens[0])
        assert dest in defaults, f"{tokens[0]} does not set {dest}; map it explicitly"
        got = vars(module._parse_args(_GOLD_REQUIRED_ARGS + tokens))[dest]
        if got != defaults[dest]:
            differ[key] = (got, defaults[dest])
    assert differ == _DOCUMENTED_TEMPLATE_DIVERGENCES


def test_every_defaulted_renderer_flag_is_accounted_for():
    """Guards the test above from going vacuous: a renderer option with a default is either
    passed from the template's config or listed, with its reason, as not passed."""
    required = {_dest(flag) for flag in _GOLD_REQUIRED_ARGS[::2]}
    defaulted = set(vars(_gold_module()._parse_args(_GOLD_REQUIRED_ARGS))) - required
    passed = {_dest(tokens[0]) for _, tokens in _template_flags()}
    excused = {_dest(flag) for flag in _RENDERER_DEFAULTS_THE_TEMPLATE_DOES_NOT_PASS}
    assert defaulted - passed == excused


def test_the_template_passes_the_batch_shape_to_the_renderer():
    keys = {key for key, _ in _template_flags()}
    assert {
        "gold_config.gradient_accumulation_steps",
        "gold_config.learning_rate",
        "gold_config.max_length",
        "gold_config.num_train_epochs",
        "gold_config.per_device_train_batch_size",
        "gold_config.response_template",
        "gold_config.save_steps",
    } <= keys


class TestTrl1Defaults:
    """Fields trl 1.x / transformers 5.16+ changed under us, stated explicitly (#676)."""

    def test_student_and_teacher_dtype_is_bfloat16(self, tmp_path):
        # ModelConfig.dtype defaults to "float32" in trl 1.x; gold.py loads both models with it.
        assert _render(tmp_path)["dtype"] == "bfloat16"
        on = _render(tmp_path, extra=["--vllm-num-servers", "1", "--lmbda", "1.0"])
        assert on["dtype"] == "bfloat16"

    def test_on_policy_states_server_mode(self, tmp_path):
        # trl 1.x defaults vllm_mode to "colocate"; every on-policy topology here uses a
        # separate server (in-allocation nodes or an external target).
        config = _render(tmp_path, extra=["--vllm-num-servers", "1", "--lmbda", "1.0"])
        assert config["vllm_mode"] == "server"

    def test_external_server_states_server_mode(self, tmp_path):
        config = _render(
            tmp_path,
            total_nodes=1,
            extra=[
                "--vllm-num-servers",
                "1",
                "--lmbda",
                "1.0",
                "--vllm-server-url",
                "http://10.0.0.1:8001",
            ],
        )
        assert config["vllm_mode"] == "server"

    def test_off_policy_does_not_emit_a_mode(self, tmp_path):
        # use_vllm is False off-policy, so the mode is never read.
        assert "vllm_mode" not in _render(tmp_path)

    def test_warmup_ratio_is_emitted_as_fractional_warmup_steps(self, tmp_path):
        config = _render(tmp_path, extra=["--warmup-ratio", "0.03"])
        assert "warmup_ratio" not in config
        assert config["warmup_steps"] == 0.03
        assert isinstance(config["warmup_steps"], float)

    def test_default_warmup_is_the_reference_ratio(self, tmp_path):
        assert _render(tmp_path)["warmup_steps"] == 0.05

    @pytest.mark.parametrize("ratio", ["1.0", "2", "-0.1"])
    def test_a_ratio_outside_the_unit_interval_is_refused(self, tmp_path, ratio):
        # warmup_steps >= 1 means a step count: a ratio of 1 would become ONE step.
        result = _render(tmp_path, extra=["--warmup-ratio", ratio], expect_rc=2)
        assert "warmup_ratio must be in [0, 1)" in result.stderr

    def test_every_emitted_upstream_key_exists_in_the_trl_and_transformers_dataclasses(
        self, tmp_path
    ):
        """TrlParser rejects unknown top-level keys, after the models load. Checked against
        the installed trl/transformers when they are importable (the step image), so a
        future removal like warmup_ratio fails here instead."""
        pytest.importorskip("trl")
        import dataclasses

        from transformers import TrainingArguments
        from trl import ModelConfig, ScriptArguments
        from trl.experimental.gold import GOLDConfig

        known = set()
        for cls in (TrainingArguments, ModelConfig, ScriptArguments, GOLDConfig):
            known |= {f.name for f in dataclasses.fields(cls)}
        try:
            from gb_steps_post_training.distillation.custom_gold_config import (
                CustomGOLDConfig,
            )

            known |= {f.name for f in dataclasses.fields(CustomGOLDConfig)}
            custom_known = True
        except ImportError:
            custom_known = False
        on = _render(tmp_path, extra=["--vllm-num-servers", "1", "--lmbda", "1.0"])
        unknown = set(on) - known
        if not custom_known:
            # Our extension keys need the in-repo CustomGOLDConfig to resolve.
            unknown -= {
                "use_liger_fused_jsd",
                "response_template",
                "vllm_num_servers",
                "use_sampled_opd_loss",
                "last_message_only",
                "clip_alpha",
                "opd_importance_sampling",
            }
        assert not unknown, f"keys no dataclass defines: {sorted(unknown)}"
