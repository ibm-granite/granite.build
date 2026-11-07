"""Tests for Shadow Residual (SR) support -- ``--tuning_algo sr``.

Covers:
- constants registration (sr is a known tuning algo mapping to the "SR" sentinel)
- the shipped autotune.yaml ``sr`` tuner section and the two dedicated SR configs
- ``build_sr_lora_config`` -- the projection of SR's search space onto LoraConfig
- the ``get_peft_config`` SR branch, which guards a **silent** degradation to SFT
- ``resolve_fsdp_strategy`` / ``assert_sr_supported`` / ``assert_no_vocab_resize``
- the per-driver rejections (DeepSpeed and the three TRL drivers)

Pure-logic tests: no model loading, no GPU, and no ``shadow_residual`` install
required -- ``autotune.shadow_residual`` imports the package lazily, so
everything here exercises fm-tune's own side of the seam.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from peft import LoraConfig, PeftType

from autotune.config import AutotuneConfig
from autotune.constants import (
    AUTOTUNE_TUNING_ALGO,
    AUTOTUNE_TUNING_TO_PEFT_TYPE,
    MLX_SUPPORTED_TUNING_ALGO,
    AutotuneTuningTypes,
)
from autotune.pipeline import AutotunePipeline
from autotune.shadow_residual import (
    SR_PEFT_TYPE,
    SR_TARGET_MODULES,
    SR_TUNING_ALGO,
    SR_WRAP_CLASS,
    assert_no_vocab_resize,
    assert_sr_supported,
    build_sr_lora_config,
    is_sr,
    resolve_fsdp_strategy,
    share_moe_routing_flag,
)
from autotune.utils import get_param_space

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "autotune" / "configs"
AUTOTUNE_YAML = CONFIG_DIR / "autotune.yaml"
SR_CONFIGS = [CONFIG_DIR / "autotune_sr_3b.yaml", CONFIG_DIR / "autotune_sr_smoke_1gpu.yaml"]

# The four keys autotune/utils.py::get_param_space actually reads.
LOAD_BEARING_KEYS = ("strategy", "values", "default", "for_tuner")


class TestConstants:
    def test_sr_is_a_known_algo(self):
        assert SR_TUNING_ALGO == "sr"
        assert SR_TUNING_ALGO in AUTOTUNE_TUNING_ALGO

    def test_sr_maps_to_the_string_sentinel(self):
        # Not a PeftType member: SR is a dual-stream *architecture* adapted with
        # stock LoRA, so it needs no PEFT type of its own. Same escape hatch as
        # "alora": "ALORA" -- the sentinel only steers fm-tune's own dispatch.
        assert AUTOTUNE_TUNING_TO_PEFT_TYPE[SR_TUNING_ALGO] == SR_PEFT_TYPE == "SR"
        assert not isinstance(AUTOTUNE_TUNING_TO_PEFT_TYPE[SR_TUNING_ALGO], PeftType)

    def test_sr_is_in_tuning_types_catalog(self):
        entry = AutotuneTuningTypes[SR_TUNING_ALGO]
        assert entry["peft_type"] == SR_PEFT_TYPE
        assert entry["tuner_name"] == "tuner.sr"

    def test_sr_is_not_mlx_supported(self):
        # Keeping sr out of this set is what makes device.py::apply_platform_guards
        # reject --backend mlx for free.
        assert SR_TUNING_ALGO not in MLX_SUPPORTED_TUNING_ALGO

    def test_wrap_class_is_the_sr_decoder_layer(self):
        # accelerate resolves this via model._no_split_modules for
        # TRANSFORMER_BASED_WRAP; a typo surfaces only at FSDP-wrap time.
        assert SR_WRAP_CLASS == "ShadowResidualDecoderLayer"

    def test_is_sr_predicate(self):
        assert is_sr({"tuning_algorithm": "sr"})
        assert not is_sr({"tuning_algorithm": "lora"})
        assert not is_sr({})
        assert not is_sr(None)


class TestShippedYamlSection:
    def _tuners(self):
        return yaml.safe_load(AUTOTUNE_YAML.read_text())["tuners_config"]

    def test_sr_section_exists(self):
        tuners = self._tuners()
        assert "sr" in tuners
        assert tuners["sr"]["title"] == "Shadow Residual"

    def test_every_hyperparam_has_the_load_bearing_keys(self):
        for name, spec in self._tuners()["sr"]["hyperparams"].items():
            for key in LOAD_BEARING_KEYS:
                assert key in spec, f"sr.{name} is missing {key!r}"

    def test_sr_specific_knobs_are_tuned(self):
        hp = self._tuners()["sr"]["hyperparams"]
        assert hp["r"]["for_tuner"] is True
        assert hp["cross_stream_r"]["for_tuner"] is True

    def test_alpha_ratio_is_for_tuner_unlike_every_other_tuner(self):
        # Load-bearing: for_tuner routes alpha_ratio into peft_kwargs, so the
        # drivers' train_kwargs.pop("alpha_ratio") returns None and they do NOT
        # compute lora_alpha = alpha_ratio * r. SR's `r` is the *minimum* rank,
        # so that formula would scale from min instead of SR's max.
        tuners = self._tuners()
        assert tuners["sr"]["hyperparams"]["alpha_ratio"]["for_tuner"] is True
        assert tuners["lora"]["hyperparams"]["alpha_ratio"]["for_tuner"] is False

    def test_no_weight_decay_hyperparam(self):
        # driver_single.py passes weight_decay to TrainingArguments explicitly; a
        # for_tuner: false "weight_decay" would be splatted in on top of it and
        # raise TypeError: got multiple values for keyword argument.
        assert "weight_decay" not in self._tuners()["sr"]["hyperparams"]

    def test_config_lookup_finds_sr(self):
        cfg = AutotuneConfig()
        cfg.load(str(AUTOTUNE_YAML))
        section = cfg.get_tuner_config_dict("sr")
        assert section
        assert section["title"] == "Shadow Residual"


class TestSrBlockIsReplicatedEverywhere:
    """Every shipped config that offers ``lora`` must also offer ``sr``.

    A missing ``sr:`` block is silent, not loud: ``get_tuner_config_dict``
    returns ``{}`` for an unknown key, ``get_param_space({})`` returns three
    empty dicts, and ``build_sr_lora_config({})`` then falls back to its own
    defaults (r=32, cross_stream_r=32, alpha=64). So ``--tuning_algo sr`` on
    such a config still trains a *valid* SR adapter -- it just trains exactly
    one fixed configuration while appearing to run a search. HPO silently
    becomes a no-op.
    """

    # The canonical search space, taken from autotune.yaml rather than restated,
    # so widening the space in one place cannot leave the replicas behind.
    def _canonical(self):
        hp = yaml.safe_load(AUTOTUNE_YAML.read_text())["tuners_config"]["sr"]["hyperparams"]
        return {name: spec["for_tuner"] for name, spec in hp.items()}

    def _all_shipped(self):
        """(label, parsed tuners_config) for every config that ships a tuner set."""
        out = []
        for path in sorted(CONFIG_DIR.glob("*.yaml")):
            doc = yaml.safe_load(path.read_text())
            if isinstance(doc, dict) and doc.get("tuners_config"):
                out.append((path.name, doc["tuners_config"]))
        return out

    def test_there_are_configs_to_check(self):
        # Guards the guard: a globbing mistake would make every assertion below
        # vacuous. The bound is deliberately well under the current inventory
        # (retiring a config is routine -- the 3b/30b task variants were folded
        # into one 8b each, and the 8b task configs were later dropped), because
        # what it has to catch is a glob that matches nothing or almost nothing,
        # not a shrinking config set.
        labels = [label for label, _ in self._all_shipped()]
        assert len(labels) >= 5, labels

    def test_every_config_with_lora_also_has_sr(self):
        missing = [label for label, tuners in self._all_shipped() if "lora" in tuners and "sr" not in tuners]
        assert not missing, f"configs offering lora but not sr: {missing}"

    def test_replicas_carry_the_canonical_search_space(self):
        canonical = self._canonical()
        for label, tuners in self._all_shipped():
            if "sr" not in tuners:
                continue
            got = {name: spec["for_tuner"] for name, spec in tuners["sr"]["hyperparams"].items()}
            assert got == canonical, f"{label} sr space diverges from autotune.yaml"

    def test_replicas_keep_the_load_bearing_keys(self):
        for label, tuners in self._all_shipped():
            if "sr" not in tuners:
                continue
            for name, spec in tuners["sr"]["hyperparams"].items():
                for key in LOAD_BEARING_KEYS:
                    assert key in spec, f"{label}: sr.{name} is missing {key!r}"

    def test_replicas_resolve_through_get_param_space(self):
        # The end-to-end path a driver actually takes: a replica must produce a
        # non-empty search space with alpha_ratio in the tuned bucket.
        for label, tuners in self._all_shipped():
            if "sr" not in tuners:
                continue
            space, defaults, flags = get_param_space(tuners["sr"])
            assert space, f"{label}: empty sr search space"
            assert flags["alpha_ratio"] is True, label
            assert flags["cross_stream_r"] is True, label
            assert flags["learning_rate"] is False, label


class TestBuildSrLoraConfig:
    def test_returns_a_plain_lora_config(self):
        config = build_sr_lora_config({"r": 32, "cross_stream_r": 32})
        assert isinstance(config, LoraConfig)
        # Everything downstream that inspects peft_config.r / .lora_alpha keeps
        # working precisely because this stays a stock LoraConfig.
        assert config.peft_type == PeftType.LORA
        assert config.task_type == "CAUSAL_LM"

    def test_targets_are_the_explicit_sr_set(self):
        config = build_sr_lora_config({"r": 16})
        assert set(config.target_modules) == set(SR_TARGET_MODULES)
        assert "cross_stream" in config.target_modules

    def test_never_all_linear(self):
        # SR's factory rejects "all-linear" outright: it expands to every
        # nn.Linear, including the forbidden K/V projections.
        config = build_sr_lora_config({"r": 16})
        assert config.target_modules != "all-linear"
        assert not isinstance(config.target_modules, str)

    def test_no_kv_projections(self):
        # Shared base-only K/V: there is no path for an adapter-side K/V delta to
        # land, so SR's _reject_kv_lora raises on these names.
        config = build_sr_lora_config({"r": 64, "cross_stream_r": 16})
        assert not ({"k_proj", "v_proj"} & set(config.target_modules))
        assert not ({"k_proj", "v_proj"} & set(config.rank_pattern))

    def test_scalar_r_is_the_minimum_rank(self):
        config = build_sr_lora_config({"r": 64, "cross_stream_r": 16})
        assert config.r == 16

    def test_rank_pattern_names_every_target(self):
        # Not tidiness: with r == min(...), a target missing from rank_pattern
        # would silently train at the minimum rank instead of the sampled one.
        config = build_sr_lora_config({"r": 64, "cross_stream_r": 16})
        assert set(config.rank_pattern) == set(SR_TARGET_MODULES)
        assert config.rank_pattern["cross_stream"] == 16
        for name in SR_TARGET_MODULES:
            if name != "cross_stream":
                assert config.rank_pattern[name] == 64, name

    def test_cross_stream_rank_override(self):
        config = build_sr_lora_config({"r": 32, "cross_stream_r": 96})
        assert config.rank_pattern["cross_stream"] == 96
        assert config.r == 32

    def test_alpha_is_ratio_times_max_rank(self):
        # SR's own default (AdapterConfig._default_alpha) is 2 * max_rank.
        config = build_sr_lora_config({"r": 32, "cross_stream_r": 96, "alpha_ratio": 2.0})
        assert config.lora_alpha == 192
        config = build_sr_lora_config({"r": 64, "cross_stream_r": 16, "alpha_ratio": 1.0})
        assert config.lora_alpha == 64

    def test_alpha_ratio_defaults_to_two(self):
        config = build_sr_lora_config({"r": 32, "cross_stream_r": 32})
        assert config.lora_alpha == 64

    def test_cross_stream_r_defaults_to_r(self):
        config = build_sr_lora_config({"r": 48})
        assert config.rank_pattern["cross_stream"] == 48
        assert config.r == 48

    def test_dropout_and_bias_are_forwarded(self):
        config = build_sr_lora_config({"r": 32, "lora_dropout": 0.1, "bias": "none"})
        assert config.lora_dropout == 0.1
        assert config.bias == "none"

    def test_every_shipped_sample_builds(self):
        # Each combination the shipped search space can sample must produce a
        # valid config -- including cross_stream_r < r, which flips which knob
        # owns the scalar default.
        hp = yaml.safe_load(AUTOTUNE_YAML.read_text())["tuners_config"]["sr"]["hyperparams"]
        for r in hp["r"]["values"]:
            for csr in hp["cross_stream_r"]["values"]:
                for ratio in hp["alpha_ratio"]["values"]:
                    config = build_sr_lora_config({"r": r, "cross_stream_r": csr, "alpha_ratio": ratio})
                    assert config.r == min(r, csr)
                    assert config.lora_alpha == int(ratio * max(r, csr))


class TestGetPeftConfigBranch:
    def test_sr_peft_type_returns_a_lora_config(self):
        # THE regression guard for this port: get_peft_config ends in
        # `else: peft_config = None  # SFT`, so a missing SR branch degrades an
        # SR run to a silent full fine-tune with no error anywhere.
        from autotune.utils import get_peft_config

        config = get_peft_config(
            model=None,
            model_name_or_path="ibm-granite/granite-4.1-3b",
            peft_type=SR_PEFT_TYPE,
            base_kwargs={"r": 32, "cross_stream_r": 32, "alpha_ratio": 2.0},
        )
        assert isinstance(config, LoraConfig)
        assert "cross_stream" in config.target_modules

    def test_sr_branch_needs_no_model(self):
        # The LORA/LOHA/LOKR/VERA branches call get_adapter_compatible_modules(model);
        # SR must not, because the model does not exist yet at that point (it is
        # built from this very config).
        from autotune.utils import get_peft_config

        config = get_peft_config(
            model=None,
            model_name_or_path="ibm-granite/granite-4.1-3b",
            peft_type=SR_PEFT_TYPE,
            base_kwargs={"r": 16},
        )
        assert config.r == 16

    def test_sft_still_falls_through_to_none(self):
        from autotune.utils import get_peft_config

        assert (
            get_peft_config(
                model=None,
                model_name_or_path="ibm-granite/granite-4.1-3b",
                peft_type=None,
                base_kwargs={},
            )
            is None
        )

    def test_pipeline_resolves_sr_to_the_sentinel(self):
        pipeline = AutotunePipeline(tuning_algo="sr", rl_algo="none", model_name_or_path="ibm-granite/granite-4.1-3b")
        assert pipeline.get_tuning_algo() == "sr"
        assert pipeline.get_peft_type() == SR_PEFT_TYPE


class TestResolveFsdpStrategy:
    def test_auto_becomes_full_shard(self):
        # estimate_fsdp_strategy sizes trainable params from a single-stream LoRA
        # assumption and under-counts SR, answering no_shard on a 3B.
        assert resolve_fsdp_strategy("auto") == "full_shard"

    def test_empty_is_rejected(self):
        # Not a strategy at all -- every other caller would read it as "no FSDP",
        # which is what "no_shard" says explicitly.
        with pytest.raises(ValueError, match="empty fsdp_strategy"):
            resolve_fsdp_strategy("")

    def test_no_shard_passes_through_with_a_warning(self, caplog):
        # fm-tune's "no_shard" is *no FSDP at all* (_build_fsdp_config returns
        # ("", {})), i.e. Accelerate's DDP wrap -- not torch's
        # ShardingStrategy.NO_SHARD, which is what breaks against
        # fsdp_use_orig_params and which fm-tune never emits. SR's factory
        # materializes on every rank when ACCELERATE_USE_FSDP is unset, and DDP's
        # constructor broadcasts rank 0's parameters to the others, so the ranks
        # agree. The cost is host RAM (one materialize per rank, not per node),
        # which is a capacity limit and therefore a warning, not a raise.
        with caplog.at_level(logging.WARNING, logger="autotune.shadow_residual"):
            assert resolve_fsdp_strategy("no_shard") == "no_shard"
        assert "host-RAM" in caplog.text

    @pytest.mark.parametrize("strategy", ["full_shard", "shard_grad_op", "hybrid_shard"])
    def test_sharding_strategies_pass_through(self, strategy):
        assert resolve_fsdp_strategy(strategy) == strategy


class TestAssertSrSupported:
    CUDA = SimpleNamespace(kind="cuda")
    # What every shipped config resolves to (config.py fills the schema default).
    FSDP = {"train_implementation": "FSDP"}

    def test_accepts_cuda_hf_sft(self):
        assert_sr_supported(self.FSDP, accel=self.CUDA, backend="hf", rl_algo="none")

    def test_rejects_mlx_backend(self):
        with pytest.raises(ValueError, match="MLX"):
            assert_sr_supported({}, accel=self.CUDA, backend="mlx")

    @pytest.mark.parametrize("kind", ["mps", "cpu"])
    def test_rejects_non_cuda(self, kind):
        with pytest.raises(ValueError, match="requires CUDA"):
            assert_sr_supported({}, accel=SimpleNamespace(kind=kind), backend="hf")

    @pytest.mark.parametrize("impl", ["DeepSpeed", "huggingface_ds", "deepspeed"])
    def test_rejects_deepspeed(self, impl):
        with pytest.raises(ValueError, match="requires FSDP"):
            assert_sr_supported({}, accel=self.CUDA, train_implementation=impl)

    def test_accepts_fsdp(self):
        assert_sr_supported({}, accel=self.CUDA, train_implementation="FSDP")

    @pytest.mark.parametrize("rl_algo", ["dpo", "kto", "ppo", "grpo"])
    def test_rejects_rl(self, rl_algo):
        with pytest.raises(ValueError, match="SFT only"):
            assert_sr_supported(self.FSDP, accel=self.CUDA, rl_algo=rl_algo)

    def test_reads_train_implementation_off_the_config(self):
        with pytest.raises(ValueError, match="requires FSDP"):
            assert_sr_supported({"train_implementation": "DeepSpeed"}, accel=self.CUDA)

    def test_rejects_missing_train_implementation(self):
        # The optimizer defaults an absent key to "huggingface_ds", i.e. the
        # DeepSpeed driver, which rejects SR -- so the guard must not let it by.
        with pytest.raises(ValueError, match="requires FSDP"):
            assert_sr_supported({}, accel=self.CUDA)

    def test_explicit_kwargs_win_over_the_config(self):
        # main.py passes these explicitly: at its guard site the CLI values have
        # not been folded into training_config yet (the optimizer does that at
        # trial-launch time), so reading them off the dict alone would no-op.
        with pytest.raises(ValueError, match="SFT only"):
            assert_sr_supported(self.FSDP, accel=self.CUDA, rl_algo="dpo")


class TestAssertNoVocabResize:
    def test_zero_new_tokens_is_a_noop(self):
        assert_no_vocab_resize(0)

    def test_added_tokens_are_rejected(self):
        # transfer_base_weights slices the upstream checkpoint at its original
        # vocab size, and resizing a PEFT-wrapped (possibly still-meta) model
        # would give a crash or a garbage row that sync_module_states broadcasts.
        with pytest.raises(ValueError):
            assert_no_vocab_resize(3)


class TestShareMoeRoutingFlag:
    def test_defaults_to_true(self):
        assert share_moe_routing_flag({}) is True
        assert share_moe_routing_flag(None) is True

    def test_honours_an_explicit_false(self):
        assert share_moe_routing_flag({"sr_share_moe_routing": False}) is False


class TestDriverRejections:
    # Every driver SR must not reach. The rejection is hard rather than a
    # fallthrough because the failure would otherwise be SILENT: PEFT's
    # inject_adapter raises only when *nothing* matched, so on a plain Granite
    # base the cross_stream target is dropped without a warning and the run
    # trains as ordinary LoRA while being reported as "sr".
    TRL_DRIVERS = [
        "autotune/trainers/driver_single_trl.py",
        "autotune/trainers/driver_multi_trl_ds.py",
        "autotune/trainers/driver_multi_trl_fsdp.py",
    ]

    @pytest.mark.parametrize("path", TRL_DRIVERS)
    def test_trl_drivers_call_the_shared_sr_rejection(self, path):
        source = (REPO_ROOT / path).read_text()
        assert "reject_sr_for_trl_driver(training_config)" in source, f"{path} does not reject SR"

    def test_the_trl_rejection_raises_for_sr_only(self):
        from autotune.shadow_residual import reject_sr_for_trl_driver

        with pytest.raises(ValueError, match="Shadow Residual"):
            reject_sr_for_trl_driver({"tuning_algorithm": SR_TUNING_ALGO})
        reject_sr_for_trl_driver({"tuning_algorithm": "lora"})
        reject_sr_for_trl_driver({})
        reject_sr_for_trl_driver(None)

    SR_DRIVERS = [
        "autotune/trainers/driver_single.py",
        "autotune/trainers/driver_multi_hf_fsdp.py",
    ]

    @pytest.mark.parametrize("path", SR_DRIVERS)
    def test_sr_drivers_do_not_wrap_an_already_wrapped_model(self, path):
        # build_sr_peft_model returns an already-PEFT-wrapped model; calling
        # get_peft_model on it again would nest a second LoraModel.
        source = (REPO_ROOT / path).read_text()
        assert "build_sr_peft_model" in source
        assert "is_sr" in source

    def test_deepspeed_driver_raises_for_sr(self):
        from autotune.trainers import driver_multi_hf_ds

        config = {
            "training_config": {"tuning_algorithm": SR_TUNING_ALGO},
            "training_rl_config": {},
            "tuner_flags": {},
            "tune_config": {},
        }
        with patch.object(driver_multi_hf_ds, "tune", MagicMock()):
            with pytest.raises(ValueError, match="Shadow Residual"):
                driver_multi_hf_ds.train_driver_multi_gpu(config)


class TestSrConfigs:
    @pytest.mark.parametrize("path", SR_CONFIGS, ids=lambda p: p.name)
    def test_config_exists_and_parses(self, path):
        cfg = AutotuneConfig()
        cfg.load(str(path))
        assert cfg.get_tuner_config_dict("sr")

    @pytest.mark.parametrize("path", SR_CONFIGS, ids=lambda p: p.name)
    def test_fsdp_only_and_sharded(self, path):
        tc = yaml.safe_load(path.read_text())["training_config"]
        assert tc["train_implementation"]["default"] == "FSDP"
        # "no_shard" is allowed but must not be the default: it drops the FSDP
        # wrap entirely, so every rank materializes its own host-RAM copy of the
        # base instead of one per node. "auto" must not be the default either --
        # resolve_fsdp_strategy overrides it, but a config should say what it
        # means. See resolve_fsdp_strategy.
        assert tc["fsdp_strategy"]["default"] == "full_shard"

    @pytest.mark.parametrize("path", SR_CONFIGS, ids=lambda p: p.name)
    def test_share_moe_routing_lives_in_training_config(self, path):
        # Not a LoraConfig field and not a TrainingArguments kwarg, so a sampled
        # hyperparam would be splatted into one of those and raise TypeError.
        tc = yaml.safe_load(path.read_text())["training_config"]
        assert tc["sr_share_moe_routing"]["default"] is True
        assert "sr_share_moe_routing" not in yaml.safe_load(path.read_text())["tuners_config"]["sr"]["hyperparams"]

    @pytest.mark.parametrize("path", SR_CONFIGS, ids=lambda p: p.name)
    def test_no_vocab_extending_tokenizer_defaults(self, path):
        # assert_no_vocab_resize raises for any added token: SR's weight transfer
        # slices the upstream checkpoint at its original vocab size.
        raw = yaml.safe_load(path.read_text())
        tk = raw.get("tokenizer_config") or {}
        for key in ("additional_special_tokens", "additional_tokens"):
            if key in tk:
                assert not tk[key]["default"], f"{path.name}: {key} must default to null"

    @pytest.mark.parametrize("path", SR_CONFIGS, ids=lambda p: p.name)
    def test_no_save_only_model_under_fsdp(self, path):
        # save_only_model=True skips the optimizer state, which breaks resume;
        # the FSDP driver's checkpoint strategy owns this decision.
        tc = yaml.safe_load(path.read_text())["training_config"]
        assert "save_only_model" not in tc

    def test_smoke_config_is_sequential(self):
        # An SR HPO trial peaks at ~12-14 GB host RAM while materializing, so the
        # smoke config runs one at a time.
        tune_config = yaml.safe_load(SR_CONFIGS[1].read_text())["tune_config"]
        assert tune_config["num_samples"]["default"] == 4
        assert tune_config["max_concurrent_trials"]["default"] == 1


class TestSingleWorkerSkipsFsdp:
    """A one-worker run must not be handed an FSDP wrap.

    FSDP refuses to shard across a single rank and silently degrades
    FULL_SHARD to NO_SHARD -- and NO_SHARD combined with the
    ``fsdp_use_orig_params=True`` the driver always sets is broken in torch:
    ``_FlatParamHandle._writeback_orig_params`` skips its early return (that is
    gated on ``uses_sharded_strategy``, which NO_SHARD is not) yet still expects
    a 1-D ``torch.Size([numel_in_shard])`` while use_orig_params keeps every
    parameter at its logical shape, so the first writeback raises
    ``Cannot writeback when the parameter shape changes``. Observed on 1-GPU SR
    runs at the very first training step, from a lora_A.weight inside an
    FSDP-wrapped LoRA submodule; the 8-rank runs are unaffected.

    This matters for SR specifically because ``fit_best_config`` sets
    ``multi_gpu = accel.supports_distributed`` (True on any CUDA box), so a
    1-GPU SR run routes its *final* training to the FSDP driver with
    ``num_workers=1``.
    """

    @pytest.mark.parametrize("strategy", ["full_shard", "shard_grad_op", "hybrid_shard"])
    def test_one_worker_gets_no_fsdp(self, strategy):
        from autotune.trainers.driver_multi_hf_fsdp import _build_fsdp_config

        assert _build_fsdp_config(strategy, 1) == ("", {})

    @pytest.mark.parametrize("strategy", ["full_shard", "shard_grad_op", "hybrid_shard"])
    def test_multiple_workers_keep_fsdp(self, strategy):
        from autotune.trainers.driver_multi_hf_fsdp import _build_fsdp_config

        resolved, config = _build_fsdp_config(strategy, 8)
        assert resolved == strategy
        assert config["fsdp_use_orig_params"] is True
        assert config["fsdp_sharding_strategy"] == strategy.upper()

    def test_no_shard_still_returns_empty(self):
        from autotune.trainers.driver_multi_hf_fsdp import _build_fsdp_config

        assert _build_fsdp_config("no_shard", 8) == ("", {})


class TestWorkerSeedsBeforeBuildingTheModel:
    """``train_loop_per_worker`` must seed before it constructs any model.

    The driver's own ``set_seed`` runs in the *trial* process, not in the Ray
    Train worker processes, and HF's ``Trainer.__init__`` seeds only after the
    model already exists -- so without a seed here every rank draws its own
    kaiming-uniform ``lora_A``. Both wrappers repair that afterwards (FSDP via
    ``fsdp_sync_module_states``, DDP via the unconditional rank-0 parameter
    broadcast in ``DistributedDataParallel.__init__``), which is why the
    divergence never showed up as a failure -- but "agree by construction" is
    the property that makes a run reproducible from its seed, and it is the one
    thing ``no_shard`` (plain DDP, every rank materializing its own weights)
    leans on. Checked by source order because the real call needs a cluster.
    """

    @staticmethod
    def _worker_source() -> str:
        source = (REPO_ROOT / "autotune/trainers/driver_multi_hf_fsdp.py").read_text()
        start = source.index("def train_loop_per_worker(")
        # Ends at the next top-level def/class.
        rest = source[start:]
        for marker in ("\ndef ", "\nclass "):
            idx = rest.find(marker, 1)
            if idx != -1:
                rest = rest[:idx]
        return rest

    @pytest.mark.parametrize(
        "builder",
        ["build_sr_peft_model(", "AutoModelForCausalLM.from_pretrained(", "Trainer("],
    )
    def test_set_seed_precedes_every_model_build(self, builder):
        source = self._worker_source()
        assert "set_seed(seed)" in source, (
            "train_loop_per_worker no longer seeds the worker process; every rank would draw its own LoRA init"
        )
        assert source.index("set_seed(seed)") < source.index(builder), (
            f"set_seed(seed) must come before {builder} in train_loop_per_worker"
        )
