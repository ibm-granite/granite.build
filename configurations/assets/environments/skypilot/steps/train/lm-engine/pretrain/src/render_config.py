# SPDX-License-Identifier: Apache-2.0

"""Render lm-engine's ``config.yml`` from the step's flat knob dict.

A port of Alexei Karve's gbansible role ``lm_engine_pretrain_job`` -- its
``templates/pretrain_config.yml.j2`` and ``defaults/main.yml``, with the
granite-4 350M muP values from ``vars/lm_engine_pretrain/nemotron-g4-350m-fused-500B.yml``
as the defaults. Alexei approved the port.

It is deliberately Python (dict -> YAML) rather than a Jinja template:

* it is unit-testable, and a golden fixture captured from the role itself keeps
  this port honest (``test/test_render_config.py``); and
* it stays clear of gbserver's Jinja environment, which has no ``bool`` filter,
  keeps unknown ``{{ }}`` literal (PreserveUndefined), and strips one trailing
  newline from every rendered config value -- all of which the role's template
  relies on not happening.

KNOBS (``DEFAULT_KNOBS``, flat on purpose: ``recipes/pretrain`` mirrors these keys
upper-cased as build parameters, and the recipe passes them back as ``config.lme``):

    experiment
    data        dataset_path sub_datasets weights col_name file_type vocab_size
                bos_token eos_token drop_tokens logical_shards num_workers seed
                sequence_length enable_data_checkpoint
    COS stream  cos_stream cos_bucket cos_root cos_cache_dir cos_keep
    tokenizer   tokenizer_name
    kernels     kernels
    model       hidden_size num_layers num_heads num_kv_heads intermediate_size
                normalization_function activation_function position_embedding_type
                rope_theta tie_word_embeddings initializer_range
                bos_token_id eos_token_id pad_token_id
    muP         m_width m_emb m_residual attention_multiplier
                attention_multiplier_method
    MoE         mlp_type moe_num_experts moe_num_experts_per_tok
                moe_shared_intermediate_size moe_router_aux_loss_coef
                moe_z_loss_coef moe_normalized_topk
    SWA         sliding_window full_attention_layers
    hybrid      mamba_hybrid attention_layers mamba_state_size
                mamba_intermediate_size mamba_num_heads mamba_conv_kernel_size
                mamba_num_groups mamba_chunk_size mamba_add_bias
                mamba_use_conv_bias mamba_activation_function
                mamba_normalization_function
    training    num_training_steps micro_batch_size gradient_accumulation_steps
                lr weight_decay adam_beta1 adam_beta2 adam_eps param_groups
                lr_decay_style warmup_steps dtype save_interval
    distributed distributed_stage fsdp_algorithm gradient_checkpointing
                gradient_checkpointing_method gradient_checkpointing_num_blocks
    logging     log_interval track_model_statistics wandb_project wandb_name
                wandb_entity wandb_group wandb_notes wandb_tags

``dataset_path`` and ``tokenizer_name`` default to "" (the role's own defaults, not
the 350M vars file's internal filesystem paths): the recipe must set both.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import yaml

#: granite-4 350M muP. ``defaults/main.yml`` overridden by the 350M vars file.
DEFAULT_KNOBS: dict[str, Any] = {
    "experiment": "nemotron-g4-350m-fused-500B",
    # ── data: nemotron-full, hf_parquet, tokenized on the fly ──
    "dataset_path": "",  # required; the recipe sets it
    "sub_datasets": None,  # None -> auto-discover every subfolder at equal weight
    "weights": None,
    "col_name": "contents",
    "file_type": "hf_parquet",  # arrow | hf_parquet
    "vocab_size": 100352,  # granite-4 tiktoken vocab
    "bos_token": None,
    "eos_token": 100257,
    "drop_tokens": "",
    "logical_shards": 512,  # multiple of nodes x gpus x num_workers
    "num_workers": 4,
    "seed": 42,
    "sequence_length": 8192,
    "enable_data_checkpoint": False,
    # ── COS on-the-fly streaming (off) ──
    "cos_stream": False,
    "cos_bucket": "cos-optimal-llm-pile",
    "cos_root": "tables/ibmdatapile/internet",
    "cos_cache_dir": "/cos-cache",
    "cos_keep": 2,
    # ── tokenizer: ids must match the model ids below ──
    "tokenizer_name": "",  # required; the recipe sets it
    # ── kernels: FA2 + fused linear cross-entropy ──
    "kernels": ["flash_attention_2", "fused_linear_cross_entropy"],
    # ── architecture: granite-4.0 350m ──
    "hidden_size": 1024,
    "num_layers": 28,
    "num_heads": 16,
    "num_kv_heads": 4,
    "intermediate_size": 2048,
    "normalization_function": "rmsnorm",
    "activation_function": "swiglu",
    "position_embedding_type": "rope",
    "rope_theta": 10000000,
    "tie_word_embeddings": True,
    "initializer_range": "0.02",
    "bos_token_id": 100257,
    "eos_token_id": 100257,
    "pad_token_id": 100256,
    # ── muP: all four set together, with param_groups mup.yml ──
    "m_width": 4,
    "m_emb": 12,
    "m_residual": 0.263,
    "attention_multiplier": 0.015625,
    "attention_multiplier_method": None,  # must stay null when the multiplier is set
    # ── MoE: off (dense MLP every layer) ──
    "mlp_type": "MLP",  # MLP | MoE
    "moe_num_experts": "",
    "moe_num_experts_per_tok": "",
    "moe_shared_intermediate_size": "",
    "moe_router_aux_loss_coef": "0.01",
    "moe_z_loss_coef": "0.0001",
    "moe_normalized_topk": True,
    # ── sliding-window attention: off (full attention on every layer) ──
    "sliding_window": "",
    "full_attention_layers": [],
    # ── hybrid Mamba2 (granite-4.0-h): off ──
    "mamba_hybrid": False,
    "attention_layers": [],  # indices kept softmax_attention when hybrid
    "mamba_state_size": 128,
    "mamba_intermediate_size": 1536,
    "mamba_num_heads": 48,
    "mamba_conv_kernel_size": 4,
    "mamba_num_groups": 1,
    "mamba_chunk_size": 256,
    "mamba_add_bias": False,
    "mamba_use_conv_bias": True,
    "mamba_activation_function": "silu",
    "mamba_normalization_function": "rmsnorm",
    # ── training: ~500B tokens, muP base LR 0.01 (mup.yml scales it by 1/m_width) ──
    "num_training_steps": 120000,
    "micro_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "lr": "0.01",
    "weight_decay": "0.1",
    "adam_beta1": "0.9",
    "adam_beta2": "0.95",
    "adam_eps": "1e-10",  # the muP value; standard-param models need 1e-8
    "param_groups": "configs/param-groups/mup.yml",
    "lr_decay_style": "cosine",
    "warmup_steps": 2000,
    "dtype": "bf16",
    "save_interval": 10000,
    # ── distributed ──
    "distributed_stage": 0,  # "" -> omit distributed_args entirely
    "fsdp_algorithm": 2,
    "gradient_checkpointing": False,
    "gradient_checkpointing_method": "block",
    "gradient_checkpointing_num_blocks": "",  # "" -> checkpoint every block
    # ── logging / W&B (off: wandb_project "" disables tracking) ──
    "log_interval": 10,
    "track_model_statistics": False,
    "wandb_project": "",
    "wandb_name": "",
    "wandb_entity": "",
    "wandb_group": "",
    "wandb_notes": "",
    "wandb_tags": [],
}

MUP_KNOBS = ("m_width", "m_emb", "m_residual", "attention_multiplier")
FILE_TYPES = ("arrow", "hf_parquet")
_MUP_PARAM_GROUPS = "mup.yml"
_SP_PARAM_GROUPS = "normal.yml"


# ── helpers ─────────────────────────────────────────────────────────────────────


def _num(value: Any) -> Any:
    """Coerce a numeric-looking string to int/float, as the role's YAML did.

    The role wrote several scalars as quoted strings (``lme_lr: "0.01"``) and the
    template interpolated them unquoted, so YAML re-read them as numbers. Keep
    that, and the knobs can carry either form.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return int(text)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
    return value


def _is_set(value: Any) -> bool:
    """Whether an optional knob carries a value.

    "", None and the literal strings "null"/"none" all mean unset: the role used
    the string forms (Ansible has no untyped null in ``-e`` overrides) and a
    recipe parameter arrives as a string too.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("", "null", "none")
    if isinstance(value, (list, tuple, dict)):
        return len(value) > 0
    return True


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "yes", "on", "1", "y")


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


# ── validate ────────────────────────────────────────────────────────────────────


def validate(knobs: dict, *, num_nodes: int, gpus_per_node: int) -> list[str]:
    """Return human-readable errors (empty = ok)."""
    errors: list[str] = []

    # 1. muP is all-or-nothing: a partial set silently trains a different model.
    set_mup = [k for k in MUP_KNOBS if _is_set(knobs.get(k))]
    if set_mup and len(set_mup) != len(MUP_KNOBS):
        missing = [k for k in MUP_KNOBS if k not in set_mup]
        errors.append(
            "muP knobs must be all set or all null: "
            f"set={sorted(set_mup)} null={sorted(missing)}"
        )

    # 2. ... and the optimizer param groups must agree with it.
    param_groups = str(knobs.get("param_groups") or "")
    if set_mup and not param_groups.endswith(_MUP_PARAM_GROUPS):
        errors.append(
            f"muP knobs are set, so param_groups must end with {_MUP_PARAM_GROUPS!r} "
            f"(got {param_groups!r})"
        )
    if not set_mup and not param_groups.endswith(_SP_PARAM_GROUPS):
        errors.append(
            f"no muP knobs are set, so param_groups must end with {_SP_PARAM_GROUPS!r} "
            f"(got {param_groups!r})"
        )

    # 3. lm-engine takes an explicit multiplier OR a method, never both.
    if _is_set(knobs.get("attention_multiplier")) and _is_set(
        knobs.get("attention_multiplier_method")
    ):
        errors.append(
            "attention_multiplier is set, so attention_multiplier_method must be null "
            f"(got {knobs.get('attention_multiplier_method')!r})"
        )

    # 4. GQA: every KV head must serve a whole number of query heads.
    num_heads = _int(knobs.get("num_heads"))
    num_kv_heads = _int(knobs.get("num_kv_heads"))
    if num_kv_heads <= 0:
        errors.append(f"num_kv_heads must be > 0 (got {knobs.get('num_kv_heads')!r})")
    elif num_heads % num_kv_heads:
        errors.append(
            f"num_heads ({num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )

    # 5. The scalable dataloader shards across dp_world_size x num_workers.
    num_workers = _int(knobs.get("num_workers"))
    if "num_workers" not in knobs or num_workers <= 0:
        errors.append(f"num_workers must be > 0 (got {knobs.get('num_workers')!r})")
    else:
        divisor = num_nodes * gpus_per_node * num_workers
        logical_shards = _int(knobs.get("logical_shards"))
        if divisor <= 0:
            errors.append(
                f"num_nodes ({num_nodes}) x gpus_per_node ({gpus_per_node}) must be > 0"
            )
        elif logical_shards % divisor:
            errors.append(
                f"logical_shards ({logical_shards}) must be a multiple of "
                f"num_nodes x gpus_per_node x num_workers "
                f"({num_nodes} x {gpus_per_node} x {num_workers} = {divisor})"
            )

    # 6. A save_interval past the end of the run writes no checkpoint at all.
    #    It need NOT divide num_training_steps -- the recipe checks its rungs.
    save_interval = _int(knobs.get("save_interval"))
    num_training_steps = _int(knobs.get("num_training_steps"))
    if save_interval <= 0:
        errors.append(f"save_interval must be > 0 (got {knobs.get('save_interval')!r})")
    elif save_interval > num_training_steps:
        errors.append(
            f"save_interval ({save_interval}) must be <= num_training_steps "
            f"({num_training_steps}), or the run saves nothing"
        )

    # 7. The arrow reader takes pre-tokenized ids from a column named "tokens".
    file_type = knobs.get("file_type")
    if file_type not in FILE_TYPES:
        errors.append(
            f"file_type must be one of {list(FILE_TYPES)} (got {file_type!r})"
        )
    elif file_type == "arrow" and knobs.get("col_name") != "tokens":
        errors.append(
            "file_type 'arrow' requires col_name 'tokens' "
            f"(got {knobs.get('col_name')!r})"
        )

    return errors


# ── render ──────────────────────────────────────────────────────────────────────


def _datasets(k: dict, train_dir: str) -> list[dict]:
    """``class_name`` routes to the restored FSDPDataset / IBM scalable loader.

    ``data_name`` and ``data_sampling_ratio`` are unused there but must be present.
    """
    class_args: dict[str, Any] = {
        "dataset_path": k["dataset_path"],
        "datasets": k["sub_datasets"] if _is_set(k.get("sub_datasets")) else None,
        "weights": k["weights"] if _is_set(k.get("weights")) else None,
        "col_name": k["col_name"],
        "file_type": k["file_type"],
        "vocab_size": _num(k["vocab_size"]),
        "bos_token": _num(k["bos_token"]) if _is_set(k.get("bos_token")) else None,
        "eos_token": _num(k["eos_token"]) if _is_set(k.get("eos_token")) else None,
        "drop_tokens": str(k.get("drop_tokens", "")),
        "logical_shards": _num(k["logical_shards"]),
        "num_workers": _num(k["num_workers"]),
        "seed": _num(k["seed"]),
        "sequence_length": _num(k["sequence_length"]),
        "enable_checkpoint": _flag(k.get("enable_data_checkpoint")),
        "checkpoint_model_weights_only": False,
        "checkpoint_folder": f"{train_dir}/dataloader_ckpt",
    }
    if _flag(k.get("cos_stream")):
        # COSParquetHandler: dataset_path is the manifest root (placeholders +
        # meta/counts.csv); shards are pulled from COS on open() and evicted after use.
        class_args.update(
            cos_stream=True,
            cos_bucket=k["cos_bucket"],
            cos_root=k["cos_root"],
            cos_cache_dir=k["cos_cache_dir"],
            cos_keep=_num(k.get("cos_keep", 2)),
        )
    return [
        {
            "class_name": "FSDPDataset",
            "data_name": k["experiment"],
            "data_sampling_ratio": 1,
            "class_args": class_args,
        }
    ]


def _sequence_mixer_block(k: dict, layer: int) -> dict:
    hybrid = _flag(k.get("mamba_hybrid"))
    attention_layers = [_int(i) for i in (k.get("attention_layers") or [])]
    if hybrid and layer not in attention_layers:
        # Mamba2 SSM layer (granite-4.0-h hybrid). The CUDA path needs
        # mamba-ssm + causal-conv1d in the image; without them lm-engine falls
        # back to the pure-torch chunked SSD scan (correct, much slower).
        return {
            "sequence_mixer_type": "mamba2",
            "state_size": _num(k["mamba_state_size"]),
            "intermediate_size": _num(k["mamba_intermediate_size"]),
            "num_heads": _num(k["mamba_num_heads"]),
            "conv_kernel_size": _num(k["mamba_conv_kernel_size"]),
            "num_groups": _num(k["mamba_num_groups"]),
            "chunk_size": _num(k.get("mamba_chunk_size", 256)),
            "add_bias": _flag(k.get("mamba_add_bias")),
            "use_conv_bias": _flag(k.get("mamba_use_conv_bias")),
            "activation_function": k.get("mamba_activation_function") or "silu",
            "normalization_function": k.get("mamba_normalization_function")
            or "rmsnorm",
        }

    block: dict[str, Any] = {
        "sequence_mixer_type": "softmax_attention",
        "num_attention_heads": _num(k["num_heads"]),
        "num_key_value_heads": _num(k["num_kv_heads"]),
        "add_bias": False,
    }
    if _is_set(k.get("attention_multiplier")):
        block["attention_multiplier"] = _num(k["attention_multiplier"])
        # Explicit multiplier and method are mutually exclusive (validate rule 3).
        block["attention_multiplier_method"] = None
    full_attention_layers = [_int(i) for i in (k.get("full_attention_layers") or [])]
    if _is_set(k.get("sliding_window")) and layer not in full_attention_layers:
        block["sliding_window"] = _num(k["sliding_window"])
    return block


def _mlp_block(k: dict) -> dict:
    if str(k.get("mlp_type") or "MLP") == "MoE":
        return {
            "mlp_type": "MoE",
            "activation_function": k["activation_function"],
            "intermediate_size": _num(k["intermediate_size"]),
            "shared_intermediate_size": _num(k["moe_shared_intermediate_size"]),
            "num_experts": _num(k["moe_num_experts"]),
            "num_experts_per_tok": _num(k["moe_num_experts_per_tok"]),
            "router_aux_loss_coefficient": _num(
                k.get("moe_router_aux_loss_coef", "0.001")
            ),
            "z_loss_coefficient": _num(k.get("moe_z_loss_coef", "0.0001")),
            "normalized_topk": _flag(k.get("moe_normalized_topk", True)),
        }
    return {
        "mlp_type": "MLP",
        "activation_function": k["activation_function"],
        "intermediate_size": _num(k["intermediate_size"]),
    }


def _model_args(k: dict) -> dict:
    num_layers = _int(k["num_layers"])
    return {
        "pretrained_config": {
            "model_type": "gpt_base",
            "vocab_size": _num(k["vocab_size"]),
            "max_position_embeddings": _num(k["sequence_length"]),
            "hidden_size": _num(k["hidden_size"]),
            "num_layers": num_layers,
            "normalization_function": k["normalization_function"],
            "layer_norm_epsilon": 1e-5,
            "initializer_range": _num(k.get("initializer_range", "0.02")),
            "bos_token_id": _num(k["bos_token_id"]),
            "eos_token_id": _num(k["eos_token_id"]),
            "pad_token_id": _num(k["pad_token_id"]),
            "embedding_dropout": 0,
            "rope_scaling": None,
            "embedding_init_method": "normal",
            "use_depth_scaled_init": True,
            "tie_word_embeddings": _flag(k.get("tie_word_embeddings")),
            "m_emb": _num(k["m_emb"]) if _is_set(k.get("m_emb")) else None,
            "m_width": _num(k["m_width"]) if _is_set(k.get("m_width")) else None,
            "m_residual": (
                _num(k["m_residual"]) if _is_set(k.get("m_residual")) else None
            ),
            "init_method": "normal",
            "rope_theta": _num(k["rope_theta"]),
            "rope_dim": None,
            "sequence_mixer_blocks": [
                _sequence_mixer_block(k, i) for i in range(num_layers)
            ],
            "mlp_blocks": [_mlp_block(k) for _ in range(num_layers)],
            "position_embedding_type": k["position_embedding_type"],
        }
    }


def _logging_args(k: dict) -> dict:
    args: dict[str, Any] = {"log_interval": _num(k.get("log_interval", 10))}
    if _flag(k.get("track_model_statistics")):
        args["track_model_statistics"] = True
    if _is_set(k.get("wandb_project")):
        # W&B is the only tracker lm-engine integrates; a project name turns it on.
        wandb: dict[str, Any] = {
            "project": k["wandb_project"],
            "name": (
                k["wandb_name"] if _is_set(k.get("wandb_name")) else k["experiment"]
            ),
        }
        for knob, key in (
            ("wandb_entity", "entity"),
            ("wandb_group", "group"),
            ("wandb_notes", "notes"),
        ):
            if _is_set(k.get(knob)):
                wandb[key] = k[knob]
        if _is_set(k.get("wandb_tags")):
            wandb["tags"] = [str(t) for t in k["wandb_tags"]]
        args["wandb_args"] = wandb
    return args


def _load_args(load: dict) -> dict:
    """Resume block, from ``resume.py``'s decision.

    ``iteration`` null means lm-engine reads ``latest_checkpointed_iteration.json``.
    ``load_dataloader_state`` stays false unless the ORIGINAL run had
    ``enable_data_checkpoint``; ``load_rng_state`` only loads when the world size
    (nodes x gpus) is unchanged.
    """
    return {
        "load_path": load["load_path"],
        "iteration": (
            _num(load["iteration"]) if _is_set(load.get("iteration")) else None
        ),
        "load_optimizer": _flag(load.get("load_optimizer", True)),
        "load_lr_scheduler": _flag(load.get("load_lr_scheduler", True)),
        "load_rng_state": _flag(load.get("load_rng_state", True)),
        "load_dataloader_state": _flag(load.get("load_dataloader_state", False)),
        "load_starting_iteration": _flag(load.get("load_starting_iteration", True)),
    }


def _distributed_args(k: dict) -> dict | None:
    """Omitted entirely when ``distributed_stage`` is unset.

    lm-engine then uses its own default (plain data parallelism), which is what
    the stock nvidia pretraining examples rely on.
    """
    stage = k.get("distributed_stage")
    if stage is None or str(stage).strip() == "":
        return None
    args: dict[str, Any] = {
        "stage": _num(stage),
        "fsdp_algorithm": _num(k.get("fsdp_algorithm", 2)),
    }
    if _flag(k.get("gradient_checkpointing")):
        args["gradient_checkpointing_method"] = (
            k.get("gradient_checkpointing_method") or "block"
        )
        blocks = k.get("gradient_checkpointing_num_blocks")
        args["gradient_checkpointing_args"] = (
            {"num_blocks": _num(blocks)} if _is_set(blocks) else {}
        )
    return args


def render(knobs: dict, *, train_dir: str, load: dict | None) -> dict:
    """The lm-engine ``config.yml`` as a dict."""
    k = dict(knobs)
    config: dict[str, Any] = {
        "datasets": _datasets(k, train_dir),
        "tokenizer_args": {"tokenizer_name": k["tokenizer_name"]},
    }
    if _is_set(k.get("kernels")):
        config["kernel_args"] = {"kernels": list(k["kernels"])}
    config["model_args"] = _model_args(k)
    config["tuning_args"] = {"tuning_method": "pretraining"}
    config["save_args"] = {
        "save_path": train_dir,
        "save_interval": _num(k["save_interval"]),
    }
    if load:
        config["load_args"] = _load_args(load)
    config["logging_args"] = _logging_args(k)
    num_training_steps = _num(k["num_training_steps"])
    config["training_parameters"] = {
        "num_training_steps": num_training_steps,
        "eval_during_training": False,
        "eval_interval": num_training_steps,
        "micro_batch_size": _num(k["micro_batch_size"]),
        "gradient_accumulation_steps": _num(k["gradient_accumulation_steps"]),
        "gradient_clipping": 1,
    }
    config["optimizer_args"] = {
        "class_name": "TorchAdamW",
        "param_groups": k["param_groups"],
        "class_args": {
            "lr": _num(k["lr"]),
            "weight_decay": _num(k["weight_decay"]),
            "betas": [
                _num(k.get("adam_beta1", "0.9")),
                _num(k.get("adam_beta2", "0.95")),
            ],
            "eps": _num(k.get("adam_eps", "1e-10")),
        },
    }
    config["lr_scheduler_args"] = {
        "lr_decay_style": k["lr_decay_style"],
        "num_warmup_steps": _num(k["warmup_steps"]),
    }
    config["mixed_precision_args"] = {"dtype": k["dtype"]}
    distributed = _distributed_args(k)
    if distributed is not None:
        config["distributed_args"] = distributed
    return config


# ── CLI ─────────────────────────────────────────────────────────────────────────


def _resolve_load(args: argparse.Namespace) -> dict | None:
    """The resume block, from either ``--load`` or ``--decision``.

    ``--load`` is inline JSON; ``--decision`` is the path to the JSON file
    ``resume.py`` writes (``{"load": dict|None, "seeded": bool, "error": str|None}``).
    ``run_pretrain.sh`` uses the latter; both are accepted so the module is usable
    by hand.
    """
    if args.load is not None and args.decision is not None:
        raise SystemExit("error: pass --load or --decision, not both")
    if args.load is not None:
        return json.loads(args.load)
    if args.decision is None:
        return None
    with open(args.decision, encoding="utf-8") as handle:
        decision = json.load(handle)
    if decision.get("error"):
        raise SystemExit(f"error: resume decision failed: {decision['error']}")
    return decision.get("load")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knobs", required=True, help="the config.lme dict, as JSON")
    parser.add_argument("--train-dir", required=True, help="save_path for this run")
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--load", help="resume block, as inline JSON")
    parser.add_argument("--decision", help="path to resume.py's decision JSON")
    parser.add_argument("--out", required=True, help="where to write config.yml")
    args = parser.parse_args(argv)

    knobs = json.loads(args.knobs)
    errors = validate(knobs, num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 2

    config = render(knobs, train_dir=args.train_dir, load=_resolve_load(args))
    with open(args.out, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, default_flow_style=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
