#!/usr/bin/env python3
"""Generate a checkpoint-fanout RL build (issue #45).

Starting from a parameters file and an eval catalog, emit a build.yaml that:
  1. runs IFRL or IdentityRL GRPO training,
  2. detects every checkpoint the trainer writes (one training output per
     checkpoint step),
  3. runs a selected set of evaluations against each checkpoint, and
  4. aggregates the results into per-checkpoint CSVs plus a combined roll-up.

The build engine dispatches each downstream target exactly once, keyed by the
binding id, so the fanout must be materialized statically: this script computes
the checkpoint schedule from the RL knobs and writes one output + one eval
target-set per checkpoint.

Usage:
    python generate_build.py --workflow ifrl \\
        --parameters-path parameters.yaml \\
        --catalog-path eval-catalog.yaml \\
        --param TOTAL_EPISODES=20480 --param SAVE_FREQ=10 \\
        --param 'EVAL_SETS=[bfcl, multilingual-eval]' \\
        --output build.yaml

Any parameter in parameters.yaml can be overridden with --param KEY=VALUE
(dot notation supported, mirroring src/gbcli/utils/buildutil.py). EVAL_SETS may
be given as a YAML/JSON list string on the command line.

The generated build.yaml is a plain recipe: run it with the usual flow, e.g.
    gb build start -f build.yaml \\
        --parameters-path parameters.yaml --space <your-space>
Note the $${...} parameter placeholders in the emitted file are resolved at
`gb build start` time against parameters.yaml (this script only expands the
knobs it needs — checkpoint schedule and eval selection — and leaves the rest
as placeholders so the existing recipe workflow keeps working).
"""

import argparse
import os
import sys

import yaml

# The environment all targets run on (matches sft-eval-full-dataset / ifrl-*).
ENVIRONMENT_URI = "space://environments/skypilot/lsf/ibm-bluevela"
# The trainer emits per-checkpoint outputs named checkpoint_<step>; the id must
# match the GB_ARTIFACT_ID the openinstruct-rl step prints for each save.
CHECKPOINT_OUTPUT_PREFIX = "checkpoint_"


# ─── YAML string quoting ──────────────────────────────────────────────────────
# The build engine substitutes $${...} placeholders into the emitted YAML and
# re-parses it (src/gbcli/utils/buildutil.py:apply_parameters). Bare placeholder
# scalars break that re-parse once the substituted value contains YAML
# metacharacters (e.g. DATASET_MIXER expands to a JSON object). Match the
# hand-written recipes: double-quote every string value, and single-quote the
# fields whose substituted value embeds double quotes (dataset mixers,
# stop_strings). Mapping keys stay unquoted (plain str -> default representer).
_SINGLE_QUOTE_TOKENS = (
    "$${DATASET_MIXER}",
    "$${DATASET_EVAL_MIXER}",
    "$${STOP_STRINGS}",
)


class _DQ(str):
    """A string emitted with double-quote style."""


class _SQ(str):
    """A string emitted with single-quote style."""


yaml.SafeDumper.add_representer(
    _DQ,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(data), style='"'
    ),
)
yaml.SafeDumper.add_representer(
    _SQ,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", str(data), style="'"
    ),
)


def _quote_values(obj):
    """Recursively wrap string *values* (not keys) in a quoting style.

    Non-string scalars (ints/bools) pass through unquoted.
    """
    if isinstance(obj, dict):
        return {k: _quote_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_quote_values(v) for v in obj]
    if isinstance(obj, str):
        if any(tok in obj for tok in _SINGLE_QUOTE_TOKENS):
            return _SQ(obj)
        return _DQ(obj)
    return obj


# ─── Parameter loading + KEY=VALUE overrides ──────────────────────────────────
# Mirrors add_parameter / add_key_value in src/gbcli/utils/buildutil.py so
# --param behaves identically to `gb build start --param`.
def add_key_value(data, key, value):
    """Add a key/value pair, supporting dot notation ('a.b.c=value')."""

    def add_branch(data_rec, prefix, key_vector, value):
        key = key_vector[0]
        if not isinstance(data_rec, dict):
            raise ValueError(
                f"param {prefix}.{key} cannot be used: prefix {prefix} is already in use."
            )
        if len(key_vector) == 1:
            data_rec[key] = value
        else:
            data_rec[key] = add_branch(
                data_rec.get(key, {}), f"{prefix}.{key}", key_vector[1:], value
            )
        return data_rec

    return add_branch(data, "", key.split("."), value)


def apply_override(data, param):
    """Apply one 'key=value' override; value is parsed as YAML for typing/lists."""
    key, sep, raw = param.partition("=")
    if not sep:
        raise ValueError(f"Invalid parameter {param!r}. Use the format 'key=value'.")
    # Parse the value as YAML so lists ('[a, b]'), ints, and bools are typed;
    # falls back to the raw string for plain identifiers.
    try:
        value = yaml.safe_load(raw.strip())
    except yaml.YAMLError:
        value = raw.strip()
    return add_key_value(data, key.strip(), value)


def load_params(parameters_path, overrides):
    with open(parameters_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
    for param in overrides:
        params = apply_override(params, param)
    return params


# ─── Checkpoint schedule ──────────────────────────────────────────────────────
def compute_checkpoint_steps(params):
    """Return the list of optimizer-update indices at which a checkpoint is saved.

    open-instruct floor-divides:
      num_updates = TOTAL_EPISODES // (NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT)
    and writes a checkpoint every SAVE_FREQ updates. We evaluate each such
    checkpoint (steps SAVE_FREQ, 2*SAVE_FREQ, ... <= num_updates). If the final
    update is not a SAVE_FREQ multiple, open-instruct still writes a final
    checkpoint, so we include num_updates as well.

    NOTE (issue #45 verification): the exact on-disk naming and the update
    indices at which grpo_fast saves must be confirmed against a real run. The
    step-id convention here (checkpoint_<step>) must match the openinstruct-rl
    step's emit loop.
    """
    total_episodes = int(params["TOTAL_EPISODES"])
    prompts = int(params["NUM_UNIQUE_PROMPTS_ROLLOUT"])
    samples = int(params["NUM_SAMPLES_PER_PROMPT_ROLLOUT"])
    save_freq = int(params["SAVE_FREQ"])

    denom = prompts * samples
    if denom <= 0:
        raise ValueError(
            "NUM_UNIQUE_PROMPTS_ROLLOUT * NUM_SAMPLES_PER_PROMPT_ROLLOUT must be > 0"
        )
    num_updates = total_episodes // denom
    if num_updates < 1:
        raise ValueError(
            f"TOTAL_EPISODES={total_episodes} yields 0 optimizer updates "
            f"(denominator {denom}); nothing to checkpoint."
        )
    if save_freq < 1:
        raise ValueError("SAVE_FREQ must be >= 1")

    steps = list(range(save_freq, num_updates + 1, save_freq))
    if not steps or steps[-1] != num_updates:
        steps.append(num_updates)
    return steps


# ─── Eval-set resolution ──────────────────────────────────────────────────────
def resolve_eval_names(params, catalog):
    """Expand EVAL_SETS (set names and/or individual eval names) to eval names.

    Preserves first-seen order and de-dups. Raises on unknown names.
    """
    raw = params.get("EVAL_SETS", [])
    if isinstance(raw, str):
        parsed = yaml.safe_load(raw)
        raw = parsed if isinstance(parsed, list) else [raw]
    if not isinstance(raw, list):
        raise ValueError(f"EVAL_SETS must be a list, got {type(raw).__name__}")

    evals = catalog["evals"]
    sets = catalog.get("sets", {})
    resolved = []
    seen = set()
    for name in raw:
        if name in sets:
            candidates = sets[name]
        elif name in evals:
            candidates = [name]
        else:
            known = sorted(set(sets) | set(evals))
            raise ValueError(
                f"Unknown EVAL_SETS entry {name!r}. Known sets/evals: {', '.join(known)}"
            )
        for eval_name in candidates:
            if eval_name not in evals:
                raise ValueError(f"set {name!r} references unknown eval {eval_name!r}")
            if eval_name not in seen:
                seen.add(eval_name)
                resolved.append(eval_name)
    if not resolved:
        raise ValueError("EVAL_SETS resolved to an empty selection")
    return resolved


# ─── Target builders ──────────────────────────────────────────────────────────
def _resources(accelerators_ph, queue_ph, memory_ph, cpus_ph=None):
    res = {}
    if accelerators_ph is not None:
        res["accelerators"] = accelerators_ph
    if cpus_ph is not None:
        res["cpus"] = cpus_ph
    res["cluster"] = "$${CLUSTER}"
    res["zone"] = queue_ph
    res["memory"] = memory_ph
    return {"resources": res}


def build_rm_server_target():
    return {
        "environment_uri": ENVIRONMENT_URI,
        "outputs": {
            "rm_server_url": {"uri": "mem://rm-server"},
            "cluster_name": {"uri": "mem://rm-server-cluster"},
        },
        "steps": [
            {
                "step_uri": "space://steps/rm-server",
                "config": {
                    "rm_server_config": {
                        "model_path": "$${RM_SERVER_MODEL}",
                        "idle_timeout": "$${RM_IDLE_TIMEOUT}",
                    },
                    "launcher_config": _resources(
                        "$${RM_ACCELERATORS}", "$${RL_QUEUE}", "$${RM_MEMORY}"
                    ),
                },
            }
        ],
    }


def build_code_server_target():
    return {
        "environment_uri": ENVIRONMENT_URI,
        "outputs": {
            "code_server_url": {"uri": "mem://code-server"},
            "cluster_name": {"uri": "mem://code-server-cluster"},
        },
        "steps": [
            {
                "step_uri": "space://steps/code-server",
                "config": {
                    "code_server_config": {
                        "module": "$${CODE_MODULE}",
                        "container_home": "$${CODE_CONTAINER_HOME}",
                        "container_pythonpath": "$${CODE_CONTAINER_PYTHONPATH}",
                        "container_path": "$${CODE_CONTAINER_PATH}",
                        "idle_timeout": "$${CODE_IDLE_TIMEOUT}",
                    },
                    "launcher_config": _resources(
                        None, "$${RL_QUEUE}", "$${CODE_MEMORY}", cpus_ph="$${CODE_CPUS}"
                    ),
                },
            }
        ],
    }


# The RL config block shared by IFRL and IdentityRL. Values are left as $${...}
# placeholders so parameters.yaml continues to drive them at `gb build start`.
def _rl_config(workflow):
    cfg = {
        "exp_name": "$${EXP_NAME}",
        "run_name": "$${RUN_NAME}",
        "rl_name": "$${RL_NAME}",
        "model_name_or_path": "$${MODEL_PATH}",
        "dataset_mixer": "$${DATASET_MIXER}",
        "dataset_eval_mixer": "$${DATASET_EVAL_MIXER}",
        "output_dir": "$${OUTPUT_DIR}",
        "checkpoint_state_dir": "$${CHECKPOINT_STATE_DIR}",
        # External services: IFRL binds both; IdentityRL binds only the RM URL.
        "rm_server_url": "{{ bindings.rm_url.binding.state }}",
        "beta": "$${BETA}",
        "num_unique_prompts_rollout": "$${NUM_UNIQUE_PROMPTS_ROLLOUT}",
        "num_samples_per_prompt_rollout": "$${NUM_SAMPLES_PER_PROMPT_ROLLOUT}",
        "cliprange_low": "$${CLIPRANGE_LOW}",
        "cliprange_high": "$${CLIPRANGE_HIGH}",
        "temperature": "$${TEMPERATURE}",
        "deepspeed_stage": "$${DEEPSPEED_STAGE}",
        "seed": "$${SEED}",
        "pad_token_id": "$${PAD_TOKEN_ID}",
        "per_device_train_batch_size": "$${PER_DEVICE_TRAIN_BATCH_SIZE}",
        "learning_rate": "$${LEARNING_RATE}",
        "max_prompt_token_length": "$${MAX_PROMPT_TOKEN_LENGTH}",
        "response_length": "$${RESPONSE_LENGTH}",
        "total_episodes": "$${TOTAL_EPISODES}",
        "kl_estimator": "$${KL_ESTIMATOR}",
        "pack_length": "$${PACK_LENGTH}",
        "thinking": "$${THINKING}",
        "vllm_tensor_parallel_size": "$${VLLM_TENSOR_PARALLEL_SIZE}",
        "num_learners_per_node": "$${NUM_LEARNERS_PER_NODE}",
        "vllm_num_engines": "$${VLLM_NUM_ENGINES}",
        "num_epochs": "$${NUM_EPOCHS}",
        "warmup_ratio": "$${WARMUP_RATIO}",
        "num_mini_batches": "$${NUM_MINI_BATCHES}",
        "save_freq": "$${SAVE_FREQ}",
        "eval_freq": "$${EVAL_FREQ}",
        "checkpoint_state_freq": "$${CHECKPOINT_STATE_FREQ}",
        "ref_policy_update_freq": "$${REF_POLICY_UPDATE_FREQ}",
        "n_gram": "$${N_GRAM}",
        "allowed_freq": "$${ALLOWED_FREQ}",
        "temp_final": "$${TEMP_FINAL}",
        "temp_change_interval": "$${TEMP_CHANGE_INTERVAL}",
        "entropy_coeff": "$${ENTROPY_COEFF}",
        "vllm_imp_ratio_cap": "$${VLLM_IMP_RATIO_CAP}",
        "gradient_checkpointing": "$${GRADIENT_CHECKPOINTING}",
        "async_mode": "$${ASYNC_MODE}",
        "apply_thinking_format_reward": "$${APPLY_THINKING_FORMAT_REWARD}",
        "apply_verifiable_reward": "$${APPLY_VERIFIABLE_REWARD}",
        "non_stop_penalty": "$${NON_STOP_PENALTY}",
        "apply_repetition_penalty": "$${APPLY_REPETITION_PENALTY}",
        "add_general_reward": "$${ADD_GENERAL_REWARD}",
        "set_weight_decay_on_bias_and_norm": "$${SET_WEIGHT_DECAY_ON_BIAS_AND_NORM}",
        "additive_format_reward": "$${ADDITIVE_FORMAT_REWARD}",
        "filter_zero_advantage": "$${FILTER_ZERO_ADVANTAGE}",
        "with_tracking": "$${WITH_TRACKING}",
        "ground_truths_key": "$${GROUND_TRUTHS_KEY}",
        "sft_messages_key": "$${SFT_MESSAGES_KEY}",
        "dataset_train_splits": "$${DATASET_TRAIN_SPLITS}",
        "dataset_eval_splits": "$${DATASET_EVAL_SPLITS}",
        "stop_strings": "$${STOP_STRINGS}",
        "importance_sampling_level": "$${IMPORTANCE_SAMPLING_LEVEL}",
        "advantage_normalization_type": "$${ADVANTAGE_NORMALIZATION_TYPE}",
        "iterator_type": "$${ITERATOR_TYPE}",
        "lr_scheduler_type": "$${LR_SCHEDULER_TYPE}",
        "loss_mode": "$${LOSS_MODE}",
        "temp_schedule_type": "$${TEMP_SCHEDULE_TYPE}",
        "dataset_local_cache_dir": "$${DATASET_LOCAL_CACHE_DIR}",
        "torchdynamo_disable": "$${TORCHDYNAMO_DISABLE}",
    }
    if workflow == "ifrl":
        cfg["code_server_url"] = "{{ bindings.code_url.binding.state }}"
    return cfg


def build_training_target(workflow, checkpoint_steps):
    inputs = {"rm_url": {"binding": "rm-server.rm_server_url"}}
    if workflow == "ifrl":
        inputs["code_url"] = {"binding": "code-server.code_server_url"}

    # One output per checkpoint step. binding.path is filled at push time with
    # the checkpoint dir the trainer emitted for that step.
    outputs = {
        f"{CHECKPOINT_OUTPUT_PREFIX}{step}": {
            "uri": "env://{{ binding.path }}",
            "type": "model",
        }
        for step in checkpoint_steps
    }

    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": inputs,
        "outputs": outputs,
        "steps": [
            {
                "step_uri": "space://steps/openinstruct-rl",
                "config": {
                    # Monitor cadence knobs (issue #45): the openinstruct-rl step
                    # reads these top-level config keys.
                    "poll_interval_seconds": "$${RL_STATUS_POLL_INTERVAL_SECONDS}",
                    "log_retrieval_mode": "periodic",
                    "log_retrieval_interval_seconds": "$${RL_LOG_SCRAPE_INTERVAL_SECONDS}",
                    # Opt into per-checkpoint artifact emission: this recipe binds
                    # the checkpoint_<step> outputs, so the step's in-run watcher
                    # must run (it's off by default for single-checkpoint recipes).
                    "emit_checkpoint_artifacts": True,
                    # How often the in-run watcher polls output_dir for newly
                    # written checkpoint dirs to emit as artifacts.
                    "checkpoint_watch_interval_seconds": "$${RL_CHECKPOINT_WATCH_INTERVAL_SECONDS}",
                    "rl_config": _rl_config(workflow),
                    "launcher_config": _resources(
                        "$${RL_ACCELERATORS}", "$${RL_QUEUE}", "$${RL_MEMORY}"
                    ),
                },
            }
        ],
    }


def _experiment_for_step(step):
    """Per-checkpoint experiment namespace so results land in distinct dirs.

    A flat NAME suffix ("<EXPERIMENT>-ckpt_<step>"), NOT a "/"-separated subdir.
    sage builds a per-eval job-script *filename* from this value
    (sage-<experiment>-<eval>.sh); a "/" turns that filename into a nested dir
    and sage's open(..., "w") fails with FileNotFoundError. Per-checkpoint
    results therefore land at <SAGE_OUTPUT_DIR>/<EXPERIMENT>-ckpt_<step>/... as
    siblings. The combined roll-up does NOT re-scan a parent tree (which is why a
    suffix is fine); it pivots the already-emitted per-checkpoint CSVs — see
    build_combined_export_target.
    """
    return "$${EXPERIMENT}-ckpt_" + str(step)


def build_sage_eval_target(eval_name, entry, step):
    experiment = _experiment_for_step(step)
    sage_cfg = {
        "gb_script": entry["gb_script"],
        "model_path": "{{ bindings.model_checkpoint.binding.path }}",
        "experiment": experiment,
        "batch_size": entry["overrides"].get("batch_size", "$${BATCH_SIZE}"),
        "max_length": entry["overrides"].get("max_length", "$${MAX_LENGTH}"),
        "num_gpus": "$${NUM_GPUS}",
        "output_dir": "$${SAGE_OUTPUT_DIR}",
        "image_id": entry["image_id"],
        "extra_env": entry["overrides"].get("extra_env", {}),
        # Eval status poll cadence (issue #45).
        "poll_interval_seconds": "$${EVAL_STATUS_POLL_INTERVAL_SECONDS}",
    }
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": {
            "model_checkpoint": {
                "binding": f"training.{CHECKPOINT_OUTPUT_PREFIX}{step}"
            }
        },
        "outputs": {
            "sage_eval_results": {
                "type": "dataset",
                "uri": f"env://$${{SAGE_OUTPUT_DIR}}/{experiment}/{entry['output_subpath']}",
            }
        },
        "steps": [
            {
                "step_uri": entry["step_uri"],
                "config": {
                    "sage_eval_config": sage_cfg,
                    "launcher_config": _resources(
                        "$${EVAL_ACCELERATORS}", "$${EVAL_QUEUE}", "$${EVAL_MEMORY}"
                    ),
                },
            }
        ],
    }


def build_bfcl_eval_target(step):
    experiment = _experiment_for_step(step)
    bfcl_cfg = {
        "model_path": "{{ bindings.model_checkpoint.binding.path }}",
        "model_id": "$${BFCL_MODEL_ID}",
        "experiment": experiment,
        "eval_name": "$${BFCL_EVAL_NAME}",
        "test_categories": "$${BFCL_TEST_CATEGORIES}",
        "num_gpus_generate": "$${BFCL_NUM_GPUS_GENERATE}",
        "num_gpus_evaluate": "$${BFCL_NUM_GPUS_EVALUATE}",
        "gpu_memory_utilization": "$${BFCL_GPU_MEMORY_UTILIZATION}",
        "output_dir": "$${BFCL_OUTPUT_DIR}",
        "poll_interval_seconds": "$${EVAL_STATUS_POLL_INTERVAL_SECONDS}",
    }
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": {
            "model_checkpoint": {
                "binding": f"training.{CHECKPOINT_OUTPUT_PREFIX}{step}"
            }
        },
        "outputs": {
            "bfcl_results": {
                "type": "dataset",
                "uri": f"env://$${{BFCL_OUTPUT_DIR}}/{experiment}/$${{BFCL_EVAL_NAME}}",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/bfcl-eval",
                "config": {
                    "bfcl_config": bfcl_cfg,
                    "launcher_config": _resources(
                        "$${EVAL_ACCELERATORS}", "$${EVAL_QUEUE}", "$${EVAL_MEMORY}"
                    ),
                },
            }
        ],
    }


def build_sage_export_target(step, sage_eval_target_names):
    """Per-checkpoint sage exporter, gated on that checkpoint's sage evals."""
    experiment = _experiment_for_step(step)
    inputs = {
        f"gate_{name}": {"binding": f"{name}.sage_eval_results"}
        for name in sage_eval_target_names
    }
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": inputs,
        "outputs": {
            "sage_export_csv": {
                "type": "dataset",
                "uri": f"env://$${{SAGE_RESULTS_DIR}}/$${{EXPERIMENT}}/ckpt_{step}-sage.csv",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/sage-export",
                "config": {
                    "sage_export_config": {
                        "experiment": experiment,
                        "export_stack": "$${EXPORT_STACK}",
                        "input_dir": "$${SAGE_OUTPUT_DIR}",
                        "output_csv": f"$${{SAGE_RESULTS_DIR}}/$${{EXPERIMENT}}/ckpt_{step}-sage.csv",
                    },
                    "launcher_config": _resources(
                        None, "$${EXPORT_QUEUE}", "$${EXPORT_MEMORY}"
                    ),
                },
            }
        ],
    }


def build_bfcl_export_target(step, bfcl_eval_target_name):
    """Per-checkpoint BFCL exporter, gated on that checkpoint's bfcl eval.

    The gate binding is the *resolved* bfcl eval target name (passed in from
    generate(), mirroring build_sage_export_target) rather than reconstructed
    here, so it stays correct if the target-name prefix or the catalog key
    changes.
    """
    experiment = _experiment_for_step(step)
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": {"gate_bfcl": {"binding": f"{bfcl_eval_target_name}.bfcl_results"}},
        "outputs": {
            "bfcl_export_csv": {
                "type": "dataset",
                "uri": f"env://$${{BFCL_RESULTS_DIR}}/$${{EXPERIMENT}}/ckpt_{step}-bfcl.csv",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/bfcl-export",
                "config": {
                    "bfcl_export_config": {
                        "experiment": experiment,
                        "eval_name": "$${BFCL_EVAL_NAME}",
                        "input_dir": "$${BFCL_OUTPUT_DIR}",
                        "output_csv": f"$${{BFCL_RESULTS_DIR}}/$${{EXPERIMENT}}/ckpt_{step}-bfcl.csv",
                        "filter": "$${BFCL_FILTER}",
                    },
                    "launcher_config": _resources(
                        None, "$${EXPORT_QUEUE}", "$${EXPORT_MEMORY}"
                    ),
                },
            }
        ],
    }


def build_combined_export_target(per_ckpt_export_bindings, has_sage, has_bfcl):
    """Final roll-up: a benchmark x checkpoint pivot of the per-checkpoint CSVs.

    Gated on every per-checkpoint export CSV (so all inputs exist), then pivots
    them into one table — rows are benchmarks (sage `model`+`metric`, plus one
    BFCL-<expt> row per bfcl eval), columns are ckpt_<step>. Unlike a sage-export
    tree re-scan, this works with the flat <EXPERIMENT>-ckpt_<step> per-checkpoint
    experiment dirs (sage-eval can't accept a "/" in the experiment name).

    Only the dirs for eval kinds actually present are set; the combined-export
    step tolerates an absent/empty dir on either side and errors only if BOTH
    yield no CSVs.
    """
    inputs = {
        f"gate_{i}": {"binding": binding}
        for i, binding in enumerate(per_ckpt_export_bindings)
    }
    cfg = {
        "sage_input_dir": "$${SAGE_RESULTS_DIR}/$${EXPERIMENT}" if has_sage else "",
        "bfcl_input_dir": "$${BFCL_RESULTS_DIR}/$${EXPERIMENT}" if has_bfcl else "",
        "output_csv": "$${SAGE_RESULTS_DIR}/$${EXPERIMENT}/combined.csv",
    }
    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": inputs,
        "outputs": {
            "combined_csv": {
                "type": "dataset",
                "uri": "env://$${SAGE_RESULTS_DIR}/$${EXPERIMENT}/combined.csv",
            }
        },
        "steps": [
            {
                "step_uri": "space://steps/combined-export",
                "config": {
                    "combined_export_config": cfg,
                    "launcher_config": _resources(
                        None, "$${EXPORT_QUEUE}", "$${EXPORT_MEMORY}"
                    ),
                },
            }
        ],
    }


def build_teardown_target(workflow, last_checkpoint_step):
    """Teardown target that shuts down server clusters once training completes."""
    inputs = {
        "gate": {
            "binding": f"training.{CHECKPOINT_OUTPUT_PREFIX}{last_checkpoint_step}"
        },
        "rm_cluster": {"binding": "rm-server.cluster_name"},
    }
    cluster_names = ["{{ bindings.rm_cluster.binding.state }}"]
    if workflow == "ifrl":
        inputs["code_cluster"] = {"binding": "code-server.cluster_name"}
        cluster_names.append("{{ bindings.code_cluster.binding.state }}")

    return {
        "environment_uri": ENVIRONMENT_URI,
        "inputs": inputs,
        "steps": [
            {
                "step_uri": "space://steps/skypilot-teardown",
                "config": {
                    "teardown_config": {
                        "cluster_names": cluster_names,
                    }
                },
            }
        ],
    }


# ─── Assembly ─────────────────────────────────────────────────────────────────
def generate(workflow, params, catalog):
    checkpoint_steps = compute_checkpoint_steps(params)
    eval_names = resolve_eval_names(params, catalog)

    targets = {}
    targets["rm-server"] = build_rm_server_target()
    if workflow == "ifrl":
        targets["code-server"] = build_code_server_target()
    targets["training"] = build_training_target(workflow, checkpoint_steps)
    targets["teardown"] = build_teardown_target(workflow, checkpoint_steps[-1])

    per_ckpt_export_bindings = []
    has_sage_export = False
    has_bfcl_export = False
    for step in checkpoint_steps:
        sage_eval_target_names = []
        bfcl_eval_target_name = None
        for eval_name in eval_names:
            entry = catalog["evals"][eval_name]
            target_name = f"eval-{eval_name}-ckpt{step}"
            if entry["category"] == "bfcl":
                targets[target_name] = build_bfcl_eval_target(step)
                bfcl_eval_target_name = target_name
            else:
                targets[target_name] = build_sage_eval_target(eval_name, entry, step)
                sage_eval_target_names.append(target_name)

        # Per-checkpoint exports, each gated on this checkpoint's evals.
        if sage_eval_target_names:
            name = f"export-sage-ckpt{step}"
            targets[name] = build_sage_export_target(step, sage_eval_target_names)
            per_ckpt_export_bindings.append(f"{name}.sage_export_csv")
            has_sage_export = True
        if bfcl_eval_target_name:
            name = f"export-bfcl-ckpt{step}"
            targets[name] = build_bfcl_export_target(step, bfcl_eval_target_name)
            per_ckpt_export_bindings.append(f"{name}.bfcl_export_csv")
            has_bfcl_export = True

    # Combined roll-up across all checkpoints, pivoting whichever eval kinds ran.
    # Only emitted when there is at least one per-checkpoint export to roll up
    # (resolve_eval_names already guarantees >=1 eval, so this holds in practice;
    # the guard keeps the combined target from being emitted with no gate inputs).
    if per_ckpt_export_bindings:
        targets["export-combined"] = build_combined_export_target(
            per_ckpt_export_bindings, has_sage_export, has_bfcl_export
        )

    build = {
        "granite.build": {
            "name": f"{workflow}-checkpoint-eval",
            "retries": {"max_retries": 5},
            # Quote string values so $${...} placeholders survive substitution
            # + re-parse; ints/bools (e.g. max_retries, catalog batch_size) stay
            # bare.
            "targets": _quote_values(targets),
        }
    }
    return build, checkpoint_steps, eval_names


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Generate a checkpoint-fanout RL build (issue #45).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--workflow",
        choices=["ifrl", "identityrl"],
        required=True,
        help="RL workflow. identityrl omits the code-server target.",
    )
    p.add_argument(
        "--parameters-path",
        default=os.path.join(here, "parameters.yaml"),
        help="Base parameters file.",
    )
    p.add_argument(
        "--catalog-path",
        default=os.path.join(here, "eval-catalog.yaml"),
        help="Eval catalog file.",
    )
    # ── Common knobs, promoted to explicit flags for discoverability ──────────
    # Each maps to the parameter named in COMMON_FLAG_PARAMS and is applied as an
    # override. An explicit --param for the same key still wins (it is applied
    # after these), so the flags are convenience, not a separate mechanism.
    common = p.add_argument_group(
        "common parameters",
        "Frequently-changed knobs. Equivalent to --param <NAME>=<value>; "
        "anything omitted falls back to the parameters file.",
    )
    common.add_argument(
        "--model-path",
        metavar="PATH",
        help="Input checkpoint to train from (MODEL_PATH). For IFRL this is the "
        "SFT checkpoint; for IdentityRL, a prior RL checkpoint.",
    )
    common.add_argument(
        "--output-dir",
        metavar="PATH",
        help="Directory the trainer writes checkpoints to (OUTPUT_DIR).",
    )
    common.add_argument(
        "--total-episodes",
        metavar="N",
        help="Total training episodes (TOTAL_EPISODES); drives the checkpoint "
        "count.",
    )
    common.add_argument(
        "--save-freq",
        metavar="N",
        help="Save a checkpoint every N optimizer updates (SAVE_FREQ); drives "
        "the checkpoint schedule.",
    )
    common.add_argument(
        "--eval-freq",
        metavar="N",
        help="Trainer's in-loop eval frequency (EVAL_FREQ).",
    )
    common.add_argument(
        "--eval-sets",
        metavar="LIST",
        help="Which evaluations to run (EVAL_SETS), e.g. 'bfcl,multilingual-eval' "
        "or 'full-eval'. Comma-separated or a YAML list.",
    )
    common.add_argument(
        "--experiment",
        metavar="NAME",
        help="Experiment namespace for eval/export outputs (EXPERIMENT). "
        "Per-checkpoint results land under <SAGE_OUTPUT_DIR>/<EXPERIMENT>/ckpt_<step>/.",
    )
    common.add_argument(
        "--log-scrape-interval",
        metavar="SECONDS",
        help="How often the training logs are downloaded and parsed for "
        "checkpoint messages (RL_LOG_SCRAPE_INTERVAL_SECONDS).",
    )
    common.add_argument(
        "--train-status-interval",
        metavar="SECONDS",
        help="How often the training job's status is checked "
        "(RL_STATUS_POLL_INTERVAL_SECONDS).",
    )
    common.add_argument(
        "--eval-status-interval",
        metavar="SECONDS",
        help="How often each evaluation job's status is checked "
        "(EVAL_STATUS_POLL_INTERVAL_SECONDS).",
    )
    p.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a parameter (dot notation supported). Repeatable. "
        "Takes precedence over the common flags above.",
    )
    p.add_argument(
        "--output",
        default=os.path.join(here, "build.yaml"),
        help="Where to write the generated build.yaml ('-' for stdout).",
    )
    p.add_argument(
        "--params-out",
        default=None,
        help="Where to write the merged parameters (base file + flags + "
        "--param). Defaults to a 'parameters-resolved.yaml' sibling of "
        "--output. This is the file to pass to `gb build start "
        "--parameters-path`, so the flags/overrides you set here are honored "
        "when the build's $${...} placeholders are resolved. Ignored when "
        "--output is '-'.",
    )
    return p.parse_args(argv)


# Maps each common flag (argparse dest) to the parameter name it overrides.
COMMON_FLAG_PARAMS = {
    "model_path": "MODEL_PATH",
    "output_dir": "OUTPUT_DIR",
    "total_episodes": "TOTAL_EPISODES",
    "save_freq": "SAVE_FREQ",
    "eval_freq": "EVAL_FREQ",
    "eval_sets": "EVAL_SETS",
    "experiment": "EXPERIMENT",
    "log_scrape_interval": "RL_LOG_SCRAPE_INTERVAL_SECONDS",
    "train_status_interval": "RL_STATUS_POLL_INTERVAL_SECONDS",
    "eval_status_interval": "EVAL_STATUS_POLL_INTERVAL_SECONDS",
}


def _flag_overrides(args):
    """Turn the provided common flags into KEY=VALUE override strings.

    EVAL_SETS accepts a comma-separated shorthand ('a,b') as well as a YAML list
    ('[a, b]'); normalize the shorthand to a YAML list so apply_override types it
    as a list rather than a bare string.
    """
    overrides = []
    for dest, name in COMMON_FLAG_PARAMS.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        if name == "EVAL_SETS" and "[" not in value:
            items = [v.strip() for v in value.split(",") if v.strip()]
            value = "[" + ", ".join(items) + "]"
        overrides.append(f"{name}={value}")
    return overrides


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    # Common flags first, then --param, so an explicit --param wins on conflict.
    params = load_params(args.parameters_path, _flag_overrides(args) + args.param)
    # Derive CHECKPOINT_STATE_DIR from OUTPUT_DIR if not explicitly set.
    if not params.get("CHECKPOINT_STATE_DIR"):
        output_dir = params.get("OUTPUT_DIR")
        if not output_dir:
            sys.exit(
                "error: OUTPUT_DIR is required (the trainer's checkpoint output "
                "directory). Set it in the parameters file, or pass --output-dir "
                "/ --param OUTPUT_DIR=<path>. It also seeds CHECKPOINT_STATE_DIR "
                "when that is left blank."
            )
        params["CHECKPOINT_STATE_DIR"] = str(output_dir).rstrip("/") + "/_state"
    with open(args.catalog_path, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)

    build, checkpoint_steps, eval_names = generate(args.workflow, params, catalog)

    dumped = yaml.safe_dump(build, sort_keys=False, default_flow_style=False)
    params_out = None
    if args.output == "-":
        sys.stdout.write(dumped)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(dumped)
        # Emit the merged parameters so the flags/--param overrides applied here
        # are honored when `gb build start` resolves the build's $${...}
        # placeholders. Without this, a flag like --model-path would set the
        # checkpoint schedule but leave $${MODEL_PATH} to be resolved from the
        # untouched base parameters file at start time.
        params_out = args.params_out or os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            "parameters-resolved.yaml",
        )
        with open(params_out, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(params, sort_keys=False))

    n_eval_targets = len(checkpoint_steps) * len(eval_names)
    total_targets = len(build["granite.build"]["targets"])
    msg = (
        f"[generate_build] workflow={args.workflow}\n"
        f"  checkpoints ({len(checkpoint_steps)}): {checkpoint_steps}\n"
        f"  evals ({len(eval_names)}): {eval_names}\n"
        f"  eval targets: {len(checkpoint_steps)} ckpts x {len(eval_names)} evals "
        f"= {n_eval_targets}\n"
        f"  total targets (incl. servers/training/exports): {total_targets}\n"
    )
    if args.output != "-":
        msg += f"  wrote build:  {args.output}\n"
        msg += f"  wrote params: {params_out}\n"
        msg += (
            "  start with:   gb build start -f "
            f"{args.output} --parameters-path {params_out} --space <space>\n"
        )
    sys.stderr.write(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
