# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

# Single-device MLX train driver (Apple Silicon). Thin orchestration layer:
# all MLX specifics live in autotune.mlx_backend.

import logging
import os
from copy import deepcopy
from typing import Any, Dict

from ray import tune

from autotune import mlx_backend
from autotune.trainers.driver_single import _load_dataset_from_file

logger = logging.getLogger(__name__)


def train_driver_single_gpu(config: Dict[str, Any]) -> Dict[str, Any]:
    """Single-device MLX training driver (SFT/LoRA/QLoRA on Apple Silicon).

    Same contract as driver_single.train_driver_single_gpu: pop the config
    sections, load data, train, and tune.report() a metric dict. All MLX
    specifics live in autotune.mlx_backend.
    """
    from autotune.logging_setup import bind_trial_id, setup_logging

    setup_logging()

    trial_id = tune.get_context().get_trial_id()
    bind_trial_id(trial_id)
    logger.info(f"[AutoTune][MLX] Trial {trial_id} config: {config}")

    local_config = deepcopy(config)
    training_config = local_config.pop("training_config")
    local_config.pop("training_rl_config", None)
    # tuner_flags is popped for its side effect (kept out of `params` below);
    # unlike driver_single, mlx_backend.translate_config takes all sampled
    # hyperparameters together rather than splitting train/peft kwargs.
    local_config.pop("tuner_flags", None)
    local_config.pop("tune_config", None)

    # local_config now holds the sampled/fixed hyperparameters keyed by name.
    params = dict(local_config)

    train_file = training_config.get("train_file")
    eval_file = training_config.get("validation_file")
    algo = training_config.get("tuning_algorithm", "lora")
    model_name_or_path = training_config.get("model_name_or_path")
    output_dir = training_config.get("output_dir")
    hpo_search = training_config.get("hpo_search", False)
    input_col = training_config.get("input_column", "input")
    output_col = training_config.get("output_column", "output")

    # Load datasets (reuse the HF-driver loader for parity across backends).
    train_raw = _load_dataset_from_file(train_file)
    eval_raw = _load_dataset_from_file(eval_file)

    # HPO uses a subset, mirroring driver_single.
    if hpo_search:
        pct = training_config.get("hpo_dataset_percentage", 0.1)
        if 0.0 < pct < 1.0:
            train_raw = train_raw.select(range(max(1, int(pct * len(train_raw)))))
            eval_raw = eval_raw.select(range(max(1, int(pct * len(eval_raw)))))

    train_inputs, train_outputs = train_raw[input_col], train_raw[output_col]
    eval_inputs, eval_outputs = eval_raw[input_col], eval_raw[output_col]
    train_records, is_chat = mlx_backend.build_records(train_inputs, train_outputs)
    eval_records, _ = mlx_backend.build_records(eval_inputs, eval_outputs)

    epochs = training_config.get("hpo_num_epochs", 1) if hpo_search else training_config.get("num_train_epochs", 1)

    mlx_model_path = mlx_backend.ensure_mlx_model(model_name_or_path, quantize=(algo == "qlora"))
    mlx_cfg = mlx_backend.translate_config(training_config, params, epochs=epochs, n_train_examples=len(train_records))

    adapter_dir = os.path.join(output_dir, "mlx_adapters", str(trial_id))
    run_result = mlx_backend.run_training(mlx_model_path, mlx_cfg, train_records, is_chat, eval_records, adapter_dir)

    # Final training only: persist the MLX-native artifact.
    if not hpo_search and training_config.get("save_model", False):
        model_name = training_config.get("output_model_name")
        dest = os.path.join(output_dir, "models", model_name)
        mlx_backend.save_output(run_result, mlx_cfg, dest)
        logger.info(f"[AutoTune][MLX] Saved artifact to {dest}")

    result = {k: run_result[k] for k in ("loss", "train_loss", "eval_loss", "done")}
    tune.report(result)
    return result
