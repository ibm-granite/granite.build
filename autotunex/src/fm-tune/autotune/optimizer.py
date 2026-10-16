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

# Hyperparameter sweeper for an existing Pipeline

import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from ray import tune
from ray.tune import Callback, ResultGrid

from autotune.config import AutotuneConfig
from autotune.constants import AUTOTUNE_DEFAULT_METRIC, AUTOTUNE_OFFLINE_RL, AUTOTUNE_ONLINE_RL, AutotunePrecision

# Local
from autotune.pipeline import AutotunePipeline
from autotune.utils import (
    get_param_space,
    get_tune_config,
    make_param_space,
    save_final_config,
)

STOP_LOSS = 0.00001

logger = logging.getLogger(__name__)


def _stop_dict_for_hpo(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the RunConfig.stop dict for an HPO trial.

    Only `loss` is capped — STOP_LOSS catches converged trials. The HF
    Trainer bounds training duration via num_train_epochs (set to
    hpo_num_epochs in every driver), so the controller does not need a
    separate training_iteration ceiling. ASHA can still STOP trials early
    via its scheduler decision; that path is independent of this dict.

    `training_config` is unused but retained in the signature so callsites
    stay stable and so future per-config logic (e.g., per-trial budgets)
    has a place to land without re-plumbing.

    Historical: earlier plans gated training_iteration here, but that
    raced with the driver's terminal tune.report() and killed the trial
    before `done=True` could be set, breaking BLDS arm updates and driver
    cleanup. See plan squiggly-bouncing-noether.md.
    """
    return {"loss": STOP_LOSS}


def _stop_dict_for_final(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the RunConfig.stop dict for the final retrain of the best HPO config.

    Same shape as _stop_dict_for_hpo: only STOP_LOSS gates the trial.
    The HF Trainer bounds training duration via num_train_epochs.
    """
    return {"loss": STOP_LOSS}


def _auto_derive_asha_max_t(tune_config: Dict[str, Any], training_config: Dict[str, Any]) -> None:
    """Set tune_config['asha_max_t'] from training_config['hpo_num_epochs'].

    ASHA's max_t bounds the rung at which a trial can be stopped. To keep
    ASHA's bracket math aligned with the trial's actual epoch budget — set
    on the HF Trainer via num_train_epochs (= hpo_num_epochs during HPO) —
    max_t must equal hpo_num_epochs. We derive it here rather than expose
    it as a YAML knob so the two stay in sync automatically.

    Mutates tune_config in place. No-op when scheduler != 'asha'. Does not
    overwrite an already-set asha_max_t (defensive: leaves a future
    explicit-override path open without us having to add API).
    """
    if tune_config.get("scheduler") != "asha":
        return
    if tune_config.get("asha_max_t") is not None:
        return
    hpo_max_iter = max(1, int(training_config.get("hpo_num_epochs", 1)))
    tune_config["asha_max_t"] = hpo_max_iter


class AutotuneOptimizer:
    """
    Ray based hyperparameter optimizer. Supports Hyperopt, BOHB, Random Search,
    Limited Discrepancy Search and Bandit Limited Discrepancy Search.
    """

    def __init__(
        self,
        pipeline: AutotunePipeline,
        config: AutotuneConfig,
        train_file: str,
        validation_file: str,
        output_dir: str = None,
        output_model_name: str = None,
        resume_from_checkpoint: bool = False,
        keep_checkpoints: bool = False,
        cluster_resources: Dict[str, Any] = None,
        tuner_callbacks: Optional[List[Callback]] = None,
        run_id: str = None,
    ):
        """
        Create the hyperparameter optimizer for a given pipeline.

        Args:
            pipeline: AutotunePipeline
                The pipeline to be optimized.
            config: AutotuneConfig
                The main configuration object containing the system, preprocessing,
                training, ray/tune sections and the hyperparameter search space.
            train_file: str
                Path to the file containing the training dataset.
            validation_file: str
                Path to the file containing the validation dataset.
            output_dir: str
                The output dir where results will be stored.
            output_model_name: str
                The name of the output model (will be a subdir of the output dir).
            resume_from_checkpoint: bool
                A boolean flag to resume the *final* training stage (the retrain
                on the best HPO config, or the --no_autotune default run) from
                the last checkpoint under ``<output_dir>/final_checkpoints/``.
                When the run is started with this flag and a saved
                ``final_config.json`` plus a checkpoint exist there, HPO is
                skipped entirely and the saved config drives the resumed final
                training (see main.py). Only consumed by the multi-GPU drivers
                and only when not in HPO search.
            keep_checkpoints: bool
                When True, skip deletion of intermediate checkpoints and training
                artifacts (final_checkpoints/, outputs/, train_results/,
                data_cache/) after final training completes. Useful for debugging.
                Has no effect during HPO trials.
            cluster_resources: Dict[str, Any]
                A dict with the ray cluster resources.
            tuner_callbacks: Optional[List[Callback]]
                A optional parameter for ray tuner callbacks.
            run_id: str
                The unique ID of the current run
        """

        # Init the members
        self.pipeline = pipeline
        self.config = config
        self.train_file = train_file
        self.validation_file = validation_file
        self.output_dir = output_dir
        self.output_model_name = output_model_name
        self.resume_from_checkpoint = resume_from_checkpoint
        self.keep_checkpoints = keep_checkpoints
        self.cluster_resources = cluster_resources
        self.run_id = run_id

        self.model_name_or_path = pipeline.get_model_name_or_path()
        self.tuning_algo = pipeline.get_tuning_algo()
        self.rl_algo = pipeline.get_rl_algo()
        self.precision = pipeline.get_precision()

        from autotune.device import detect_accelerator

        self.accel = detect_accelerator()

        # Ensure that if SFT is used, the precision is set to FP32 or BF16
        assert self.precision in (
            AutotunePrecision.BF16,
            AutotunePrecision.FP32,
        ), "Precision must be BF16 (GPU/MPS) or FP32 (MPS/CPU)."

        self.trainer_fn = None
        self.tuner_callbacks = tuner_callbacks
        self.time_budget_s = None
        self.best_config = None

        # Log the initialization
        logger.info(f"[AutoTune] Optimizer initialized with model: {self.model_name_or_path}")
        logger.info(f"[AutoTune] Tuning algorithm: {self.tuning_algo}")  # can be `none` for online RL
        logger.info(f"[AutoTune] RL algorithm: {self.rl_algo}")  # can be `none` for non-RL tuning
        logger.info(f"[AutoTune] Precision: {self.precision}")
        logger.info(f"[AutoTune] Output dir: {self.output_dir}")
        logger.info(f"[AutoTune] Output model name: {self.output_model_name}")
        logger.info(f"[AutoTune] Resume from checkpoint: {self.resume_from_checkpoint}")
        logger.info(f"[AutoTune] Keep checkpoints: {self.keep_checkpoints}")
        logger.info(f"[AutoTune] Run ID: {self.run_id}")

    def setup_pipeline(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Setup the optimizer to handle the AutotunePipeline.

        Returns:
            A tuple of dict containing the tune_config, training_config
            and param_space (i.e., the hyperparameter search space).
        """

        # Get the config sections (deep copies)
        tune_config = deepcopy(self.config.get_tune_config_dict())
        training_config = deepcopy(self.config.get_training_config_dict())
        training_rl_config = deepcopy(self.config.get_training_rl_config_dict())

        # Set the multi-gpu training
        multi_gpu = True if training_config.get("num_gpus_per_trial") > 1 else False
        self.pipeline.set_multi_gpu(multi_gpu)
        logger.info(f"[AutoTune] Multi-GPU training per trial: {multi_gpu}")
        print(f"[AutoTune] Multi-GPU training per trial: {multi_gpu}")
        if multi_gpu:
            logger.info("[AutoTune] Using the multi-gpu training driver")
            print("[AutoTune] Using the multi-gpu training driver")
        else:
            logger.info("[AutoTune] Using the single-gpu training driver")
            print("[AutoTune] Using the single-gpu training driver")

        # Determine the number of concurrent trials (disable for now)
        num_concurrent_trials = tune_config.get("max_concurrent_trials")
        self.time_budget_s = tune_config.get("time_budget_s", None)
        print(f"[AutoTune] Max concurrent trials: {num_concurrent_trials}")
        print(f"[AutoTune] Time budget (s): {self.time_budget_s}")
        logger.info(f"[AutoTune] Max concurrent trials: {num_concurrent_trials}")
        logger.info(f"[AutoTune] Time budget (s): {self.time_budget_s}")

        # Get the tuning type and tuner config
        peft_type = self.pipeline.get_peft_type()
        tuning_algo = self.pipeline.get_tuning_algo()
        rl_algo = self.pipeline.get_rl_algo()
        tuner_config = deepcopy(self.config.get_tuner_config_dict(tuning_algo))
        tuner_rl_config = deepcopy(self.config.get_tuner_rl_config_dict(rl_algo)) if rl_algo != "none" else {}

        # Check if offline RL of sft/peft tuning
        if rl_algo in ["none", "dpo", "orpo", "kto"]:
            assert len(tuner_config) > 0 and "hyperparams" in tuner_config.keys()
            if len(tuner_rl_config) > 0 and "hyperparams" in tuner_rl_config.keys():
                tuner_config["hyperparams"].update(tuner_rl_config["hyperparams"])
        else:  # online RL
            assert len(tuner_config) == 0
            tuner_config = tuner_rl_config

        # Get the tunable param spaces, default values and tuner flags from the config
        param_space, default_values, tuner_flags = get_param_space(tuner_config)

        # Keep ASHA's max_t aligned with the per-epoch report cadence: max_t
        # must equal hpo_num_epochs so ASHA's bracket math knows the trial's
        # actual epoch budget (set on HF Trainer via num_train_epochs).
        # Mutates tune_config in place; no-op when scheduler != "asha".
        _auto_derive_asha_max_t(tune_config, training_config)

        # Get the tune config (overwrites the ray.tune config)
        tune_config = get_tune_config(tune_config, default_values)

        # Update the training config
        training_config["peft_type"] = peft_type
        training_config["tuning_algorithm"] = tuning_algo
        training_config["rl_algorithm"] = rl_algo
        training_config["model_name_or_path"] = self.model_name_or_path
        training_config["train_file"] = self.train_file
        training_config["validation_file"] = self.validation_file
        training_config["metric"] = tune_config.get("metric")
        training_config["output_dir"] = self.output_dir
        training_config["output_model_name"] = self.output_model_name
        training_config["time_budget_s"] = self.time_budget_s

        # Merge tokenizer customization into training_config
        for k, v in self.config.get_tokenizer_config_dict().items():
            if v is not None:
                training_config[k] = v

        print(f"[AutoTune] Training config: {training_config}")
        print(f"[AutoTune] Training config (RL): {training_rl_config}")
        logger.info(f"[AutoTune] Training config: {training_config}")
        logger.info(f"[AutoTune] Training config (RL): {training_rl_config}")

        # Update the param space
        param_space["training_config"] = training_config
        param_space["training_rl_config"] = training_rl_config
        param_space["tune_config"] = tune_config
        param_space["tuner_flags"] = tuner_flags

        return tune_config, param_space

    def setup_default_config(self) -> Dict[str, Any]:
        # Get the default hyperparameter config for the tuning and RL algorithms
        default_rl_config = self.config.get_default_rl_config_dict(self.rl_algo)
        default_config = self.config.get_default_config_dict(self.tuning_algo)
        if self.rl_algo in ["none"] + AUTOTUNE_OFFLINE_RL:  # sft/peft or offline RL
            default_config.update(default_rl_config)
        else:  # online RL
            default_config.update(default_rl_config)

        # Set the multi-gpu training
        multi_gpu = True if default_config.get("training_config").get("num_gpus_per_trial") > 1 else False
        self.pipeline.set_multi_gpu(multi_gpu)
        logger.info(f"[AutoTune] Multi-GPU training per trial: {multi_gpu}")
        print(f"[AutoTune] Multi-GPU training per trial: {multi_gpu}")
        if multi_gpu:
            logger.info("[AutoTune] Using the multi-gpu training driver")
            print("[AutoTune] Using the multi-gpu training driver")
        else:
            logger.info("[AutoTune] Using the single-gpu training driver")
            print("[AutoTune] Using the single-gpu training driver")

        default_config["training_config"]["peft_type"] = self.pipeline.get_peft_type()
        default_config["training_config"]["tuning_algorithm"] = self.pipeline.get_tuning_algo()
        default_config["training_config"]["rl_algorithm"] = self.pipeline.get_rl_algo()  # can be None
        default_config["training_config"]["model_name_or_path"] = self.model_name_or_path
        default_config["training_config"]["train_file"] = self.train_file
        default_config["training_config"]["validation_file"] = self.validation_file
        default_config["training_config"]["metric"] = AUTOTUNE_DEFAULT_METRIC
        default_config["training_config"]["output_dir"] = self.output_dir
        default_config["training_config"]["output_model_name"] = self.output_model_name

        # Merge tokenizer customization into training_config
        for k, v in self.config.get_tokenizer_config_dict().items():
            if v is not None:
                default_config["training_config"][k] = v

        return default_config

    def fit_best_config(
        self, use_default: bool = False, do_checkpoint: bool = True, saved_config: dict = None
    ) -> ResultGrid:
        """
        Train remotely in a single trial the best hyperparameter config or
        the default hyperparameter config. This should be done using all
        available GPUs in the cluster.

        Args:
            use_default: bool
                Flag indicating that the default hyperparameter config is used.
            do_checkpoint: bool
                Whether to enable checkpointing for the final/default run.
            saved_config: dict
                A previously saved final config (from
                ``final_checkpoints/final_config.json``) to resume from. When
                provided it takes precedence over ``use_default`` / the in-memory
                ``best_config``, and the config is NOT re-saved (we loaded it from
                disk). See ``autotune.utils.load_final_config`` and main.py.
        """

        # Resolve the config source: saved (resume) > HPO best > default.
        if saved_config is not None:
            best_config = deepcopy(saved_config)
        elif not use_default:
            assert self.best_config is not None, "Unable to train the best config."
            best_config = deepcopy(self.best_config)
        else:
            best_config = self.setup_default_config()

        print(f"[AutoTune] Fitting the best/default config: {best_config}")
        logger.info(f"[AutoTune] Fitting the best/default config: {best_config}")

        # Persist the resolved config to final_checkpoints/final_config.json so a
        # later --resume_from_checkpoint run can skip HPO and resume final
        # training from the last checkpoint. Save here, before the pops below.
        # save_final_config sanitizes non-JSON-serializable values (e.g. the Ray
        # search-alg/scheduler objects that ride inside the HPO best config's
        # tune_config; they are rebuilt on resume, never consumed from the file).
        # Best-effort: a save failure must never abort the final training run.
        # Skip entirely when resuming (we just loaded from that file).
        if saved_config is None:
            try:
                saved_path = save_final_config(self.output_dir, best_config)
                if saved_path:
                    logger.info(f"[AutoTune] Saved final config to: {saved_path}")
            except Exception as e:
                logger.warning(f"[AutoTune] Could not save final config (resume unavailable): {e}")

        # Setup: tune_config, training_config and param_space
        training_config = best_config.pop("training_config")
        training_rl_config = best_config.pop("training_rl_config")
        tune_config = best_config.pop("tune_config")
        tuner_flags = best_config.pop("tuner_flags")
        tuner_rl_flags = {} if "tuner_rl_flags" not in best_config.keys() else best_config.pop("tuner_rl_flags")

        # Output the best/default config
        print(f"[AutoTune] Best/default hyperparameters: {best_config}")
        logger.info(f"[AutoTune] Best/default hyperparameters: {best_config}")

        # Make the param space from the hyperparameter config
        param_space, default_values = make_param_space(best_config)

        # Set the training driver (multi-gpu by default)
        train_implementation = training_config.get("train_implementation", "huggingface_ds")
        num_gpus_per_trial = training_config.get("num_gpus_per_trial", 1)
        multi_gpu = self.accel.supports_distributed  # CUDA: True (unchanged); MPS/CPU: False → single driver
        self.pipeline.set_multi_gpu(multi_gpu)

        # Set the training driver based on the train implementation
        rl_algo = self.pipeline.get_rl_algo()
        if self.pipeline.get_multi_gpu() is False:  # Single GPU training
            if rl_algo in AUTOTUNE_OFFLINE_RL:
                from autotune.trainers.driver_single_trl import train_driver_single_gpu

                self.trainer_fn = train_driver_single_gpu
            elif rl_algo in AUTOTUNE_ONLINE_RL:
                raise ValueError(
                    f"Online RL algorithms {AUTOTUNE_ONLINE_RL} are not supported for single GPU training."
                )
            else:  # no RL, use tuning with SFT or PEFT
                if training_config.get("backend", "torch") == "mlx":
                    from autotune.trainers.driver_single_mlx import train_driver_single_gpu
                else:
                    from autotune.trainers.driver_single import train_driver_single_gpu

                self.trainer_fn = train_driver_single_gpu
        else:  # Multi-GPU training
            # Set the training driver based on the train implementation
            self.pipeline.set_multi_gpu(True)
            if rl_algo in AUTOTUNE_ONLINE_RL:
                from autotune.trainers.driver_multi_verl import train_driver_multi_gpu

                self.trainer_fn = train_driver_multi_gpu
            else:
                if rl_algo in AUTOTUNE_OFFLINE_RL:
                    if train_implementation == "FSDP":
                        print("[AutoTune] Using TRL FSDP training implementation.")
                        logger.info("[AutoTune] Using TRL FSDP training implementation.")
                        from autotune.trainers.driver_multi_trl_fsdp import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu
                    else:
                        print("[AutoTune] Using TRL DS training implementation.")
                        logger.info("[AutoTune] Using TRL DS training implementation.")
                        from autotune.trainers.driver_multi_trl_ds import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu
                else:  # no offline RL, use tuning with SFT or PEFT
                    if train_implementation == "FSDP":
                        print("[AutoTune] Using HuggingFace FSDP training implementation.")
                        logger.info("[AutoTune] Using HuggingFace FSDP training implementation.")
                        from autotune.trainers.driver_multi_hf_fsdp import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu
                    else:
                        print("[AutoTune] Using HuggingFace DS training implementation.")
                        logger.info("[AutoTune] Using HuggingFace DS training implementation.")
                        from autotune.trainers.driver_multi_hf_ds import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu

        # Get the max concurrent trials (this is used to maximize GPU usage)
        max_concurrent_trials = tune_config.get("max_concurrent_trials")
        print(f"[AutoTune] Using max concurrent trials: {max_concurrent_trials}")
        logger.info(f"[AutoTune] Using max concurrent trials: {max_concurrent_trials}")
        print("[AutoTune] No time budget set, running just one trial.")
        logger.info("[AutoTune] No time budget set, running just one trial.")

        # Update the tune config for fitting the best/default config. We want
        # to maximize GPU usage for the final training, so we set
        # max_concurrent_trials to the number of GPUs per trial (if multi-gpu) or 1
        tune_config["search_alg"] = "random"
        tune_config["scheduler"] = "fifo"
        tune_config["num_samples"] = 1
        tune_config["max_concurrent_trials"] = 1  # run just one trial
        tune_config = get_tune_config(tune_config, default_values)
        tune_config["time_budget_s"] = None  # no time budget for the best config run

        # Update the training config
        training_config["save_model"] = True
        training_config["eval_test"] = True
        training_config["hpo_search"] = False
        training_config["do_checkpoint"] = do_checkpoint  # enable checkpointing for final/default config
        # Final-stage checkpoint resume is controlled by --resume_from_checkpoint,
        # independent of --restore (which resumes a failed HPO sweep upstream).
        training_config["resume_from_checkpoint"] = self.resume_from_checkpoint
        training_config["keep_checkpoints"] = self.keep_checkpoints

        # Get the resources per trial. The training drivers will allocate the GPUs
        num_gpus_per_trial = training_config.get("num_gpus_per_trial", 1)
        training_config["num_workers"] = num_gpus_per_trial * max_concurrent_trials  # maximize GPU usage

        # Update the param space
        if len(tuner_rl_flags) > 0:
            tuner_flags.update(tuner_rl_flags)  # merge the tuner and offline RL flags (if any)
        param_space["training_config"] = training_config
        param_space["training_rl_config"] = training_rl_config
        param_space["tune_config"] = tune_config
        param_space["tuner_flags"] = tuner_flags

        # Setup param search space and resources per worker
        experiment_name = "ray_results"
        storage_path = os.path.join(self.output_dir)

        # Safety checks
        assert self.trainer_fn is not None
        print(f"[AutoTune] Storage path (best config): {storage_path}")
        print(f"[AutoTune] Experiment name (best config): {experiment_name}")

        # Set resources per trial. The assumption is 1 GPU per trial.
        if self.pipeline.get_multi_gpu() is False:  # Single-device training (CUDA 1-GPU, or MPS/CPU)
            num_cpus = 1
            from autotune.device import ray_num_gpus

            num_gpus = ray_num_gpus(self.accel, 1)
        else:  # Multi-GPU training
            num_cpus = 1
            num_gpus = 0

        print(f"[AutoTune] Training driver using {num_gpus_per_trial * max_concurrent_trials} GPUs per trial.")
        logger.info(f"[AutoTune] Training driver using {num_gpus_per_trial * max_concurrent_trials} GPUs per trial.")

        # Prepare the trainable function and resources per trial
        resource_group = tune.PlacementGroupFactory(bundles=[{"CPU": num_cpus, "GPU": num_gpus}], strategy="PACK")
        trainable = tune.with_resources(
            tune.with_parameters(self.trainer_fn),
            resources=resource_group,  # {"gpu": 1, "cpu": 1}
        )

        # Start a new ray.tune run or restore a previous one
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(**tune_config),
            run_config=tune.RunConfig(
                name=experiment_name,
                storage_path=storage_path,
                stop=_stop_dict_for_final(training_config),
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute=tune_config.get("metric"),
                    num_to_keep=1,
                ),
            ),
        )

        results: ResultGrid = tuner.fit()
        return results

    def fit(self) -> ResultGrid:
        """
        Run the Ray Tune based hyperparameter optimizer.

        Returns:
            A ResultGrid instance holding the results of all trials run during
            the hyperparameter optimization.
        """

        # Setup: tune_config and param_space
        tune_config, param_space = self.setup_pipeline()

        # Set the training driver
        rl_algo = self.pipeline.get_rl_algo()
        training_config = param_space["training_config"]
        train_implementation = training_config.get("train_implementation", "huggingface_ds")
        if self.pipeline.get_multi_gpu() is False:  # Single GPU training
            if rl_algo in AUTOTUNE_OFFLINE_RL:
                from autotune.trainers.driver_single_trl import train_driver_single_gpu

                self.trainer_fn = train_driver_single_gpu
            elif rl_algo in AUTOTUNE_ONLINE_RL:
                raise ValueError(
                    f"Online RL algorithms {AUTOTUNE_ONLINE_RL} are not supported for single GPU training."
                )
            else:  # no offline RL, use tuning with SFT or PEFT
                if training_config.get("backend", "torch") == "mlx":
                    from autotune.trainers.driver_single_mlx import train_driver_single_gpu
                else:
                    from autotune.trainers.driver_single import train_driver_single_gpu

                self.trainer_fn = train_driver_single_gpu
        else:  # Multi-GPU training
            # Set the training driver based on the train implementation
            self.pipeline.set_multi_gpu(True)
            if rl_algo in AUTOTUNE_ONLINE_RL:  # if online RL
                from autotune.trainers.driver_multi_verl import train_driver_multi_gpu

                self.trainer_fn = train_driver_multi_gpu
            else:  # either offline RL or no RL, use the train implementation from the config
                if rl_algo in AUTOTUNE_OFFLINE_RL:
                    if train_implementation == "FSDP":
                        print("[AutoTune] Using TRL FSDP training implementation.")
                        logger.info("[AutoTune] Using TRL FSDP training implementation.")
                        from autotune.trainers.driver_multi_trl_fsdp import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu
                    else:
                        print("[AutoTune] Using TRL DS training implementation.")
                        logger.info("[AutoTune] Using TRL DS training implementation.")
                        from autotune.trainers.driver_multi_trl_ds import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu
                else:  # no offline RL, use tuning with SFT or PEFT
                    if train_implementation == "FSDP":
                        print("[AutoTune] Using HuggingFace FSDP training implementation.")
                        logger.info("[AutoTune] Using HuggingFace FSDP training implementation.")
                        from autotune.trainers.driver_multi_hf_fsdp import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu
                    else:
                        print("[AutoTune] Using HuggingFace DS training implementation.")
                        logger.info("[AutoTune] Using HuggingFace DS training implementation.")
                        from autotune.trainers.driver_multi_hf_ds import train_driver_multi_gpu

                        self.trainer_fn = train_driver_multi_gpu

        # Setup param search space and resources per worker
        experiment_name = "ray_results"
        storage_path = os.path.join(self.output_dir)

        assert self.trainer_fn is not None
        print(f"[AutoTune] Storage path: {storage_path}")
        print(f"[AutoTune] Experiment name: {experiment_name}")

        # Get the resources per trial. The training drivers will allocate the GPUs
        num_gpus_per_trial = training_config.get("num_gpus_per_trial", 1)
        training_config["num_workers"] = num_gpus_per_trial
        training_config["hpo_search"] = True

        # Set resources per trial. The assumption is 1 GPU per trial.
        if self.pipeline.get_multi_gpu() is False:
            # Single GPU training
            assert num_gpus_per_trial == 1, "Single GPU training requires num_gpus_per_trial == 1"
            num_cpus = 1
            from autotune.device import ray_num_gpus

            num_gpus = ray_num_gpus(self.accel, num_gpus_per_trial)
        else:  # Multi-GPU training
            print(f"[AutoTune] Training driver (trial) using {num_gpus_per_trial} GPUs per trial.")
            logger.info(f"[AutoTune] Training driver (trial) using {num_gpus_per_trial} GPUs per trial.")
            num_cpus = 1
            num_gpus = 0

        print(f"[AutoTune] Training driver using {num_gpus_per_trial} GPUs per trial.")
        logger.info(f"[AutoTune] Training driver using {num_gpus_per_trial} GPUs per trial.")

        # Prepare the trainable function and resources per trial
        resource_group = tune.PlacementGroupFactory(bundles=[{"CPU": num_cpus, "GPU": num_gpus}], strategy="PACK")
        trainable = tune.with_resources(
            tune.with_parameters(self.trainer_fn),
            resources=resource_group,  # {"gpu": 1, "cpu": 1}
        )

        # Start a new ray.tune run. A crashed HPO sweep is NOT restored — if HPO
        # fails the user simply reruns the job. (Final-training resume is handled
        # separately via --resume_from_checkpoint; see main.py / fit_best_config.)
        if self.time_budget_s is not None:
            # If time budget is set, we use the time budget as the stop condition
            print(f"[AutoTune] Using time budget of {self.time_budget_s} seconds as the stop condition.")
            logger.info(f"[AutoTune] Using time budget of {self.time_budget_s} seconds as the stop condition.")
            tune_config["time_budget_s"] = self.time_budget_s
        else:
            # Otherwise, we use the training iteration as the stop condition
            print("[AutoTune] No time budget set")
            logger.info("[AutoTune] No time budget set")
            tune_config["time_budget_s"] = None

        # Create a new tuner object
        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(**tune_config),
            run_config=tune.RunConfig(
                name=experiment_name,
                storage_path=storage_path,
                stop=_stop_dict_for_hpo(training_config),
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_score_attribute=tune_config.get("metric"),
                    num_to_keep=1,
                ),
                callbacks=self.tuner_callbacks,
            ),
        )

        results: ResultGrid = tuner.fit()

        # Check for errored trials
        num_errors = results.num_errors
        if num_errors > 0:
            logger.warning(f"[AutoTune] {num_errors}/{len(results)} HPO trials errored")
            print(f"[AutoTune] {num_errors}/{len(results)} HPO trials errored")
        if num_errors == len(results):
            raise RuntimeError(f"All {num_errors} HPO trials failed. Check individual trial logs for details.")

        # Get the best config from successful trials
        metric = tune_config.get("metric")
        mode = tune_config.get("mode")
        best_result = results.get_best_result(metric=metric, mode=mode)
        self.best_config = deepcopy(best_result.metrics.get("config"))

        return results
