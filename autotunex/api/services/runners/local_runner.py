# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import os
import ray
import sys
import models
import shutil
import logging
import paths
from typing import Optional
from ray.tune import Callback
from ray.tune.experiment.trial import Trial
from services.impl.runner import Runner
from autotune.config import AutotuneConfig
from autotune.pipeline import AutotunePipeline
from services import db_service, logging_service, file_service
from autotune.optimizer import AutotuneOptimizer
from services.logging_service import BufferedLogHandler
from models import ModelSource
from utils import (
    write_inference_script,
    generate_bash_script,
    generate_readme,
    generate_install_bash_script,
)
from autotune.utils import set_seed, save_hpo_history, cleanup, generate_unique_id
from autotune.validation import validate_config_for_pipeline
from utils import parse_result

logger = logging.getLogger("LocalRunner")
db: db_service.Database = db_service.Database()


# Redirect print statements to logger
class PrintLogger:
    def __init__(self, logger):
        self.logger = logger
        self.buffer = ""

    def write(self, message):
        # Add message to buffer
        self.buffer += message

        # Process complete lines in buffer
        if "\n" in self.buffer:
            lines = self.buffer.split("\n")
            # Keep the last piece if it doesn't end with newline
            self.buffer = lines.pop()

            # Log each complete line
            for line in lines:
                if line.strip():  # Only log non-empty lines
                    self.logger.info(line.strip())

    def flush(self):
        # Log any remaining content in buffer when flush is called
        if self.buffer.strip():
            self.logger.info(self.buffer.strip())
            self.buffer = ""

    def isatty(self):
        return False

    def fileno(self):
        return 1


class CustomLoggerCallback(Callback):
    def __init__(
        self,
        job_id=None,
        handler: Optional[BufferedLogHandler] = None,
        db: db_service.Database = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.job_id = job_id
        self.handler = handler
        self.db = db

    def on_trial_start(self, iteration, trials, trial: Trial):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)
        data = {
            "id": trial.trial_id,
            "job_id": self.handler.get_job_id(),
            "status": models.TrialStatus.RUNNING,
            "config": trial.config,
            # "config": {"tune_config": "trial.config"},
        }
        self.db.insert_trial_sync(data=data)

        self.logger.info(
            f"::::::::::::::: Trial_{trial.trial_id} Initialized :::::::::::::::\n"
        )

        self.logger.info(f"trial_id: {trial.trial_id}")
        self.logger.info(f"iterations: {iteration}")
        self.logger.info(f"trial_fn: {trial.trainable_name}")
        self.logger.info(f"trial_status: {trial.status}\n")
        self.logger.info(">>>>>>>>>>>>> trial_config <<<<<<<<<<<<<\n")
        for key, value in trial.config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info(
            f"::::::::::::::: Trial_{trial.trial_id} Started :::::::::::::::\n"
        )
        if self.handler:
            self.handler.flush()

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)
        self.logger.info(f"--------- Trial_{trial.trial_id} Result Start -----------")
        self.logger.info(f"trial_id: {trial.trial_id}")
        self.logger.info(f"iterations: {iteration}")
        self.logger.info(f"trial_fn: {trial.trainable_name}")
        self.logger.info(f"trial_config: {trial.config}")
        self.logger.info(f"trial_status: {trial.status}")
        self.logger.info(f"Trial {trial.trial_id} reported result: {result}")
        self.logger.info(f"--------- Result for Trial_{trial.trial_id} End -----------")
        self.logger.info(f"......... TRIAL_JOB_ID.....{self.job_id}.............")

        try:
            result["job_id"] = self.job_id
            result["trial_id"] = trial.trial_id
            result["metric"] = "loss"
            result["metrics"] = parse_result(result)
            logger.info(f"Parsed result: {result}")
            self.db.insert_result_sync(metadata=result)
        except Exception as e:
            logger.exception(f"error occured on_trial_result: {e}")

        self.db.update_trial_status_sync(
            trial_id=trial.trial_id, status=models.TrialStatus.COMPLETED
        )

        if self.handler:
            self.handler.set_trial_id(None)
            self.handler.flush()

    def on_trial_complete(self, iteration, trials, trial, **info):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)
        self.logger.info(f"--------- Trial_{trial.trial_id} Completed -----------")
        if self.handler:
            self.handler.set_trial_id(None)
            self.handler.flush()

    def on_trial_error(self, iteration, trials, trial: Trial, **info):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)
        self.logger.error(f"Error occured during trial_{trial.trial_id} execution")
        self.db.update_trial_status_sync(
            trial_id=trial.trial_id, status=models.TrialStatus.ERROR
        )
        if self.handler:
            self.handler.set_trial_id(None)
            self.handler.flush()


class LocalRunner(Runner):
    def __init__(
        self,
        job_id: str,
        run_config: models.TuningConfig,
        db: db_service.Database,
        logging_handler: logging_service.BufferedLogHandler,
    ):
        super().__init__(job_id, run_config)
        self.db = db
        self.logging_handler = logging_handler

    def initialize_logger(self):
        self.logging_handler.set_trial_id(None)
        logger.setLevel(level=logging.INFO)
        self.logging_handler.set_job_id(job_id=str(self.job_id))
        logger.addHandler(self.logging_handler)
        sys.stdout = PrintLogger(logger)
        sys.stderr = PrintLogger(logger)

    def run(self):
        self.initialize_logger()
        logger.info(f"Starting Job: {self.job_id}")
        logger.info(f"config_id: {self.run_config.config_id}")
        logger.info(f"dataset_id: {self.run_config.dataset_id}")
        logger.info(f"model_name_or_path: {self.run_config.model}")
        logger.info(f"experiment_name: {self.run_config.experiment_name}")
        logger.info(f"autotune: {self.run_config.autotune}")

        try:
            AUTOTUNE_RESULTS_PATH = paths.results_path()
            logger.info(f"AUTOTUNE_RESULTS_PATH={AUTOTUNE_RESULTS_PATH}")
            output_dir = f"{AUTOTUNE_RESULTS_PATH}/output/{str(self.job_id)}"
            self.check_and_create_folder(output_dir)
        except Exception:
            logger.exception("Something went wrong while folder creation")

        # Load the main config file. Read the algorithm selection from the
        # config record's tuner_type / rl_tuner_type columns — the same fields
        # the GB runner maps to --tuning_algo / --rl_algo (gb_runner.build_start_cmd),
        # so local and remote select the same algorithm from the same config.
        # The pipeline requires both as strings and treats "none" as "not used".
        config_record = self.db.get_config_sync(self.run_config.config_id)
        config_data = config_record["config_data"]
        tuning_algo = config_record.get("tuner_type") or "none"
        rl_algo = config_record.get("rl_tuner_type") or "none"
        dataset = self.db.get_dataset_sync(self.run_config.dataset_id)

        # Write reward function from job config to file (Online RL)
        reward_code = getattr(self.run_config, "reward_function_code", None)
        reward_fn_name = getattr(self.run_config, "reward_function_name", None)
        if reward_code:
            reward_path = os.path.join(output_dir, "reward_function.py")
            with open(reward_path, "w") as f:
                f.write(reward_code)
            rl_config = config_data.get("training_rl_config", {})
            if rl_config.get("reward_function_path"):
                rl_config["reward_function_path"]["default"] = reward_path
            if reward_fn_name and rl_config.get("reward_function_name"):
                rl_config["reward_function_name"]["default"] = reward_fn_name
            logger.info(f"Reward function written to: {reward_path}")

        # Set the seed
        set_seed(self.run_config.seed)

        # Note: No need to specify the runtime_env in ray.init()
        # in the driver script.
        logger.info("[Connecting to ray cluster...]")

        if self.run_config.ray_address is not None:
            if not ray.is_initialized():
                ray.init(address=self.run_config.ray_address)
            logger.info(
                f"Connected to ray cluster on CCC at {self.run_config.ray_address}"
            )
        else:
            if not ray.is_initialized():
                ray.init(runtime_env={"env_vars": {"JOB_ID": str(self.job_id)}})
            logger.info("Connected to local cluster or remote OpenShift cluster")

        # Set the local caching directory. Results will be stored here
        # before they are synced to remote storage. This env variable is ignored
        # if `storage_path` below is set to a local directory.
        # os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = self.run_config.output_dir
        os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # os.environ["WORLD_SIZE"] = "1"

        # Get the best result and HPO history
        try:
            # Validate model source for local runner
            if hasattr(
                self.run_config, "model_source"
            ) and self.run_config.model_source in (
                ModelSource.DMF,
                ModelSource.CUSTOM_PATH,
            ):
                error_msg = (
                    "Local runner does not support DMF or Custom Path models. "
                    "These model sources can only be used with Granite Build (GB) runner. "
                    "Please use GB runner for DMF or Custom Path model fine-tuning."
                )
                logger.error(error_msg)
                raise NotImplementedError(error_msg)

            # Cleaning up the ray_results folder
            if self.run_config.cleanup:
                logger.info("[Cleaning up the ray results...]")
                folder = os.path.join(output_dir, "ray_results")
                cleanup(folder)

            # Create the main config (AutotuneConfig)
            config = AutotuneConfig()
            config.from_dict(config=config_data)

            # Create the tuning pipeline (AutotunePipeline) — the current API
            # takes tuning_algo + rl_algo (from the config record's tuner_type /
            # rl_tuner_type), matching how the GB runner drives main.py.
            pipeline = AutotunePipeline(
                tuning_algo=tuning_algo,
                rl_algo=rl_algo,
                model_name_or_path=self.run_config.model,
            )
            logger.info(f"Pipeline created: {pipeline}")

            # Algorithm-aware config validation (online RL normalizes tuning_algo
            # to "none", so validate against the pipeline's resolved algos).
            validate_config_for_pipeline(
                config,
                tuning_algo=pipeline.get_tuning_algo(),
                rl_algo=pipeline.get_rl_algo(),
            )

            # Unique run id for this run (consumed by the optimizer).
            run_id = generate_unique_id()
            logger.info(f"[AutoTune] Starting run: {run_id}")

            # Create the hyperparameter optimizer (AutotuneOptimizer)
            optimizer = AutotuneOptimizer(
                pipeline=pipeline,
                config=config,
                train_file=f"{file_service.get_training_file_path(self.run_config.dataset_id)}/{dataset.get('name')}/{dataset.get('name')}_train.{dataset.get('data_format', 'jsonl')}",
                validation_file=f"{file_service.get_training_file_path(self.run_config.dataset_id)}/{dataset.get('name')}/{dataset.get('name')}_validation.{dataset.get('data_format', 'jsonl')}",
                output_dir=output_dir,
                output_model_name=self.run_config.experiment_name,
                resume_from_checkpoint=False,
                keep_checkpoints=False,
                cluster_resources=ray.cluster_resources(),
                run_id=run_id,
                tuner_callbacks=[
                    CustomLoggerCallback(
                        job_id=self.job_id,
                        handler=self.logging_handler,
                        db=self.db,
                    )
                ],
            )
            logger.info(f"Hyper Parameter Optimizer created: {optimizer}")
            # Run HPO in a distributed manner
            result_grid = optimizer.fit()
            # For logging result to job level
            self.logging_handler.set_trial_id(trial_id=None)
            logger.info(f"result_grid created: {result_grid}")

            for i in range(len(result_grid)):
                res = result_grid[i]
                if res.error:
                    raise RuntimeError(res.error)

            metric = config.get_metric()
            mode = config.get_mode()
            logger.info(f"Tune Config Metrics: {metric}")

            # Save the HPO trial history
            if self.run_config.save_history:
                save_hpo_history(
                    result_grid=result_grid,
                    metric=metric,
                    mode=mode,
                    output_dir=output_dir,
                    run_name=self.run_config.experiment_name,
                )
            self.logging_handler.flush()

            # Train the best or the default configuration (if any)
            print("[Training best config...]")
            result_grid = optimizer.fit_best_config()
            best_result = result_grid.get_best_result(metric=metric, mode=mode)
            logger.info(f"Trained model with best result: {best_result.metrics}")
            self.logging_handler.flush()

            ray_result_folder = os.path.join(output_dir, "ray_results")
            current_folder = os.path.join(
                output_dir, "models", f"{self.run_config.experiment_name}"
            )

            # Creating Master Weights folder
            weights_folder_path = os.path.join(
                output_dir, f"{self.run_config.experiment_name}_weights"
            )

            try:
                os.makedirs(weights_folder_path, exist_ok=True)
                logger.info(f"Created Directory'{weights_folder_path}'")
            except Exception as e:
                logger.error(f"An error occured while creating folder: {e}")

            # Move weights to weights folder
            try:
                shutil.move(current_folder, weights_folder_path)
                logger.info(f"Moved '{current_folder}' to '{weights_folder_path}'")
            except Exception as e:
                logger.error(f"An error occured while moving folder: {e}")

            # Adding scripts
            try:
                generate_readme(weights_folder_path)
                generate_bash_script(weights_folder_path)
                write_inference_script(
                    self.run_config.model,
                    self.run_config.experiment_name,
                    weights_folder_path,
                    tuning_algo == "alora",
                )
                generate_install_bash_script(weights_folder_path)
            except Exception as e:
                logger.error(f"An error occured while adding: {e}")

            ray_result_path = file_service.zip_folder(
                ray_result_folder,
                f"{self.run_config.experiment_name}_ray_results.zip",
                os.path.join(output_dir, "results"),
            )
            logger.info(f"ray_result_path: {ray_result_path}")
            tuned_model_path = file_service.zip_folder(
                weights_folder_path,
                f"{self.run_config.experiment_name}_weights.zip",
                os.path.join(output_dir, "results"),
            )
            logger.info(f"tuned_model_path: {tuned_model_path}")
            self.db.update_job_status_sync(
                id=self.job_id, status=models.JobStatus.COMPLETED
            )

            # Cleaning up the ray_results folder
            if self.run_config.cleanup:
                # cleanup(weights_folder_path)
                logger.info("[Cleaning up the ray results...]")
                cleanup(ray_result_folder)

        except (Exception, RuntimeError) as e:
            self.db.update_job_status_sync(
                id=self.job_id, status=models.JobStatus.ERROR
            )
            self.logging_handler.set_trial_id(None)
            logger.exception(f"Job Failed due to error: {e}")
            ray.shutdown()

        logger.info("Shutting down ray...")
        ray.shutdown()
        logger.info("Done.")
        self.logging_handler.flush()

    def check_and_create_folder(self, folder_path):
        """
        Check if a folder exists, and create it if it doesn't.

        Parameters:
        folder_path (str): The path of the folder to check and create.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created: {folder_path}")
        else:
            print(f"Folder already exists: {folder_path}")
