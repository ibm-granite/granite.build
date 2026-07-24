# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import models
import yaml
from services import db_service, gb_service, logging_service
from services.impl.runner import Runner
from services.yaml_service import YAMLManager
from utilites.granite_build import GraniteBuild
from utils import extract_github_url

logger = logging.getLogger("GBRunner")
gb: gb_service.GBService = gb_service.GBService()

# Repo holding the autotune training library, cloned by the Granite Build runtime.
# Override GB_TUNE_REPO to point at a fork or an enterprise host.
DEFAULT_TUNE_REPO = "github.com/ibm-research/fm-tune.git"


def get_tune_repo() -> str:
    """Return the training-library repo URL for Granite Build spec generation."""
    return os.getenv("GB_TUNE_REPO", DEFAULT_TUNE_REPO)


class GBRunner(Runner):
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

    async def run(self):
        self.logging_handler.set_job_id(job_id=self.job_id)
        logger.addHandler(self.logging_handler)
        try:
            await self._run_inner()
        except Exception as e:
            await self.db.update_job_status(
                id=self.job_id, status=models.JobStatus.ERROR
            )
            logger.error(f"Error while submitting job to granite build: {str(e)}")
        finally:
            self.logging_handler.flush()
            logger.removeHandler(self.logging_handler)

    async def _run_inner(self):
        logger.info(f"Initializing Job: {self.job_id}")
        logger.info(f"config_id: {self.run_config.config_id}")
        logger.info(f"datase_id: {self.run_config.dataset_id}")
        logger.info(f"model_name_or_path: {self.run_config.model}")
        logger.info(
            f"model_source: {getattr(self.run_config, 'model_source', 'huggingface')}"
        )
        logger.info(f"experiment_name: {self.run_config.experiment_name}")

        config_dict = await self.db.get_config(self.run_config.config_id)
        dataset_dict = await self.db.get_dataset(self.run_config.dataset_id)

        logger.info(
            config_dict.get("config_data", {})
            .get("tune_config", {})
            .get("max_concurrent_trials")
        )
        logger.info(
            config_dict.get("config_data", {})
            .get("training_config", {})
            .get("num_gpus_per_trial")
        )
        # logger.info(
        #     config_dict.get("config_data", {})
        #     .get("training_config", {})
        #     .get("num_cpus_per_worker")
        # )

        max_concurrent_trials = (
            config_dict.get("config_data", {})
            .get("tune_config", {})
            .get("max_concurrent_trials")
            .get("default", 1)
        )
        num_gpus_per_trial = (
            config_dict.get("config_data", {})
            .get("training_config", {})
            .get("num_gpus_per_trial")
            .get("default", 1)
        )
        # num_cpus_per_worker = (
        #     config_dict.get("config_data", {})
        #     .get("training_config", {})
        #     .get("num_cpus_per_worker")
        #     .get("default", 1)
        # )

        logger.info(dataset_dict)
        experiment = f"temp_yaml/jobs/{self.job_id}/{self.run_config.experiment_name}"
        # Get reward function from job config (passed via TuningConfig model)
        reward_code = getattr(self.run_config, "reward_function_code", None)
        reward_fn_name = getattr(self.run_config, "reward_function_name", None)

        config_data = config_dict.get("config_data", {})

        # If reward function code provided, update config paths before YAML serialization
        if reward_code:
            rl_config = config_data.get("training_rl_config", {})
            if rl_config.get("reward_function_path"):
                rl_config["reward_function_path"]["default"] = "/tmp/reward_function.py"
            if reward_fn_name and rl_config.get("reward_function_name"):
                rl_config["reward_function_name"]["default"] = reward_fn_name
            logger.info(
                "Reward function provided, will inject into build.yaml as /tmp/reward_function.py"
            )

        yaml_string = yaml.dump(
            config_data,
            default_flow_style=False,
            sort_keys=False,
            Dumper=yaml.SafeDumper,
        )

        # Determine if model is from DMF (Lakehouse URI) or HuggingFace
        model_source = getattr(
            self.run_config, "model_source", models.ModelSource.HUGGINGFACE
        )
        is_dmf_model = model_source == models.ModelSource.DMF

        # For DMF models, construct Lakehouse URI from additional_info
        if is_dmf_model:
            additional_info = getattr(self.run_config, "additional_info", None)
            if additional_info:
                # Construct Lakehouse URI from DMF model metadata
                namespace = additional_info.get("namespace")
                base_model = additional_info.get("base_model")
                revision = additional_info.get("revision")

                if namespace and base_model and revision:
                    model_uri = f"lh://prod/{namespace}/models/model_shared/{base_model}/{revision}"
                    logger.info(
                        f"Constructed DMF model URI from additional_info: {model_uri}"
                    )
                    logger.info(
                        f"DMF model metadata: namespace={namespace}, base_model={base_model}, revision={revision}"
                    )
                else:
                    # Fallback to using model field directly if additional_info is incomplete
                    model_uri = self.run_config.model
                    logger.warning(
                        f"Incomplete additional_info, using model field directly: {model_uri}"
                    )
            else:
                # Fallback to using model field directly if no additional_info
                model_uri = self.run_config.model
                logger.warning(
                    f"No additional_info provided for DMF model, using model field: {model_uri}"
                )

            build = GraniteBuild.create_default_build(
                name="autotunex",
                model_uri=model_uri,
                dataset_uri=dataset_dict["artifact_url"],
                output_uri=f"lh://prod/granite_dot_build.public/models/model_shared/autotunex_{self.run_config.experiment_name}/{self.job_id}/",
                github_url=get_tune_repo(),
                config=yaml_string,
                config_name=config_dict.get("name", "config").replace(" ", "_"),
            )
        else:
            logger.info(f"Using HuggingFace model: {self.run_config.model}")
            build = GraniteBuild.create_default_build(
                name=f"autotunex-{self.run_config.experiment_name}",
                dataset_uri=dataset_dict["artifact_url"],
                output_uri=f"hf://huggingface.co/ibm-research/autotunex_{self.job_id.split('-')[0]}/",
                github_url=get_tune_repo(),
                config=yaml_string,
                config_name=config_dict.get("name", "config").replace(" ", "_"),
            )
        # Inject reward function code into build.yaml additional_files
        if reward_code:
            build.add_additional_file("/tmp/reward_function.py", reward_code)
            logger.info(
                "Reward function injected into build.yaml as /tmp/reward_function.py"
            )

        build.update_start_command(
            await self.build_start_cmd(self.run_config, self.job_id)
        )

        # Update compute configuration
        build.update_compute_config(
            num_gpus_per_node=(max_concurrent_trials * num_gpus_per_trial),
            # num_cpus_per_node=(max_concurrent_trials * num_gpus_per_trial * 8), # Hardcoded to 32 CPU as we got error in GB with 64 CPU
            num_cpus_per_node=32,
            num_nodes=1,
            total_memory_per_node="256Gi",
        )
        yaml_file = YAMLManager(f"{experiment}/build.yaml")
        yaml_file.write_yaml(build.to_dict())

        logger.info(f"build.yaml created: {experiment}")
        self.logging_handler.flush()
        logger.debug("current dir", os.getcwd())
        logger.info(f"starting tuning in gb: {experiment}")
        command = [
            "build",
            "start",
            "-f",
            f"{experiment}/build.yaml",
            "--tag",
            "autotunex",
        ]
        logger.debug(f"command for build start: {command}")
        result = await gb.command_executor(command)
        output = result.strip().replace("\r", "\n")
        logger.info(f"Command executor output: {output}")
        url, build_id = extract_github_url(result)
        logger.info(f"Github PR url: {url} - {build_id}")
        if url is not None or build_id is not None:
            await self.db.update_job_status(
                id=self.job_id, status=models.JobStatus.RUNNING
            )
            task = models.Task(
                job_id=str(self.job_id),
                type=models.TaskType.TUNING,
                pr_url=url.strip() if url else None,
                build_id=build_id.strip() if build_id else None,
            )
            task_dict = await self.db.get_task_by_job_id(
                job_id=self.job_id, type=models.TaskType.TUNING
            )
            if task_dict is not None:
                task.id = task_dict.get("id")
                await self.db.update_task(task=task)
                logger.debug(f"Tuning task updated: {task.id}")
            else:
                task_id = await self.db.insert_task(task=task)
                logger.debug(f"Tuning task inserted: {task_id}")
            logger.debug(f"Tuning task: {task}")
        else:
            logger.error("Github PR not found")
            raise Exception("Github PR not found")
        self.logging_handler.flush()

    async def build_start_cmd(self, run: models.TuningConfig, job_id: str):
        configuration = await self.db.get_config(run.config_id)
        dataset = await self.db.get_dataset(run.dataset_id)
        AUTOTUNE_SERVER_BRIDGE_URL = os.getenv("AUTOTUNE_SERVER_BRIDGE_URL")

        # Determine model path based on model source
        model_source = getattr(run, "model_source", models.ModelSource.HUGGINGFACE)
        if model_source == models.ModelSource.DMF:
            # For DMF models, use Granite Build binding template
            model_path = "{{ bindings.model_to_tune.binding.path }}"
        else:
            # For HuggingFace models, use direct string
            model_path = run.model

        cmd = (
            f"export CUDA_HOME=/usr/local/cuda-12.4 && "
            f"export LOG_PATH=$OUTPUT_PATH && "
            f"python main.py "
            f"--config_file /tmp/{configuration.get('name')}.yaml "
            f"--train_file {{{{ bindings.dataset_files.binding.path }}}}/{dataset.get('name')}_train.{dataset.get('data_format', 'jsonl')} "
            f"--validation_file {{{{ bindings.dataset_files.binding.path }}}}/{dataset.get('name')}_validation.{dataset.get('data_format', 'jsonl')} "
            f"--model_name_or_path {model_path} "
            f"--tuning_algo {configuration.get('tuner_type')} "
            f"{'--rl_algo ' + configuration.get('rl_tuner_type') + ' ' if configuration.get('rl_tuner_type') is not None else ''}"
            f"--run_name {run.experiment_name} "
            f"--output_dir $OUTPUT_PATH "
            f"--output_model_name {run.experiment_name} "
            f"--cleanup --save_history "
            f"{run.autotune is False and '--no_autotune ' or ''}"
            f"--job_id {job_id} "
            f"--autotunex_server_url {AUTOTUNE_SERVER_BRIDGE_URL}"
        )
        return cmd
