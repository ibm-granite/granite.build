# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class CustomCodeConfig(BaseModel):
    github_url: str
    setup_command: str
    start_command: str
    dir_to_save: str = "."


class K8sConfig(BaseModel):
    image: str
    additional_files: Dict[str, str]


class ComputeConfig(BaseModel):
    num_gpus_per_node: int
    num_nodes: int
    num_cpus_per_node: int


class StepConfigDict(BaseModel):
    custom_code_config: Dict[str, Any]
    k8s: Dict[str, Any]
    compute_config: Dict[str, Any]


class StepConfig(BaseModel):
    step_uri: str
    config: StepConfigDict


class InputOutput(BaseModel):
    uri: str


class Target(BaseModel):
    environment_uri: str
    inputs: Dict[str, InputOutput]
    outputs: Dict[str, InputOutput]
    steps: List[StepConfig]


class GraniteBuild(BaseModel):
    name: str
    targets: Dict[str, Target]

    def update_setup_command(self, new_setup_command: str) -> None:
        """
        Update the setup command in the custom code config

        Args:
            new_setup_command: The new setup command to use
        """
        current_config = self.targets["custom"].steps[0].config
        current_custom_code = current_config.custom_code_config

        self.targets["custom"].steps[0].config = StepConfigDict(
            custom_code_config={
                "github_url": current_custom_code["github_url"],
                "setup_command": new_setup_command,
                "start_command": current_custom_code["start_command"],
                "dir_to_save": current_custom_code.get("dir_to_save", "."),
            },
            k8s=current_config.k8s,
            compute_config=current_config.compute_config,
        )

    def update_start_command(self, new_start_command: str) -> None:
        """
        Update the start command in the custom code config

        Args:
            new_start_command: The new start command to use
        """
        current_config = self.targets["custom"].steps[0].config.custom_code_config
        self.targets["custom"].steps[0].config = StepConfigDict(
            custom_code_config={
                "github_url": current_config["github_url"],
                "setup_command": current_config["setup_command"],
                "start_command": new_start_command,
                "dir_to_save": current_config.get("dir_to_save", "."),
            },
            k8s=self.targets["custom"].steps[0].config.k8s,
            compute_config=self.targets["custom"].steps[0].config.compute_config,
        )

    def get_setup_command(self) -> str:
        """
        Get the current setup command

        Returns:
            str: The current setup command
        """
        return (
            self.targets["custom"]
            .steps[0]
            .config["custom_code_config"]["setup_command"]
        )

    def get_start_command(self) -> str:
        """
        Get the current start command

        Returns:
            str: The current start command
        """
        return (
            self.targets["custom"]
            .steps[0]
            .config["custom_code_config"]["start_command"]
        )

    def add_additional_file(self, file_path: str, content: str) -> None:
        """Add a file to k8s.additional_files in the build config.

        Args:
            file_path: The path where the file will be available in the container
                       (e.g., /tmp/reward_function.py)
            content: The file content as a string
        """
        self.targets["custom"].steps[0].config.k8s["additional_files"][
            file_path
        ] = content

    def update_compute_config(
        self,
        num_gpus_per_node: Optional[int] = None,
        num_cpus_per_node: Optional[int] = None,
        num_nodes: Optional[int] = None,
        total_memory_per_node: Optional[str] = None,
    ) -> None:
        """
        Update the compute configuration

        Args:
            num_gpus_per_node: Number of GPUs per node
            num_cpus_per_node: Number of CPUs per node
            num_nodes: Number of nodes
            total_memory_per_node: Available memory per node
        """
        current_config = self.targets["custom"].steps[0].config
        current_compute = current_config.compute_config

        self.targets["custom"].steps[0].config = StepConfigDict(
            custom_code_config=current_config.custom_code_config,
            k8s=current_config.k8s,
            compute_config={
                "num_gpus_per_node": (
                    num_gpus_per_node
                    if num_gpus_per_node is not None
                    else current_compute["num_gpus_per_node"]
                ),
                "num_cpus_per_node": (
                    num_cpus_per_node
                    if num_cpus_per_node is not None
                    else current_compute["num_cpus_per_node"]
                ),
                "num_nodes": (
                    num_nodes if num_nodes is not None else current_compute["num_nodes"]
                ),
                "total_memory_per_node": (
                    total_memory_per_node
                    if total_memory_per_node is not None
                    else current_compute["total_memory_per_node"]
                ),
            },
        )

    @classmethod
    def create_default_build(
        cls,
        name: str,
        dataset_uri: str,
        output_uri: str,
        github_url: str,
        config: str,
        config_name: str = "config",
        image: str = "us.icr.io/cil15-shared-registry/autotunex/build-runtime:21",
        environment_uri: str = "space://environments/{{ space.variables.DEFAULT_ENVIRONMENT }}",
        model_uri: Optional[str] = None,
    ) -> "GraniteBuild":
        """
        Create a default GraniteBuild configuration

        Args:
            name: Name of the build
            dataset_uri: URI for the dataset files
            output_uri: URI for the output
            github_url: GitHub repository URL
            config: Configuration YAML content as string
            config_name: Name for the config file (default: "config")
            image: Docker image to use
            environment_uri: Environment URI
            model_uri: Optional Lakehouse URI for DMF model (e.g., lh://prod/...)

        Returns:
            GraniteBuild: Configured GraniteBuild instance
        """
        print("Creating default GraniteBuild configuration")

        step_config = StepConfigDict(
            custom_code_config={
                "github_url": github_url,
                "setup_command": 'git checkout stage && pip install -e ".[full]" && pip list && nvidia-smi',
                "start_command": (
                    "export CUDA_HOME=/usr/local/cuda-12.4 && "
                    "export LOG_PATH=$OUTPUT_PATH && "
                    "python main.py "
                    f"--config_file /tmp/{config_name}.yaml "
                    "--train_file {{ bindings.dataset_files.binding.path }}/llml_train.jsonl "
                    "--validation_file {{ bindings.dataset_files.binding.path }}/llml_validation.jsonl "
                    "--model_name_or_path ibm-granite/granite-4.0-micro "
                    "--tuning_type lora "
                    "--run_name test-0311 "
                    "--output_dir $OUTPUT_PATH "
                    "--output_model_name test-0311 "
                    "--cleanup --save_history"
                ),
                "dir_to_save": ".",
            },
            k8s={
                "image": image,
                "additional_files": {f"/tmp/{config_name}.yaml": config},
            },
            compute_config={
                "num_gpus_per_node": 4,
                "num_nodes": 1,
                "num_cpus_per_node": 32,
            },
        )

        step = StepConfig(step_uri="space://steps/custom_code", config=step_config)

        # Build inputs dictionary based on whether model_uri is provided
        inputs = {"dataset_files": InputOutput(uri=dataset_uri)}
        if model_uri:
            # For DMF models, add model_to_tune input
            inputs["model_to_tune"] = InputOutput(uri=model_uri)

        target = Target(
            environment_uri=environment_uri,
            inputs=inputs,
            outputs={"custom": InputOutput(uri=output_uri)},
            steps=[step],
        )

        return cls(name=name, targets={"custom": target})

    def to_dict(self) -> Dict:
        """Convert the build configuration to a dictionary"""
        return {"granite.build": self.model_dump()}

    def to_yaml(self) -> str:
        """Convert the build configuration to YAML string"""
        import yaml

        def str_presenter(dumper, data):
            if len(data.split("\n")) > 1:  # check for multiline string
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.represent_scalar("tag:yaml.org,2002:str", data)

        # Register the custom presenter
        yaml.add_representer(str, str_presenter)

        # Use the custom dumper
        return yaml.dump(
            self.to_dict(), default_flow_style=False, sort_keys=False, width=1000
        )
