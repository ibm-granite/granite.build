from pathlib import Path

import pytest
import yaml


class TestS3PullStep:
    def test_step_yaml_exists(self):
        step_yaml = Path("src/gbserver/builtins/steps/s3pull/step.yaml")
        assert step_yaml.exists(), f"{step_yaml} does not exist"

    def test_step_yaml_valid(self):
        step_yaml = Path("src/gbserver/builtins/steps/s3pull/step.yaml")
        with open(step_yaml) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "s3pull"
        assert "s3pull_config" in config["config"]
        assert "environment_configs" in config

    def test_step_yaml_has_runpod_config(self):
        step_yaml = Path("src/gbserver/builtins/steps/s3pull/step.yaml")
        with open(step_yaml) as f:
            config = yaml.safe_load(f)
        env_configs = config["environment_configs"]
        assert "Runpod" in env_configs
        assert env_configs["Runpod"]["launchers"]["s3pull"]["type"] == "runpod"

    def test_step_yaml_has_bash_config(self):
        step_yaml = Path("src/gbserver/builtins/steps/s3pull/step.yaml")
        with open(step_yaml) as f:
            config = yaml.safe_load(f)
        env_configs = config["environment_configs"]
        assert "Bash" in env_configs

    def test_bash_command_script_exists(self):
        script = Path("src/gbserver/builtins/steps/s3pull/bash_scripts/s3pull/command.sh")
        assert script.exists(), f"{script} does not exist"

    def test_bash_command_script_uses_aws_s3(self):
        script = Path("src/gbserver/builtins/steps/s3pull/bash_scripts/s3pull/command.sh")
        content = script.read_text()
        assert "aws s3" in content or "rclone" in content
