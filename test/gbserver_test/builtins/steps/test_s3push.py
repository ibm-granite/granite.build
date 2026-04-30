from pathlib import Path

import pytest
import yaml


class TestS3PushStep:
    def test_step_yaml_exists(self):
        step_yaml = Path("src/gbserver/builtins/steps/s3push/step.yaml")
        assert step_yaml.exists()

    def test_step_yaml_valid(self):
        step_yaml = Path("src/gbserver/builtins/steps/s3push/step.yaml")
        with open(step_yaml) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "s3push"
        assert "s3push_config" in config["config"]

    def test_step_yaml_has_runpod_config(self):
        step_yaml = Path("src/gbserver/builtins/steps/s3push/step.yaml")
        with open(step_yaml) as f:
            config = yaml.safe_load(f)
        env_configs = config["environment_configs"]
        assert "Runpod" in env_configs

    def test_bash_command_script_exists(self):
        script = Path("src/gbserver/builtins/steps/s3push/bash_scripts/s3push/command.sh")
        assert script.exists()

    def test_bash_command_script_uses_aws_s3(self):
        script = Path("src/gbserver/builtins/steps/s3push/bash_scripts/s3push/command.sh")
        content = script.read_text()
        assert "aws s3" in content
