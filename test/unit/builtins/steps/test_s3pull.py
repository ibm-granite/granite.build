"""Unit tests for the builtin s3pull step.

After the env-class-match migration, each `environment_configs` block of the
original multi-env step.yaml lives in its own per-env file at
``src/gbserver/builtins/steps/<env-class-lower>/s3pull/``.  The tests below
exercise the Bash and Runpod splits as representatives of the migration shape;
the `Bash` split also carries the `bash_scripts/` payload referenced by its
launcher.
"""

from pathlib import Path

import yaml

BUILTINS_STEPS = Path("src/gbserver/builtins/steps")
BASH_S3PULL_YAML = BUILTINS_STEPS / "bash" / "s3pull" / "step.yaml"
RUNPOD_S3PULL_YAML = BUILTINS_STEPS / "runpod" / "s3pull" / "step.yaml"
BASH_S3PULL_SCRIPT = (
    BUILTINS_STEPS / "bash" / "s3pull" / "bash_scripts" / "s3pull" / "command.sh"
)


class TestS3PullStep:
    def test_bash_split_step_yaml_exists(self):
        assert BASH_S3PULL_YAML.exists(), f"{BASH_S3PULL_YAML} does not exist"

    def test_runpod_split_step_yaml_exists(self):
        assert RUNPOD_S3PULL_YAML.exists(), f"{RUNPOD_S3PULL_YAML} does not exist"

    def test_step_yaml_valid(self):
        with open(BASH_S3PULL_YAML) as f:
            config = yaml.safe_load(f)
        assert config["name"] == "s3pull"
        assert "s3pull_config" in config["config"]
        assert "environment_configs" in config

    def test_step_yaml_has_runpod_config(self):
        with open(RUNPOD_S3PULL_YAML) as f:
            config = yaml.safe_load(f)
        env_configs = config["environment_configs"]
        assert "Runpod" in env_configs
        assert env_configs["Runpod"]["launchers"]["s3pull"]["type"] == "runpod"

    def test_step_yaml_has_bash_config(self):
        with open(BASH_S3PULL_YAML) as f:
            config = yaml.safe_load(f)
        env_configs = config["environment_configs"]
        assert "Bash" in env_configs

    def test_bash_command_script_exists(self):
        assert BASH_S3PULL_SCRIPT.exists(), f"{BASH_S3PULL_SCRIPT} does not exist"

    def test_bash_command_script_uses_aws_s3(self):
        content = BASH_S3PULL_SCRIPT.read_text()
        assert "aws s3" in content or "rclone" in content
