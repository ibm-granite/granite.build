# Copyright Granite.Build Authors
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

"""Shared pytest fixtures for fm-tune unit tests."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_config_dict():
    """Load a minimal AutotuneConfig-shaped dict from YAML."""
    return yaml.safe_load((FIXTURES / "sample_config.yaml").read_text())


@pytest.fixture
def sample_config_yaml_path():
    """Path to the sample YAML config (for AutotuneConfig.load)."""
    return str(FIXTURES / "sample_config.yaml")


@pytest.fixture
def sample_train_jsonl(tmp_path):
    """Five-row simple-format JSONL training file."""
    p = tmp_path / "train.jsonl"
    rows = [{"input": f"q{i}", "output": f"a{i}"} for i in range(5)]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


@pytest.fixture
def sample_chat_jsonl(tmp_path):
    """Five-row chat-format JSONL training file."""
    p = tmp_path / "chat.jsonl"
    rows = [
        {
            "conversation": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(5)
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


@pytest.fixture
def mock_ray_trial():
    """Minimal stand-in for ray.tune.experiment.trial.Trial."""
    trial = MagicMock()
    trial.trial_id = "trial-0001"
    trial.config = {"learning_rate": 1e-4}
    return trial
