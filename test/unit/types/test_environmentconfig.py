#!/usr/bin/env python3

# Copyright LLM.build Authors
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

import logging
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from gbserver.types.environmentconfig import EnvironmentConfig


def _sf():
    return {
        "provider": "efs",
        "mount_point": "/mnt/gb-shared",
        "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
    }


def _aws_env(config):
    return {
        "name": "e",
        "type": "Skypilot",
        "subtype": "aws",
        "config": config,
        "assetstores": [],
    }


_SF = {
    "provider": "efs",
    "mount_point": "/mnt/gb-shared",
    "efs": {"file_system_id": "fs-0abc123", "region": "us-east-1"},
}


def test_shared_filesystem_allowed_on_skypilot_aws():
    cfg = EnvironmentConfig.model_validate(
        {
            "name": "e",
            "type": "Skypilot",
            "subtype": "aws",
            "config": {
                "default_cloud": "aws",
                "shared_workdir": "/mnt/gb-shared/gbroot",
                "shared_filesystem": _sf(),
            },
        }
    )
    assert cfg.config["shared_filesystem"]["mount_point"] == "/mnt/gb-shared"


def test_shared_filesystem_requires_shared_workdir():
    cfg = {"default_cloud": "aws", "shared_filesystem": _SF}
    with pytest.raises(ValidationError, match="requires 'shared_workdir'"):
        EnvironmentConfig.model_validate(_aws_env(cfg))


def test_shared_workdir_must_be_under_mount_point():
    cfg = {
        "default_cloud": "aws",
        "shared_workdir": "/somewhere/else",
        "shared_filesystem": _SF,
    }
    with pytest.raises(
        ValidationError, match="must be under shared_filesystem.mount_point"
    ):
        EnvironmentConfig.model_validate(_aws_env(cfg))


def test_shared_workdir_must_be_absolute():
    cfg = {
        "default_cloud": "aws",
        "shared_workdir": "gbroot",
        "shared_filesystem": _SF,
    }
    with pytest.raises(
        ValidationError, match="must be under shared_filesystem.mount_point"
    ):
        EnvironmentConfig.model_validate(_aws_env(cfg))


def test_valid_shared_filesystem_with_workdir_subdir():
    cfg = {
        "default_cloud": "aws",
        "shared_workdir": "/mnt/gb-shared/gbroot",
        "shared_filesystem": _SF,
    }
    env = EnvironmentConfig.model_validate(_aws_env(cfg))
    assert (env.config or {}).get("shared_workdir") == "/mnt/gb-shared/gbroot"


def test_shared_workdir_may_equal_mount_point():
    cfg = {
        "default_cloud": "aws",
        "shared_workdir": "/mnt/gb-shared",
        "shared_filesystem": _SF,
    }
    EnvironmentConfig.model_validate(_aws_env(cfg))  # no raise


def test_legacy_shared_workdir_alone_still_valid():
    env = EnvironmentConfig.model_validate(
        {
            "name": "e",
            "type": "Skypilot",
            "subtype": "slurm",
            "config": {"shared_workdir": "/shared"},
            "assetstores": [],
        }
    )
    assert (env.config or {}).get("shared_workdir") == "/shared"


def test_committed_fixture_env_validates():
    p = (
        Path(__file__).parents[3]
        / "test-data/integration/ibm/buildrunner/skypilot/aws/shared-fs/space"
        / "environments/skypilot/aws-shared-fs/environment.yaml"
    )
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    env = EnvironmentConfig.model_validate(data)
    wd = (env.config or {}).get("shared_workdir")
    mp = env.config["shared_filesystem"]["mount_point"]
    assert wd and (wd == mp or wd.startswith(mp.rstrip("/") + "/"))


def test_shared_filesystem_rejected_off_aws():
    with pytest.raises(ValueError, match="only supported on a Skypilot/aws"):
        EnvironmentConfig.model_validate(
            {"name": "e", "type": "K8s", "config": {"shared_filesystem": _sf()}}
        )


def test_shared_filesystem_rejected_when_default_cloud_not_aws():
    # subtype gate passes but default_cloud (what mount/teardown actually key on)
    # is non-aws -> reject at load rather than mounting/tearing down on k8s.
    with pytest.raises(ValueError, match="requires 'default_cloud: aws'"):
        EnvironmentConfig.model_validate(
            {
                "name": "e",
                "type": "Skypilot",
                "subtype": "aws",
                "config": {
                    "default_cloud": "kubernetes",
                    "shared_workdir": "/mnt/gb-shared/gbroot",
                    "shared_filesystem": _sf(),
                },
            }
        )


def test_shared_filesystem_rejected_when_default_cloud_unset():
    # Unset default_cloud => skypilot _get_cloud() falls back to k8s, so the
    # gate must reject it too (not just an explicit non-aws value).
    with pytest.raises(ValueError, match="requires 'default_cloud: aws'"):
        EnvironmentConfig.model_validate(
            {
                "name": "e",
                "type": "Skypilot",
                "subtype": "aws",
                "config": {
                    "shared_workdir": "/mnt/gb-shared/gbroot",
                    "shared_filesystem": _sf(),
                },
            }
        )


def test_hf_inline_coexist_warns(caplog):
    with caplog.at_level(logging.WARNING):
        EnvironmentConfig.model_validate(
            {
                "name": "e",
                "type": "Skypilot",
                "subtype": "aws",
                "config": {
                    "default_cloud": "aws",
                    "shared_workdir": "/mnt/gb-shared/gbroot",
                    "shared_filesystem": _sf(),
                },
                "assetstores": [
                    {
                        "store_uri": "space://assetstores/hf",
                        "pull": [
                            {
                                "mode": "default",
                                "config": {
                                    "inline": True,
                                    "cache_path": "/tmp/hf_cache",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    assert "will not cache to the shared filesystem" in caplog.text


def test_hf_local_cache_path_coexist_warns(caplog):
    with caplog.at_level(logging.WARNING):
        EnvironmentConfig.model_validate(
            {
                "name": "e",
                "type": "Skypilot",
                "subtype": "aws",
                "config": {
                    "default_cloud": "aws",
                    "shared_workdir": "/mnt/gb-shared/gbroot",
                    "shared_filesystem": _sf(),
                },
                "assetstores": [
                    {
                        "store_uri": "space://assetstores/hf",
                        "pull": [
                            {
                                "mode": "default",
                                "config": {
                                    "cache_path": "/tmp/hf_cache",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    assert "will not cache to the shared filesystem" in caplog.text


def test_hf_cache_path_under_mount_point_no_warn(caplog):
    with caplog.at_level(logging.WARNING):
        EnvironmentConfig.model_validate(
            {
                "name": "e",
                "type": "Skypilot",
                "subtype": "aws",
                "config": {
                    "default_cloud": "aws",
                    "shared_workdir": "/mnt/gb-shared/gbroot",
                    "shared_filesystem": _sf(),
                },
                "assetstores": [
                    {
                        "store_uri": "space://assetstores/hf",
                        "pull": [
                            {
                                "mode": "default",
                                "config": {
                                    "cache_path": "/mnt/gb-shared/hf_cache",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    assert "will not cache to the shared filesystem" not in caplog.text
