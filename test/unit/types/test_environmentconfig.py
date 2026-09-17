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

import pytest

from gbserver.types.environmentconfig import EnvironmentConfig


def _sf():
    return {
        "provider": "efs",
        "mount_point": "/mnt/gb-shared",
        "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
    }


def test_shared_filesystem_allowed_on_skypilot_aws():
    cfg = EnvironmentConfig.model_validate(
        {
            "name": "e",
            "type": "Skypilot",
            "subtype": "aws",
            "config": {"shared_filesystem": _sf()},
        }
    )
    assert cfg.config["shared_filesystem"]["mount_point"] == "/mnt/gb-shared"


def test_shared_filesystem_rejected_off_aws():
    with pytest.raises(ValueError, match="only supported on a Skypilot/aws"):
        EnvironmentConfig.model_validate(
            {"name": "e", "type": "K8s", "config": {"shared_filesystem": _sf()}}
        )


def test_shared_filesystem_and_shared_workdir_both_set_errors():
    with pytest.raises(ValueError, match="exactly one"):
        EnvironmentConfig.model_validate(
            {
                "name": "e",
                "type": "Skypilot",
                "subtype": "aws",
                "config": {"shared_filesystem": _sf(), "shared_workdir": "/mnt/x"},
            }
        )


def test_hf_inline_coexist_warns(caplog):
    with caplog.at_level(logging.WARNING):
        EnvironmentConfig.model_validate(
            {
                "name": "e",
                "type": "Skypilot",
                "subtype": "aws",
                "config": {"shared_filesystem": _sf()},
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
