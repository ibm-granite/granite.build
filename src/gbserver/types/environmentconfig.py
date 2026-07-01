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

"""
The environment type.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from gbserver.types.config import Config

ENVIRONMENT_FILENAME = "environment.yaml"


class ClusterSshConfigs(Config):
    """Inline cluster SSH configs keyed by cloud.

    Each cloud holds a list of ``Host`` entries rendered verbatim into
    ``~/.<cloud>/config`` (the OpenSSH file SkyPilot's slurm/lsf provisioners
    read). Each entry is a mapping whose keys are the **exact OpenSSH directive
    names** from that file — ``Host``, ``HostName``, ``User``, ``Port``,
    ``IdentityFile``, ``IdentitiesOnly``, etc. — so the environment.yaml mirrors
    the config file 1:1 with no key translation.

    The ``Host`` value is the cluster alias SkyPilot references and is always
    literal; every other directive value is resolved by exact-name lookup against
    the environment's secrets, falling back to the literal when no secret matches.
    Multiple hosts per cloud are supported, so one environment can describe
    several clusters.

    Attributes:
        slurm: Host entries rendered into ``~/.slurm/config``.
        lsf: Host entries rendered into ``~/.lsf/config``.
    """

    slurm: Optional[List[Dict[str, Any]]] = None
    lsf: Optional[List[Dict[str, Any]]] = None


class AwsCredentialProfile(Config):
    """One profile in ``~/.aws/credentials``.

    Credential values are secret-resolved (secret-name-or-literal) so only
    secret *names* ever appear in environment.yaml. Materialized so the SkyPilot
    API server's boto3 can provision AWS and SkyPilot can upload the file to
    remote nodes for S3 access.

    Attributes:
        profile: The INI section name (e.g. ``default``).
        aws_access_key_id: Access key id (secret-name-or-literal).
        aws_secret_access_key: Secret access key (secret-name-or-literal).
        aws_session_token: Optional session token (secret-name-or-literal).
    """

    profile: str = "default"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None


class StoreLoad(Config):
    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class StorePush(Config):
    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class AssetStoreEnvironmentConfig(Config):
    store_uri: str = ""
    load: List[StoreLoad] = Field(default_factory=list)
    push: List[StorePush] = Field(default_factory=list)


class EnvironmentConfig(Config):
    """The environment.yaml file.

    Attributes:
        name: The user-facing name of the environment.
        type: The environment class identifier (e.g. ``Skypilot``, ``K8s``).
        config: Free-form environment-class-specific config block.
        assetstores: Per-environment assetstore mappings.
    """

    name: str
    type: str
    config: Dict = Field(default_factory=dict)
    assetstores: List[AssetStoreEnvironmentConfig] = Field(default_factory=list)
