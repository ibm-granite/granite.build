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

from typing import Dict, List, Optional, Union

from pydantic import Field

from gbserver.types.config import Config

ENVIRONMENT_FILENAME = "environment.yaml"


class ClusterSshHost(Config):
    """One SSH ``Host`` stanza for a SkyPilot slurm/lsf cluster.

    Rendered into ``~/.<cloud>/config`` (the OpenSSH file SkyPilot's slurm/lsf
    provisioners read). Each connection field is resolved by exact-name lookup
    against the environment's secrets; if no matching secret exists the literal
    value is used as-is. ``host`` (the cluster alias SkyPilot references) is
    always literal.

    Attributes:
        host: SSH ``Host`` alias / SkyPilot cluster name (literal).
        hostname: ``HostName`` value (secret-name-or-literal).
        user: ``User`` value (secret-name-or-literal).
        port: ``Port`` value (secret-name-or-literal).
        identity_file: ``IdentityFile`` path to an on-host key (secret-name-or-literal).
        options: Extra SSH directives; values are secret-name-or-literal.
    """

    host: str
    hostname: Optional[str] = None
    user: Optional[str] = None
    port: Optional[Union[int, str]] = None
    identity_file: Optional[str] = None
    options: Dict[str, str] = Field(default_factory=dict)


class ClusterSshConfigs(Config):
    """Inline cluster SSH configs keyed by cloud.

    Each list is rendered to ``~/.<cloud>/config``. Multiple hosts per cloud are
    supported (one ``Host`` block each), so a single environment can describe
    several clusters.

    Attributes:
        slurm: Hosts rendered into ``~/.slurm/config``.
        lsf: Hosts rendered into ``~/.lsf/config``.
    """

    slurm: Optional[List[ClusterSshHost]] = None
    lsf: Optional[List[ClusterSshHost]] = None


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
