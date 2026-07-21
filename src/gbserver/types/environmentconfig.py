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

import logging
from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from gbserver.types.config import Config

# Module-local logger (standard logging) to avoid a types -> gbserver.utils import cycle.
logger = logging.getLogger(__name__)

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
    # Type of the ``pull`` entries on an assetstore (the input side). ``mode``
    # selects *how* an asset is loaded, but this is only meaningful in k8s
    # (afm_mount/cos_mount/cos_pull/dmf_pull/hf_pull branch there). Every other
    # environment dispatches by store *type* and ignores ``mode``; the preferred
    # value is ``"default"`` (or unset), and non-default values are accepted for
    # backwards compatibility but log a deprecation warning
    # (see Environment._warn_non_default_mode).
    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class StorePush(Config):
    # See StoreLoad.mode: meaningful only in k8s; non-k8s environments ignore it
    # (prefer ``"default"``; legacy values warn but are accepted).
    mode: Optional[str] = None
    config: Dict = Field(default_factory=dict)


class AssetStoreEnvironmentConfig(Config):
    store_uri: str = ""
    # ``pull`` (input) pairs with ``push`` (output) and matches the
    # ``pullasset_*``/``pushasset_*`` handler names. The key was formerly ``load``;
    # see ``_accept_legacy_load_key`` for the deprecated alias.
    pull: List[StoreLoad] = Field(default_factory=list)
    push: List[StorePush] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_load_key(cls, data: Any) -> Any:
        """Accept the pre-rename ``load`` key as a deprecated alias for ``pull``.

        The assetstore input key was renamed ``load`` -> ``pull``. Existing space
        configs may still use ``load``; map it onto ``pull`` and warn, rather than
        failing, so those configs keep working.

        :param data: the raw input for this model (a dict when parsed from YAML).
        :returns: the (possibly rewritten) input.
        """
        if isinstance(data, dict) and "load" in data:
            if "pull" not in data:
                data["pull"] = data.pop("load")
            else:
                # Both given: prefer the explicit ``pull`` and drop the legacy key.
                data.pop("load")
            logger.warning(
                "assetstores entry '%s': the 'load' key is deprecated; rename it to "
                "'pull' ('load' still works for now).",
                data.get("store_uri", ""),
            )
        return data


class EnvironmentConfig(Config):
    """The environment.yaml file.

    Attributes:
        name: The user-facing name of the environment.
        type: The environment class identifier (e.g. ``Skypilot``, ``K8s``).
        config: Free-form environment-class-specific config block.
        assetstores: Per-environment assetstore mappings.
        subtype: Optional free-form discriminator distinguishing environments
            that share the same ``type`` (e.g. the skypilot endpoints are all
            class ``Skypilot`` but differ by ``kubernetes``/``slurm``/``aws``/
            ``lsf``).  Used by the ``SpaceURI`` resolver to gate steps that
            declare a ``subtypes`` list on their ``environment_configs``: a step
            with a non-empty list matches only environments whose ``subtype`` is
            in it (exact string match).  Any string is accepted — there is no
            predefined set.  When unset, the environment matches only steps that
            declare no ``subtypes`` (steps with an empty list are universal).
    """

    name: str
    type: str
    config: Dict = Field(default_factory=dict)
    assetstores: List[AssetStoreEnvironmentConfig] = Field(default_factory=list)
    subtype: Optional[str] = None
