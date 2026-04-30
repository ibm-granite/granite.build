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

"""Access assets in Lakehouse."""

import os
import time
from pathlib import Path
from typing import Dict, Union

from gbcommon.uri.lh import LhType, LhURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.types.artifact import ArtifactType
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def _import_lakehouse():
    """Lazy import lakehouse library. Raises ImportError with clear message if not installed."""
    try:
        from lakehouse.api import ConfigMap
        from lakehouse.assets.dataset import Dataset
        from lakehouse.assets.fileset import Fileset
        from lakehouse.assets.model import Model
        from lakehouse.assets.table import Table

        return ConfigMap, Dataset, Fileset, Model, Table
    except ImportError as e:
        raise ImportError(
            "The 'lakehouse' (dmf-lib) library is required for Lakehouse asset storage. "
            "Install it with: pip install dmf-lib"
        ) from e


class Lhstore(Assetstore):
    """A class for storing and accessing Lakehouse assets."""

    def __init__(self, uri: Union[URI, str], **kwargs):
        _import_lakehouse()  # fail fast with clear error
        super().__init__(uri, **kwargs)

    @classmethod
    def get_supported_uri_classes(self):
        return [LhURI]

    def get_metadata(self, uri: URI) -> Dict:
        lhenv = (
            self.config.config["env"]
            if self.config is not None
            and isinstance(self.config.config, dict)
            and "env" in self.config.config
            else "STAGING"
        )
        lhtokenkey = (
            self.config.config["token_secretname"]
            if self is not None
            and self.config is not None
            and isinstance(self.config.config, dict)
            and "token_secretname" in self.config.config
            else "LAKEHOUSE_TOKEN"
        )
        return {"env": lhenv, "token_secretname": lhtokenkey}

    def get_subdir(self, uri: URI) -> str:
        """Get the subdir."""
        assert isinstance(uri, LhURI)
        asset_type = uri.get_lh_type()
        subdir_path = ""
        match asset_type:
            case LhType.TABLE:
                subdir_path = uri.get_lh_table_name()
            case LhType.MODEL:
                subdir_path = str(Path(uri.get_lh_model_label()) / uri.get_lh_model_revision())
            case LhType.FILESET:
                subdir_path = str(Path(uri.get_lh_fileset_label()) / uri.get_lh_fileset_version())
            case LhType.DATASET:
                subdir_path = uri.get_lh_dataset_name()
        return subdir_path

    def get_relpath(self, uri: URI) -> str:
        """Get the relpath."""
        lhtokensecret = (
            self.config.config["token_secretname"]
            if self is not None
            and self.config is not None
            and isinstance(self.config.config, dict)
            and "token_secretname" in self.config.config
            else "LAKEHOUSE_TOKEN"
        )
        lhenv = (
            self.config.config["env"]
            if self is not None
            and self.config is not None
            and isinstance(self.config.config, dict)
            and "env" in self.config.config
            else "STAGING"
        )
        if self.secrets is not None and lhtokensecret in self.secrets:
            token = self.secrets[lhtokensecret]
            if token == "":
                raise ValueError(
                    f"failed to get the Lakehouse token from the secrets using the key '{lhtokensecret}'"
                )
        else:
            token = os.getenv(lhtokensecret, "")
            if token == "":
                raise ValueError(
                    f"failed to get the Lakehouse token from the env variable '{lhtokensecret}'"
                )
        logger.info("Getting a Lakehouse client for the env '%s'", lhenv)
        from gbserver.utils.lakehouse_utils import create_lakehouse_iceberg

        ConfigMap, Dataset, Fileset, Model, Table = _import_lakehouse()
        lh = create_lakehouse_iceberg(
            config="map", conf_map=ConfigMap(environment=lhenv, token=token)
        )
        assert isinstance(uri, LhURI)
        asset_type = uri.get_lh_type()
        cos_path = ""
        start_time = time.time()
        # Will throw UnauthorizedException
        match asset_type:
            case LhType.TABLE:
                table = Table(
                    lh=lh,
                    namespace=uri.get_lh_namespace(),
                    table_name=uri.get_lh_table_name(),
                )
                cos_path = os.path.join(table.cos_location(), "data", "*.parquet")
            case LhType.MODEL:
                model = Model(lh=lh)
                cos_path = model.cos_location(
                    namespace=uri.get_lh_namespace(),
                    model=uri.get_lh_model_label(),
                    table=uri.get_lh_table_name(),
                    revision=uri.get_lh_model_revision(),
                )
            case LhType.FILESET:
                fileset = Fileset(
                    lh=lh,
                    namespace=uri.get_lh_namespace(),
                    table=uri.get_lh_table_name(),
                )
                cos_path = fileset.cos_location(
                    label=uri.get_lh_fileset_label(),
                    version=uri.get_lh_fileset_version(),
                )
            case LhType.DATASET:
                dataset = Dataset(
                    lh=lh,
                    namespace=uri.get_lh_namespace(),
                    table_name=uri.get_lh_table_name(),
                    dataset_name=uri.get_lh_dataset_name(),
                )
                cos_path = dataset.cos_location()
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 30.0:  # Takes more than 30 seconds
            logger.warning(f"Taking too long retrieving cos_path elapsed_time={elapsed_time}")
        rel_path = cos_path.removeprefix("s3a://")
        slash_loc = rel_path.find("/")
        rel_path = rel_path[slash_loc + 1 :]
        return rel_path

    def get_asset_type(self, uri: URI) -> ArtifactType:
        assert isinstance(uri, LhURI)
        lhtype = uri.get_lh_type()
        if lhtype is None:
            return ArtifactType.UNDEFINED
        return LhType.convert_to_artifact_type(lhtype)
