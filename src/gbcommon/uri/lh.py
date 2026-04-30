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

"""URI pointing to assets in Lakehouse"""

from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Type
from urllib.parse import ParseResult, urlparse, urlunparse

from gbcommon.uri.uri import URI
from gbserver.types.artifact import ArtifactType
from gbserver.types.constants import GB_PUBLIC_ARTIFACT_NAMESPACE, LAKEHOUSE_ENVIRONMENT
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

URLSEGMENT_TABLES = "tables"
URLSEGMENT_MODELS = "models"
URLSEGMENT_FILES = "filesets"
URLSEGMENT_DATASETS = "datasets"

DEFAULT_MODEL_REVISION = "granite-dot-build"
DEFAULT_FILESET_VERSION = "granite-dot-build"

LH_URI_SCHEME = "lh"


class LhType(StrEnum):
    """Different types of LhURI"""

    TABLE = auto()
    DATASET = auto()
    MODEL = auto()
    FILESET = auto()

    @classmethod
    def convert_to_artifact_type(cls: Type[Self], lhtype: "LhType") -> ArtifactType:
        """Returns the corresponding artifact type."""
        if lhtype is None:
            return ArtifactType.UNDEFINED
        match (lhtype):
            case LhType.TABLE:
                return ArtifactType.TABLE
            case LhType.DATASET:
                return ArtifactType.DATASET
            case LhType.MODEL:
                return ArtifactType.MODEL
            case LhType.FILESET:
                return ArtifactType.FILESET
            case _:
                return ArtifactType.UNDEFINED


# Indexes of elements when splitting on '/' in the uri
NAMESPACE_INDEX = 1
TYPE_INDEX = 2
TABLENAME_INDEX = 3
DATASET_INDEX = 4
MODEL_LABEL_INDEX = 4
FILESET_LABEL_INDEX = 4
MODEL_REVISION_INDEX = 5
FILESET_REVISION_INDEX = 5

STAGING_HOST = "staging"
PRODUCTION_HOST = "prod"


class LhURI(URI):
    """Defines support for a g.b url for entities stored in Lakehouse.  The uri encodes the following
        host
        type of asset (table, dataset, model, file)
        namespace - lakhouse namespace holding the table that holds this entity
        name - varies depending on type
            table: table name
            dataset: <tablename>/<datasetname>
            model: <modellabel>/<revision>

        format is as follows:

            table: lh://(prod|staging|<hostname>)/<namespace>/tables/<tablename>
            model: lh://(prod|staging|<hostname>)/<namespace>/models/<tablename>/<modellabel>/<revision>
            fileset: lh://(prod|staging|<hostname>)/<namespace>/filesets/<tablename>/<filesetlabel>/<version>
            dataset : lh://(prod|staging|<hostname>)/<namespace>/datasets/<tablename>/<dataset name>


    Args:
        URI (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self: Self,
        uri: Optional[ParseResult] = None,
        context: Optional[str] = None,
        secrets: Optional[Dict] = None,
        **kwargs: Dict,
    ) -> None:
        """
        Override to make sure model/fileset uris always have a revision/version.
        If the uri does NOT include a revision, then we add default g.b  revision.version
        """
        original_uri = uri
        if uri is not None:
            lh_type = LhURI.__get_lh_type(uri.path)
            newuri = None
            if lh_type is LhType.MODEL:
                revision = LhURI.__get_lh_model_revision(uri.path)
                uristr = urlunparse(uri)
                if not revision in uristr:
                    if not uristr.endswith("/"):
                        revision = f"/{revision}"
                    newuri = f"{uristr}{revision}"
            elif lh_type is LhType.FILESET:
                version = LhURI.__get_lh_fileset_version(uri.path)
                uristr = urlunparse(uri)
                if not version in uristr:
                    if not uristr.endswith("/"):
                        version = f"/{version}"
                    newuri = f"{uristr}{version}"
            if newuri is not None:
                uri = urlparse(newuri)

        super().__init__(uri, context, secrets, **kwargs)
        try:
            table_name = self.get_lh_table_name()
            if table_name == "":
                raise ValueError(f"The table name cannot be empty in a {LH_URI_SCHEME}:// URI")
            lh_type = self.get_lh_type()
            if lh_type is None:
                supported_types = self.get_supported_lh_types()
                raise ValueError(f"unsupported lh_type, supported ones are {supported_types}")
        except Exception as e:
            raise ValueError(f"failed to create from uri: {original_uri}") from e

    @staticmethod
    def _get_uri_from_name(
        uri_suffix: str,
        lh_env: str = LAKEHOUSE_ENVIRONMENT,
        lh_type: LhType = LhType.TABLE,
        namespace: str = GB_PUBLIC_ARTIFACT_NAMESPACE,
    ):
        lh_env = lh_env.lower()
        if lh_env not in (STAGING_HOST, PRODUCTION_HOST):
            # Convert old hostnames to new identifiers.
            if "staging" in lh_env:
                lh_env = STAGING_HOST
            else:
                lh_env = PRODUCTION_HOST
        return urlunparse(
            (
                LH_URI_SCHEME,
                lh_env,
                namespace + "/" + LhURI._get_urisegment_for_type(lh_type) + "/" + uri_suffix,
                None,
                None,
                None,
            )
        )

    @staticmethod
    def get_model_uri(
        table_name: str,
        model_label: str,
        model_revision: str = DEFAULT_MODEL_REVISION,
        lh_env: str = LAKEHOUSE_ENVIRONMENT,
        namespace: str = GB_PUBLIC_ARTIFACT_NAMESPACE,
    ) -> str:
        """Get the model URI."""
        if not model_revision:
            model_revision = DEFAULT_MODEL_REVISION
        suffix = table_name + "/" + model_label + "/" + model_revision
        return LhURI._get_uri_from_name(suffix, lh_env, LhType.MODEL, namespace)

    @staticmethod
    def get_fileset_uri(
        table_name: str,
        fileset_label: str,
        fileset_version: str = DEFAULT_FILESET_VERSION,
        lh_env: str = LAKEHOUSE_ENVIRONMENT,
        namespace: str = GB_PUBLIC_ARTIFACT_NAMESPACE,
    ) -> str:
        """Get the fileset URI."""
        if not fileset_version:
            fileset_version = DEFAULT_FILESET_VERSION
        suffix = table_name + "/" + fileset_label + "/" + fileset_version
        return LhURI._get_uri_from_name(suffix, lh_env, LhType.FILESET, namespace)

    @staticmethod
    def get_dataset_uri(
        dataset_name: str,
        table_name: str,
        lh_env: str = LAKEHOUSE_ENVIRONMENT,
        namespace: str = GB_PUBLIC_ARTIFACT_NAMESPACE,
    ) -> str:
        """Get the dataset URI."""
        suffix = table_name + "/" + dataset_name
        return LhURI._get_uri_from_name(suffix, lh_env, LhType.DATASET, namespace)

    @staticmethod
    def get_table_uri(
        table_name: str,
        lh_env: str = LAKEHOUSE_ENVIRONMENT,
        namespace: str = GB_PUBLIC_ARTIFACT_NAMESPACE,
    ) -> str:
        """Get the table URI."""
        suffix = table_name
        return LhURI._get_uri_from_name(suffix, lh_env, LhType.TABLE, namespace)

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [LH_URI_SCHEME]

    @staticmethod
    def _get_urisegment_for_type(lh_type: LhType = LhType.TABLE) -> str:
        match lh_type:
            case LhType.TABLE:
                return URLSEGMENT_TABLES
            case LhType.MODEL:
                return URLSEGMENT_MODELS
            case LhType.FILESET:
                return URLSEGMENT_FILES
            case LhType.DATASET:
                return URLSEGMENT_DATASETS
        return ""

    def exists(self: Self, force: bool = False) -> bool:
        # TODO: Fix logic
        return True

    def is_accessible(self: Self) -> bool:
        # TODO: Fix logic
        return True

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        # TODO: Fix logic
        raise NotImplementedError("LhURI pull is not implemented")

    def delete(self: Self) -> bool:
        """Delete the Lakehouse resource referenced by this URI.

        Only TABLE type deletion is currently supported via BaseLakehouseStorage.

        Returns:
            True if deletion succeeded, False on error.

        Raises:
            NotImplementedError: If the LH type is not TABLE.
        """
        lh_type = self.get_lh_type()
        if lh_type == LhType.TABLE:
            from gbserver.storage.lh.lh_storage import BaseLakehouseStorage

            try:
                BaseLakehouseStorage().delete_table_in_namespace(
                    namespace=self.get_lh_namespace(),
                    table_name=self.get_lh_table_name(),
                )
                return True
            except Exception as e:
                logger.warning("Could not delete LH artifact %s: %s", self, e)
                return False
        raise NotImplementedError(f"LhURI delete is not implemented for LH type {lh_type}")

    def get_lh_namespace(self: Self) -> str:
        """Get the lakehouse namespace."""
        assert self.uri is not None, "self.uri is None"
        return self.uri.path.split("/")[NAMESPACE_INDEX]

    def get_lh_environment(self: Self) -> str:
        """Get the lakehouse environment PROD/STAGING"""
        assert self.uri is not None, "self.uri is None"
        assert self.uri.hostname is not None, "self.uri.hostname is None"
        return self.uri.hostname

    def is_prod(self: Self) -> bool:
        """Return True if this URI points to the production Lakehouse environment."""
        return self.get_lh_environment() == PRODUCTION_HOST

    @staticmethod
    def __get_lh_type(uri_path: str) -> Optional[LhType]:
        splits = uri_path.split("/")
        if len(splits) <= TYPE_INDEX:
            return None
        type_pattern = splits[TYPE_INDEX]
        if type_pattern == URLSEGMENT_TABLES:
            return LhType.TABLE
        if type_pattern == URLSEGMENT_MODELS:
            return LhType.MODEL
        if type_pattern == URLSEGMENT_FILES:
            return LhType.FILESET
        if type_pattern == URLSEGMENT_DATASETS:
            return LhType.DATASET
        return None

    @staticmethod
    def get_supported_lh_types() -> List[str]:
        """Get all the supported Lakehouse types."""
        return [
            URLSEGMENT_TABLES,
            URLSEGMENT_MODELS,
            URLSEGMENT_FILES,
            URLSEGMENT_DATASETS,
        ]

    def get_lh_type(self: Self) -> Optional[LhType]:
        """Get the current Lakehouse type."""
        assert self.uri is not None, "self.uri is None"
        uristr = self.uri.path
        return LhURI.__get_lh_type(uristr)

    def get_artifact_type(self) -> ArtifactType:
        """Return the artifact type by converting the Lakehouse resource type.

        Returns:
            ArtifactType: The artifact type, or UNDEFINED if the LH type is unknown.
        """
        return LhType.convert_to_artifact_type(self.get_lh_type())

    def get_lh_table_name(self: Self) -> str:
        """Get the current Lakehouse table name."""
        assert self.uri is not None, "self.uri is None"
        return self.uri.path.split("/")[TABLENAME_INDEX]

    def get_lh_dataset_name(self: Self) -> str:
        """Get the current Lakehouse dataset name."""
        assert self.get_lh_type() == LhType.DATASET, f"URI is not a dataset uri {self.uri}"
        assert self.uri is not None, "self.uri is None"
        parts = self.uri.path.split("/")
        assert len(parts) >= DATASET_INDEX + 1, f"URI did not include a dataset name: {self.uri}"
        return parts[DATASET_INDEX]

    def get_lh_model_label(self: Self) -> str:
        """Get the current Lakehouse model label."""
        assert self.get_lh_type() == LhType.MODEL, f"URI is not a model uri {self.uri}"
        assert self.uri is not None, "self.uri is None"
        parts = self.uri.path.split("/")
        assert len(parts) >= MODEL_LABEL_INDEX + 1, f"URI did not include a model label: {self.uri}"
        return parts[MODEL_LABEL_INDEX]

    @staticmethod
    def __get_lh_model_revision(uri_path: str) -> str:
        parts = uri_path.split("/")
        if len(parts) >= MODEL_REVISION_INDEX + 1 and len(parts[MODEL_REVISION_INDEX]) > 0:
            return parts[MODEL_REVISION_INDEX]
        return DEFAULT_MODEL_REVISION

    def get_lh_model_revision(self: Self) -> str:
        """Get the current Lakehouse model revision."""
        assert self.get_lh_type() == LhType.MODEL, f"URI is not a model uri {self.uri}"
        assert self.uri is not None, "self.uri is None"
        return LhURI.__get_lh_model_revision(self.uri.path)
        # parts = self.uri.path.split("/")
        # if len(parts) >= MODEL_REVISION_INDEX+1 and len(parts[MODEL_REVISION_INDEX]) > 0:
        #     return parts[MODEL_REVISION_INDEX]
        # else:
        #     return DEFAULT_MODEL_REVISION

    def get_lh_fileset_label(self: Self) -> str:
        """Get the current Lakehouse fileset label."""
        assert self.get_lh_type() == LhType.FILESET, f"URI is not a fileset uri {self.uri}"
        assert self.uri is not None, "self.uri is None"
        parts = self.uri.path.split("/")
        assert (
            len(parts) >= FILESET_LABEL_INDEX + 1
        ), f"URI did not include a fileset label: {self.uri}"
        return parts[FILESET_LABEL_INDEX]

    @staticmethod
    def __get_lh_fileset_version(uri_path: str) -> str:
        parts = uri_path.split("/")
        if len(parts) >= FILESET_REVISION_INDEX + 1 and len(parts[FILESET_REVISION_INDEX]) > 0:
            return parts[FILESET_REVISION_INDEX]
        return DEFAULT_FILESET_VERSION

    def get_lh_fileset_version(self: Self) -> str:
        """Get the current Lakehouse fileset version."""
        assert self.get_lh_type() == LhType.FILESET, f"URI is not a fileset uri {self.uri}"
        assert self.uri is not None, "self.uri is None"
        return LhURI.__get_lh_fileset_version(self.uri.path)
        # parts = self.uri.path.split("/")
        # if len(parts) >= FILESET_REVISION_INDEX+1 and len(parts[FILESET_REVISION_INDEX]) > 0:
        #     return parts[FILESET_REVISION_INDEX]
        # else:
        #     return DEFAULT_FILESET_VERSION

    def get_metadata(self: Self) -> Any:
        lh_type = self.get_lh_type()
        assert lh_type is not None, "lh_type is None"
        md = {
            "uri": self.get_uristr(self),
            "type": lh_type.value,
            "namespace": self.get_lh_namespace(),
            "type_segment": self._get_urisegment_for_type(lh_type),
            "table_name": self.get_lh_table_name(),
        }
        if lh_type == LhType.FILESET:
            md["fileset_label"] = self.get_lh_fileset_label()
            md["fileset_version"] = self.get_lh_fileset_version()
        elif lh_type == LhType.MODEL:
            md["model_label"] = self.get_lh_model_label()
            md["model_revision"] = self.get_lh_model_revision()
        elif lh_type == LhType.DATASET:
            md["dataset_name"] = self.get_lh_dataset_name()
        return md
