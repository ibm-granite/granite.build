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

from pathlib import Path
from typing import Any, Optional, Self, Union

import pandas as pd

from gbserver.utils.optional_imports import HAS_LAKEHOUSE

if HAS_LAKEHOUSE:
    from lakehouse.assets.table import Table
    from lakehouse.core import TableDetails

    from gbserver.storage.lh.lh_storage import BaseLakehouseStorage

from gbserver.types.constants import (
    GB_PUBLIC_ARTIFACT_NAMESPACE,
    GRANITE_DOT_BUILD_ADMIN_NAMESPACE,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

LH_SCHEME = "lh"
DEFAULT_LH_HOST = "ui.dmf.vpc-int.res.ibm.com"


class LakehouseLoader(BaseLakehouseStorage):

    default_namespace: str = GB_PUBLIC_ARTIFACT_NAMESPACE
    logger: Any = None  # Set in the initializer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = logger

    # @staticmethod
    # def get_lh_uri(namespace:str, table_name:str, host:str=DEFAULT_LH_HOST) -> str:
    #     """ Get the referencable URI for a given lakehouse table in the given namespace.

    #     Args:
    #         namespace (str): _description_
    #         table_name (str): _description_
    #         host (str, optional): _description_. Defaults to DEFAULT_LH_HOST.

    #     Returns:
    #         str: _description_
    #     """
    #     uri = f"https://{host}/lakehouse/{namespace}/{table_name}"
    #     #parse = urlparse(uri)
    #     #print(f"parse={parse}")
    #     return uri

    # @staticmethod
    # def get_gb_uri(namespace:str, table_name:str, host:str=DEFAULT_LH_HOST) -> str:
    #     """ Get the g.b pseudo-URI for a the named table in the given namespace.
    #     Considered a pseudo-URI as it may only be referencable via g.b.

    #     Args:
    #         namespace (str): _description_
    #         table_name (str): _description_
    #         host (str, optional): _description_. Defaults to DEFAULT_LH_HOST.

    #     Returns:
    #         str: _description_
    #     """
    #     uri = f"lh://{host}?ns={namespace}&table={table_name}"
    #     #parse = urlparse(uri)
    #     #print(f"parse={parse}")
    #     return uri

    # @staticmethod
    # @deprecated(message="in favor of parse_gb_uri()")
    # def parse_uri(lakehouse_uri:str) -> tuple[str,str]:
    #     return LakehouseLoader.parse_gb_uri(lakehouse_uri)

    # @staticmethod
    # def parse_gb_uri(lakehouse_uri:str) -> tuple[str,str]:
    #     parse = urlparse(lakehouse_uri)
    #     if parse.scheme != LH_SCHEME:
    #         raise ValueError(f"URI scheme must be {LH_SCHEME}")
    #     #print(f"parse={parse}")
    #     if len(parse.query) == 0:
    #         raise ValueError("URI does not contain query parameters containing namespace and/or table name")
    #     params = parse.query.split("&")
    #     if len(params) != 2:
    #         raise ValueError("URI does not contain namespace and table name")
    #     _, namespace = params[0].split("=")
    #     _, table_name = params[1].split("=")
    #     #ParseResult(scheme='http', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html',
    #     #    params='', query='', fragment='')
    #     return namespace, table_name

    # def _get_gb_uri(self:Self, table_name:str) -> str:
    #     parse = urlparse(self.lh.aws_endpoint)
    #     uri = LakehouseLoader.get_gb_uri(self.namespace,table_name, parse.hostname)
    #     #uri = self.lh.aws_endpoint + "?ns=" + self.namespace + "&table=" + table_name
    #     #uri = str(uri).replace("https",LH_SCHEME, 1)
    #     #uri = str(uri).replace("http",LH_SCHEME, 1)
    #     #parse = urlparse(uri)
    #     #print(f"parse={parse}")
    #     return uri

    def __get_table(self: Self, namespace: str, table_name: str) -> Table:
        try:
            table = Table(lh=self.lh, namespace=namespace, table_name=table_name)
            return table
        except Exception as exc:
            raise exc

    def upload(
        self: Self,
        file_path: Union[str, Path],
        table_name: str,
        namespace: Optional[str] = None,
    ) -> None:
        """Upload the file into a new table with the given name in this instance's namespace.
        File types supported include .parquet and .jsonl

        Args:
            file_path (Path): _description_
            table_name (str): _description_
            namespace(str): if not provided then use this instance's namespace

        Raises:
            ValueError: unsupported file type/extension.

        """
        if namespace is None:
            namespace = self.default_namespace
        self.logger.info(
            f"Begin upload of {file_path} to table {namespace}.{table_name}"
        )
        if isinstance(file_path, Path):
            file_path = str(file_path.name)
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        elif file_path.endswith(".jsonl"):
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"File extension not supported: {file_path}")

        table_details = TableDetails(
            namespace=namespace,
            name=table_name,
            is_public=True,
            # properties={"property1":"test_value"}
        )
        # pa_table = pa.Table.from_pandas(df)
        # namespace(str): if not provided then use this instance's namespace
        # table = Table.from_arrow_schema(lh=self.lh, schema=pa_table.schema, table_details=table_details)
        # table.append_dataframe(df=df)
        Table.from_dataframe(lh=self.lh, df=df, table_details=table_details)

        self.logger.info(
            f"Done upload of {file_path} to table {namespace}.{table_name}"
        )

    def download(
        self: Self, table_name: str, file_path: Path, namespace: Optional[str] = None
    ) -> None:
        """
        Download the contents of the given table in this instance's namespace to the named file.
        The extension determines the format of the resulting file.

        Args:
            table_name(str): name of table to load.
            file_path (Path): where to put the table contents.
            namespace(str): if not provided then use this instance's namespace

        Raises:
            ValueError: file type/extension is not supported.

        """
        if namespace is None:
            namespace = self.default_namespace
        if isinstance(file_path, Path):
            file_path = str(file_path.name)
        self.logger.info(
            f"Begin download of from table {namespace}.{table_name} to {file_path}"
        )
        table = self.__get_table(namespace, table_name)
        df = table.to_pandas()
        if file_path.endswith(".parquet"):
            df.to_parquet(file_path)
        elif file_path.endswith(".jsonl"):
            df.to_json(file_path, orient="records", lines=True)
        else:
            raise ValueError(f"File extension not supported: {file_path}")
        self.logger.info(
            f"End  download of table {namespace}.{table_name} to {file_path}"
        )


if __name__ == "__main__":
    ll = LakehouseLoader(default_namespace=GRANITE_DOT_BUILD_ADMIN_NAMESPACE)
    # ll = LakehouseLoader()
    print(f"ll={ll}")
