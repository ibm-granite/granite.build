import filecmp
import os
import random
import tempfile

import pandas as pd
import pytest

from gbserver.storage.lh_loader import LakehouseLoader
from gbserver.types.constants import GRANITE_DOT_BUILD_ADMIN_NAMESPACE

pytestmark = pytest.mark.ibm

from gbserver.types.constants import GB_PUBLIC_ARTIFACT_NAMESPACE
from gbserver.utils.logger import get_logger

namespace = GB_PUBLIC_ARTIFACT_NAMESPACE
# namespace = GRANITE_DOT_BUILD_ADMIN_NAMESPACE
logger = get_logger(__name__)


@pytest.mark.skip("Disabled until LakehouseLoader class is really being used.")
class TestLakehouseLoader:

    def setup_method(self, method):
        self.ll = self.get_loader(namespace)
        random_number = random.randint(
            1, 1000000
        )  # So we have very low probablity of colliding with other test runs.
        self.table_name = "test_lhloader_" + str(random_number)
        logger.info(f"Deleting table {namespace}.{self.table_name}")
        self.ll.delete_table_in_namespace(namespace=namespace, table_name=self.table_name)

    def teardown_method(self, method):
        logger.info(f"Deleting table {namespace}.{self.table_name}")
        self.ll.delete_table_in_namespace(namespace=namespace, table_name=self.table_name)

    def get_loader(self, namespace: str):
        ll = LakehouseLoader(default_namespace=namespace)
        return ll

    # def test_parse_uri():
    #     ll = get_loader()
    #     table_name = "tn"
    #     uri = ll._get_gb_uri(table_name)
    #     ns, tn = ll.parse_gb_uri(uri)
    #     assert ns == namespace
    #     assert tn == table_name

    def test_jsonl_loader(self):
        self.do_test_loader(".jsonl")

    def test_parquet_loader(self):
        self.do_test_loader(".parquet")

    def do_test_loader(self, file_extension: str):

        data = {"col_1": [3, 2, 1, 0], "col_2": ["a", "b", "c", "d"]}
        df = pd.DataFrame.from_dict(data)

        with tempfile.TemporaryDirectory() as tmp:
            upload_file = os.path.join(tmp, "upload" + file_extension)
            download_file = os.path.join(tmp, "download" + file_extension)
            if file_extension == ".parquet":
                df.to_parquet(path=upload_file)
            elif file_extension == ".jsonl":
                df.to_json(path_or_buf=upload_file, orient="records", lines=True)
            else:
                raise ValueError("Unexpected extension configured in test")
            self.ll.upload(file_path=upload_file, table_name=self.table_name)
            self.ll.download(self.table_name, download_file)
            assert filecmp.cmp(
                upload_file, download_file, shallow=False
            ), f"Uploaded file is not the same as the downloaded file"
