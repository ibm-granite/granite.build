from typing import Self

import pytest
from lib.lineage.lineage import AbstractLineageTest
from lib.lineage.mock_lineage_service import MockLineageService

from gbserver.lineage.wandb_jobstats import WandBLineageStore

pytestmark = pytest.mark.ibm


class TestWandBLineage(AbstractLineageTest):

    def _get_tested_lineage_storage(self: Self):
        store = WandBLineageStore.__new__(WandBLineageStore)
        store._service = MockLineageService()
        return store
