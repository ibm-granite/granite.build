from typing import Self

import pytest
from lib.lineage.lineage import AbstractLineageTest

from gbserver.lineage.wandb_jobstats import WandBLineageStore

pytestmark = pytest.mark.ibm


class TestWandBLineage(AbstractLineageTest):

    def _get_tested_lineage_storage(self: Self):
        return WandBLineageStore()
