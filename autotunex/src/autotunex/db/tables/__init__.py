# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""ORM tables mirroring ``resources/autotunex_schema.sql``.

One module per table. Importing this package registers every table on
``Base.metadata``, which is what Alembic autogenerate and
:func:`autotunex.db.session.create_schema` rely on — so re-export all of them
here even though nothing imports some directly.

The mapping is a deliberate transcription of the live MySQL schema, defects
included. ``docs/schema-review.md`` lists what is wrong and why it was left
alone. The one exception is ``jobs.precision``, dropped at the maintainer's
request with its values backfilled into ``config_snapshot``.
"""

from __future__ import annotations

from autotunex.db.tables.configurations import ConfigurationTable
from autotunex.db.tables.datasets import DatasetTable
from autotunex.db.tables.gb_tasks import GbTaskTable
from autotunex.db.tables.jobs import JobTable
from autotunex.db.tables.log_entries import LogEntryTable
from autotunex.db.tables.results import ResultTable
from autotunex.db.tables.training_metrics import TrainingMetricTable
from autotunex.db.tables.trials import TrialTable
from autotunex.db.tables.users import UserTable

__all__ = [
    "ConfigurationTable",
    "DatasetTable",
    "GbTaskTable",
    "JobTable",
    "LogEntryTable",
    "ResultTable",
    "TrainingMetricTable",
    "TrialTable",
    "UserTable",
]
