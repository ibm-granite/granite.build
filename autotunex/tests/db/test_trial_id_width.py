"""The trial id and its foreign key are ``VARCHAR(16)`` wide."""

from __future__ import annotations

from sqlalchemy import String

from autotunex.db.tables.results import ResultTable
from autotunex.db.tables.trials import TrialTable


def test_trial_id_column_is_16_wide() -> None:
    column_type = TrialTable.__table__.c.id.type

    assert isinstance(column_type, String)
    assert column_type.length == 16


def test_result_trial_id_column_is_16_wide() -> None:
    column_type = ResultTable.__table__.c.trial_id.type

    assert isinstance(column_type, String)
    assert column_type.length == 16
