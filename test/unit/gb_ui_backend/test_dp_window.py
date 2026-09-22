"""The Data Processing window ceiling, and why it is where it is.

The page could not reach past 30 days, which made the pipelines users were
looking for unreachable — most of the history is old. Raising the ceiling is
safe only while the scan stays inside its row cap, because exceeding that cap
drops the *oldest* builds, i.e. exactly the ones a wider window was opened for.
"""

from gb_ui_backend.api import data_processing as dp


def test_window_ceiling_allows_six_months():
    """3 and 6 month options in the UI must be accepted by the API."""
    assert dp._MAX_WINDOW_DAYS >= 180


def test_window_ceiling_stays_inside_the_scan_row_cap():
    """The ceiling and the scan's row cap are coupled; this pins the coupling.

    Measured against a production database: ~1,600 builds in 90 days and 24,180
    in 365, so a year would exceed the 10,000-row cap while 180 days does not.
    A future raise past roughly 200 days needs the cap raised in the same change,
    or the page silently under-reports — which is the bug the truncation warning
    exists to expose, not to excuse.
    """
    assert dp._MAX_WINDOW_DAYS <= 200, (
        "raising the window past ~200 days requires raising the scan row cap in "
        "_scan_datasets_async at the same time; see the constant's comment"
    )


def test_both_endpoints_share_the_ceiling():
    """Neither endpoint may keep its own limit — they are queried as a pair."""
    import inspect

    for fn in (dp.get_lineage, dp.recent_datasets):
        src = inspect.getsource(fn)
        assert (
            "_MAX_WINDOW_DAYS" in src
        ), f"{fn.__name__} does not use the shared ceiling"
        assert "le=30" not in src, f"{fn.__name__} still hardcodes the old 30-day cap"
