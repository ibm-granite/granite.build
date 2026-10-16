"""The Data Processing window ceiling, and why it is where it is.

The page could not reach past 30 days, which made the pipelines users were
looking for unreachable — most of the history is old. Raising the ceiling is
safe only while the scan stays inside its row cap, because exceeding that cap
drops the *oldest* builds, i.e. exactly the ones a wider window was opened for.

Measured against a production database (rolling windows, 2026-09-29):

    30 days      826
    90 days    1,365
    180 days   4,813   <- the ceiling this module pins
    200 days   5,341
    365 days  22,154

An earlier version of this file asserted only ``_MAX_WINDOW_DAYS <= 200``. Review
pointed out that this references nothing: it would keep passing if the scan's row
cap were dropped to 100, so it pinned none of the coupling its own docstring
claimed to pin. The cap is now a named constant and these tests use it.
"""

from gb_ui_backend.api import data_processing as dp

# Builds present in a window of exactly _MAX_WINDOW_DAYS, measured (not derived).
_MEASURED_AT_CEILING = 4_813

# Busiest sustained rate in the same history: the 200-365 day band, ~102/day
# against ~27/day over the last month. Used to document the ceiling's *limit*,
# not to assert it away — see test_at_peak_volume_the_window_truncates.
_PEAK_OBSERVED_BUILDS_PER_DAY = 102


def test_window_ceiling_allows_six_months():
    """3 and 6 month options in the UI must be accepted by the API."""
    assert dp._MAX_WINDOW_DAYS >= 180


def test_both_endpoints_share_the_ceiling():
    """Neither endpoint may keep its own limit — they are queried as a pair."""
    import inspect

    for fn in (dp.get_lineage, dp.recent_datasets):
        src = inspect.getsource(fn)
        assert (
            "_MAX_WINDOW_DAYS" in src
        ), f"{fn.__name__} does not use the shared ceiling"
        assert "le=30" not in src, f"{fn.__name__} still hardcodes the old 30-day cap"


def test_the_scan_uses_the_named_row_cap():
    """The cap must not drift back to an inline literal at the call site.

    It was one, which is why no test could reference it.
    """
    import inspect

    src = inspect.getsource(dp._scan_datasets_async)
    assert "_SCAN_ROW_CAP" in src, "the scan no longer uses the named row cap"
    assert "limit=10000" not in src, "the row cap has drifted back to a literal"


def test_measured_volume_at_the_ceiling_fits_the_row_cap():
    """The actual reason 180 is acceptable: measured volume, not extrapolation."""
    assert _MEASURED_AT_CEILING <= dp._SCAN_ROW_CAP, (
        f"a {dp._MAX_WINDOW_DAYS}-day window held {_MEASURED_AT_CEILING:,} builds "
        f"when last measured, over the {dp._SCAN_ROW_CAP:,}-row cap"
    )


def test_ceiling_and_row_cap_must_change_together():
    """Tripwire on the pair, so neither can move without the other being considered.

    Deliberately pins both literal values. Either one changing is a decision that
    needs fresh builds-per-day numbers from a real deployment — the figures in this
    module's docstring are a snapshot of one deployment on one day, and the
    distribution behind them is uneven enough that extrapolating from them is how
    the previous overclaim happened.
    """
    assert (dp._MAX_WINDOW_DAYS, dp._SCAN_ROW_CAP) == (180, 10_000), (
        "window ceiling or scan row cap changed: re-measure builds-per-day against "
        "a real deployment, confirm the truncation warning still fires at the new "
        "cap, and update this test and the constants together"
    )


def test_at_peak_volume_the_window_truncates():
    """Document the ceiling's limit rather than asserting it does not exist.

    At the busiest rate this deployment has actually sustained, a full-ceiling
    window exceeds the cap. That is not a bug to fix here — raising the cap to
    cover it would reintroduce the ~220MB transfer that ruled out a 365-day window
    in the first place — but it must not be a surprise either. The truncation
    warning is what makes it acceptable, so this asserts both halves: that the
    overflow is real, and that a full page is reported as possibly incomplete.
    """
    at_peak = dp._MAX_WINDOW_DAYS * _PEAK_OBSERVED_BUILDS_PER_DAY
    assert at_peak > dp._SCAN_ROW_CAP, (
        "peak-volume arithmetic no longer overflows the cap — if that is because "
        "the cap was raised, this test and the constant's comment need rewriting"
    )

    from gb_ui_backend.services.gbserver_source import _truncation_warning

    full_page = [object()] * dp._SCAN_ROW_CAP
    warning = _truncation_warning(full_page, dp._SCAN_ROW_CAP)
    assert warning is not None, "a capped page must say it may be incomplete"
    assert str(dp._SCAN_ROW_CAP) in warning
