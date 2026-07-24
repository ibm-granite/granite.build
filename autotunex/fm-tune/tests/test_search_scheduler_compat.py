"""Tests for autotune.utils.validate_search_alg_scheduler_combo.

The compatibility matrix is derived from per-searcher source-code analysis;
see plans silky-cantering-lovelace.md (per-searcher ASHA compatibility) and
sandy-relaxing-vole.md (scheduler trim) for the rationale.

Supported schedulers after the trim: {fifo, asha, hyperbandforbohb}.
"""

import pytest

from autotune.utils import (
    _SEARCH_ALG_SCHEDULER_COMPAT,
    validate_search_alg_scheduler_combo,
)


class TestCompatibleCombos:
    """Combinations that must NOT raise."""

    def test_random_with_fifo(self):
        validate_search_alg_scheduler_combo("random", "fifo")

    def test_random_with_asha(self):
        validate_search_alg_scheduler_combo("random", "asha")

    def test_lds_with_fifo(self):
        validate_search_alg_scheduler_combo("lds", "fifo")

    def test_lds_with_asha(self):
        validate_search_alg_scheduler_combo("lds", "asha")

    def test_blds_with_fifo(self):
        validate_search_alg_scheduler_combo("blds", "fifo")

    def test_blds_with_asha(self):
        validate_search_alg_scheduler_combo("blds", "asha")

    def test_hyperopt_with_fifo(self):
        validate_search_alg_scheduler_combo("hyperopt", "fifo")

    def test_bohb_with_hyperbandforbohb(self):
        validate_search_alg_scheduler_combo("bohb", "hyperbandforbohb")

    def test_none_inputs_default_to_random_fifo(self):
        # None on either side must resolve to defaults and not raise.
        validate_search_alg_scheduler_combo(None, None)
        validate_search_alg_scheduler_combo(None, "asha")
        validate_search_alg_scheduler_combo("lds", None)


class TestRejectedCombos:
    """Combinations that must raise ValueError."""

    def test_hyperopt_with_asha_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            validate_search_alg_scheduler_combo("hyperopt", "asha")

    def test_bohb_with_asha_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            validate_search_alg_scheduler_combo("bohb", "asha")

    def test_bohb_with_fifo_rejected(self):
        # BOHB requires HyperBandForBOHB to inject hyperband_info.budget into
        # the result dict; without it, the multi-fidelity model never updates.
        with pytest.raises(ValueError, match="not compatible"):
            validate_search_alg_scheduler_combo("bohb", "fifo")

    # Removed-scheduler tokens (hyperband, median_stopping) are no longer
    # supported by get_scheduler. The validator must reject them for every
    # searcher that does NOT have them in its allowed set — which after the
    # trim means every searcher.
    def test_lds_with_removed_hyperband_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            validate_search_alg_scheduler_combo("lds", "hyperband")

    def test_random_with_removed_median_stopping_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            validate_search_alg_scheduler_combo("random", "median_stopping")

    def test_blds_with_removed_median_stopping_rejected(self):
        with pytest.raises(ValueError, match="not compatible"):
            validate_search_alg_scheduler_combo("blds", "median_stopping")

    def test_error_message_lists_allowed_schedulers(self):
        with pytest.raises(ValueError) as exc_info:
            validate_search_alg_scheduler_combo("hyperopt", "asha")
        msg = str(exc_info.value)
        assert "hyperopt" in msg
        assert "asha" in msg
        assert "fifo" in msg  # the only allowed scheduler appears in the list


class TestUnknownSearcher:
    """Unknown searcher names should pass through (deferred to dispatch)."""

    def test_unknown_searcher_does_not_raise(self):
        validate_search_alg_scheduler_combo("optuna", "asha")
        validate_search_alg_scheduler_combo("nevergrad", "fifo")


class TestMatrixSelfConsistency:
    """The matrix itself should match what each searcher actually accepts."""

    def test_every_searcher_allows_fifo(self):
        # FIFO is a no-op scheduler; every searcher must support it.
        for searcher, allowed in _SEARCH_ALG_SCHEDULER_COMPAT.items():
            if searcher == "bohb":
                # BOHB is the documented exception: locked to HyperBandForBOHB.
                continue
            assert "fifo" in allowed, f"{searcher!r} should allow fifo (no-op scheduler) but doesn't"

    def test_every_allowed_combo_passes_validator(self):
        for searcher, allowed in _SEARCH_ALG_SCHEDULER_COMPAT.items():
            for scheduler in allowed:
                validate_search_alg_scheduler_combo(searcher, scheduler)

    def test_no_removed_schedulers_in_matrix(self):
        # Sanity guard: the trim removed hyperband and median_stopping;
        # neither token should appear in any searcher's allowed set.
        removed = {"hyperband", "median_stopping"}
        for searcher, allowed in _SEARCH_ALG_SCHEDULER_COMPAT.items():
            assert allowed.isdisjoint(removed), (
                f"{searcher!r} still references a removed scheduler: {sorted(allowed & removed)}"
            )
