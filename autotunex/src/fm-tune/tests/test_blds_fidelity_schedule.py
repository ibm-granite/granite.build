"""Tests for BLDS fidelity_schedule validation and top-rung injection.

The schedule's final entry — the "top rung" — defines the most-expensive
evaluation tier and may be < 1.0. See plan sunny-noodling-fermat.md.
"""

import pytest
from ray import tune
from ray.tune.search import Searcher

from autotune.blds import BanditLimitedDiscrepancySearch, IncrementalBanditLDS


def _build_optimizer(schedule):
    return IncrementalBanditLDS(
        max_discrepancy=1,
        variables=["X"],
        values=[["a", "b"]],
        fidelity_schedule=schedule,
    )


class TestScheduleValidation:
    def test_accepts_top_rung_below_one(self):
        opt = _build_optimizer([0.1, 0.25, 0.5])
        assert opt.fidelity_schedule[-1] == 0.5

    def test_accepts_default_ladder(self):
        opt = _build_optimizer([0.1, 0.25, 0.5, 1.0])
        assert opt.fidelity_schedule[-1] == 1.0

    def test_accepts_single_rung(self):
        opt = _build_optimizer([0.3])
        assert opt.num_rungs == 1
        assert opt.fidelity_schedule[-1] == 0.3

    def test_accepts_top_rung_exactly_one(self):
        opt = _build_optimizer([1.0])
        assert opt.num_rungs == 1

    def test_rejects_zero_rung(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _build_optimizer([0.0, 0.5])

    def test_rejects_negative_rung(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _build_optimizer([-0.1, 0.5])

    def test_rejects_above_one(self):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            _build_optimizer([0.5, 1.5])

    def test_rejects_non_monotone(self):
        with pytest.raises(ValueError, match="monotone"):
            _build_optimizer([0.5, 0.3, 1.0])

    def test_rejects_duplicate_rungs(self):
        # Strict monotone increase: equal entries are rejected.
        with pytest.raises(ValueError, match="monotone"):
            _build_optimizer([0.25, 0.5, 0.5])

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            _build_optimizer([])


class TestTopRungInjection:
    """The injected `_blds_top_rung_pct` lets drivers detect top-rung trials."""

    def _make_searcher(self, schedule):
        space = {"X": tune.choice(["a", "b", "c"])}
        return BanditLimitedDiscrepancySearch(
            space=space,
            metric="loss",
            mode="min",
            max_discrepancy=1,
            num_samples=10,
            fidelity_schedule=schedule,
        )

    def test_injects_top_rung_pct_with_full_ladder(self):
        searcher = self._make_searcher([0.1, 0.5, 1.0])
        cfg = searcher.suggest("t1")
        assert cfg["training_config"]["_blds_top_rung_pct"] == 1.0

    def test_injects_top_rung_pct_below_one(self):
        searcher = self._make_searcher([0.1, 0.25, 0.5])
        cfg = searcher.suggest("t1")
        assert cfg["training_config"]["_blds_top_rung_pct"] == 0.5

    def test_top_rung_pct_constant_across_trials(self):
        """Every emission carries the same top-rung value."""
        space = {"X": tune.choice(["a", "b", "c", "d"])}
        searcher = BanditLimitedDiscrepancySearch(
            space=space,
            metric="loss",
            mode="min",
            max_discrepancy=2,
            num_samples=20,
            fidelity_schedule=[0.25, 0.5],
        )
        seen = set()
        for i in range(8):
            cfg = searcher.suggest(f"t{i}")
            if cfg == Searcher.FINISHED:
                break
            seen.add(cfg["training_config"]["_blds_top_rung_pct"])
        assert seen == {0.5}, f"expected only 0.5, got {seen}"

    def test_first_emission_is_at_rung_zero(self):
        """The very first trial emits at fidelity_schedule[0], not the top
        rung — even when the schedule has only two rungs and the gap is small."""
        searcher = self._make_searcher([0.4, 0.6])
        cfg = searcher.suggest("t1")
        assert cfg["training_config"]["hpo_dataset_percentage"] == 0.4
        assert cfg["training_config"]["_blds_top_rung_pct"] == 0.6
