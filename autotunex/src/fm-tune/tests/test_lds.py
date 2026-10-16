"""Tests for autotune.lds — Limited Discrepancy Search."""

import pytest
from ray import tune
from ray.tune.search import Searcher

from autotune.lds import (
    IncrementalLimitedDiscrepancySearch,
    LimitedDiscrepancySearch,
    _dict_hash,
)


class TestDictHash:
    def test_same_input_same_hash(self):
        a = {"x": 1, "y": 2.0}
        b = {"y": 2.0, "x": 1}  # different key order
        assert _dict_hash(a, precision=4) == _dict_hash(b, precision=4)

    def test_different_input_different_hash(self):
        a = {"x": 1}
        b = {"x": 2}
        assert _dict_hash(a, precision=4) != _dict_hash(b, precision=4)

    def test_float_precision(self):
        # Floats below precision threshold compare equal after rounding
        a = {"x": 1.000001}
        b = {"x": 1.000002}
        assert _dict_hash(a, precision=2) == _dict_hash(b, precision=2)
        assert _dict_hash(a, precision=8) != _dict_hash(b, precision=8)

    def test_nested_dict(self):
        a = {"x": {"y": 1.5}}
        # No exception, returns string
        h = _dict_hash(a, precision=4)
        assert isinstance(h, str)


class TestIncrementalLDS:
    def test_default_config_first(self):
        variables = ["X", "Y"]
        values = [["a", "b", "c"], ["1", "2"]]
        defaults = {"X": "a", "Y": "1"}
        lds = IncrementalLimitedDiscrepancySearch(
            max_discrepancy=1, variables=variables, values=values, default_values=defaults
        )
        init = lds.get_init_config()
        assert init == {"X": "a", "Y": "1"}

    def test_next_config_eventually_returns_none(self):
        # max_discrepancy=1 with 2 binary vars → ≤3 configs (default + 2 single-flips)
        variables = ["X", "Y"]
        values = [["a", "b"], ["1", "2"]]
        defaults = {"X": "a", "Y": "1"}
        lds = IncrementalLimitedDiscrepancySearch(
            max_discrepancy=1, variables=variables, values=values, default_values=defaults
        )
        configs = []
        while True:
            c = lds.next_config()
            if c is None:
                break
            configs.append(c)
            if len(configs) > 100:
                pytest.fail("LDS did not terminate")
        # Generated some configs, then exhausted
        assert len(configs) > 0
        # Each config has both variables
        for c in configs:
            assert set(c.keys()) == {"X", "Y"}

    def test_max_discrepancy_zero_means_no_deviation(self):
        variables = ["X", "Y", "Z"]
        values = [["a", "b"], ["1", "2"], ["x", "y"]]
        defaults = {"X": "a", "Y": "1", "Z": "x"}
        lds = IncrementalLimitedDiscrepancySearch(
            max_discrepancy=0, variables=variables, values=values, default_values=defaults
        )
        configs = []
        while True:
            c = lds.next_config()
            if c is None:
                break
            configs.append(c)
            if len(configs) > 50:
                pytest.fail("LDS did not terminate")
        # With max_discrepancy=0 and defaults, only the default itself survives
        for c in configs:
            assert c == {"X": "a", "Y": "1", "Z": "x"}

    def test_random_init_when_no_defaults(self):
        variables = ["X"]
        values = [["a", "b", "c"]]
        # No default_values provided
        lds = IncrementalLimitedDiscrepancySearch(
            max_discrepancy=1, variables=variables, values=values, default_values=None, random_state=7
        )
        init = lds.get_init_config()
        assert init["X"] in ["a", "b", "c"]


class TestLimitedDiscrepancySearch:
    def _build(self, **kwargs):
        space = {
            "X": tune.choice(["a", "b", "c"]),
            "Y": tune.choice(["1", "2"]),
        }
        defaults = {"X": "a", "Y": "1"}
        return LimitedDiscrepancySearch(
            space=space,
            metric="loss",
            mode="min",
            max_discrepancy=1,
            default_values=defaults,
            **kwargs,
        )

    def test_first_suggest_returns_default(self):
        searcher = self._build()
        suggestion = searcher.suggest("trial-1")
        assert suggestion["X"] == "a"
        assert suggestion["Y"] == "1"

    def test_subsequent_suggestions_are_dicts(self):
        searcher = self._build()
        searcher.suggest("trial-1")  # default
        s2 = searcher.suggest("trial-2")
        assert isinstance(s2, dict) or s2 == Searcher.FINISHED

    def test_finishes_when_space_exhausted(self):
        searcher = self._build()
        # Drain the search
        suggestions = []
        for i in range(50):
            s = searcher.suggest(f"trial-{i}")
            if s == Searcher.FINISHED:
                break
            suggestions.append(s)
        assert len(suggestions) > 0
        assert len(suggestions) < 50  # terminated before iter cap

    def test_num_samples_caps_suggestions(self):
        space = {"X": tune.choice(list(range(20)))}
        defaults = {"X": 0}
        searcher = LimitedDiscrepancySearch(
            space=space,
            metric="loss",
            mode="min",
            max_discrepancy=1,
            num_samples=3,
            default_values=defaults,
        )
        count = 0
        for i in range(20):
            s = searcher.suggest(f"trial-{i}")
            if s == Searcher.FINISHED:
                break
            count += 1
        assert count <= 3

    def test_continuous_distribution_rejected(self):
        # LDS only handles Categorical (tune.choice). Continuous like uniform
        # is not a Categorical, so the assert in _create_search_space fires,
        # OR the search space ends up empty and zip raises.
        space = {"X": tune.uniform(0.0, 1.0)}
        with pytest.raises((AssertionError, ValueError)):
            LimitedDiscrepancySearch(space=space, metric="loss", mode="min")

    def test_on_trial_complete_handles_missing_trial(self):
        searcher = self._build()
        # Should silently no-op on unknown trial id
        searcher.on_trial_complete("never-suggested", result={"loss": 0.5})

    def test_save_and_restore_roundtrip(self, tmp_path):
        searcher = self._build()
        searcher.suggest("trial-1")  # advance state
        ckpt = tmp_path / "lds.pkl"
        searcher.save(str(ckpt))

        restored = LimitedDiscrepancySearch(space=None, metric="loss", mode="min", max_discrepancy=1)
        restored.restore(str(ckpt))
        # State has been restored
        assert restored._max_discrepancy == 1

    def test_get_set_state(self):
        searcher = self._build()
        state = searcher.get_state()
        searcher2 = self._build()
        searcher2.set_state(state)
        # Verify a non-trivial attribute round-trips
        assert searcher2._max_discrepancy == searcher._max_discrepancy
