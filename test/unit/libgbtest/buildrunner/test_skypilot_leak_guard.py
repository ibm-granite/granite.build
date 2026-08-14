"""Unit tests for the SkyPilot cluster leak guard (no AWS, fake sky module)."""

import sys
import types

import pytest

from libgbtest.buildrunner import skypilot_leak_guard as guard


class _FakeSky:
    """Minimal stand-in for the SkyPilot SDK.

    ``status()`` returns the current cluster list directly and ``get(x)``
    returns its argument, so the production call ``sky.get(sky.status())``
    yields the list. ``down`` records the name and can be made to raise.
    """

    def __init__(self, clusters):
        # clusters: list of {"name": ...} dicts
        self._clusters = list(clusters)
        self.downed = []
        self.fail_on = set()  # names for which down() raises

    def status(self, *a, **k):
        return self._clusters

    def get(self, x):
        return x

    def down(self, name, purge=False):
        self.downed.append(name)
        if name in self.fail_on:
            raise RuntimeError(f"boom downing {name}")
        return name

    def add(self, name):
        self._clusters.append({"name": name})


def _install(monkeypatch, clusters):
    fake = _FakeSky(clusters)
    monkeypatch.setitem(sys.modules, "sky", fake)
    return fake


def test_snapshot_selects_only_gb_clusters(monkeypatch):
    _install(monkeypatch, [{"name": "gb-aaa"}, {"name": "gb-bbb"}, {"name": "other"}])
    assert guard.snapshot_gb_clusters() == {"gb-aaa", "gb-bbb"}


def test_snapshot_returns_empty_when_sky_missing(monkeypatch):
    # Setting sys.modules['sky'] = None makes `import sky` raise ImportError.
    monkeypatch.setitem(sys.modules, "sky", None)
    assert guard.snapshot_gb_clusters() == set()


def test_down_new_downs_only_new_gb_clusters(monkeypatch):
    fake = _install(monkeypatch, [{"name": "gb-old"}, {"name": "gb-new"}, {"name": "keep"}])
    leaked = guard.down_new_gb_clusters(before={"gb-old"})
    assert leaked == {"gb-new"}
    assert fake.downed == ["gb-new"]


def test_down_new_noop_when_nothing_new(monkeypatch):
    fake = _install(monkeypatch, [{"name": "gb-old"}])
    assert guard.down_new_gb_clusters(before={"gb-old"}) == set()
    assert fake.downed == []


def test_down_new_best_effort_on_error(monkeypatch):
    fake = _install(monkeypatch, [{"name": "gb-a"}, {"name": "gb-b"}])
    fake.fail_on = {"gb-a"}
    leaked = guard.down_new_gb_clusters(before=set())
    # Both attempted despite gb-a raising; leaked set still complete.
    assert leaked == {"gb-a", "gb-b"}
    assert set(fake.downed) == {"gb-a", "gb-b"}


def test_should_fail_only_when_leaked_and_passed():
    passed = types.SimpleNamespace(passed=True)
    failed = types.SimpleNamespace(passed=False)
    assert guard._should_fail({"gb-x"}, passed) is True
    assert guard._should_fail(set(), passed) is False
    assert guard._should_fail({"gb-x"}, failed) is False
    assert guard._should_fail({"gb-x"}, None) is False
