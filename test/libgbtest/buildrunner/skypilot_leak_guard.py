#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test-only safety net: tear down SkyPilot clusters leaked by early failures.

SkyPilot AWS tests provision real EC2. On the happy path each step's own
cleanup runs ``sky down`` when the step reaches a terminal state, but on an
early test failure (timeout mid-provision, an exception in the wait thread,
verify aborting before the build finishes) that cleanup may never fire and the
EC2 instance leaks.

Cluster names are ``gb-<launch_id[:12]>`` where ``launch_id`` is a random UUID
minted at launch, so the guard cannot target clusters by build id (and a
mid-provision leak is not even recorded in the env's internal state). Instead
it discovers clusters from ``sky status`` and does a session-scoped
snapshot-diff: snapshot the ``gb-*`` clusters up before a test, then after the
test ``sky down`` any ``gb-*`` cluster that appeared. Only NEW clusters are
downed, so pre-existing developer clusters on a shared host are never touched.
"""

import logging

import pytest

from libgbtest.constants import GBTEST_SKIP_BUILD_TEARDOWN

logger = logging.getLogger(__name__)

GB_CLUSTER_PREFIX = "gb-"


def _list_gb_clusters() -> set[str]:
    """Names of all currently-up ``gb-*`` SkyPilot clusters.

    Defensive: if ``sky`` is not importable or the status call fails, log and
    return an empty set so the guard can never break a test.
    """
    try:
        import sky
    except Exception as e:  # sky not installed in this environment
        logger.warning("skypilot_leak_guard: sky not importable: %s", e)
        return set()
    try:
        clusters = sky.get(sky.status())
    except Exception as e:
        logger.warning("skypilot_leak_guard: sky.status failed: %s", e)
        return set()
    names: set[str] = set()
    for c in clusters or []:
        name = c.get("name") if isinstance(c, dict) else getattr(c, "name", None)
        if name and name.startswith(GB_CLUSTER_PREFIX):
            names.add(name)
    return names


def snapshot_gb_clusters() -> set[str]:
    """Snapshot ``gb-*`` clusters currently up. Call before the test runs."""
    return _list_gb_clusters()


def down_new_gb_clusters(before: set[str]) -> set[str]:
    """``sky down`` every ``gb-*`` cluster that appeared since ``before``.

    Best-effort per cluster: a failure to down one is logged and the rest are
    still attempted. Returns the set of leaked cluster names (those that
    appeared during the test), regardless of teardown success.
    """
    leaked = _list_gb_clusters() - before
    if not leaked:
        return set()
    try:
        import sky
    except Exception as e:
        logger.warning(
            "skypilot_leak_guard: %d leaked clusters but sky not importable: %s",
            len(leaked),
            e,
        )
        return leaked
    for name in sorted(leaked):
        try:
            logger.warning(
                "skypilot_leak_guard: tearing down leaked cluster %s", name
            )
            sky.get(sky.down(name, purge=True))
        except Exception as e:
            logger.error("skypilot_leak_guard: failed to down %s: %s", name, e)
    return leaked


def _should_fail(leaked: set[str], rep) -> bool:
    """True iff a cluster leaked AND the test itself passed.

    We never pile a teardown assertion on top of a test that already failed —
    that would mask the primary failure.
    """
    return bool(leaked) and rep is not None and rep.passed


def _leak_guard(request):
    """Generator body of the autouse fixture (separated for unit testing)."""
    if GBTEST_SKIP_BUILD_TEARDOWN:
        yield
        return
    before = snapshot_gb_clusters()
    try:
        yield
    finally:
        leaked = down_new_gb_clusters(before)
        rep = getattr(request.node, "rep_call", None)
        if _should_fail(leaked, rep):
            pytest.fail(
                "SkyPilot clusters leaked (torn down by guard): "
                f"{sorted(leaked)}"
            )


@pytest.fixture(autouse=True)
def skypilot_cluster_leak_guard(request):
    """Autouse guard: tear down any gb-* cluster this test leaked.

    Activated per test tree by importing this fixture into that tree's
    ``conftest.py``. Requires the ``pytest_runtest_makereport`` hook below so
    ``request.node.rep_call`` reflects the test outcome.
    """
    yield from _leak_guard(request)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each phase's report on the item so the fixture can read rep_call."""
    rep = yield
    setattr(item, "rep_" + rep.when, rep)
    return rep
