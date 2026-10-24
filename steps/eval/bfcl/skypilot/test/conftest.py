"""Pytest configuration for the bfcl-eval step's own tests.

Three jobs; the first two are the same as distill-corpus-prep's conftest:

1. Put this step's ``src/`` on ``sys.path`` so the tests import the modules under test
   directly, without installing anything.

2. Skip collection of the tests whose modules under test import ``bfcl_eval`` — the
   Berkeley Function Calling Leaderboard harness itself, which this repo does not vendor.
   It is a dependency of the step's *image* (see pyproject.toml / uv.lock / Dockerfile),
   not of this repository, so in a plain granite.build checkout it is absent and those
   files would ERROR at import rather than skip. The remaining tests cover the parts that
   are ours: the shell contract, shard merging, tag repair, and the step template.

   To run the full set, do it inside the step's own environment where the harness is
   present::

       uv sync --locked && uv run pytest test

   ``make unit-tests`` deliberately does not, so that editing src/ stays a fast loop.
"""

import importlib.util
import sys
from pathlib import Path

_OWN_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_OWN_SRC) not in sys.path:
    sys.path.insert(0, str(_OWN_SRC))

# Import bfcl_eval at module scope, so collection itself fails without it.
_NEEDS_HARNESS = [
    "test_resolve_test_categories.py",
    "test_sample_test_ids.py",
    "test_shard_test_ids.py",
]

collect_ignore: list[str] = []

# find_spec rather than a try/import: it answers "is it installed?" without paying the
# harness's (heavy) import side effects during collection.
if importlib.util.find_spec("bfcl_eval") is None:
    collect_ignore += _NEEDS_HARNESS

# 3. Skip the per-cluster build test unless this repo's OWN build-runner harness imports.
#
# test/lsf/ subclasses libgbtest.buildrunner.buildtest, which resolves from the repo root
# (test/ is on pytest's pythonpath) but then imports psutil — present in a repo-root .venv,
# absent from the throwaway environment `make unit-tests` uses. Without this guard that
# ImportError is a collection ERROR that takes the WHOLE suite down with it, so the 62
# offline tests never run and the failure looks like a broken step rather than a missing
# dependency (measured: `Interrupted: 1 error during collection`, 0 tests run).
#
# Importing the real symbol rather than probing for psutil by name: what matters is whether
# the harness is usable, not which of its dependencies is missing today.
try:  # noqa: SIM105
    import libgbtest.buildrunner.buildtest  # noqa: F401
except ImportError:
    collect_ignore.append("lsf")
