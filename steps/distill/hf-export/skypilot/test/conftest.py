"""Resolve the upstream distillation package for the ported unit tests.

IDENTICAL IN EVERY PORTED DISTILLATION STEP.

These suites test code this repo does not own. ``gb_steps_post_training.distillation``
is delivered to the step at RUN time from a checkout on the shared filesystem (see
``code_config`` in step-template.yaml), so it is not importable from a bare
granite.build checkout and must not be vendored: its ``tokenizer_identity`` module
writes a guard whose reader is the trainer in that same checkout, and a second copy is
how the two halves drift apart.

So the ported suites are GATED. Point ``GB_DISTILL_CODE_DIR`` at a checkout to run
them; without one they are skipped and only the hermetic template tests run. Be
honest about what that means: a green ``make test`` in CI proves the STEP CONTRACT
holds, not that the upstream code does.

    make test                                          # contract tests only
"""

import os
import sys
from pathlib import Path

# The trainer package is vendored in-tree (steps/distill/src, see VENDORED.md) and baked
# into the shared images, so the suite needs no external checkout.
_PKG_PARENT = Path(__file__).resolve().parents[3] / "src"

# Files that import the upstream package at module scope. Listing them explicitly
# (rather than ignoring on ImportError) keeps a genuine breakage in this repo's own
# tests from being silently skipped.
# The step's OWN src/ too. `make test` sets PYTHONPATH to it, but a suite that only works
# under one invocation is a suite people stop running: a ported entrypoint imported by flat
# name (prep_corpus, merge_shards) must resolve however pytest was started. Harmless for a
# step whose entrypoint is bash.
_OWN_SRC = Path(__file__).resolve().parent.parent / "src"
if _OWN_SRC.is_dir() and str(_OWN_SRC) not in sys.path:
    sys.path.insert(0, str(_OWN_SRC))

_NEEDS_UPSTREAM = ["test_export_hf_model.py"]


def _delivered_package_is_readable() -> bool:
    """Whether the vendored package is present and readable."""
    try:
        return (_PKG_PARENT / "gb_steps_post_training").is_dir()
    except OSError:
        # PermissionError above; also ELOOP/ENAMETOOLONG from a mangled override.
        return False


collect_ignore = []
if _delivered_package_is_readable():
    # Prepended, not appended: an installed copy of the same name would otherwise win
    # and the suite would test something other than the delivered code.
    sys.path.insert(0, str(_PKG_PARENT))
else:
    collect_ignore = list(_NEEDS_UPSTREAM)
