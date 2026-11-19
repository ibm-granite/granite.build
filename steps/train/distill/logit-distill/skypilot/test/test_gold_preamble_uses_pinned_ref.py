# SPDX-License-Identifier: Apache-2.0

"""gold.py's preamble pre-load must read the PINNED revision's ref, not "main".

The pre-load path (_read_cached_sha -> _preload_kernel_into_mapping) installs
a cached kernel directly into transformers' _KERNEL_MODULE_MAPPING, bypassing
_HUB_KERNEL_MAPPING entirely. If it read "refs/main" it would install the
main-branch build the pin is meant to avoid; the pin would only work by the
accident of the main build failing to import and the try block falling through
to the online path. This test verifies the source code reads the pinned
revision's ref file, so the pre-load, the probe and the prefetch all resolve
the same commit. See the review of PR 540 (klwu, item 2).

Reads gold.py from disk rather than importing it: the module imports trl at
module scope, which the repo venv does not ship (the step images do).
"""

import re
from pathlib import Path


def _gold_source() -> str:
    """The source of gold.py, read from the vendored package on disk."""
    # test file: steps/train/distill/logit-distill/skypilot/test/<this>.py
    # gold.py:   steps/train/distill/src/gb_steps_post_training/distillation/gold.py
    # 3 parents up from the test dir: .../logit-distill/skypilot/test -> .../distill/
    gold_path = (
        Path(__file__).parent.parent.parent.parent
        / "src"
        / "gb_steps_post_training"
        / "distillation"
        / "gold.py"
    )
    assert gold_path.exists(), f"could not find gold.py at {gold_path}"
    return gold_path.read_text()


def test_read_cached_sha_calls_pinned_revision():
    """_read_cached_sha must call pinned_revision(_repo_id), not hardcode "main"."""
    source = _gold_source()

    assert "pinned_revision(_repo_id)" in source, (
        "gold.py's _read_cached_sha does not call pinned_revision(_repo_id); "
        "it may still read refs/main, bypassing the mamba-ssm pin. "
        "See hub_kernel_pins.py and PR 540 review item 2."
    )


def test_read_cached_sha_does_not_read_refs_main():
    """The ref path must not contain the literal "main" for pinned repos."""
    source = _gold_source()

    # Find the _read_cached_sha function body
    match = re.search(
        r"def _read_cached_sha\(.*?\n(.*?)\n    def ", source, re.DOTALL
    )
    assert match, "could not locate _read_cached_sha in gold.py's source"
    body = match.group(1)

    # Strip comments (the fix's own explanation mentions "main", so the
    # literal appears in prose, not in code).
    code_only = re.sub(r"#.*", "", body)

    # The only ref name in the path must come from pinned_revision, not "main"
    assert '"main"' not in code_only, (
        "gold.py's _read_cached_sha still contains the literal \"main\"; "
        "it must read refs/<pinned_revision(repo_id)> so the pre-load, "
        "the probe and the prefetch all resolve the same commit."
    )
