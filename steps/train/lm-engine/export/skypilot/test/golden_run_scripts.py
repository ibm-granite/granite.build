# SPDX-License-Identifier: Apache-2.0

"""Regenerate ``golden_run_scripts.json`` from the template.

Run it after a deliberate change to the launcher's ``run:`` block, and read the
diff: this file's whole job is to make that diff show up in review.

    cd steps/train/lm-engine/export/skypilot
    GB_ENVIRONMENT=STANDALONE ../../../../../.venv/bin/python test/golden_run_scripts.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_skypilot_render import _CASES, _GOLDEN_PATH, _render  # noqa: E402

if __name__ == "__main__":
    golden = {name: _render(over).splitlines() for name, over in _CASES.items()}
    _GOLDEN_PATH.write_text(json.dumps(golden, indent=2) + "\n")
    print(f"wrote {_GOLDEN_PATH}")
