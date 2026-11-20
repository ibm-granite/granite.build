# SPDX-License-Identifier: Apache-2.0

"""Put the step's own ``src/`` on ``sys.path``.

The suites import the step's entrypoints by flat module name (``render_config``),
the same way the step's ``run:`` block does. ``make test`` sets ``PYTHONPATH`` to
``src/``, but a suite that only passes under one invocation is a suite people stop
running, so resolve it here too and the bare ``pytest test`` works as well.
"""

import sys
from pathlib import Path

_OWN_SRC = Path(__file__).resolve().parent.parent / "src"
if _OWN_SRC.is_dir() and str(_OWN_SRC) not in sys.path:
    sys.path.insert(0, str(_OWN_SRC))
