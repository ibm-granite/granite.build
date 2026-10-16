"""Put this step's ``src/`` on ``sys.path`` so the tests import it by flat name."""

import sys
from pathlib import Path

_OWN_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_OWN_SRC) not in sys.path:
    sys.path.insert(0, str(_OWN_SRC))
