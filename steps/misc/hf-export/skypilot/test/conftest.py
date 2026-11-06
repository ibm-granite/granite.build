"""Put the step's own src/ on sys.path for the unit tests.

export_hf_model.py and its resume gate (src/hf_export_state/) are what the image bakes in
at HF_EXPORT_HOME, so the suite imports them from src/ however pytest was started.
"""

import sys
from pathlib import Path

_OWN_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_OWN_SRC) not in sys.path:
    sys.path.insert(0, str(_OWN_SRC))
