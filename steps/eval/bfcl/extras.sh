# SPDX-License-Identifier: Apache-2.0

# bfcl family image setup (runs as root during the build; README.md).
#
# BFCL v3 (`bfcl-v3`) runs in its own venv, /opt/bfcl-v3: its harness,
# bfcl-eval 2025.8.6.2, is the same distribution (import name bfcl_eval) as the
# v4 release the job venv carries for `bfcl-v4`, with conflicting pins, so the
# two cannot share one environment. granite_evals.benchmarks.bfcl_v3 runs it as
# a child process (python -I). Hash-locked in bfcl-v3-requirements.txt; the
# bfcl-eval wheel itself (whose sha256 bfcl_v3.py records as the dataset pin:
# BFCL's data ship inside it) must come as that wheel, never the sdist.
uv venv --python /usr/local/bin/python3 /opt/bfcl-v3
uv pip install --no-cache --python /opt/bfcl-v3/bin/python --require-hashes --only-binary bfcl-eval \
    -r /tmp/extras/bfcl-v3-requirements.txt
/opt/bfcl-v3/bin/python -I -c "import bfcl_eval.constants.model_config, bfcl_eval.eval_checker.eval_runner; \
from importlib.metadata import version; print('bfcl-v3 env', version('bfcl-eval'))"
uv pip freeze --python /opt/bfcl-v3/bin/python > /opt/bfcl-v3/granite-freeze.txt
chmod -R a+rX /opt/bfcl-v3
