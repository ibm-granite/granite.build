"""Isolate the test suite from a developer's local ``.env``.

``tests/conftest.py`` imports ``autotunex.main``, whose module-level ``create_app()``
constructs ``Settings()`` — which reads ``.env`` from the working directory. A real
``make dev`` config therefore leaks into collection: at best it aborts the run before a
single test is collected (a non-default ``job_backend`` fails validation), at worst it
points the suite at whatever ``AUTOTUNEX_DATABASE_URL`` and credentials ``.env`` carries.
CI never hits either, because ``.env`` is gitignored and absent there.

Dropping ``env_file`` before the first ``Settings()`` reproduces CI exactly: settings come
from defaults and explicitly-exported environment variables only. Per-test overrides are
unaffected — ``tests/conftest.py``'s ``make_settings`` constructs instances directly.
"""

from __future__ import annotations

from autotunex.core.config import Settings

Settings.model_config["env_file"] = None
