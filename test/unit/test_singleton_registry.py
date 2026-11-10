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

"""Every lazily cached module global in src/ is either restored per test or
deliberately exempt, so a new singleton can't silently leak between tests (#605).

A "lazily cached global" is a module-level name initialized to None/False that a
function in the same module rebinds with ``global``.
"""

import ast
import sys
from pathlib import Path

# The unit conftest is loaded by pytest under its rootdir-relative module name.
conftest = next(
    m
    for name, m in sys.modules.items()
    if name.endswith("conftest") and hasattr(m, "_OTHER_SINGLETONS")
)

SRC = Path(__file__).resolve().parents[2] / "src"

# Process-wide state that holds no storage, db or GB_HOME_DIR-bound resource, so
# reusing it across tests is harmless. Each entry must say why.
EXEMPT = {
    # Saved/restored as a pair via singleton_storage._singleton_state().
    ("gbserver.storage.singleton_storage", "__STORAGE"),
    ("gbserver.storage.singleton_storage", "__STORAGE_FACTORY"),
    # Set to True for every test by the conftest fixture itself.
    ("gbserver.cli", "process_initialized"),
    # Idempotent installs / one-time configuration and log-once flags.
    ("gbserver.resilience.transport_retry", "_INSTALLED"),
    ("gbserver.utils.logger", "__LOGGER_CONFIGURED"),
    ("gbserver.types.gbserverenvconfig", "_LOADED_EXTRA_SERVER_RUNTIME_CONFIGS"),
    ("gbserver.storage.sql.cert_file", "_SSL_CERT_FILE"),
    ("gbcli.cli", "_gb_environment_warned"),
    ("gbcommon.utils.fs_lock", "_rename_fallback_logged"),
    ("gbserver.api.autotunex_proxy", "_upstream_unreachable_logged"),
    # Remote-service clients/executors, not tied to a local db or home dir.
    ("gbserver.api.autotunex_proxy", "_client"),
    ("gbserver.environment.skypilot", "_LOADED_CLOUD_CONFIG"),
    ("gbserver.environment.skypilot", "_SKY_EXECUTOR"),
    ("gbserver.utils.cloud_logquery", "_LOG_MANAGER"),
    ("gbserver.utils.cloud_logquery_server", "_LOG_SERVER_MANAGER"),
    ("gb_ui_backend.services.cloud_logs", "_client"),
    ("gb_ui_backend.services.chat_agents.dashboard_tools", "_artifacts_http_client"),
    # AI analysis background task state; never started by unit tests.
    ("gb_ui_backend.api.ai", "_trigger_analyzing"),
    ("gb_ui_backend.api.ai", "_trigger_task"),
    ("gb_ui_backend.services.ai_daemon", "_daemon_instance"),
    ("gb_ui_backend.services.ai_prompts", "_last_prompt_refresh"),
}


def _lazily_cached_globals() -> set[tuple[str, str]]:
    found = set()
    for path in SRC.rglob("*.py"):
        if "build" in path.relative_to(SRC).parts:
            continue
        tree = ast.parse(path.read_text())
        unset = set()
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(
                node.value, ast.Constant
            ):
                if node.value.value in (None, False):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    unset |= {t.id for t in targets if isinstance(t, ast.Name)}
        rebound = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Global)
            for name in node.names
        }
        module = ".".join(path.relative_to(SRC).with_suffix("").parts)
        found |= {(module, name) for name in unset & rebound}
    return found


def test_every_cached_global_is_restored_or_exempt():
    restored = {(module, name) for module, name, _ in conftest._OTHER_SINGLETONS}
    unclassified = _lazily_cached_globals() - restored - EXEMPT
    assert not unclassified, (
        f"New lazily cached globals {sorted(unclassified)}: add each to "
        "_OTHER_SINGLETONS in test/unit/conftest.py if it holds storage, a db or "
        "anything under GB_HOME_DIR, else to EXEMPT here with the reason."
    )


def test_no_stale_entries():
    present = _lazily_cached_globals()
    restored = {(module, name) for module, name, _ in conftest._OTHER_SINGLETONS}
    stale = (restored | EXEMPT) - present
    assert not stale, f"Entries no longer match a cached global: {sorted(stale)}"
