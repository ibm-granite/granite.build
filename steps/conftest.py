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

"""Give the step tests under ``steps/`` the repo test suite's ``test/conftest.py``.

pytest only loads a conftest from a test's own directory and its parents, so a
step test (``steps/<path>/<env>/test/...``, run by ``make test``) would otherwise
miss everything ``test/conftest.py`` sets up: SPS secret loading from
``GBTEST_SPS_IBMCLOUD_API_KEY`` (``IBM_CLOUD_API_KEY``, ``HF_TOKEN``, GitHub tokens,
...), the GB* env baseline and restore, the mock-mode fixtures, the ``ibm``
marker's credential check, the SkyPilot xdist grouping, and the stall watchdog.

It can't be pulled in with ``pytest_plugins`` (pytest rejects that outside the
rootdir conftest) or ``import conftest`` (this file is also module ``conftest``).
So it is loaded from its path under its own module name and registered as a
plugin. Its fixtures are then global, and pytest replays the historic hooks
(``pytest_addoption``, ``pytest_configure``) for it. The ``collect_ignore`` it
defines is not applied: those paths are relative to ``test/``.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

# The repo test suite's conftest, and the name it is registered under here.
_REPO_CONFTEST = Path(__file__).resolve().parent.parent / "test" / "conftest.py"
_PLUGIN_NAME = "gb_repo_test_conftest"


def _already_loaded(config: pytest.Config) -> bool:
    """Return True when ``test/conftest.py`` is already a plugin of this session.

    That happens when one pytest run spans both trees (e.g. ``pytest test steps``):
    pytest has loaded it as an ordinary conftest, and registering it again would
    run its hooks twice and re-add its command-line options (an error).

    Args:
        config: The pytest config whose plugin manager is checked.

    Returns:
        True if a registered plugin is that file, or was registered by this module.
    """
    pm = config.pluginmanager
    if pm.has_plugin(_PLUGIN_NAME):
        return True
    for plugin in pm.get_plugins():
        plugin_file = getattr(plugin, "__file__", None)
        if plugin_file and Path(plugin_file).resolve() == _REPO_CONFTEST:
            return True
    return False


def _load_repo_conftest() -> ModuleType:
    """Import ``test/conftest.py`` under :data:`_PLUGIN_NAME`.

    Returns:
        The loaded module.

    Raises:
        ImportError: If the file is missing or cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(_PLUGIN_NAME, _REPO_CONFTEST)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the repo test conftest at {_REPO_CONFTEST}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so anything it imports that looks it up by name finds it.
    sys.modules[_PLUGIN_NAME] = module
    spec.loader.exec_module(module)
    return module


def pytest_configure(config: pytest.Config) -> None:
    """Register ``test/conftest.py`` as a plugin, unless the session already has it.

    Args:
        config: The pytest config to register the plugin with.
    """
    if _already_loaded(config):
        return
    config.pluginmanager.register(_load_repo_conftest(), _PLUGIN_NAME)
