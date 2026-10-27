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

One of its fixtures is overridden below: ``_check_test_env``, whose credential
check for ``ibm``-marked tests doesn't fit step tests.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pytest

# The repo test suite's conftest. It is registered under its path, the name pytest
# would give it as a conftest plugin.
_REPO_CONFTEST = Path(__file__).resolve().parent.parent / "test" / "conftest.py"
# Its module name in sys.modules. Not "conftest": that is this file's own name.
_MODULE_NAME = "gb_repo_test_conftest"


def _already_loaded(pm: pytest.PytestPluginManager) -> bool:
    """Return True when ``test/conftest.py`` is already a plugin of this session.

    That happens when one pytest run spans both trees (e.g. ``pytest test steps``)
    and pytest loaded it as an ordinary conftest first. Registering it again would
    run its hooks twice and re-add its command-line options (an error). The other
    order is covered by the registration name: pytest names a conftest plugin by
    its path, so when it reaches ``test/conftest.py`` after this module registered
    it under that name, it reuses this registration instead of importing it again.

    Args:
        pm: The plugin manager to check.

    Returns:
        True if a registered plugin is that file.
    """
    if pm.has_plugin(str(_REPO_CONFTEST)):
        return True
    for plugin in pm.get_plugins():
        plugin_file = getattr(plugin, "__file__", None)
        if plugin_file and Path(plugin_file).resolve() == _REPO_CONFTEST:
            return True
    return False


def _load_repo_conftest() -> ModuleType:
    """Import ``test/conftest.py`` as module :data:`_MODULE_NAME`.

    Returns:
        The loaded module.

    Raises:
        ImportError: If the file is missing or cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _REPO_CONFTEST)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the repo test conftest at {_REPO_CONFTEST}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so anything it imports that looks it up by name finds it.
    sys.modules[_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _check_test_env() -> Iterator[None]:
    """Override ``test/conftest.py``'s guard of the same name for step tests.

    Keeps its refusal to run with ``GB_ENVIRONMENT=PROD`` but drops its
    ``check_cloud_config()`` for ``ibm``-marked tests, which asserts that
    ``IBM_CLOUD_API_KEY`` and a GitHub token are set. ``ibm`` step tests use the
    local space and an SSH key file (e.g. the lsf distill-* and nemo-rl slurm-sft
    tests), so they don't need those. A test that does need a credential gates on
    it itself (e.g. the byoc slurm-bluevela test's ``IBM_CLOUD_API_KEY`` skip). As
    the fixture closest to ``steps/``, this one is used instead of the plugin's.
    """
    from gbserver.types.constants import GB_ENVIRONMENT

    assert GB_ENVIRONMENT != "PROD", "Refusing to run tests with GB_ENVIRONMENT=PROD."
    yield


def pytest_addoption(
    parser: pytest.Parser, pluginmanager: pytest.PytestPluginManager
) -> None:
    """Register ``test/conftest.py`` as a plugin, unless the session already has it.

    Done here, not in ``pytest_configure``, so the plugin's own
    ``pytest_addoption`` (``--buildtest-yaml``, ``--build-yaml``) runs before
    pytest parses the command line. This file is one of the initial conftests
    (it sits between the rootdir and the step tests), so pytest calls this hook
    early enough, and replays the plugin's ``pytest_addoption`` as it registers.

    Args:
        parser: The pytest option parser (unused; the plugin adds its own options).
        pluginmanager: The plugin manager to register the plugin with.
    """
    if _already_loaded(pluginmanager):
        return
    pluginmanager.register(_load_repo_conftest(), str(_REPO_CONFTEST))
