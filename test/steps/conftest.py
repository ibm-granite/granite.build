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

"""Auto-mark everything under ``test/steps/`` as an ``extended`` test.

``make publish-step`` copies a step's per-cluster build tests into this tree (see
``steps/README.md``, "Two test modes"). Those copies are real-infra tests that
should be **discoverable/runnable from VSCode** and run in the extended suite, yet
stay **out of the quick / PR selections**. Rather than relying on each copied test
to carry ``@extended_testing_only`` (the copies are generated, not hand-edited), we
apply the ``extended`` marker to every item collected under this directory here, so
``quick-tests`` (and any ``not extended`` selection) deselects them while
``extended-tests`` runs them — no dedicated ``step_build_test`` marker required.
"""

from pathlib import Path

import pytest

# Root of the Mode-2 tree; every test collected below it is auto-marked.
_STEPS_DIR = Path(__file__).parent


def _item_under_steps(item: pytest.Item) -> bool:
    """Return True if ``item`` is collected from within ``test/steps/``.

    :param item: the pytest test item being collected.
    :returns: True when the item's file lives under this conftest's directory.
    """
    try:
        return _STEPS_DIR in item.path.parents
    except AttributeError:  # very old pytest without ``item.path``
        return str(_STEPS_DIR) in str(item.fspath)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Tag every test under ``test/steps/`` with the ``extended`` marker.

    :param items: the list of collected pytest items (modified in place).
    """
    for item in items:
        if _item_under_steps(item):
            item.add_marker(pytest.mark.extended)
