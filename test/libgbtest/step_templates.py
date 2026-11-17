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

"""steps/scripts/render_step_template.py for tests.

The renderer is a standalone script (common.mk runs it; it is not an installed
module), so tests load it by path. Both the step tests under steps/ and the unit
tests use these helpers, so moving the script or changing its API touches one
place.
"""

import importlib.util
import pathlib
from functools import lru_cache
from types import ModuleType
from typing import Any, Dict, Union

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RENDER_STEP_TEMPLATE = REPO_ROOT / "steps" / "scripts" / "render_step_template.py"


@lru_cache(maxsize=None)
def render_step_template() -> ModuleType:
    """Return the render_step_template module, loaded once per process.

    Returns:
        The module (``expand``, ``render``, ``main``, ``IncludeError``, ...).
    """
    spec = importlib.util.spec_from_file_location(
        "render_step_template", RENDER_STEP_TEMPLATE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expanded_step_template(template: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Return a step template as `make space` renders it, parsed.

    Args:
        template: the step-template.yaml.

    Returns:
        The template with its gb-include lines expanded, loaded as YAML.
        ``${IMAGE_REF}`` is left as is.
    """
    return yaml.safe_load(render_step_template().expand(pathlib.Path(template)))
