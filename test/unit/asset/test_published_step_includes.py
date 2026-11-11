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

"""A published step.yaml that uses gb-include matches a fresh render of its source.

`make publish-step` inlines the files a step-template.yaml names on its
`{#- gb-include <path> -#}` lines (steps/scripts/render_step_template.py). A shared
file such as steps/train/shared/trainer_args.j2 feeds several steps, so editing it
leaves every including step's committed step.yaml stale until each is re-published.
`make check-published` catches that but is not run in CI; this test is. It renders
with the image ref already committed in step.yaml, as check-published does.
"""

import importlib.util
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STEPS = REPO_ROOT / "steps"
SCRIPT = STEPS / "scripts" / "render_step_template.py"
_INCLUDE = re.compile(r"^ *\{#- gb-include ", re.M)
# check-published: sed -n 's#.*image_id: *"docker:\(.*\)".*#\1#p' | head -1
_IMAGE_REF = re.compile(r'image_id: *"docker:(.*)"')


def _load():
    spec = importlib.util.spec_from_file_location("render_step_template", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _including_templates() -> list:
    return sorted(
        p.relative_to(STEPS).as_posix()
        for p in STEPS.rglob("step-template.yaml")
        if _INCLUDE.search(p.read_text(encoding="utf-8"))
    )


def _published(template: pathlib.Path) -> pathlib.Path:
    env_dir = template.parent
    return (
        REPO_ROOT
        / "configurations"
        / "assets"
        / "environments"
        / env_dir.name
        / "steps"
        / env_dir.parent.relative_to(STEPS)
        / "step.yaml"
    )


def test_some_step_uses_gb_include():
    assert _including_templates(), "no step-template.yaml under steps/ uses gb-include"


@pytest.mark.parametrize("rel", _including_templates())
def test_published_step_matches_its_render(rel):
    template = STEPS / rel
    published = _published(template)
    assert published.is_file(), f"{rel} has no published step.yaml at {published}"
    text = published.read_text(encoding="utf-8")
    refs = [m.group(1) for line in text.splitlines() if (m := _IMAGE_REF.search(line))]
    assert _load().render(template, refs[0] if refs else "") == text, (
        f"{published.relative_to(REPO_ROOT)} is stale: run `make publish-step` in "
        f"steps/{template.parent.relative_to(STEPS)} and commit the result"
    )
