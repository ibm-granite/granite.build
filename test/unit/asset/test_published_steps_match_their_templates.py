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

"""Every published step asset must be what `make publish-step` would write.

Steps are authored at steps/<name>/<env>/step-template.yaml and PUBLISHED into
configurations/assets/environments/<env>/steps/<name>/step.yaml, which is what a
`space://steps/<name>` URI actually resolves to. The two are a source and a build
product, and nothing enforced that until this file.

THE DRIFT THIS EXISTS TO CATCH, because it happened. vllm-server's allocation
lifetime cap (max_lifetime_seconds plus its watchdog, ~40 lines) was added to the
ASSET, with an asset-side unit test to match, and `make publish-step` was never run.
The authoring template stayed behind for as long as nobody diffed them. Both halves of
that are quietly dangerous:

* a later `make publish-step` for any unrelated reason DELETES the newer work, because
  publish renders template -> asset and never reads the asset;
* meanwhile `make space` and every step test that reads the template describe a step
  that is not the one the cluster runs.

Neither shows up as a failure. The asset keeps working, the template keeps passing its
own tests, and the two describe different steps.

So this is a build-product freshness check, of the same kind as a committed lockfile
matching its manifest: if it fails, the fix is `make publish-step` in the step's
directory (after deciding which side is actually correct -- the answer is not always
the template, and in the vllm-server case it was not).
"""

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STEPS_DIR = REPO_ROOT / "steps"
ASSETS_DIR = REPO_ROOT / "configurations" / "assets" / "environments"


def _pairs():
    """(name, env, template, asset) for every authored step that is also published.

    Parametrised over what is on disk rather than a hand-kept list, so a newly ported
    step is covered without being added anywhere.
    """
    found = []
    for template in sorted(STEPS_DIR.glob("*/*/step-template.yaml")):
        name = template.parent.parent.name
        env = template.parent.name
        asset = ASSETS_DIR / env / "steps" / name / "step.yaml"
        if asset.is_file():
            found.append(pytest.param(name, env, template, asset, id=f"{name}-{env}"))
    return found


PAIRS = _pairs()


def test_there_is_something_to_check():
    """A glob that silently matches nothing is a test suite that silently passes."""
    assert PAIRS, "no step-template.yaml / published step.yaml pairs found on disk"


@pytest.mark.parametrize("name,env,template,asset", PAIRS)
def test_the_published_asset_matches_its_template(name, env, template, asset):
    template_text = template.read_text(encoding="utf-8")
    asset_text = asset.read_text(encoding="utf-8")

    # publish-step substitutes ${IMAGE_REF} from the step's Makefile. No step that is
    # both templated that way and published exists today (steps/eval uses it and is
    # not published), so the comparison is a plain one. If that changes, this test has
    # to learn the substitution rather than start failing for a legitimate difference
    # -- hence an explicit, self-explaining skip rather than silence.
    if "${IMAGE_REF}" in template_text:
        pytest.skip(
            f"{name} templates ${{IMAGE_REF}}; teach this test the substitution that "
            "publish-step applies before comparing"
        )

    assert asset_text == template_text, (
        f"{name} ({env}): the published asset and its authoring template have "
        f"diverged.\n"
        f"  template: {template.relative_to(REPO_ROOT)}\n"
        f"  asset:    {asset.relative_to(REPO_ROOT)}\n"
        "Decide which side is correct FIRST -- the asset is what the cluster runs, so "
        "it may well be the newer one -- port the difference into the template, then "
        f"run `make publish-step` in steps/{name}/{env}/."
    )
