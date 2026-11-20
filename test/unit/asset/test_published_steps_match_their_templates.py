#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Every published step asset must be what `make publish-step` would write.

Steps are authored at steps/<name>/<env>/step-template.yaml and PUBLISHED into
configurations/assets/environments/<env>/steps/<name>/step.yaml, which is what a
`space://steps/<name>` URI actually resolves to. The two are a source and a build
product, and nothing enforced that until this file. It renders the template the way
`make publish-step` does -- gb-include expansion through
steps/scripts/render_step_template.py, then ${IMAGE_REF} taken from the image ref
committed in the asset, as `make check-published` reads it -- so image steps and
include steps are covered too.

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
import re

import pytest
from libgbtest.step_templates import render_step_template as _load_renderer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
STEPS_DIR = REPO_ROOT / "steps"
ASSETS_DIR = REPO_ROOT / "configurations" / "assets" / "environments"

render_step_template = _load_renderer()

# check-published: sed -n 's#.*image_id: *"docker:\(.*\)".*#\1#p' | head -1
_IMAGE_REF = re.compile(r'image_id: *"docker:(.*)"')


def _committed_image_ref(asset_text):
    """The image ref committed in the asset, read as check-published reads it."""
    for line in asset_text.splitlines():
        match = _IMAGE_REF.search(line)
        if match:
            return match.group(1)
    return ""


def _pairs():
    """(name, env, template, asset) for every authored step that is also published.

    Parametrised over what is on disk rather than a hand-kept list, so a newly ported
    step is covered without being added anywhere.
    """
    found = []
    # `**`, not `*/*`: a nested step (steps/distill/gold/skypilot) is published at
    # steps/distill/gold, and a one-level glob silently skipped all twelve of them.
    for template in sorted(STEPS_DIR.glob("**/step-template.yaml")):
        name = template.parent.parent.relative_to(STEPS_DIR).as_posix()
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
    asset_text = asset.read_text(encoding="utf-8")
    rendered = render_step_template.render(template, _committed_image_ref(asset_text))

    assert asset_text == rendered, (
        f"{name} ({env}): the published asset and its rendered authoring template "
        f"have diverged.\n"
        f"  template: {template.relative_to(REPO_ROOT)}\n"
        f"  asset:    {asset.relative_to(REPO_ROOT)}\n"
        "Decide which side is correct FIRST -- the asset is what the cluster runs, so "
        "it may well be the newer one -- port the difference into the template, then "
        f"run `make publish-step` in steps/{name}/{env}/."
    )


# What `make publish-step` copies VERBATIM next to the rendered step.yaml (common.mk):
# the step's src/ tree, USAGE.md as README.md, and .gbignore. The step.yaml check above
# never looked at these, and on PR 540 they drifted two review rounds in a row (a
# published src/ still calling a renamed function, a README missing a section) while CI
# stayed green. Bytecode is not part of the copy.
_SKIP_PARTS = {"__pycache__"}
_SKIP_SUFFIXES = {".pyc"}


def _tree(root):
    """{relative path: bytes} for every file under root, bytecode excluded."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not _SKIP_PARTS.intersection(path.relative_to(root).parts)
        and path.suffix not in _SKIP_SUFFIXES
    }


def _verbatim_copies(template, asset):
    """(label, source, published) for each verbatim copy publish-step makes."""
    step_dir, published = template.parent, asset.parent
    copies = []
    src = step_dir / "src"
    if src.is_dir() and any(src.iterdir()):
        copies.append(("src/", src, published / "src"))
    if (step_dir / "USAGE.md").is_file():
        copies.append(("README.md", step_dir / "USAGE.md", published / "README.md"))
    if (step_dir / ".gbignore").is_file():
        copies.append((".gbignore", step_dir / ".gbignore", published / ".gbignore"))
    return copies


@pytest.mark.parametrize("name,env,template,asset", PAIRS)
def test_the_published_copies_match_their_sources(name, env, template, asset):
    stale = []
    for label, source, published in _verbatim_copies(template, asset):
        if source.is_dir():
            want = _tree(source)
            have = _tree(published) if published.is_dir() else {}
            stale += [
                f"{label}{rel}"
                for rel in sorted(want.keys() | have.keys())
                if want.get(rel) != have.get(rel)
            ]
        elif not published.is_file() or published.read_bytes() != source.read_bytes():
            stale.append(label)

    assert not stale, (
        f"{name} ({env}): published copies differ from their sources: "
        f"{', '.join(stale)}\n"
        f"  published under: {asset.parent.relative_to(REPO_ROOT)}\n"
        f"Run `make publish-step` in steps/{name}/{env}/ and commit the result "
        "(`make check-published` there shows the diff)."
    )
