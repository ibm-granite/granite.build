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
COMMON_MK = STEPS_DIR / "common.mk"
ASSETS_DIR = REPO_ROOT / "configurations" / "assets" / "environments"
# common.mk's PUBLISH_TEST_DIR / PUBLISH_TESTDATA_DIR roots.
PUBLISHED_TESTS_DIR = REPO_ROOT / "test" / "steps"
PUBLISHED_TESTDATA_DIR = REPO_ROOT / "test-data" / "steps"

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


# What `make publish-step` copies into the top-level test trees (common.mk): each
# per-cluster SUBDIR of the step's test/ into test/steps/<path>/<env>/, and each subdir
# of its test-data/ into test-data/steps/<path>/<env>/. Loose files in test/ and
# test-data/ are not copied, and neither are hidden subdirs (the recipe globs `*/`).
# __pycache__ is pruned after the copy. Each copied buildtest.yaml then has its LOCAL
# space_uri rewritten to MODE2_SPACE_URI by a two-expression `sed -E`, mirrored below
# from common.mk's SPACE_URI_VALUE_RE / FILE_SPACE_URI_RE / REMOTE_SPACE_URI_RE
# (_MIRRORED_FROM_COMMON_MK pins that source text). [[:space:]] is POSIX whitespace; a sed
# line never holds the newline, so that drops out.
_SPACE_URI_VALUE = rb"^space_uri:[ \t\r\f\v]*[\"']?"
_FILE_SPACE_URI = re.compile(_SPACE_URI_VALUE + rb"file://")
_REMOTE_SPACE_URI = re.compile(
    _SPACE_URI_VALUE
    + rb"([a-zA-Z][a-zA-Z0-9+.-]*://|[a-zA-Z0-9_][a-zA-Z0-9._-]*@[a-zA-Z0-9.-]+:)"
)
# The substitution both sed expressions make: s#^space_uri:.*#space_uri: <uri>#
_SPACE_URI_LINE = re.compile(rb"^space_uri:.*")


def _mode2_space_uri(name, env):
    """common.mk's MODE2_SPACE_URI: one `../` per dir of the copied buildtest.yaml.

    MODE2_PARENTS := test-data steps $(subst /, ,$(STEP_PATH)) $(STEP_ENV) cluster
    """
    depth = 2 + len(name.split("/")) + 1 + 1
    return "../" * depth + "configurations/spaces/local"


def _rewrite_space_uri(text, uri):
    """The publish-step sed over one buildtest.yaml, line by line.

    1. a file:// space_uri line is rewritten;
    2. then every space_uri line that is NOT remote is rewritten -- so a bare path is
       too, and a remote URI (git+ssh://, https://, user@host:path) is left alone.
    """
    line_with_uri = b"space_uri: " + uri.encode()
    out = []
    for line in text.split(b"\n"):
        if _FILE_SPACE_URI.match(line):
            line = _SPACE_URI_LINE.sub(lambda _: line_with_uri, line, count=1)
        if not _REMOTE_SPACE_URI.match(line):
            line = _SPACE_URI_LINE.sub(lambda _: line_with_uri, line, count=1)
        out.append(line)
    return b"\n".join(out)


def _copied_subdirs(root):
    """{relative path: bytes} for `cp -R root/*/ dest/`: non-hidden subdirs only."""
    copied = {}
    if not root.is_dir():
        return copied
    for sub in sorted(root.iterdir()):
        # A test/__pycache__/ is matched by `*/` too; publish-step prunes it after the copy.
        if (
            sub.is_dir()
            and not sub.name.startswith(".")
            and sub.name not in _SKIP_PARTS
        ):
            copied.update(
                {f"{sub.name}/{rel}": data for rel, data in _tree(sub).items()}
            )
    return copied


def _build_test_copies(name, env, template):
    """(label, wanted tree, published dir) for the two build-test copies publish-step makes."""
    step_dir = template.parent
    uri = _mode2_space_uri(name, env)
    fixtures = {
        rel: (
            _rewrite_space_uri(data, uri)
            if rel.rsplit("/", 1)[-1] == "buildtest.yaml"
            else data
        )
        for rel, data in _copied_subdirs(step_dir / "test-data").items()
    }
    return [
        ("test/steps/", _copied_subdirs(step_dir / "test"), PUBLISHED_TESTS_DIR),
        ("test-data/steps/", fixtures, PUBLISHED_TESTDATA_DIR),
    ]


# The common.mk text the helpers above transcribe. If publish-step changes any of it, the
# mirror here is stale: update both together (whitespace-insensitive match).
_MIRRORED_FROM_COMMON_MK = (
    """SPACE_URI_VALUE_RE := ^space_uri:[[:space:]]*["'"'"']?""",
    "FILE_SPACE_URI_RE := $(SPACE_URI_VALUE_RE)file://",
    "REMOTE_SPACE_URI_RE := $(SPACE_URI_VALUE_RE)([a-zA-Z][a-zA-Z0-9+.-]*://"
    "|[a-zA-Z0-9_][a-zA-Z0-9._-]*@[a-zA-Z0-9.-]+:)",
    "MODE2_PARENTS := test-data steps $(subst /, ,$(STEP_PATH)) $(STEP_ENV) cluster",
    "MODE2_SPACE_URI ?= $(MODE2_DOTDOT)configurations/spaces/local",
    "PUBLISH_TEST_DIR ?= $(REPO_ROOT)/test/steps/$(STEP_PATH)/$(STEP_ENV)",
    "PUBLISH_TESTDATA_DIR ?= $(REPO_ROOT)/test-data/steps/$(STEP_PATH)/$(STEP_ENV)",
    "TEST_DIR ?= test",
    "for d in $(TEST_DIR)/*/; do",
    "for d in test-data/*/; do",
    "-name __pycache__ -type d -prune -exec rm -rf {} +",
    '@find "$(PUBLISH_TESTDATA_DIR)" -name buildtest.yaml -exec',
    "-e '\\#$(FILE_SPACE_URI_RE)#s#^space_uri:.*#space_uri: $(MODE2_SPACE_URI)#'",
    "-e '\\#$(REMOTE_SPACE_URI_RE)#!s#^space_uri:.*#space_uri: $(MODE2_SPACE_URI)#'",
)


def test_publish_step_still_does_what_this_file_mirrors():
    common_mk = " ".join(COMMON_MK.read_text(encoding="utf-8").split())
    missing = [
        text
        for text in _MIRRORED_FROM_COMMON_MK
        if " ".join(text.split()) not in common_mk
    ]
    assert not missing, (
        "steps/common.mk's publish-step no longer contains what this file mirrors, so "
        f"the build-test drift check may be comparing the wrong thing: {missing}"
    )


def test_the_space_uri_rewrite_mirrors_publish_step():
    """The sed mirror, on the cases common.mk's comment names."""
    uri = "../../../../../configurations/spaces/local"
    want = f"space_uri: {uri}".encode()
    for local in (
        b"space_uri: ../../space",
        b"space_uri: 'file://../../space'",
        b'space_uri:\t"file:///abs/space"',
        b"space_uri: ../../a@b:c/space",
        b"space_uri: ../x://y",
    ):
        assert _rewrite_space_uri(local, uri) == want, local
    for unchanged in (
        b"space_uri: git+ssh://git@example.com/org/repo.git",
        b"space_uri: 'https://example.com/space'",
        b"space_uri: git@example.com:org/repo.git",
        b"  space_uri: ../../space",
        b"# space_uri: ../../space",
        b"name: space_uri",
    ):
        assert _rewrite_space_uri(unchanged, uri) == unchanged, unchanged
    assert _rewrite_space_uri(b"a: 1\nspace_uri: ../s\nb: 2\n", uri) == (
        b"a: 1\n" + want + b"\nb: 2\n"
    )


@pytest.mark.parametrize("name,env,template,asset", PAIRS)
def test_the_published_build_tests_match_their_sources(name, env, template, asset):
    stale = []
    for label, want, published_root in _build_test_copies(name, env, template):
        published = published_root / name / env
        have = _tree(published) if published.is_dir() else {}
        stale += [
            f"{label}{name}/{env}/{rel}"
            for rel in sorted(want.keys() | have.keys())
            if want.get(rel) != have.get(rel)
        ]

    assert not stale, (
        f"{name} ({env}): published build tests or fixtures differ from their "
        f"sources (after publish-step's space_uri rewrite): {', '.join(stale)}\n"
        f"Run `make publish-step` in steps/{name}/{env}/ and commit the result "
        "(`make check-published` there shows the diff)."
    )
