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

"""`make publish-step` repoints only LOCAL buildtest.yaml space_uris.

publish-step copies a step's per-cluster fixtures into test-data/steps/ and rewrites
each buildtest.yaml's ``space_uri`` to configurations/spaces/local, so a Mode-1 test
that read the locally rendered ``../../space`` resolves the PUBLISHED step in Mode 2.

A fixture whose ``space_uri`` is a remote git space (e.g. the IBM
``git+ssh://github.ibm.com/...`` space, used by tests marked ``ibm``) already resolves
the step from that remote space in both modes. Rewriting it would silently point an
IBM-cluster test at the local space, whose environments lack the IBM secrets and
cluster definitions, so those values must be copied through unchanged.

"Local" follows libgbtest's ``_resolve_space_uri``: a bare path or a ``file://`` URI.
Anything else is remote: any other ``<scheme>://`` URI, and the scp-like
``user@host:path`` form (copied unchanged rather than silently repointed).

The test runs the real steps/common.mk against a throwaway step in a temp tree:
REPO_ROOT is derived from common.mk's location, so publishing writes only under
``tmp_path``.
"""

import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMON_MK = REPO_ROOT / "steps" / "common.mk"

MODE2_URI = "../../../../../configurations/spaces/local"
LOCAL_URIS = {
    "local-bare": "../../space",
    "local-file": "file://../../space",
    "local-quoted": '"../../space"',
    "local-single-quoted": "'../../space'",
    # Local paths that contain remote-looking pieces must still be rewritten:
    "local-at-in-path": "../../git@mirror/space",
    "local-scp-lookalike": "../../a@b:c/space",
    "local-home-scp-lookalike": "~/mirror@x:1/space",
    # The create-framework-step skill's fixture template writes exactly this line; the
    # remote URI in its comment must not make the local value look remote.
    "local-template-comment": (
        "../../space         # IBM: "
        "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config"
    ),
}
REMOTE_URIS = {
    "ibm-git-ssh": "git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config",
    "quoted-https": '"https://github.ibm.com/granite-dot-build/gb-test.git"',
    "plain-git": "git://github.ibm.com/granite-dot-build/gb-test.git",
    "scp-user": "alice@github.ibm.com:granite-dot-build/gb-test.git",
    "scp-git": "git@github.ibm.com:granite-dot-build/gb-test.git",
    "single-quoted-git": "'git://github.ibm.com/granite-dot-build/gb-test.git'",
    "ssh": "ssh://git@github.ibm.com/granite-dot-build/gb-test.git",
    "remote-with-comment": "git+ssh://github.ibm.com/o/r.git@b  # kept as written",
}


def _make_step(root: pathlib.Path) -> pathlib.Path:
    """Create a minimal public-image step at ``root/steps/fake/skypilot``.

    Args:
        root: temp directory standing in for the repo root.

    Returns:
        The step directory, with one per-cluster fixture per entry in
        ``LOCAL_URIS`` and ``REMOTE_URIS``.
    """
    (root / "steps").mkdir()
    shutil.copy(COMMON_MK, root / "steps" / "common.mk")
    step = root / "steps" / "fake" / "skypilot"
    step.mkdir(parents=True)
    (step / "Makefile").write_text("STEP_NAME := fake\ninclude ../../common.mk\n")
    (step / "step-template.yaml").write_text("name: fake\n")
    (step / ".gbignore").write_text("**/*.md\nsrc/**\n")  # common.mk requires one
    (step / "USAGE.md").write_text("# fake\n")
    for cluster, uri in {**LOCAL_URIS, **REMOTE_URIS}.items():
        (step / "test" / cluster).mkdir(parents=True)
        (step / "test" / cluster / f"test_{cluster.replace('-', '_')}.py").write_text(
            ""
        )
        (step / "test-data" / cluster).mkdir(parents=True)
        (step / "test-data" / cluster / "buildtest.yaml").write_text(
            f"targets: [t]\nspace_uri: {uri}\ntests: [runner]\n"
        )
    return step


def _published_space_uri(root: pathlib.Path, cluster: str) -> str:
    """Return the space_uri line's value from a published fixture.

    Args:
        root: the temp repo root that was published into.
        cluster: the per-cluster fixture dir name.

    Returns:
        The text after ``space_uri:``, stripped.
    """
    path = (
        root / "test-data" / "steps" / "fake" / "skypilot" / cluster / "buildtest.yaml"
    )
    line = next(
        ln for ln in path.read_text().splitlines() if ln.startswith("space_uri:")
    )
    return line.split(":", 1)[1].strip()


@pytest.fixture(name="published", scope="module")
def fixture_published(tmp_path_factory) -> pathlib.Path:
    """Run ``make publish-step`` once on the throwaway step; return the temp root."""
    if shutil.which("make") is None:
        pytest.skip("make not installed")
    root = tmp_path_factory.mktemp("repo")
    step = _make_step(root)
    subprocess.run(
        ["make", "-s", "publish-step", "PUBLISH_REQUIRE_IMAGE=false"],
        cwd=step,
        check=True,
        capture_output=True,
        timeout=60,
    )
    return root


@pytest.mark.parametrize("cluster", sorted(LOCAL_URIS))
def test_local_space_uri_is_repointed_at_the_shared_space(published, cluster):
    """A local space_uri becomes the depth-derived configurations/spaces/local."""
    assert _published_space_uri(published, cluster) == MODE2_URI


@pytest.mark.parametrize("cluster", sorted(REMOTE_URIS))
def test_remote_space_uri_is_copied_unchanged(published, cluster):
    """A remote space_uri (e.g. github.ibm.com) survives publish-step verbatim."""
    assert _published_space_uri(published, cluster) == REMOTE_URIS[cluster]
