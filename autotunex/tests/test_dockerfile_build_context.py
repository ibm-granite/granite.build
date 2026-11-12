"""``autotunex/Dockerfile`` and ``Dockerfile.aio`` build from the repository root.

Both install granite.build from the checkout they are built from (issue #489), so
their build context is the repo root, and what enters it is set by an allowlist
ignore file beside each Dockerfile — BuildKit reads ``<Dockerfile>.dockerignore``
in place of the root ``.dockerignore``, which serves the gbserver image and
excludes ``autotunex/`` altogether.

Like ``test_dockerfile_variants.py`` these read the files as text: nothing in CI
builds the images, so a COPY source the ignore file drops would otherwise surface
only as a failed toolchain build.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

_AUTOTUNEX = Path(__file__).resolve().parents[1]
_REPO = _AUTOTUNEX.parent
_DOCKERFILE = _AUTOTUNEX / "Dockerfile"
_AIO = _AUTOTUNEX / "Dockerfile.aio"


def _text(path: Path) -> str:
    """Return ``path``'s text."""
    return path.read_text(encoding="utf-8")


def _instructions(content: str) -> str:
    """Return ``content`` without its comment lines."""
    return "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("#"))


def _stage_body(content: str, name: str) -> str:
    """Return the instructions between ``AS <name>`` and the next ``FROM``."""
    after = content.split(f" AS {name}\n", 1)[1]
    return after.split("\nFROM ", 1)[0]


def _copy_sources(content: str) -> list[str]:
    """Return every context-relative source of a ``COPY`` (``--from`` copies excluded)."""
    sources: list[str] = []
    for line in content.splitlines():
        if not line.startswith("COPY ") or "--from=" in line:
            continue
        sources += [token for token in line.split()[1:-1] if not token.startswith("--")]
    return sources


def _ignore_rules(dockerfile: Path) -> list[str]:
    """Return the non-comment rules of the ignore file BuildKit pairs with ``dockerfile``."""
    lines = _text(dockerfile.with_name(dockerfile.name + ".dockerignore")).splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _in_context(path: str, rules: list[str]) -> bool:
    """Return whether ``path`` reaches the build context under ``rules``.

    An approximation of BuildKit's matcher, enough for the patterns these files
    use: the last matching rule wins, a rule naming a directory covers everything
    under it, and a leading ``**/`` also matches at the top level.
    """
    parts = path.strip("/").split("/")
    candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    included = True
    for rule in rules:
        negate = rule.startswith("!")
        pattern = (rule[1:] if negate else rule).rstrip("/")
        patterns = [pattern, pattern[3:]] if pattern.startswith("**/") else [pattern]
        if any(fnmatch.fnmatchcase(c, p) for c in candidates for p in patterns):
            included = negate
    return included


def test_each_dockerfile_has_an_allowlist_ignore_file() -> None:
    for dockerfile in (_DOCKERFILE, _AIO):
        assert _ignore_rules(dockerfile)[0] == "*", f"{dockerfile.name}: start from `*`"


def test_every_copy_source_exists_at_the_repo_root() -> None:
    for dockerfile in (_DOCKERFILE, _AIO):
        for source in _copy_sources(_text(dockerfile)):
            assert (_REPO / source).exists(), f"{dockerfile.name}: {source} is not in the repo"


def test_every_copy_source_survives_the_ignore_file() -> None:
    for dockerfile in (_DOCKERFILE, _AIO):
        rules = _ignore_rules(dockerfile)
        for source in _copy_sources(_text(dockerfile)):
            assert _in_context(source, rules), f"{dockerfile.name}: the ignore file drops {source}"


def test_local_env_files_never_enter_the_context() -> None:
    # src/ux/.env holds a dev proxy target that must not be baked into the SPA.
    for dockerfile in (_DOCKERFILE, _AIO):
        rules = _ignore_rules(dockerfile)
        for secret in (
            "autotunex/src/ux/.env",
            "autotunex/src/api-bridge/.env",
            "src/gbserver/.env",
        ):
            assert not _in_context(secret, rules), f"{dockerfile.name}: {secret} would be sent"


def test_no_dockerfile_copies_the_license() -> None:
    # Both packages declare their license as text; neither build reads the file.
    for dockerfile in (_DOCKERFILE, _AIO):
        assert not any(s.endswith("LICENSE") for s in _copy_sources(_text(dockerfile)))


def test_the_aio_context_keeps_frontend_build_outputs_and_local_env_out() -> None:
    rules = _ignore_rules(_AIO)

    for path in (
        "frontend/node_modules",
        "frontend/apps/standalone/node_modules",
        "frontend/apps/standalone/.next",
        "frontend/apps/standalone/out",
        "frontend/apps/standalone/.env.local",
        "frontend/apps/standalone/certificates",
    ):
        assert not _in_context(path, rules), f"{path} would be sent"
    assert _in_context("frontend/apps/standalone/package.json", rules)


def test_the_aio_image_does_not_clone_granite_build() -> None:
    content = _text(_AIO)

    assert "git clone" not in _instructions(content)
    assert "GB_REF" not in content
    assert "GB_REPO" not in content


def test_the_aio_dashboard_is_built_by_the_repos_own_recipe() -> None:
    body = _instructions(_stage_body(_text(_AIO), "gb-builder"))

    assert "make build-frontend" in body
    assert "yarn build" not in body
    # A local .env*.local would inline a loopback API URL into the static export.
    assert body.index("rm -f frontend/.env*.local") < body.index("make build-frontend")


def test_the_aio_image_requires_a_version_tag() -> None:
    body = _instructions(_stage_body(_text(_AIO), "gb-builder"))

    assert "git describe --tags --match 'v[0-9]*'" in body


def test_the_aio_image_carries_one_granite_build() -> None:
    # INSTALL_AUTOTUNE_CORE=1 used to add granite.build@stable to the AutoTuneX venv
    # beside the cloned tree in the gb venv: two versions in one image.
    instructions = _instructions(_text(_AIO))

    assert "granite-build" not in instructions
    assert 'GB_PKG="-e /opt/granite.build"' in _stage_body(_text(_AIO), "py-builder")
