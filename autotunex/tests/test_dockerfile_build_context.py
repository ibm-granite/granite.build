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
import re
import tomllib
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


def _bind_sources(content: str) -> list[str]:
    """Return every context-relative source of a ``RUN --mount=type=bind`` (``from=`` excluded)."""
    sources: list[str] = []
    for mount in re.findall(r"--mount=(\S+)", content):
        options = dict(option.split("=", 1) for option in mount.split(","))
        if options.get("type") == "bind" and "from" not in options:
            sources.append(options["source"])
    return sources


def _runs(stage: str) -> list[str]:
    """Return each ``RUN`` in ``stage`` as one string, continuation lines included."""
    return _instructions(stage).split("\nRUN ")[1:]


def _scm_pretend_env() -> str:
    """Return the variable setuptools_scm reads a pretend version for granite.build from."""
    name = tomllib.loads(_text(_REPO / "pyproject.toml"))["project"]["name"]
    return "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_" + re.sub(r"[-_.]+", "_", name).upper()


def _ignore_rules(dockerfile: Path) -> list[str]:
    """Return the non-comment rules of the ignore file BuildKit pairs with ``dockerfile``."""
    lines = _text(dockerfile.with_name(dockerfile.name + ".dockerignore")).splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def _in_context(path: str, rules: list[str]) -> bool:
    """Return whether ``path`` reaches the build context under ``rules``.

    An approximation of BuildKit's matcher, enough for the patterns these files
    use: the last matching rule wins, a rule naming a directory covers everything
    under it, and a leading ``**/`` also matches at the top level. A ``/**/``
    matches zero directories too, so ``frontend/**/out`` covers ``frontend/out``.
    """
    parts = path.strip("/").split("/")
    candidates = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    included = True
    for rule in rules:
        negate = rule.startswith("!")
        pattern = (rule[1:] if negate else rule).rstrip("/")
        patterns = [pattern, pattern.replace("/**/", "/")]
        if pattern.startswith("**/"):
            patterns += [p[3:] for p in patterns]
        if any(fnmatch.fnmatchcase(c, p) for c in candidates for p in patterns):
            included = negate
    return included


def test_each_dockerfile_has_an_allowlist_ignore_file() -> None:
    for dockerfile in (_DOCKERFILE, _AIO):
        assert _ignore_rules(dockerfile)[0] == "*", f"{dockerfile.name}: start from `*`"


def test_every_copy_and_bind_source_exists_at_the_repo_root() -> None:
    for dockerfile in (_DOCKERFILE, _AIO):
        content = _text(dockerfile)
        for source in _copy_sources(content) + _bind_sources(content):
            assert (_REPO / source).exists(), f"{dockerfile.name}: {source} is not in the repo"


def test_every_copy_and_bind_source_survives_the_ignore_file() -> None:
    for dockerfile in (_DOCKERFILE, _AIO):
        rules = _ignore_rules(dockerfile)
        content = _text(dockerfile)
        for source in _copy_sources(content) + _bind_sources(content):
            assert _in_context(source, rules), f"{dockerfile.name}: the ignore file drops {source}"


def test_known_junk_inside_the_allowlisted_trees_stays_out() -> None:
    # Untracked files under a re-included tree enter the context, and granite.build's
    # `"*"` package data would ship them in the wheel.
    for dockerfile in (_DOCKERFILE, _AIO):
        rules = _ignore_rules(dockerfile)
        for junk in (
            "src/gbserver/dist/granite_build-0.0.0.tar.gz",
            "test/run.log",
            "src/.coverage",
            "test/.tox/py312",
        ):
            assert not _in_context(junk, rules), f"{dockerfile.name}: {junk} would be sent"
    assert not _in_context("steps/train/fm-tune/bash/space/space.yaml", _ignore_rules(_AIO))


def test_local_env_files_never_enter_the_context() -> None:
    # src/ux/.env holds a dev proxy target that must not be baked into the SPA.
    for dockerfile in (_DOCKERFILE, _AIO):
        rules = _ignore_rules(dockerfile)
        for secret in (
            "autotunex/src/ux/.env",
            "autotunex/src/api-bridge/.env",
            "src/gbserver/.env",
            "autotunex/src/ux/.env.local",
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
        "frontend/certificates/localhost-key.pem",
        "frontend/.next",
        "frontend/out",
    ):
        assert not _in_context(path, rules), f"{path} would be sent"
    assert _in_context("frontend/apps/standalone/package.json", rules)


def test_no_dockerfile_copies_git_into_a_layer() -> None:
    # A copied .git would publish the builder's config, remotes and reflogs, and make
    # every commit, fetch or rebase invalidate every layer after it.
    for dockerfile in (_DOCKERFILE, _AIO):
        sources = _copy_sources(_text(dockerfile))
        assert not [s for s in sources if s.rstrip("/") == ".git"], dockerfile.name


def test_every_granite_build_install_can_read_the_version_tags() -> None:
    # setuptools_scm stamps the version at build time; without .git (or GB_VERSION)
    # it falls back to 0.0.0, below the llmb CLI's floor.
    stages = [
        (_DOCKERFILE, _stage_body(_text(_DOCKERFILE), "gb-wheel"), "pip wheel"),
        (_AIO, _stage_body(_text(_AIO), "py-builder"), "/opt/granite.build"),
    ]
    for dockerfile, stage, marker in stages:
        builds = [run for run in _runs(stage) if marker in run and "pip" in run]
        assert builds, f"{dockerfile.name}: no granite.build install found"
        for run in builds:
            assert "source=.git" in run, f"{dockerfile.name}: a granite.build install has no .git"


def test_the_aio_runtime_does_not_ship_a_git_directory() -> None:
    # .git is only bind-mounted, but a builder can leave the empty mount point behind.
    runs = _runs(_stage_body(_text(_AIO), "py-builder"))
    last_mount = max(i for i, run in enumerate(runs) if "source=.git" in run)

    assert any("rm -rf /opt/granite.build/.git" in run for run in runs[last_mount + 1 :])


def test_gb_version_stands_in_for_a_missing_tag() -> None:
    # Worktrees and shallow CI clones reach no v* tag; GB_VERSION is their way through.
    stages = [
        (_DOCKERFILE, _stage_body(_text(_DOCKERFILE), "gb-wheel")),
        (_AIO, _stage_body(_text(_AIO), "py-builder")),
    ]
    for dockerfile, stage in stages:
        body = _instructions(stage)
        assert 'ARG GB_VERSION=""' in body, dockerfile.name
        assert f"ENV {_scm_pretend_env()}=${{GB_VERSION}}" in body, dockerfile.name
        check = next(run for run in _runs(stage) if "git" in run and "describe" in run)
        assert "$GB_VERSION" in check, f"{dockerfile.name}: GB_VERSION does not skip the tag check"
        assert "exit 1" in check, f"{dockerfile.name}: a missing tag no longer fails the build"


def test_the_aio_pip_installs_reuse_a_pip_cache() -> None:
    # The granite.build tree changes with every commit, so these layers rerun often.
    stage = _stage_body(_text(_AIO), "py-builder")

    assert "PIP_NO_CACHE_DIR" not in _instructions(stage)
    for run in _runs(stage):
        if "pip install" in run:
            assert "--mount=type=cache,target=/root/.cache/pip" in run, run.splitlines()[0]


def test_the_aio_image_does_not_clone_granite_build() -> None:
    content = _text(_AIO)

    assert "git clone" not in _instructions(content)
    assert "GB_REF" not in content
    assert "GB_REPO" not in content


def test_the_aio_dashboard_is_built_by_the_repos_own_recipe() -> None:
    body = _instructions(_stage_body(_text(_AIO), "dashboard"))

    assert "make build-frontend" in body
    assert "yarn build" not in body
    # A local .env*.local would inline a loopback API URL into the static export.
    assert body.index("rm -f frontend/.env*.local") < body.index("make build-frontend")


def test_the_aio_dashboard_rebuilds_only_when_its_inputs_change() -> None:
    # Anything else copied in (above all .git) would rerun yarn install and
    # `next build` for commits that never touched frontend/.
    stage = _stage_body(_text(_AIO), "dashboard")

    assert sorted(_copy_sources(stage) + _bind_sources(stage)) == ["Makefile", "frontend/"]


def test_the_aio_image_ships_the_docs_search_docs_reads() -> None:
    # dashboard_tools._docs_root() resolves docs/ beside src/ in the editable install.
    assert "docs/" in _copy_sources(_stage_body(_text(_AIO), "py-builder"))
    assert _in_context("docs/README.md", _ignore_rules(_AIO))


def test_the_aio_image_carries_one_granite_build() -> None:
    # INSTALL_AUTOTUNE_CORE=1 used to add granite.build@stable to the AutoTuneX venv
    # beside the cloned tree in the gb venv: two versions in one image.
    instructions = _instructions(_text(_AIO))

    assert "granite-build" not in instructions
    assert "-e /opt/granite.build" in _stage_body(_text(_AIO), "py-builder")


def test_autotunex_pulls_no_dependency_from_a_url() -> None:
    # A direct reference (`granite.build @ git+https://...@stable`) installs a
    # granite.build other than the one in this checkout.
    project = tomllib.loads(_text(_AUTOTUNEX / "pyproject.toml"))
    deps = list(project["project"]["dependencies"])
    for extra in project["project"].get("optional-dependencies", {}).values():
        deps += extra

    assert not [dep for dep in deps if " @ " in dep]
    assert "metadata" not in project.get("tool", {}).get("hatch", {})
