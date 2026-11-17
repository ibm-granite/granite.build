#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Store snapshots of a space repo and its base_uris repos in gb_spaces.

The space is then served from the database (``space_uri:
db://spaces/<name>``) with no git access at build or request time. See
gbserver.spaces.db_space.
"""

import contextvars
import io
import os
import posixpath
import re
import subprocess
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import click
import yaml

from gbcommon.types.gbenvconfig import DEFAULT_GH_DOMAIN
from gbserver.spaces.db_space import (
    SNAPSHOT_MAX_ENTRIES,
    SNAPSHOT_MAX_UNCOMPRESSED_BYTES,
    check_own_space_uri,
    is_db_uri,
    make_db_uri,
)
from gbserver.spaces.local_secrets_isolation import check_local_secrets_collision
from gbserver.spaces.repo_identity import is_scp_uri, normalize_repo_uri, same_repo
from gbserver.storage import singleton_storage
from gbserver.storage.stored_space import RepoSnapshot, StoredSpace
from gbserver.types.constants import (
    GBSERVER_GITHUB_TOKEN,
    SPACE_REPO_CONFIG_BRANCH_NAME,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.git_retry import GIT_STALL_ENV
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_utc_time

logger = get_logger(__name__)

SPACE_YAML = "space.yaml"
DEFAULT_GIT_TIMEOUT = 600.0

# What builds resolve from a base_uris repo. Docs, images and other binaries
# stay out: they don't compress and every gb_spaces read loads the archive.
DEFAULT_BASE_URI_PATHS = ("steps", "templates")

# base_uris entry schemes that name a git repo (plus the scp form).
_GIT_SCHEMES = frozenset({"https", "http", "ssh", "git", "git+ssh", "git+https"})


@dataclass
class _Source:
    """Where to snapshot a repo from."""

    location: str  # local checkout dir, or a clonable URL
    ref: Optional[str]  # None: the repo's default branch
    source_uri: str  # normalized https://host/owner/repo (or file://)
    # ref is only the deployment default: fall back to the default branch if
    # the repo lacks it, as git-backed loading does.
    ref_is_default: bool = False


# Per git command; set from --git-timeout.
_git_timeout: contextvars.ContextVar[float] = contextvars.ContextVar(
    "gb_space_snapshot_git_timeout", default=DEFAULT_GIT_TIMEOUT
)


def _git(args: List[str], cwd: Optional[Path] = None) -> bytes:
    """Run git and return its stdout; raise ClickException on failure or timeout.

    The GitHub token (if any) goes in an auth header, never in argv. A stalled
    transfer is abandoned (GIT_STALL_ENV) and the whole command is killed after
    --git-timeout, which also covers SSH; git never prompts for credentials.
    """
    env = dict(os.environ, **GIT_STALL_ENV)
    if GBSERVER_GITHUB_TOKEN:
        # Scoped to the GHE host: never sent to a host a space.yaml names.
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0=f"http.https://{DEFAULT_GH_DOMAIN}/.extraHeader",
            GIT_CONFIG_VALUE_0=f"Authorization: token {GBSERVER_GITHUB_TOKEN}",
        )
    timeout = _git_timeout.get()
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        shown = " ".join(re.sub(r"//[^@/]+@", "//****@", a) for a in args)
        raise click.ClickException(
            f"git {shown} took longer than {timeout:g}s (--git-timeout)"
        ) from e
    if result.returncode != 0:
        raise click.ClickException(
            f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _split_ref(uri: str) -> Tuple[str, Optional[str]]:
    """Split a pip-style ``...owner/repo.git@ref[#frag]`` into (uri-without-ref, ref).

    The fragment is dropped. Works for URLs and the scp form alike. The ref
    starts at an ``@`` in the repo segment and may itself contain ``/``.
    """
    base = uri.partition("#")[0]
    if "://" in base:
        scheme_end = base.index("://") + 3
        slash = base.find("/", scheme_end)
        start = slash if slash >= 0 else len(base)
    else:
        start = base.find(":") + 1  # scp form host:path (0 when no ':')
    prefix, path = base[:start], base[start:]
    segments = path.split("/")
    non_empty = [i for i, seg in enumerate(segments) if seg]
    if len(non_empty) < 2 or "@" not in segments[non_empty[1]]:
        return base, None
    i = non_empty[1]
    name, _, ref_head = segments[i].partition("@")
    ref = "/".join([ref_head, *segments[i + 1 :]])
    if not name or not ref_head:
        return base, None
    return prefix + "/".join([*segments[:i], name]), ref


def _resolve_source(
    repo: str,
    ref: Optional[str],
    source_uri: Optional[str],
    default_ref: Optional[str],
) -> _Source:
    """Turn a --*-repo value (URL in any form, or local checkout) into a _Source."""
    local = Path(repo).expanduser()
    if "://" not in repo and local.is_dir():
        if source_uri is None:
            try:
                origin = _git(["remote", "get-url", "origin"], cwd=local).decode()
                source_uri = origin.strip()
            except click.ClickException:
                source_uri = str(local.resolve())
        return _Source(
            location=str(local.resolve()),
            ref=ref or default_ref,
            source_uri=normalize_repo_uri(source_uri),
            ref_is_default=not ref and default_ref is not None,
        )
    url, url_ref = _split_ref(repo)
    return _Source(
        location=url,
        ref=ref or url_ref or default_ref,
        source_uri=normalize_repo_uri(source_uri or url),
        ref_is_default=not (ref or url_ref) and default_ref is not None,
    )


def _clone_url(url: str, https: bool = False) -> str:
    """The URL to clone ``url`` from.

    With ``https``, always ``https://host/owner/repo.git`` (authenticated with
    GBSERVER_GITHUB_TOKEN on the GHE host), whatever form ``url`` is written in.
    Otherwise as written, with git+ssh:// mapped onto ssh://git@ for git.
    """
    if https:
        normalized = normalize_repo_uri(url)
        if normalized.startswith("https://"):
            return f"{normalized}.git"
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "git+ssh":
        netloc = parsed.netloc if "@" in parsed.netloc else f"git@{parsed.netloc}"
        return parsed._replace(scheme="ssh", netloc=netloc).geturl()
    return url


def _archive(repo_dir: Path, sha: str, paths: Optional[Tuple[str, ...]]) -> bytes:
    """``git archive`` of the tree at ``sha``, limited to ``paths`` if given.

    Paths absent from the tree are skipped with a warning.
    """
    if not paths:
        return _git(["archive", "--format=zip", sha], cwd=repo_dir)
    present = [
        p
        for p in paths
        if _git(["ls-tree", "--name-only", sha, "--", p], cwd=repo_dir).strip()
    ]
    for missing in sorted(set(paths) - set(present)):
        click.echo(f"warning: {missing} not found at {sha[:12]}; skipped", err=True)
    if not present:
        raise click.ClickException(f"none of {list(paths)} found at {sha[:12]}")
    return _git(["archive", "--format=zip", sha, "--", *present], cwd=repo_dir)


def _rev(repo_dir: Path, rev: str) -> Optional[str]:
    """The commit ``rev`` names in ``repo_dir``, or None if it doesn't resolve."""
    try:
        out = _git(
            ["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], cwd=repo_dir
        )
    except click.ClickException:
        return None
    return out.decode().strip() or None


def _local_commit(repo_dir: Path, source: _Source) -> str:
    """The commit to snapshot from a local checkout (no fetch is done).

    ``origin/<ref>`` wins over a local ``<ref>``: the snapshot should match the
    remote branch, not unpushed or stale local work, which is warned about.
    """
    if source.ref:
        remote = _rev(repo_dir, f"origin/{source.ref}")
        local = _rev(repo_dir, source.ref)
        if remote and local and remote != local:
            click.echo(
                f"warning: local {source.ref} ({local[:12]}) differs from"
                f" origin/{source.ref}; snapshotting origin/{source.ref}"
                f" ({remote[:12]}). Fetch first if origin is behind.",
                err=True,
            )
        if remote or local:
            return remote or local  # type: ignore[return-value]
        if not source.ref_is_default:
            raise click.ClickException(f"ref {source.ref!r} not found in {repo_dir}")
        click.echo(
            f"warning: no {source.ref} branch in {repo_dir}; using its default"
            " branch",
            err=True,
        )
        source.ref = None
    head = _rev(repo_dir, "origin/HEAD") or _rev(repo_dir, "HEAD")
    if head is None:
        raise click.ClickException(f"no commit found in {repo_dir}")
    return head


def _snapshot(
    source: _Source, paths: Optional[Tuple[str, ...]] = None, https: bool = False
) -> Tuple[bytes, str]:
    """Return (zip of the tree at source.ref, commit sha). No history included.

    ``git archive`` emits only the committed tree: no .git, no untracked files.
    ``paths`` limits it to those top-level paths.
    """
    if Path(source.location).is_dir():
        repo_dir = Path(source.location)
        sha = _local_commit(repo_dir, source)
        return _archive(repo_dir, sha, paths), sha
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "repo"

        def clone(ref: Optional[str]) -> None:
            branch = [f"--branch={ref}"] if ref else []
            _git(
                [
                    "clone",
                    "--quiet",
                    "--depth=1",
                    "--single-branch",
                    *branch,
                    _clone_url(source.location, https),
                    str(clone_dir),
                ]
            )

        try:
            clone(source.ref)
        except click.ClickException:
            if not (source.ref and source.ref_is_default):
                raise
            click.echo(
                f"warning: no {source.ref} branch in {source.location}; using its"
                " default branch",
                err=True,
            )
            source.ref = None
            clone(None)
        if source.ref is None:
            source.ref = (
                _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=clone_dir)
                .decode()
                .strip()
            )
        sha = _git(["rev-parse", "HEAD"], cwd=clone_dir).decode().strip()
        return _archive(clone_dir, sha, paths), sha


def _check_size(data: bytes, label: str) -> None:
    """Refuse a snapshot zip over the entry or size caps unpacking enforces."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
    total = sum(i.file_size for i in infos)
    if len(infos) > SNAPSHOT_MAX_ENTRIES or total > SNAPSHOT_MAX_UNCOMPRESSED_BYTES:
        raise click.ClickException(
            f"{label} snapshot too large: {len(infos)} entries, {total} bytes"
            f" (limits {SNAPSHOT_MAX_ENTRIES}, {SNAPSHOT_MAX_UNCOMPRESSED_BYTES})"
        )


def _read_space_yaml(space_zip: bytes) -> str:
    """The text of the space snapshot's top-level space.yaml; refuse if it has none."""
    with zipfile.ZipFile(io.BytesIO(space_zip)) as zf:
        try:
            return zf.read(SPACE_YAML).decode("utf-8")
        except KeyError as e:
            raise click.ClickException(
                f"the space snapshot has no top-level {SPACE_YAML}"
            ) from e


def _replace_space_yaml(space_zip: bytes, text: str) -> bytes:
    """Return the zip with space.yaml's content replaced (all else unchanged)."""
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(space_zip)) as src,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst,
    ):
        for info in src.infolist():
            data = (
                text.encode("utf-8") if info.filename == SPACE_YAML else src.read(info)
            )
            dst.writestr(info, data)
    return out.getvalue()


def _base_uris(space_yaml: str) -> List[str]:
    """The ``base_uris`` list of space.yaml's text, as strings (empty if absent)."""
    data = yaml.safe_load(space_yaml) or {}
    return [str(b) for b in (data.get("base_uris") or [])]


def _is_remote(uri: str) -> bool:
    """Whether a base_uris entry is fetched over the network (not file:/db:)."""
    if is_db_uri(uri):
        return False
    if "://" not in uri:
        return is_scp_uri(uri)
    return urllib.parse.urlparse(uri).scheme not in ("", "file")


def _is_git_entry(entry: str) -> bool:
    """Whether a base_uris entry names a git repo (URL or scp form)."""
    if "://" not in entry:
        return _is_remote(entry)  # scp form
    return urllib.parse.urlparse(entry).scheme in _GIT_SCHEMES


def _subdirectory(entry: str) -> str:
    """The ``#subdirectory=`` of a base_uris entry, without slashes at the ends."""
    fragment = entry.partition("#")[2]
    return urllib.parse.parse_qs(fragment).get("subdirectory", [""])[0].strip("/")


def _check_local_entry(entry: str) -> None:
    """A relative file entry is served from inside the snapshot; refuse escapes."""
    parsed = urllib.parse.urlparse(entry)
    if parsed.scheme not in ("", "file"):
        return
    path = (parsed.netloc + parsed.path) if parsed.scheme == "file" else entry
    if not path or path.startswith("/"):
        return  # absolute: a host path, as for any space
    if posixpath.normpath(path).split("/")[0] == "..":
        raise click.ClickException(
            f"base_uri {entry!r} is outside the space repo, so it can't be served"
            " from the snapshot; name its repo as a git base_uri instead"
        )


def _parse_checkouts(values: Tuple[str, ...]) -> List[Tuple[str, Path]]:
    """--checkout values: DIR (matched by its origin) or REPO=DIR."""
    checkouts: List[Tuple[str, Path]] = []
    for value in values:
        repo, sep, path = value.partition("=")
        explicit = bool(sep) and not Path(value).expanduser().is_dir()
        directory = Path(path if explicit else value).expanduser()
        if not directory.is_dir():
            raise click.ClickException(f"--checkout {value!r}: no such directory")
        if not explicit:
            origin = _git(["remote", "get-url", "origin"], cwd=directory)
            repo = origin.decode().strip()
        checkouts.append((repo, directory.resolve()))
    return checkouts


def _base_uris_nodes(space_yaml: str) -> List[yaml.ScalarNode]:
    """The scalar nodes of space.yaml's top-level ``base_uris`` list, in order."""
    root = yaml.compose(space_yaml, Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.MappingNode):
        return []
    for key, value in root.value:
        if isinstance(key, yaml.ScalarNode) and key.value == "base_uris":
            if not isinstance(value, yaml.SequenceNode):
                break
            nodes = value.value
            lo, hi = value.start_mark.index, value.end_mark.index
            for node in nodes:
                # Rewritable only if the entry's own text is in the list: an
                # alias (*name) resolves to a node defined elsewhere, and an
                # anchored entry (&name) may be referenced elsewhere.
                if (
                    not isinstance(node, yaml.ScalarNode)
                    or not lo <= node.start_mark.index < hi
                    or space_yaml[node.start_mark.index] == "&"
                    or sum(n is node for n in nodes) > 1
                ):
                    raise click.ClickException(
                        f"{SPACE_YAML} base_uris must be plain strings (no YAML"
                        " anchors or aliases) to rewrite"
                    )
            return nodes
    return []


def _rewrite_base_uris(space_yaml: str, entries: List[str], new: List[str]) -> str:
    """Replace changed base_uris entries in space.yaml's text, and nothing else.

    Each entry is replaced at its exact position from the YAML parser, so the
    same URL elsewhere (another field, a comment) is untouched, as are comments
    and layout. A quoted entry keeps its quotes. The result must parse to
    exactly ``new``, in order.
    """
    nodes = _base_uris_nodes(space_yaml)
    if [n.value for n in nodes] != entries:
        raise click.ClickException(f"could not locate the base_uris of {SPACE_YAML}")
    text = space_yaml
    # Last first, so earlier positions stay valid.
    for node, old, rep in reversed(list(zip(nodes, entries, new))):
        if old == rep:
            continue
        quote = node.style if node.style in ("'", '"') else ""
        start, end = node.start_mark.index, node.end_mark.index
        text = f"{text[:start]}{quote}{rep}{quote}{text[end:]}"
    try:
        rewritten = _base_uris(text)
    except yaml.YAMLError as e:
        raise click.ClickException(f"rewritten {SPACE_YAML} does not parse: {e}") from e
    if rewritten != new:
        raise click.ClickException(f"could not rewrite the base_uris of {SPACE_YAML}")
    return text


def _load_row(
    storage,
    space_name: str,
    create: bool,
    space_repo: Optional[str],
    lakehouse_namespace: Optional[str],
) -> Tuple[StoredSpace, bool]:
    """The row to snapshot, and whether it is new (--create).

    A new row's git_repo_uri is set later, from the snapshot's source.

    Raises:
        click.ClickException: No such row without --create, --create without
            --space-repo, or a space_uri naming another space.
    """
    row = storage.get_by_name(space_name)
    is_new = row is None
    if row is None:
        if not create:
            raise click.ClickException(
                f"space {space_name!r} not found (use --create to add it)"
            )
        if space_repo is None:
            raise click.ClickException("--space-repo is required with --create")
        row = StoredSpace(
            name=space_name, git_repo_uri="", lakehouse_namespace=lakehouse_namespace
        )
    try:
        check_own_space_uri(row)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    return row, is_new


def _snapshot_space(
    row: StoredSpace,
    is_new: bool,
    space_repo: Optional[str],
    space_ref: Optional[str],
    space_source_uri: Optional[str],
    use_https: bool,
) -> Tuple[_Source, bytes, str]:
    """Snapshot the space repo: (its source, zip, commit).

    It must be the row's repo (git_repo_uri, in any form), which secrets and
    PRs key on; a new row takes the source as its git_repo_uri.
    """
    space_src = _resolve_source(
        space_repo or row.git_repo_uri,
        space_ref,
        space_source_uri,
        SPACE_REPO_CONFIG_BRANCH_NAME,
    )
    if is_new:
        row.git_repo_uri = space_src.source_uri
    elif not same_repo(row.git_repo_uri, space_src.source_uri):
        raise click.ClickException(
            f"snapshot source {space_src.source_uri} is not space {row.name!r}'s"
            f" repo {row.git_repo_uri}; pass --space-source-uri if it is a mirror"
        )
    click.echo(f"space: {space_src.location} @ {space_src.ref}")
    space_zip, space_sha = _snapshot(space_src, https=use_https)
    return space_src, space_zip, space_sha


def _snapshot_base_uris(
    space_name: str,
    entries: List[str],
    checkouts: Tuple[str, ...],
    base_uri_paths: Tuple[str, ...],
    use_https: bool,
    skip: bool,
) -> Tuple[List[Optional[RepoSnapshot]], List[str]]:
    """Snapshot space.yaml's git base_uris: (slots, entries rewritten to db://).

    One snapshot per (repo, ref), at its first entry's index; later entries for
    it reuse that slot. Each takes ``base_uri_paths`` under every entry's
    #subdirectory. A repo is snapshotted from its --checkout, else cloned.
    With ``skip``, nothing is snapshotted and the entries stay as written.
    """
    slots: List[Optional[RepoSnapshot]] = [None] * len(entries)
    new_entries = list(entries)
    groups: Dict[Tuple[str, Optional[str]], List[int]] = {}
    for i, entry in enumerate([] if skip else entries):
        if is_db_uri(entry):
            raise click.ClickException(
                f"base_uri {entry!r} is already db://; snapshot from the space's"
                " own repo"
            )
        if not _is_git_entry(entry):
            _check_local_entry(entry)
            continue
        url, ref = _split_ref(entry)
        groups.setdefault((normalize_repo_uri(url).lower(), ref), []).append(i)
    local = _parse_checkouts(checkouts)
    for (_, ref), indices in groups.items():
        first = indices[0]
        url = _split_ref(entries[first])[0]
        repo = normalize_repo_uri(url)
        checkout = next((d for r, d in local if same_repo(r, url)), None)
        src = _resolve_source(str(checkout or url), ref, repo, None)
        click.echo(
            f"base_uris[{first}]: {src.location} @ {src.ref or 'default branch'}"
        )
        subdirs = sorted({_subdirectory(entries[i]) for i in indices})
        paths = tuple(
            f"{sd}/{p}" if sd else p for sd in subdirs for p in base_uri_paths
        )
        data, sha = _snapshot(src, paths, https=use_https)
        _check_size(data, f"base_uris[{first}]")
        snapshot = RepoSnapshot(
            source_uri=repo,
            source_ref=src.ref,
            source_commit=sha,
            snapshot_time=get_utc_time(),
            entry=entries[first],
        )
        snapshot.save_archive(data)
        slots[first] = snapshot
        for i in indices:
            fragment = entries[i].partition("#")[2]
            new_entries[i] = make_db_uri(space_name, first) + (
                f"#{fragment}" if fragment else ""
            )
            click.echo(f"base_uris[{i}]: {entries[i]} -> {new_entries[i]}")
    for repo, directory in local:
        if not any(
            same_repo(repo, _split_ref(entries[g[0]])[0]) for g in groups.values()
        ):
            click.echo(
                f"warning: --checkout {directory} ({repo}) matches no base_uris"
                " entry; unused",
                err=True,
            )
    return slots, new_entries


@click.command()
@click.option("--space-name", required=True, help="gb_spaces row to update.")
@click.option(
    "--space-repo",
    default=None,
    help="Space repo: a URL (https://, git+ssh://, ssh://, git@host:o/r) or a"
    " local checkout. Default: the row's current repo or snapshot source.",
)
@click.option(
    "--space-ref",
    default=None,
    help=f"Branch to snapshot (default: {SPACE_REPO_CONFIG_BRANCH_NAME}).",
)
@click.option(
    "--space-source-uri",
    default=None,
    help="Repo to record as the source (default: --space-repo, or a local"
    " checkout's origin). Secret groups are matched on it.",
)
@click.option(
    "--checkout",
    "checkouts",
    multiple=True,
    help="Local checkout to snapshot a base_uris repo from: DIR (matched by its"
    " origin) or REPO=DIR; repeatable. Repos without one are cloned.",
)
@click.option(
    "--https",
    "use_https",
    is_flag=True,
    help="Clone remote repos over https://<host>/<owner>/<repo>.git with"
    " GBSERVER_GITHUB_TOKEN, even where they are written as git+ssh://.",
)
@click.option(
    "--base-uri-path",
    "base_uri_paths",
    multiple=True,
    default=DEFAULT_BASE_URI_PATHS,
    show_default=True,
    help="Path to include from each base_uris repo (under its #subdirectory);"
    " repeatable.",
)
@click.option(
    "--no-base-uris",
    is_flag=True,
    help="Snapshot only the space repo; base_uris stay as written.",
)
@click.option(
    "--switch/--no-switch",
    default=True,
    show_default=True,
    help="Serve the space from the snapshot (space_uri: db://spaces/<name>)."
    " With --no-switch the snapshot is stored and space_uri is left as is.",
)
@click.option(
    "--create",
    is_flag=True,
    help="Create the row if it does not exist.",
)
@click.option(
    "--lakehouse-namespace", default=None, help="For --create: lakehouse namespace."
)
@click.option(
    "--git-timeout",
    type=float,
    default=DEFAULT_GIT_TIMEOUT,
    show_default=True,
    help="Seconds before any one git command (e.g. a clone) is killed.",
)
@click.option("--dry-run", is_flag=True, help="Report what would be stored.")
@pass_environment
def cli(
    ctx: CliEnvironment,
    space_name: str,
    space_repo: Optional[str],
    space_ref: Optional[str],
    space_source_uri: Optional[str],
    checkouts: Tuple[str, ...],
    use_https: bool,
    base_uri_paths: Tuple[str, ...],
    no_base_uris: bool,
    switch: bool,
    create: bool,
    lakehouse_namespace: Optional[str],
    git_timeout: float,
    dry_run: bool,
):
    """Snapshot a space repo and the repos its base_uris name into gb_spaces.

    Only the tree at the branch head is stored, no git history. In the stored
    space.yaml, each git base_uris[i] becomes db://spaces/<name>/base_uri_<i>,
    in place, so lookup order is unchanged. git_repo_uri, the space's identity,
    is unchanged: the snapshot must come from that repo.
    """
    _git_timeout.set(git_timeout)
    storage = singleton_storage.get_admin_storage().space_storage
    row, is_new = _load_row(
        storage, space_name, create, space_repo, lakehouse_namespace
    )
    space_src, space_zip, space_sha = _snapshot_space(
        row, is_new, space_repo, space_ref, space_source_uri, use_https
    )
    space_yaml = _read_space_yaml(space_zip)
    entries = _base_uris(space_yaml)
    slots, new_entries = _snapshot_base_uris(
        space_name, entries, checkouts, base_uri_paths, use_https, no_base_uris
    )
    if new_entries != entries:
        space_yaml = _rewrite_base_uris(space_yaml, entries, new_entries)
        space_zip = _replace_space_yaml(space_zip, space_yaml)
    for still_remote in (e for e in new_entries if _is_remote(e)):
        click.echo(
            f"warning: base_uri {still_remote} is still fetched remotely", err=True
        )
    _check_size(space_zip, "space")

    space = RepoSnapshot(
        source_uri=space_src.source_uri,
        source_ref=space_src.ref,
        source_commit=space_sha,
        snapshot_time=get_utc_time(),
    )
    space.save_archive(space_zip)

    updated = row.model_copy(
        update={"space": space, "base_uris": None if no_base_uris else slots}
    )
    if switch:
        updated.space_uri = make_db_uri(space_name)

    # Same guard create-spaces applies: no two spaces may share local secrets.
    if is_db_uri(updated.space_uri):
        check_local_secrets_collision(
            updated,
            [
                s
                for s in cast(List[StoredSpace], storage.get_by_uuid(None) or [])
                if s.name != space_name
            ],
        )

    click.echo(
        f"space {space_name} ({updated.git_repo_uri}): space_uri"
        f" {row.get_space_uri()} -> {updated.get_space_uri()};"
        f" space {space_src.source_uri}@{space_sha[:12]} ({len(space_zip)} bytes)"
        + "".join(
            f", base_uris[{i}] {snap.source_uri}@{(snap.source_commit or '')[:12]}"
            f" ({len(snap.archive or '')} bytes stored)"
            for i, snap in enumerate(slots)
            if snap
        )
    )
    if dry_run:
        click.echo("dry run: nothing stored")
        return
    if is_new:
        storage.add(updated)
    else:
        storage.update(updated, create_if_not_exist=False)
    click.echo("stored")
