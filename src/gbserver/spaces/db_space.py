#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Spaces whose definitions are stored in gb_spaces instead of a git repo.

A row with ``space_uri: db://spaces/<name>`` is served from its own ``space``
snapshot (StoredSpace.space). The notation could name another row, but serving
one row from another's snapshot is not supported: every db-backed row holds
its own copy. Its ``git_repo_uri`` stays the space's git repo. The
snapshot's space.yaml names each snapshotted base_uris entry as
``db://spaces/<name>/base_uri_<i>`` (``base_uris[i]``), served from that row's
``base_uris[i]`` snapshot, in its original position.
Either may carry a ``#subdirectory=`` fragment.

Snapshots are unpacked into a local cache (gbserver.spaces.snapshot_cache) and
served as ``file://``, so no git access is needed. Secret lookup, space
matching and build PRs keep using ``git_repo_uri``, never the local path.
"""

import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Optional

from gbcommon.uri.git import GitURI
from gbserver.spaces import snapshot_cache
from gbserver.spaces.repo_identity import normalize_repo_uri, to_config_uri
from gbserver.storage.stored_space import RepoSnapshot, StoredSpace
from gbserver.utils.archive import extract_archive
from gbserver.utils.logger import get_logger

if TYPE_CHECKING:
    from gbserver.build.space import Space

logger = get_logger(__name__)

DB_SCHEME = "db"
DB_SPACES_HOST = "spaces"
SPACE_SLOT: Final = "space"
BASE_URI_SLOT_PREFIX: Final = "base_uri_"
_BASE_URI_SLOT = re.compile(r"base_uri_(0|[1-9][0-9]*)")

# Snapshots are admin-written, not client uploads, so the caps only guard
# against a corrupt row and are looser than the build-archive defaults.
SNAPSHOT_MAX_ENTRIES = 20_000
SNAPSHOT_MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024


@dataclass(frozen=True)
class DbUri:
    """A parsed ``db://spaces/<space_name>[/<slot>][#subdirectory=...]``."""

    space_name: str
    slot: str = SPACE_SLOT  # "space", or "base_uri_<i>"
    subdirectory: str = ""

    @property
    def base_uri_index(self) -> Optional[int]:
        """``i`` for a ``base_uri_<i>`` slot, None for the space slot."""
        if self.slot == SPACE_SLOT:
            return None
        return int(self.slot[len(BASE_URI_SLOT_PREFIX) :])


def is_db_uri(uri: Optional[str]) -> bool:
    """Return whether ``uri`` uses the db:// scheme."""
    return urllib.parse.urlparse((uri or "").strip()).scheme == DB_SCHEME


def make_db_uri(space_name: str, base_uri_index: Optional[int] = None) -> str:
    """The db:// URI of a row's space snapshot, or of its ``base_uris[i]`` one."""
    uri = f"{DB_SCHEME}://{DB_SPACES_HOST}/{urllib.parse.quote(space_name, safe='')}"
    if base_uri_index is None:
        return uri
    return f"{uri}/{BASE_URI_SLOT_PREFIX}{base_uri_index}"


def _is_slot(name: str) -> bool:
    """Whether ``name`` is a snapshot slot: "space" or "base_uri_<i>"."""
    return name == SPACE_SLOT or bool(_BASE_URI_SLOT.fullmatch(name))


def parse_db_uri(uri: str) -> DbUri:
    """Parse a db:// space URI.

    Raises:
        ValueError: Not a well-formed ``db://spaces/<name>[/base_uri_<i>]``.
    """
    parsed = urllib.parse.urlparse(uri.strip())
    parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
    if (
        parsed.scheme != DB_SCHEME
        or parsed.netloc != DB_SPACES_HOST
        or not (len(parts) == 1 or (len(parts) == 2 and _is_slot(parts[1])))
    ):
        raise ValueError(
            f"invalid db space URI {uri!r}; expected"
            f" {DB_SCHEME}://{DB_SPACES_HOST}/<space_name>[/base_uri_<i>]"
        )
    subdir = urllib.parse.parse_qs(parsed.fragment).get("subdirectory", [""])[0]
    slot = parts[1] if len(parts) == 2 else SPACE_SLOT
    return DbUri(space_name=parts[0], slot=slot, subdirectory=subdir)


def _get_row(space_name: str) -> StoredSpace:
    """The gb_spaces row named ``space_name``; ValueError if there is none."""
    # Local import: singleton storage pulls in the storage backends.
    from gbserver.storage.singleton_storage import get_admin_storage

    row = get_admin_storage().space_storage.get_by_name(space_name)
    if row is None:
        raise ValueError(f"db space URI names unknown space {space_name!r}")
    return row


def _get_snapshot(row: StoredSpace, ref: DbUri) -> RepoSnapshot:
    """The snapshot a db:// URI's slot names in ``row``; ValueError if it has none."""
    index = ref.base_uri_index
    if index is None:
        snapshot = row.space
    else:
        slots = row.base_uris or []
        snapshot = slots[index] if index < len(slots) else None
    if snapshot is None or not snapshot.archive:
        raise ValueError(f"space {row.name!r} has no {ref.slot} snapshot")
    return snapshot


def portable_uri(uri: str) -> str:
    """Map a ``file://`` URI inside an unpacked snapshot back to its db:// URI.

    The unpacked path exists only on this host and changes with each snapshot,
    so anything persisted or handed to another process uses this instead. The
    cache path names the space and slot, so no lookup is needed. Other URIs
    are returned unchanged.
    """
    if not uri.startswith("file://"):
        return uri
    # Decode: as_uri() percent-encodes the path (e.g. a quoted space name).
    path = Path(urllib.parse.unquote(urllib.parse.urlparse(uri).path))
    cached = snapshot_cache.parse_cached_path(path)
    if cached is None:
        return uri
    space_name, slot, subpath = cached
    ref = DbUri(space_name=space_name, slot=slot)
    db_uri = make_db_uri(space_name, ref.base_uri_index)
    return db_uri + (f"#subdirectory={subpath}" if subpath else "")


def _unpacker(snapshot: RepoSnapshot, label: str):
    """Extract a snapshot's zip into a given directory, within the size caps."""

    def extract(dest: Path) -> None:
        if not extract_archive(
            snapshot.load_archive(),
            dest,
            archive_format="zip",
            max_entries=SNAPSHOT_MAX_ENTRIES,
            max_uncompressed_bytes=SNAPSHOT_MAX_UNCOMPRESSED_BYTES,
        ):
            raise ValueError(f"{label} is not a readable zip archive")

    return extract


@dataclass(frozen=True)
class ResolvedDbUri:
    """A db:// URI unpacked onto the local filesystem."""

    file_uri: str
    # The snapshot's source repo, as GitURI.get_gb_space_config_uri would give
    # it (git+ssh://host/o/r.git@<ref>); file_uri when the source is unknown.
    identity_uri: str


def resolve_db_uri(uri: str, row: Optional[StoredSpace] = None) -> ResolvedDbUri:
    """Unpack the snapshot a db:// URI names and return where it landed.

    Args:
        uri: A ``db://spaces/...`` URI.
        row: The already-loaded row it names, to skip the lookup.
    """
    ref = parse_db_uri(uri)
    if row is None or row.name != ref.space_name:
        row = _get_row(ref.space_name)
    snapshot = _get_snapshot(row, ref)
    label = f"space {row.name!r} {ref.slot} snapshot"
    assert snapshot.archive is not None  # _get_snapshot checked
    path = snapshot_cache.materialize(
        row.name, ref.slot, snapshot.archive, _unpacker(snapshot, label), label
    )
    root = path.resolve()
    if ref.subdirectory:
        # Contained: '..' or an absolute path would reach another copy (or
        # another space's) in the shared cache root.
        path = (root / ref.subdirectory).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"db space URI {uri!r}: subdirectory escapes the snapshot")
    file_uri = path.resolve().as_uri()
    identity_uri = (
        to_config_uri(snapshot.source_uri, snapshot.source_ref)
        if snapshot.source_uri
        else file_uri
    )
    return ResolvedDbUri(file_uri=file_uri, identity_uri=identity_uri)


def check_own_space_uri(stored_space: StoredSpace) -> None:
    """Refuse a db:// space_uri naming another row (not supported).

    Raises:
        ValueError: space_uri is a db:// URI other than the row's own.
    """
    uri = stored_space.space_uri
    if is_db_uri(uri) and parse_db_uri(uri or "").space_name != stored_space.name:
        raise ValueError(
            f"space {stored_space.name!r}: space_uri {uri!r} names another space;"
            f" serving a space from another's snapshot is not supported, use"
            f" {make_db_uri(stored_space.name)}"
        )


def local_uri(uri: str) -> str:
    """A db:// URI as the ``file://`` URI of its unpacked snapshot; others as is."""
    return resolve_db_uri(uri).file_uri if is_db_uri(uri) else uri


def _source_row(stored_space: StoredSpace) -> StoredSpace:
    """The row itself, once its space_uri is checked to name it."""
    check_own_space_uri(stored_space)
    return stored_space


def get_pr_repo_uri(stored_space: StoredSpace) -> Optional[str]:
    """The ``https://host/owner/repo`` holding a space's build PRs, or None.

    That is git_repo_uri in any form, also for a db-backed space. None for a
    local space, or when the repo can't be told.
    """
    try:
        uri = normalize_repo_uri(stored_space.git_repo_uri)
    except ValueError as e:
        logger.warning("space %r has no PR repo: %s", stored_space.name, e)
        return None
    return None if uri.startswith("file://") else uri


def get_space_config_uri(stored_space: StoredSpace, token: Optional[str] = None) -> str:
    """URI to pull a stored space's definitions from (space.yaml, environments...).

    A db-backed space is unpacked and returned as ``file://``. Its space.yaml
    may still name ``db://`` base_uris, which only Space resolves; callers
    that only read the space's own files need nothing more.
    """
    uri = stored_space.get_space_uri()
    if is_db_uri(uri):
        return resolve_db_uri(uri, row=_source_row(stored_space)).file_uri
    if token is None:
        return GitURI.get_gb_space_config_uri(uri=uri)
    return GitURI.get_gb_space_config_uri(uri=uri, token=token)


def _db_identity_uri(stored_space: StoredSpace) -> Optional[str]:
    """git_repo_uri as a git-backed load would report it (git+ssh, @ref).

    Token-free: the branch comes from the snapshot instead of a GitHub probe.
    """
    snapshot = _source_row(stored_space).space
    try:
        return to_config_uri(
            stored_space.git_repo_uri, snapshot.source_ref if snapshot else None
        )
    except ValueError:
        return None  # not a repo URI; Space falls back to the snapshot source


def create_space(
    stored_space: StoredSpace,
    username: Optional[str] = None,
    token: Optional[str] = None,
    force_fetch: bool = False,
) -> "Space":
    """Build the Space for a stored space, git- or db-backed."""
    # Local import: gbserver.build pulls in the whole build engine.
    from gbserver.build.space import Space

    uri = stored_space.get_space_uri()
    if is_db_uri(uri):
        # Space resolves db:// itself; it is identified by its git repo.
        logger.info("space %s is served from %s", stored_space.name, uri)
        return Space(
            uri=uri,
            username=username,
            force_fetch=force_fetch,
            identity_uri=_db_identity_uri(stored_space),
            stored_space=stored_space,  # holds the archives: no re-read
        )
    uri = get_space_config_uri(stored_space, token=token)
    logger.info("for the space name %s using the URI: %s", stored_space.name, uri)
    return Space(uri=uri, username=username, force_fetch=force_fetch)
