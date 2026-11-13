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

"""Spaces whose definitions are stored in gb_spaces instead of a git repo.

A row with ``space_uri: db://spaces/<name>`` is served from the ``space``
snapshot (StoredSpace.space) of row ``<name>`` -- normally itself; another
name makes it an alias. Its ``git_repo_uri`` stays the space's git repo. The
snapshot's space.yaml names its assets repo as
``db://spaces/<name>/assets``, served from that row's ``assets`` snapshot.
Either may carry a ``#subdirectory=`` fragment.

Snapshots are unpacked into a local directory named by content digest and
served as ``file://``, so no git access is needed. Secret lookup, space
matching and build PRs keep using ``git_repo_uri``, never the local path.
"""

import hashlib
import os
import shutil
import tempfile
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Final, Literal, Optional, cast

from gbcommon.uri.git import GitURI
from gbserver.spaces.repo_identity import normalize_repo_uri, to_config_uri
from gbserver.storage.stored_space import RepoSnapshot, StoredSpace
from gbserver.utils.archive import extract_archive
from gbserver.utils.logger import get_logger

if TYPE_CHECKING:
    from gbserver.build.space import Space

logger = get_logger(__name__)

DB_SCHEME = "db"
DB_SPACES_HOST = "spaces"
Slot = Literal["space", "assets"]
SPACE_SLOT: Final = "space"
ASSETS_SLOT: Final = "assets"

# Snapshots are admin-written, not client uploads, so the caps only guard
# against a corrupt row and are looser than the build-archive defaults.
SNAPSHOT_MAX_ENTRIES = 20_000
SNAPSHOT_MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024

# Written last, so a directory without it is a partial extraction.
_COMPLETE_MARKER = ".gb-snapshot-complete"


@dataclass(frozen=True)
class DbUri:
    """A parsed ``db://spaces/<space_name>[/<slot>][#subdirectory=...]``."""

    space_name: str
    slot: Slot = SPACE_SLOT
    subdirectory: str = ""


def is_db_uri(uri: Optional[str]) -> bool:
    """Return whether ``uri`` uses the db:// scheme."""
    return urllib.parse.urlparse((uri or "").strip()).scheme == DB_SCHEME


def make_db_uri(space_name: str, slot: Slot = SPACE_SLOT) -> str:
    """Build the db:// URI for a row's snapshot slot."""
    uri = f"{DB_SCHEME}://{DB_SPACES_HOST}/{urllib.parse.quote(space_name, safe='')}"
    return uri if slot == SPACE_SLOT else f"{uri}/{slot}"


def parse_db_uri(uri: str) -> DbUri:
    """Parse a db:// space URI.

    Raises:
        ValueError: Not a well-formed ``db://spaces/<name>[/space|/assets]``.
    """
    parsed = urllib.parse.urlparse(uri.strip())
    parts = [urllib.parse.unquote(p) for p in parsed.path.split("/") if p]
    if (
        parsed.scheme != DB_SCHEME
        or parsed.netloc != DB_SPACES_HOST
        or not 1 <= len(parts) <= 2
        or (len(parts) == 2 and parts[1] not in (SPACE_SLOT, ASSETS_SLOT))
    ):
        raise ValueError(
            f"invalid db space URI {uri!r}; expected"
            f" {DB_SCHEME}://{DB_SPACES_HOST}/<space_name>[/{ASSETS_SLOT}]"
        )
    subdir = urllib.parse.parse_qs(parsed.fragment).get("subdirectory", [""])[0]
    slot = cast(Slot, parts[1] if len(parts) == 2 else SPACE_SLOT)
    return DbUri(space_name=parts[0], slot=slot, subdirectory=subdir)


def get_snapshot_cache_root() -> Path:
    """Root under which snapshots are unpacked, one dir per archive digest.

    Per user: a shared, predictable path would let another local user plant a
    tree for a digest that is computable from the (readable) archive.
    """
    return Path(tempfile.gettempdir()) / f"gb-db-spaces-{os.getuid()}"


def _private_root() -> Path:
    """The cache root, created 0700; refused if not ours or writable by others."""
    root = get_snapshot_cache_root()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = root.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise ValueError(
            f"refusing snapshot cache {root}: not owned by this user or writable"
            " by others"
        )
    return root


def unpack_snapshot(snapshot: RepoSnapshot, label: str) -> Path:
    """Unpack a snapshot into its content-addressed cache dir and return it.

    An unchanged snapshot is reused across builds and requests (no per-call
    leak on the rest-server); a refreshed one gets a new dir. Extraction goes
    to a sibling temp dir renamed into place, so readers never see a partial
    tree. A published tree is never deleted, since a reader may be using it.

    Raises:
        ValueError: The snapshot has no archive, or it is not a usable zip.
    """
    if not snapshot.archive:
        raise ValueError(f"{label} has no archive")
    digest = hashlib.sha256(snapshot.archive.encode("utf-8")).hexdigest()[:32]
    root = _private_root()
    target = root / digest
    if (target / _COMPLETE_MARKER).is_file():
        return target
    tmp = Path(tempfile.mkdtemp(prefix=f".{digest}-", dir=root))
    try:
        if not extract_archive(
            snapshot.load_archive(),
            tmp,
            archive_format="zip",
            max_entries=SNAPSHOT_MAX_ENTRIES,
            max_uncompressed_bytes=SNAPSHOT_MAX_UNCOMPRESSED_BYTES,
        ):
            raise ValueError(f"{label} is not a readable zip archive")
        (tmp / _COMPLETE_MARKER).touch()
        try:
            os.rename(tmp, target)
            return target
        except OSError:
            pass  # target exists
        if (target / _COMPLETE_MARKER).is_file():
            return target  # a concurrent extraction won the rename
        # A damaged tree (marker gone) may still be in use: leave it, and
        # serve this extraction from its own sibling.
        alt = root / f"{digest}-{uuid.uuid4().hex[:8]}"
        os.rename(tmp, alt)
        logger.warning("%s: %s is damaged; unpacked to %s", label, target, alt)
        return alt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _get_row(space_name: str) -> StoredSpace:
    # Local import: singleton storage pulls in the storage backends.
    from gbserver.storage.singleton_storage import get_admin_storage

    row = get_admin_storage().space_storage.get_by_name(space_name)
    if row is None:
        raise ValueError(f"db space URI names unknown space {space_name!r}")
    return row


def _get_snapshot(row: StoredSpace, slot: Slot) -> RepoSnapshot:
    snapshot = row.space if slot == SPACE_SLOT else row.assets
    if snapshot is None or not snapshot.archive:
        raise ValueError(f"space {row.name!r} has no {slot} snapshot")
    return snapshot


# Unpacked dir -> the db URI it came from, so a path under the cache can be
# turned back into a URI that means the same on another host (portable_uri).
_origins: Dict[Path, str] = {}


def portable_uri(uri: str) -> str:
    """Map a ``file://`` URI inside an unpacked snapshot back to its db:// URI.

    The unpacked path exists only on this host and changes with each snapshot,
    so anything persisted or handed to another process uses this instead.
    Other URIs are returned unchanged.
    """
    if not uri.startswith("file://"):
        return uri
    path = Path(uri[len("file://") :])
    for root, db_uri in list(_origins.items()):
        if path == root or root in path.parents:
            rel = path.relative_to(root).as_posix()
            return db_uri + (f"#subdirectory={rel}" if rel != "." else "")
    return uri


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
    snapshot = _get_snapshot(row, ref.slot)
    path = unpack_snapshot(snapshot, f"space {row.name!r} {ref.slot} snapshot")
    _origins[path.resolve()] = make_db_uri(row.name, ref.slot)
    if ref.subdirectory:
        path = path / ref.subdirectory
    file_uri = path.resolve().as_uri()
    identity_uri = (
        to_config_uri(snapshot.source_uri, snapshot.source_ref)
        if snapshot.source_uri
        else file_uri
    )
    return ResolvedDbUri(file_uri=file_uri, identity_uri=identity_uri)


def _source_row(stored_space: StoredSpace) -> StoredSpace:
    """The row holding a db-backed space's snapshots (itself, or its alias target)."""
    name = parse_db_uri(stored_space.get_space_uri()).space_name
    return stored_space if name == stored_space.name else _get_row(name)


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
        )
    uri = get_space_config_uri(stored_space, token=token)
    logger.info("for the space name %s using the URI: %s", stored_space.name, uri)
    return Space(uri=uri, username=username, force_fetch=force_fetch)
