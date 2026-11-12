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

"""Store a snapshot of a space repo and its assets repo in gb_spaces.

The space is then served from the database (``git_repo_uri:
db://spaces/<name>``) with no git access at build or request time. See
gbserver.spaces.db_space.
"""

import io
import os
import subprocess
import tempfile
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import click
import yaml

from gbserver.spaces.db_space import (
    ASSETS_SLOT,
    SNAPSHOT_MAX_ENTRIES,
    SNAPSHOT_MAX_UNCOMPRESSED_BYTES,
    get_identity_uri,
    is_db_uri,
    make_db_uri,
    parse_db_uri,
)
from gbserver.spaces.local_secrets_isolation import check_local_secrets_collision
from gbserver.spaces.repo_identity import normalize_repo_uri, same_repo, to_config_uri
from gbserver.storage import singleton_storage
from gbserver.storage.stored_space import RepoSnapshot, StoredSpace
from gbserver.types.constants import (
    GBSERVER_GITHUB_TOKEN,
    SPACE_REPO_CONFIG_BRANCH_NAME,
)
from gbserver.types.context import CliEnvironment, pass_environment
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import get_utc_time

logger = get_logger(__name__)

SPACE_YAML = "space.yaml"


@dataclass
class _Source:
    """Where to snapshot a repo from."""

    location: str  # local checkout dir, or a clonable URL
    ref: str
    source_uri: str  # normalized https://host/owner/repo (or file://)


def _git(args: List[str], cwd: Optional[Path] = None) -> bytes:
    """Run git, with the GitHub token (if any) as an auth header, not in argv."""
    env = dict(os.environ)
    if GBSERVER_GITHUB_TOKEN:
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="http.extraHeader",
            GIT_CONFIG_VALUE_0=f"Authorization: token {GBSERVER_GITHUB_TOKEN}",
        )
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=False, capture_output=True
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _split_ref(uri: str) -> Tuple[str, Optional[str]]:
    """Split a pip-style ``...repo.git@ref[#frag]`` into (uri-without-ref, ref)."""
    if "://" not in uri:
        return uri, None  # scp form or path: '@' is userinfo, not a ref
    parsed = urllib.parse.urlparse(uri)
    path, sep, ref = parsed.path.rpartition("@")
    if not sep or "/" in ref:
        return parsed._replace(fragment="").geturl(), None
    return parsed._replace(path=path, fragment="").geturl(), ref


def _resolve_source(
    repo: str, ref: Optional[str], source_uri: Optional[str], default_ref: str
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
        )
    url, url_ref = _split_ref(repo)
    return _Source(
        location=url,
        ref=ref or url_ref or default_ref,
        source_uri=normalize_repo_uri(source_uri or url),
    )


def _clone_url(url: str) -> str:
    """git understands ssh:// and https://; map git+ssh:// onto ssh://git@."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "git+ssh":
        netloc = parsed.netloc if "@" in parsed.netloc else f"git@{parsed.netloc}"
        return parsed._replace(scheme="ssh", netloc=netloc).geturl()
    return url


def _snapshot(source: _Source) -> Tuple[bytes, str]:
    """Return (zip of the tree at source.ref, commit sha). No history included.

    ``git archive`` emits only the committed tree: no .git, no untracked files.
    """
    if Path(source.location).is_dir():
        repo_dir = Path(source.location)
        for candidate in (source.ref, f"origin/{source.ref}"):
            try:
                commit = _git(
                    ["rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"],
                    cwd=repo_dir,
                )
                break
            except click.ClickException:
                continue
        else:
            raise click.ClickException(
                f"ref {source.ref!r} not found in {repo_dir} (nor origin/{source.ref})"
            )
        sha = commit.decode().strip()
        return _git(["archive", "--format=zip", sha], cwd=repo_dir), sha
    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "repo"
        _git(
            [
                "clone",
                "--quiet",
                "--depth=1",
                "--single-branch",
                f"--branch={source.ref}",
                _clone_url(source.location),
                str(clone_dir),
            ]
        )
        sha = _git(["rev-parse", "HEAD"], cwd=clone_dir).decode().strip()
        return _git(["archive", "--format=zip", sha], cwd=clone_dir), sha


def _check_size(data: bytes, label: str) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
    total = sum(i.file_size for i in infos)
    if len(infos) > SNAPSHOT_MAX_ENTRIES or total > SNAPSHOT_MAX_UNCOMPRESSED_BYTES:
        raise click.ClickException(
            f"{label} snapshot too large: {len(infos)} entries, {total} bytes"
            f" (limits {SNAPSHOT_MAX_ENTRIES}, {SNAPSHOT_MAX_UNCOMPRESSED_BYTES})"
        )


def _read_space_yaml(space_zip: bytes) -> str:
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
    data = yaml.safe_load(space_yaml) or {}
    return [str(b) for b in (data.get("base_uris") or [])]


def _is_remote(uri: str) -> bool:
    """Whether a base_uris entry is fetched over the network (not file:/db:)."""
    if is_db_uri(uri):
        return False
    if "://" not in uri:
        head, sep, tail = uri.partition(":")  # scp form: [user@]host:owner/repo
        return bool(sep and "/" not in head and tail and not tail.startswith("/"))
    return urllib.parse.urlparse(uri).scheme not in ("", "file")


def _remote_git_base_uris(base_uris: List[str]) -> List[str]:
    return [b for b in base_uris if _is_remote(b)]


def _rewrite_assets_base_uris(
    space_yaml: str, assets_source_uri: str, db_assets_uri: str
) -> Tuple[str, Dict[str, str]]:
    """Point every base_uris entry for the assets repo at the db snapshot.

    Text substitution of the exact entries, so comments and layout survive.
    A ``#subdirectory=`` fragment is carried over.
    """
    rewrites: Dict[str, str] = {}
    for base_uri in _base_uris(space_yaml):
        if is_db_uri(base_uri) or not same_repo(base_uri, assets_source_uri):
            continue
        fragment = urllib.parse.urlparse(base_uri).fragment if "://" in base_uri else ""
        rewrites[base_uri] = db_assets_uri + (f"#{fragment}" if fragment else "")
    text = space_yaml
    for old, new in rewrites.items():
        if text.count(old) != 1:
            raise click.ClickException(
                f"cannot rewrite base_uri {old!r}: it occurs {text.count(old)}"
                f" times in {SPACE_YAML}"
            )
        text = text.replace(old, new)
    after = _base_uris(text)
    for new in rewrites.values():
        if new not in after:
            raise click.ClickException(f"rewritten {SPACE_YAML} lost base_uri {new!r}")
    return text, rewrites


def _secret_group_names(space_yaml: str, uri: str) -> Optional[frozenset]:
    """Names of the IBM secret groups matching ``uri``; None if not ibmcloud."""
    data = yaml.safe_load(space_yaml) or {}
    secret_manager = data.get("secret_manager") or {}
    if secret_manager.get("type") != "ibmcloud":
        return None
    from gbserver.spacesecretmanager.ibmcloudspacesecretmanager import (
        IbmcloudSpaceSecretManager,
    )

    manager = IbmcloudSpaceSecretManager(
        uri=uri, **(secret_manager.get("config") or {})
    )
    return frozenset(manager.get_secret_groups().values())


def _old_identity_uris(row: StoredSpace) -> Optional[Tuple[str, str]]:
    """(bare, config-form) URIs secrets matched on before this snapshot."""
    if is_db_uri(row.git_repo_uri):
        if row.space is None or not row.space.source_uri:
            return None
        return row.space.source_uri, to_config_uri(
            row.space.source_uri, row.space.source_ref
        )
    uri = row.git_repo_uri
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme == "file" or not parsed.scheme and "://" not in uri:
        return None
    config_uri = (
        uri
        if parsed.scheme == "git+ssh" and "@" in parsed.path
        else to_config_uri(uri, SPACE_REPO_CONFIG_BRANCH_NAME)
    )
    return uri, config_uri


def _verify_secret_groups(
    row: StoredSpace, space_yaml: str, source_uri: str, ref: str
) -> None:
    """Refuse a snapshot whose identity would resolve different secret groups."""
    old = _old_identity_uris(row)
    if old is None:
        return
    new = (source_uri, to_config_uri(source_uri, ref))
    for old_uri, new_uri in zip(old, new):
        if old_uri == new_uri:
            continue  # same string, so the same regex matches
        try:
            before = _secret_group_names(space_yaml, old_uri)
        except ImportError as e:
            raise click.ClickException(
                f"cannot compare secret groups ({e}); install the IBM Cloud SDK"
                " or pass --skip-secret-group-check"
            ) from e
        if before is None:
            return
        after = _secret_group_names(space_yaml, new_uri)
        if before != after:
            raise click.ClickException(
                f"secret groups differ: {old_uri} -> {sorted(before)},"
                f" {new_uri} -> {sorted(after or [])}."
                " Use --skip-secret-group-check to override."
            )
        click.echo(f"secret groups for {new_uri}: {sorted(after or [])}")


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
    "--assets-repo",
    default=None,
    help="Assets repo URL or local checkout. Default: the git base_uris entry"
    f" in the snapshot's {SPACE_YAML}.",
)
@click.option(
    "--assets-ref",
    default=None,
    help="Branch to snapshot (default: the base_uris entry's @ref).",
)
@click.option("--assets-source-uri", default=None, help="As --space-source-uri.")
@click.option("--no-assets", is_flag=True, help="Snapshot only the space repo.")
@click.option(
    "--switch/--no-switch",
    default=True,
    show_default=True,
    help="Point git_repo_uri at the snapshot (db://spaces/<name>). With"
    " --no-switch the snapshot is stored but the row keeps using git.",
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
    "--skip-secret-group-check",
    is_flag=True,
    help="Do not compare IBM secret groups before and after the switch.",
)
@click.option("--dry-run", is_flag=True, help="Report what would be stored.")
@pass_environment
def cli(
    ctx: CliEnvironment,
    space_name: str,
    space_repo: Optional[str],
    space_ref: Optional[str],
    space_source_uri: Optional[str],
    assets_repo: Optional[str],
    assets_ref: Optional[str],
    assets_source_uri: Optional[str],
    no_assets: bool,
    switch: bool,
    create: bool,
    lakehouse_namespace: Optional[str],
    skip_secret_group_check: bool,
    dry_run: bool,
):
    """Snapshot a space repo (and its assets repo) into gb_spaces.

    Only the tree at the branch head is stored, no git history. The stored
    space.yaml's base_uris entry for the assets repo is rewritten to
    db://spaces/<name>/assets.
    """
    storage = singleton_storage.get_admin_storage().space_storage
    row = storage.get_by_name(space_name)
    if row is None:
        if not create:
            raise click.ClickException(
                f"space {space_name!r} not found (use --create to add it)"
            )
        if space_repo is None:
            raise click.ClickException("--space-repo is required with --create")
        row = StoredSpace(
            name=space_name,
            git_repo_uri=make_db_uri(space_name),
            lakehouse_namespace=lakehouse_namespace,
        )
        is_new = True
    else:
        is_new = False
    if space_repo is None:
        space_repo = get_identity_uri(row)
        if is_db_uri(space_repo):
            raise click.ClickException(
                f"space {space_name!r} has no recorded source; pass --space-repo"
            )
    if is_db_uri(row.git_repo_uri) and parse_db_uri(row.git_repo_uri).space_name != (
        space_name
    ):
        raise click.ClickException(
            f"space {space_name!r} is an alias of {row.git_repo_uri}; snapshot"
            " that space instead"
        )

    # Space snapshot.
    space_src = _resolve_source(
        space_repo, space_ref, space_source_uri, SPACE_REPO_CONFIG_BRANCH_NAME
    )
    click.echo(f"space: {space_src.location} @ {space_src.ref}")
    space_zip, space_sha = _snapshot(space_src)
    space_yaml = _read_space_yaml(space_zip)

    # Assets snapshot.
    assets: Optional[RepoSnapshot] = None
    if not no_assets:
        remote = _remote_git_base_uris(_base_uris(space_yaml))
        if assets_repo is None:
            if len(remote) != 1:
                raise click.ClickException(
                    f"{SPACE_YAML} has {len(remote)} remote base_uris ({remote});"
                    " pass --assets-repo or --no-assets"
                )
            assets_repo = remote[0]
        assets_src = _resolve_source(
            assets_repo, assets_ref, assets_source_uri, SPACE_REPO_CONFIG_BRANCH_NAME
        )
        if not any(same_repo(b, assets_src.source_uri) for b in remote):
            raise click.ClickException(
                f"no {SPACE_YAML} base_uris entry refers to {assets_src.source_uri};"
                " pass --no-assets or the right --assets-repo"
            )
        if assets_ref is None:
            # Snapshot the branch space.yaml asks for, even from a local checkout.
            entry_refs = sorted(
                {
                    ref
                    for b in remote
                    if same_repo(b, assets_src.source_uri) and (ref := _split_ref(b)[1])
                }
            )
            if len(entry_refs) > 1:
                raise click.ClickException(
                    f"{SPACE_YAML} names {assets_src.source_uri} at several refs"
                    f" {entry_refs}; pass --assets-ref"
                )
            if entry_refs:
                assets_src.ref = entry_refs[0]
        click.echo(f"assets: {assets_src.location} @ {assets_src.ref}")
        assets_zip, assets_sha = _snapshot(assets_src)
        _check_size(assets_zip, "assets")
        space_yaml, rewrites = _rewrite_assets_base_uris(
            space_yaml, assets_src.source_uri, make_db_uri(space_name, ASSETS_SLOT)
        )
        for old, new in rewrites.items():
            click.echo(f"base_uris: {old} -> {new}")
        space_zip = _replace_space_yaml(space_zip, space_yaml)
        assets = RepoSnapshot(
            source_uri=assets_src.source_uri,
            source_ref=assets_src.ref,
            source_commit=assets_sha,
            snapshot_time=get_utc_time(),
        )
        assets.save_archive(assets_zip)
    for still_remote in _remote_git_base_uris(_base_uris(space_yaml)):
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

    if switch and not is_new and not skip_secret_group_check:
        _verify_secret_groups(row, space_yaml, space_src.source_uri, space_src.ref)

    updated = row.model_copy(update={"space": space, "assets": assets})
    if switch:
        updated.git_repo_uri = make_db_uri(space_name)

    # Same guard create-spaces applies: no two spaces may share local secrets.
    if is_db_uri(updated.git_repo_uri):
        check_local_secrets_collision(
            updated,
            [
                s
                for s in cast(List[StoredSpace], storage.get_by_uuid(None) or [])
                if s.name != space_name
            ],
        )

    click.echo(
        f"space {space_name}: git_repo_uri {row.git_repo_uri} -> {updated.git_repo_uri};"
        f" space {space_src.source_uri}@{space_sha[:12]} ({len(space_zip)} bytes)"
        + (
            f", assets {assets.source_uri}@{(assets.source_commit or '')[:12]}"
            f" ({len(assets.load_archive())} bytes)"
            if assets
            else ""
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
