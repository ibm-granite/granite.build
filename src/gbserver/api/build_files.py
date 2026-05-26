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

"""REST endpoints for inspecting an LSF build's remote-file outputs.

Three endpoints, registered on the shared builds_api:
  - GET /builds/{id}/files          — directory listing (optional substring filter)
  - GET /builds/{id}/files/search   — recursive content grep
  - GET /builds/{id}/file/download  — streamed file bytes (capped large)

Path resolution: ``path`` is relative to the build root
(``{workspace_remote_dir}/llm-build-{build_id}``).

Auth matches PUT /builds/{id}/update (owner or space/super admin).
Every user-supplied path passes through validate_subpath() and then
resolve_and_check_real_path() before it hits a shell or SFTP call — do
not bypass those helpers.
"""

import re
import shlex
from datetime import datetime
from pathlib import PurePosixPath
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import quote

from fastapi import HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from gbserver.api.build_files_paths import (
    authorize_build_access,
    lookup_build,
    resolve_and_check_real_path,
    validate_subpath,
)
from gbserver.api.builds import builds_api
from gbserver.api.lsf_tunnel import open_lsf_tunnel
from gbserver.environment.lsf_paths import build_remote_root_dir
from gbserver.storage.singleton_storage import SingletonAdminStorage, get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import (
    BUILD_FILES_DOWNLOAD_MAX_BYTES,
    BUILD_FILES_GREP_LINE_MAX_BYTES,
    BUILD_FILES_GREP_MAX_CONTEXT,
    BUILD_FILES_GREP_MAX_HITS,
    BUILD_FILES_LIST_MAX_ENTRIES,
    BUILD_FILES_PEEK_MAX_BYTES,
    BUILD_FILES_PEEK_MAX_LINES,
    BUILD_FILES_STAT_BATCH_MAX,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------- models


class GrepHit(BaseModel):
    path: str
    """Path of the matching file, relative to the build root."""
    line: int
    text: str
    is_match: bool = True
    """False for context lines emitted when ``before``/``after`` > 0."""
    size: Optional[int] = None
    """File size in bytes; populated when ``stat=true``."""
    mtime: Optional[int] = None
    """File mtime as Unix epoch seconds; populated when ``stat=true``."""


class FileEntry(BaseModel):
    path: str
    """Path of the entry, relative to the build root."""
    type: str
    """One of ``file``, ``dir``, ``symlink``, ``other``."""
    size: int
    """File size in bytes; 0 for directories."""
    mtime: int
    """Mtime as Unix epoch seconds."""


# --------------------------------------------------------------------- helpers


def _pick_environment_uri(build: StoredBuild) -> str:
    """Return the most recent target run's environment_uri for this build.

    Build-root listings still need an SSH tunnel, which is keyed by
    environment_uri. We don't persist environment on the build, so we
    borrow it from any of its target runs.
    """
    storage: SingletonAdminStorage = get_admin_storage()
    target_runs = cast(
        list[StoredTargetRun],
        storage.target_storage.get_by_where({"build_id": build.uuid}),
    )
    if not target_runs:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"build {build.uuid!r} has no target runs to ssh through",
        )
    target = max(target_runs, key=lambda t: t.started_at or datetime.min)
    return target.environment_uri


def _reject_pattern_control_chars(pattern: str) -> None:
    """Reject patterns with chars that break shell quoting or grep -F semantics.

    `shlex.quote` makes the pattern safe for the shell, and `grep -F`
    treats it as a literal — but newlines split into separate patterns
    and NULs terminate strings in C-level libraries, so we still 400 on
    those.
    """
    if any(c in pattern for c in ("\x00", "\n", "\r")):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "pattern contains illegal characters",
        )


def _no_match_or_500(rc: int, stdout: str, stderr: str, what: str) -> List[str]:
    """Translate a `... | grep ... | head` exit code into hits or HTTPException.

    grep exits 1 when there are no matches — that's not an error here,
    return []. rc>=2 is a real failure (or a stage before grep failed
    under pipefail).
    """
    if rc == 0:
        # Split on '\n' only — not str.splitlines(), which also breaks on
        # embedded '\r'. Source lines from tqdm progress bars and other
        # \r-heavy output must stay intact so _GREP_LINE_RE parses the
        # whole `path:line:text` triple as one record (not as fragments
        # whose `:NNNN:` substrings get re-parsed as fake hits).
        return [ln for ln in (stdout or "").split("\n") if ln]
    if rc == 1 and not stdout:
        return []
    err = (stderr or "").lower()
    if "no such file" in err or "cannot access" in err:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
    raise HTTPException(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        f"{what} failed: {stderr.strip() or 'unknown error'}",
    )


# ---------------------------------------------------------------- /files/search


# grep -H output: <path><sep><lineno><sep><text>, where sep is ':' for
# match lines and '-' for context lines (when -A/-B is set). Parsing is
# split into two strategies depending on the separator:
#
#   * Match lines (':'): we anchor on the FIRST ':<digits>:' that
#     appears after the known absolute prefix (build_root). This stops
#     a long matched source line whose content contains its own
#     ":NNNN:" substring (e.g. pickled config dumps in training logs)
#     from being re-parsed as a fake hit, with the inline content
#     landing in `path` and the suffix in `text`. Filenames containing
#     ':' are pathological and not handled.
#   * Context lines ('-'): we anchor on the RIGHTMOST '-<digits>-' (the
#     greedy `.+` strategy) because filenames frequently contain
#     "-<digits>-" segments (UUIDs, run IDs) and the path needs to
#     extend across them. Greedy matching backtracks to the unambiguous
#     trailing '-<digits>-' boundary.
_GREP_MATCH_RE = re.compile(r"^(?P<rel>.*?):(?P<lineno>\d+):(?P<text>.*)$")
_GREP_CONTEXT_RE = re.compile(r"^(?P<path>.+)-(?P<lineno>\d+)-(?P<text>.*)$")


def _parse_grep_line(
    ln: str, build_root: PurePosixPath
) -> Optional[Tuple[str, int, str, bool]]:
    """Parse one line of ``grep -H -n`` output into (rel_path, lineno, text, is_match).

    With ``-A``/``-B`` set, grep emits ``<path>-<lineno>-<text>`` for
    context lines and ``<path>:<lineno>:<text>`` for match lines.

    Match lines are anchored against ``build_root`` (the known absolute
    prefix) and split on the first ':<digits>:' after it, so embedded
    ":NNNN:" substrings in the matched content can't masquerade as the
    record boundary. Context lines fall back to greedy/rightmost split,
    which preserves filenames containing "-<digits>-" segments.
    """
    root_str = str(build_root)
    # Match line: ':' separator. Anchor against the build_root prefix.
    if ln.startswith(root_str):
        suffix = ln[len(root_str):]
        m = _GREP_MATCH_RE.match(suffix)
        if m is not None:
            try:
                lineno = int(m.group("lineno"))
            except ValueError:
                return None
            rel = m.group("rel").lstrip("/") or "."
            return rel, lineno, m.group("text"), True

    # Context line: '-' separator. Filenames legitimately contain
    # "-<digits>-" segments, so use the rightmost-split fallback.
    m = _GREP_CONTEXT_RE.match(ln)
    if m is None:
        return None
    abs_path = m.group("path")
    try:
        lineno = int(m.group("lineno"))
    except ValueError:
        return None
    try:
        rel = str(PurePosixPath(abs_path).relative_to(build_root))
    except ValueError:
        return None
    return rel, lineno, m.group("text"), False


async def _remote_stat_batch(
    tunnel, paths: List[PurePosixPath]
) -> Dict[str, Tuple[int, int]]:
    """Return ``{abs_path: (size, mtime_epoch)}`` for the given paths.

    Single batched ``stat`` call. Paths missing from the result (e.g.
    deleted between grep and stat) are simply omitted from the dict —
    callers leave size/mtime as None for those.
    """
    if not paths:
        return {}
    quoted = " ".join(shlex.quote(str(p)) for p in paths)
    cmd = f"stat -c '%n\t%s\t%Y' -- {quoted}"
    rc, stdout, _stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    out: Dict[str, Tuple[int, int]] = {}
    if rc not in (0, 1):  # 1 just means some paths were missing
        return out
    for ln in (stdout or "").splitlines():
        parts = ln.split("\t")
        if len(parts) < 3:
            continue
        try:
            out[parts[0]] = (int(parts[1]), int(parts[2]))
        except ValueError:
            continue
    return out


@builds_api.get(
    "/{build_id}/files/search",
    response_model=List[GrepHit],
)
async def search_files(
    request: Request,
    build_id: str,
    pattern: str = Query(..., min_length=1, max_length=512),
    path: str = Query(".", min_length=1),
    ignore_case: bool = Query(False),
    regex: bool = Query(False),
    before: int = Query(0, ge=0, le=BUILD_FILES_GREP_MAX_CONTEXT),
    after: int = Query(0, ge=0, le=BUILD_FILES_GREP_MAX_CONTEXT),
    stat: bool = Query(False),
) -> List[GrepHit]:
    """Recursively grep for ``pattern`` under ``path``.

    Defaults to literal substring match (``grep -F``). Set ``regex=true``
    to enable extended regex (``grep -E``). ``before``/``after`` add
    context lines — context entries are returned with ``is_match=false``.
    Skips binary files (``-I``). Caps total hits (matches + context) at
    ``BUILD_FILES_GREP_MAX_HITS`` and truncates each line's text to
    ``BUILD_FILES_GREP_LINE_MAX_BYTES`` bytes. With ``stat=true`` each
    hit's owning file is annotated with ``size`` and ``mtime``. Returns
    ``[]`` when the pattern doesn't match anything.
    """
    _reject_pattern_control_chars(pattern)

    build = lookup_build(build_id)
    authorize_build_access(request, build)
    environment_uri = _pick_environment_uri(build)

    async with open_lsf_tunnel(build.space_name, environment_uri) as (
        tunnel,
        cfg,
    ):
        build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
        candidate = validate_subpath(build_root, path)
        real = await resolve_and_check_real_path(tunnel, build_root, candidate)

        logger.info(
            "[build-files] search build=%s ignore_case=%s regex=%s "
            "before=%s after=%s stat=%s",
            build_id,
            ignore_case,
            regex,
            before,
            after,
            stat,
        )

        flags = "-r -n -I -H"
        flags += " -E" if regex else " -F"
        if ignore_case:
            flags += " -i"
        if before:
            flags += f" -B {before}"
        if after:
            flags += f" -A {after}"
        # `-H` forces filenames in the output for both single-file and
        # recursive targets, so we don't need a trailing slash hack on
        # the search root. (A trailing '/' on a regular-file target made
        # grep fail with "Not a directory" and surfaced as a 500.)
        # pipefail propagates grep's rc past head.
        cmd = (
            f"set -o pipefail; "
            f"grep {flags} -- {shlex.quote(pattern)} {shlex.quote(str(real))} "
            f"| head -n {BUILD_FILES_GREP_MAX_HITS}"
        )

        rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
        lines = _no_match_or_500(rc, stdout or "", stderr or "", "search")

        hits: List[GrepHit] = []
        for ln in lines:
            if ln == "--":
                # grep emits this between non-adjacent context groups.
                continue
            parsed = _parse_grep_line(ln, build_root)
            if parsed is None:
                continue
            rel, lineno, text, is_match = parsed
            # Replace embedded '\r' (e.g. from tqdm progress bars) with a
            # space so the rendered text is readable, then apply the byte
            # cap on the cleaned-up string.
            if "\r" in text:
                text = text.replace("\r", " ")
            if len(text) > BUILD_FILES_GREP_LINE_MAX_BYTES:
                text = text[:BUILD_FILES_GREP_LINE_MAX_BYTES]
            hits.append(
                GrepHit(path=rel, line=lineno, text=text, is_match=is_match)
            )

        if stat and hits:
            distinct_rels: List[str] = []
            seen: set[str] = set()
            for h in hits:
                if h.path not in seen:
                    seen.add(h.path)
                    distinct_rels.append(h.path)
            # Cap stat batch — surplus files keep size/mtime as None.
            batched = distinct_rels[:BUILD_FILES_STAT_BATCH_MAX]
            abs_paths = [build_root / r for r in batched]
            stats = await _remote_stat_batch(tunnel, abs_paths)
            # Map back from abs path string to (size, mtime).
            rel_to_meta: Dict[str, Tuple[int, int]] = {}
            for rel, abs_p in zip(batched, abs_paths):
                meta = stats.get(str(abs_p))
                if meta is not None:
                    rel_to_meta[rel] = meta
            for h in hits:
                meta = rel_to_meta.get(h.path)
                if meta is not None:
                    h.size, h.mtime = meta
        return hits


# ---------------------------------------------------------------------- /files


_FIND_TYPE_MAP = {"f": "file", "d": "dir", "l": "symlink"}


def _parse_find_printf(line: str, real: PurePosixPath) -> Optional[FileEntry]:
    """Parse one ``find -printf '%P\\t%y\\t%s\\t%T@\\n'`` line into a FileEntry.

    ``%P`` is the path with the search root stripped, so we re-anchor it
    at ``real``'s relpath under the build root. ``%T@`` is float epoch.
    """
    parts = line.split("\t")
    if len(parts) < 4:
        return None
    p_rel_to_real, type_char, size_s, mtime_s = parts[0], parts[1], parts[2], parts[3]
    if not p_rel_to_real:
        return None
    try:
        size = int(size_s)
        # `%T@` is e.g. "1700000000.1234567890"; truncate to whole seconds.
        mtime = int(float(mtime_s))
    except ValueError:
        return None
    type_ = _FIND_TYPE_MAP.get(type_char, "other")
    return FileEntry(
        path=str(real / p_rel_to_real),
        type=type_,
        size=size,
        mtime=mtime,
    )


@builds_api.get(
    "/{build_id}/files",
    response_model=Union[List[str], List[FileEntry]],
)
async def list_files(
    request: Request,
    build_id: str,
    path: str = Query(".", min_length=1),
    recursive: bool = Query(False),
    pattern: Optional[str] = Query(None, min_length=1, max_length=256),
    regex: bool = Query(False),
    stat: bool = Query(False),
) -> Union[List[str], List[FileEntry]]:
    """List entries under the resolved path, returning paths relative to
    the build root, sorted lexicographically. Includes both files and
    directories (no trailing slash) and dotfiles.

    With ``recursive=true`` the subtree is walked (capped at
    ``BUILD_FILES_LIST_MAX_ENTRIES`` entries). Symlinks are listed as
    their own entries; their targets are not followed.

    With ``pattern`` set, the listing is filtered server-side by literal
    substring (``grep -F``), or extended regex when ``regex=true``
    (``grep -E``). Returns ``[]`` when the pattern doesn't match
    anything.

    With ``stat=true`` the response is a list of ``FileEntry`` objects
    (path, type, size, mtime) instead of bare path strings — this lets
    callers prioritize by recency/size and skip directories without a
    second round-trip. The pattern filter is applied to the path
    component in this mode.
    """
    if pattern is not None:
        _reject_pattern_control_chars(pattern)

    build = lookup_build(build_id)
    authorize_build_access(request, build)
    environment_uri = _pick_environment_uri(build)

    async with open_lsf_tunnel(build.space_name, environment_uri) as (
        tunnel,
        cfg,
    ):
        build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
        candidate = validate_subpath(build_root, path)
        real = await resolve_and_check_real_path(tunnel, build_root, candidate)

        logger.info(
            "[build-files] list build=%s recursive=%s filtered=%s regex=%s stat=%s",
            build_id,
            recursive,
            pattern is not None,
            regex,
            stat,
        )
        logger.debug("[build-files] list real=%s build_root=%s", real, build_root)

        if stat:
            return await _list_files_stat(
                tunnel, build_root, real, recursive, pattern, regex
            )

        grep_flag = "-E" if regex else "-F"
        quoted = shlex.quote(str(real))
        if recursive:
            base = f"find {quoted} -mindepth 1"
        else:
            base = f"ls -1A -- {quoted}"

        # pipefail in both branches so a failing producer (e.g. ls
        # permission denied) propagates past grep/head instead of being
        # masked by their success.
        if pattern is not None:
            cmd = (
                f"set -o pipefail; {base} "
                f"| grep {grep_flag} -- {shlex.quote(pattern)} "
                f"| head -n {BUILD_FILES_LIST_MAX_ENTRIES}"
            )
        elif recursive:
            cmd = f"set -o pipefail; {base} | head -n {BUILD_FILES_LIST_MAX_ENTRIES}"
        else:
            cmd = base

        rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)

        if pattern is not None:
            lines = _no_match_or_500(rc, stdout or "", stderr or "", "listing")
        else:
            if rc != 0:
                err = (stderr or "").lower()
                if "no such file" in err or "cannot access" in err:
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    f"listing failed: {stderr.strip() or 'unknown error'}",
                )
            lines = [ln for ln in (stdout or "").splitlines() if ln]

        if recursive:
            # find emits absolute paths.
            rels = [str(PurePosixPath(ln).relative_to(build_root)) for ln in lines]
        else:
            # ls -1A emits bare names rooted at `real`.
            rels = [str((real / name).relative_to(build_root)) for name in lines]
        rels.sort()
        return rels


async def _list_files_stat(
    tunnel,
    build_root: PurePosixPath,
    real: PurePosixPath,
    recursive: bool,
    pattern: Optional[str],
    regex: bool,
) -> List[FileEntry]:
    """``stat=true`` branch of list_files: ``find -printf`` → FileEntry list.

    A single ``find`` call gathers path/type/size/mtime in one shot, so we
    don't pay a per-entry stat round-trip. The pattern filter is applied
    in Python on the path component to keep the shell pipeline simple.
    """
    quoted = shlex.quote(str(real))
    maxdepth = "" if recursive else "-maxdepth 1"
    # %P is path-relative-to-search-root (empty for the root itself, which
    # -mindepth 1 already excludes), %y is type, %s is size, %T@ is mtime.
    printf_fmt = r"%P\t%y\t%s\t%T@\n"
    cmd = (
        f"set -o pipefail; "
        f"find {quoted} -mindepth 1 {maxdepth} -printf {shlex.quote(printf_fmt)} "
        f"| head -n {BUILD_FILES_LIST_MAX_ENTRIES}"
    )
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    if rc != 0:
        err = (stderr or "").lower()
        if "no such file" in err or "cannot access" in err:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"listing failed: {stderr.strip() or 'unknown error'}",
        )

    entries: List[FileEntry] = []
    for ln in (stdout or "").splitlines():
        if not ln:
            continue
        entry = _parse_find_printf(ln, real)
        if entry is None:
            continue
        # Re-anchor entry.path to be relative to build_root (currently
        # absolute because we passed `real` which is absolute).
        try:
            entry.path = str(PurePosixPath(entry.path).relative_to(build_root))
        except ValueError:
            continue
        entries.append(entry)

    if pattern is not None:
        if regex:
            try:
                rx = re.compile(pattern)
            except re.error as e:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"invalid regex: {e}",
                ) from e
            entries = [e for e in entries if rx.search(e.path)]
        else:
            entries = [e for e in entries if pattern in e.path]

    entries.sort(key=lambda e: e.path)
    return entries


# ------------------------------------------------------------- /file/download


async def _stream_sftp_file(tunnel, remote_path: str) -> AsyncIterator[bytes]:
    """Yield chunks of a remote file via SFTP, closing the client on exit."""
    chunk_size = 256 * 1024
    sftp = await tunnel.start_sftp()
    try:
        async with sftp.open(remote_path, "rb", encoding=None) as fh:
            while True:
                chunk = await fh.read(chunk_size)
                if not chunk:
                    return
                yield chunk
    finally:
        sftp.exit()


def _content_disposition(filename: str) -> str:
    """RFC 5987 Content-Disposition value with an ASCII fallback + UTF-8 form."""
    ascii_fallback = (
        filename.encode("ascii", "replace").decode("ascii").replace('"', "_")
    ) or "download.bin"
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(filename, safe='')}"
    )


def _validate_peek_args(
    head: Optional[int], tail: Optional[int], range_: Optional[str]
) -> Optional[Tuple[str, Tuple[int, ...]]]:
    """Return ``(mode, args)`` if exactly one peek arg is set, else None.

    Modes: ``("head", (n,))``, ``("tail", (n,))``, ``("range", (start, end))``.
    Raises 400 if more than one is set or if ``range`` is malformed.
    """
    set_count = sum(x is not None for x in (head, tail, range_))
    if set_count == 0:
        return None
    if set_count > 1:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "head, tail, and range are mutually exclusive",
        )
    if head is not None:
        return "head", (head,)
    if tail is not None:
        return "tail", (tail,)
    assert range_ is not None
    try:
        start_s, end_s = range_.split("-", 1)
        start, end = int(start_s), int(end_s)
    except ValueError as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "range must be of the form start-end (1-indexed line numbers)",
        ) from e
    if start < 1 or end < start:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "range requires 1 <= start <= end",
        )
    return "range", (start, end)


async def _peek_text(
    tunnel,
    real: PurePosixPath,
    mode: str,
    args: Tuple[int, ...],
) -> str:
    """Run head/tail/sed against ``real`` and return decoded text.

    The output bytes are capped at ``BUILD_FILES_PEEK_MAX_BYTES`` via a
    trailing ``head -c``. Bytes are decoded as UTF-8 with replacement so
    a binary chunk doesn't 500.
    """
    quoted = shlex.quote(str(real))
    if mode == "head":
        producer = f"head -n {args[0]} -- {quoted}"
    elif mode == "tail":
        producer = f"tail -n {args[0]} -- {quoted}"
    else:
        # `sed -n 'A,Bp; Bq'` exits as soon as line B is printed; cheaper
        # than scanning the rest of a huge file.
        start, end = args
        producer = f"sed -n {shlex.quote(f'{start},{end}p;{end}q')} -- {quoted}"
    cmd = (
        f"set -o pipefail; {producer} | head -c {BUILD_FILES_PEEK_MAX_BYTES}"
    )
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    # head -c truncating its input causes the producer to die with
    # SIGPIPE → rc=141 under pipefail; that's success here.
    if rc not in (0, 141):
        err = (stderr or "").lower()
        if "no such file" in err or "cannot access" in err:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"peek failed: {stderr.strip() or 'unknown error'}",
        )
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")
    return stdout or ""


@builds_api.get("/{build_id}/file/download")
async def download_file(
    request: Request,
    build_id: str,
    path: str = Query(..., min_length=1),
    head: Optional[int] = Query(None, ge=1, le=BUILD_FILES_PEEK_MAX_LINES),
    tail: Optional[int] = Query(None, ge=1, le=BUILD_FILES_PEEK_MAX_LINES),
    range_: Optional[str] = Query(
        None, alias="range", pattern=r"^\d+-\d+$"
    ),
) -> Response:
    """Download or peek at a remote file.

    Default (no peek param): streams the file as
    ``application/octet-stream``. Rejects directories with 400 and files
    larger than ``BUILD_FILES_DOWNLOAD_MAX_BYTES`` with 413 before any
    bytes are streamed.

    Peek mode (set exactly one of ``head=N``, ``tail=N``, ``range=A-B``):
    returns ``text/plain; charset=utf-8`` with the requested slice of
    the file. Output bytes are capped at ``BUILD_FILES_PEEK_MAX_BYTES``
    (~256 KiB by default). The file size cap does **not** apply in peek
    mode — tailing the last 200 lines of a 50 GiB log is the use case.
    """
    peek = _validate_peek_args(head, tail, range_)

    build = lookup_build(build_id)
    authorize_build_access(request, build)
    environment_uri = _pick_environment_uri(build)

    if peek is not None:
        # Peek mode: bounded output, no streaming.
        async with open_lsf_tunnel(build.space_name, environment_uri) as (
            tunnel,
            cfg,
        ):
            build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
            candidate = validate_subpath(build_root, path)
            real = await resolve_and_check_real_path(tunnel, build_root, candidate)

            # Reject directories explicitly — head/tail on a directory
            # would error from the shell, but the message is clearer here.
            _size, is_dir = await _remote_stat(tunnel, real)
            if is_dir:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "peek endpoint requires a file, not a directory",
                )

            mode, args = peek
            logger.info(
                "[build-files] peek build=%s mode=%s args=%s",
                build_id,
                mode,
                args,
            )
            text = await _peek_text(tunnel, real, mode, args)
            return Response(
                content=text,
                media_type="text/plain; charset=utf-8",
            )

    # Tunnel lifecycle must outlive the streaming response body, so we open
    # it manually here and close it inside the body's finally on success or
    # in the except below if anything fails before we hand off to streaming.
    ctx = open_lsf_tunnel(build.space_name, environment_uri)
    tunnel, cfg = await ctx.__aenter__()
    try:
        build_root = build_remote_root_dir(cfg.workspace_remote_dir, build.uuid)
        candidate = validate_subpath(build_root, path)
        real = await resolve_and_check_real_path(tunnel, build_root, candidate)

        size, is_dir = await _remote_stat(tunnel, real)
        if is_dir:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "download endpoint requires a file, not a directory",
            )
        if size > BUILD_FILES_DOWNLOAD_MAX_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                {
                    "message": "file exceeds download cap",
                    "size": size,
                    "cap": BUILD_FILES_DOWNLOAD_MAX_BYTES,
                },
            )

        logger.info(
            "[build-files] download build=%s size=%d",
            build_id,
            size,
        )

        filename = real.name or "download.bin"

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _stream_sftp_file(tunnel, str(real)):
                    yield chunk
            finally:
                await ctx.__aexit__(None, None, None)

        return StreamingResponse(
            body(),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": _content_disposition(filename),
                "Content-Length": str(size),
            },
        )
    except BaseException:
        # Pre-stream failure: close the tunnel now.
        await ctx.__aexit__(None, None, None)
        raise


async def _remote_stat(tunnel, target: PurePosixPath) -> tuple[int, bool]:
    """Return (size, is_dir) for `target`. 404 if missing, 500 otherwise."""
    cmd = f"stat -c '%s\t%F' -- {shlex.quote(str(target))}"  # literal TAB
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    if rc != 0:
        err = (stderr or "").strip().lower()
        if "no such file" in err or "cannot stat" in err:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "path not found")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"stat failed: {stderr.strip() or 'unknown error'}",
        )
    first = (stdout or "").splitlines()[0] if stdout else ""
    parts = first.split("\t")
    if len(parts) < 2:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"unexpected stat output: {first!r}",
        )
    try:
        size = int(parts[0])
    except ValueError as e:
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"unexpected stat size: {parts[0]!r}",
        ) from e
    return size, parts[1].startswith("directory")
