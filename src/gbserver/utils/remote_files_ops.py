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

"""Remote file-operation helpers shared by the build-files and environment-files
REST APIs.

These read a remote path tree over an already-open SSH tunnel: directory
listing (``find``/``ls``), recursive content grep (``grep -Z``), batched
``stat``, head/tail/range peek, and streamed SFTP download. Every function is
rooted at a caller-supplied ``root`` (a ``PurePosixPath``) — a build root or a
environment-folder root — and is otherwise agnostic to what the root represents.

This module is pure machinery and deliberately framework-free: it raises the
domain errors defined below (``RemoteFileError`` subclasses) rather than
``fastapi.HTTPException``, so it lives under ``gbserver/utils/`` alongside the
other low-level helpers (``ssh_tunnel``, ``filesystem``, …) with no web-layer
dependency. The thin ``api/`` handlers that call these functions translate the
domain errors into HTTP responses (see ``gbserver.api.utils``).

SECURITY: callers MUST resolve and containment-check every remote path (via
``build_files_paths.validate_subpath`` + ``resolve_and_check_real_path``, rooted
at ``root``) before passing it here. Nothing in this module re-validates paths;
it quotes with ``shlex.quote`` for shell safety but trusts that ``real`` is
already confined to ``root``.
"""

import re
import shlex
from pathlib import PurePosixPath
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union
from urllib.parse import quote

from pydantic import BaseModel

from gbserver.types.constants import (
    BUILD_FILES_GREP_LINE_MAX_BYTES,
    BUILD_FILES_GREP_MAX_HITS,
    BUILD_FILES_LIST_MAX_ENTRIES,
    BUILD_FILES_PEEK_MAX_BYTES,
    BUILD_FILES_STAT_BATCH_MAX,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------- errors


class RemoteFileError(Exception):
    """Base for remote-file-op failures the API layer maps to an HTTP status.

    Keeping this framework-free (no ``fastapi`` dependency) is what lets this
    module live under ``gbserver/utils/``. The ``api/`` handlers catch these
    and translate to ``HTTPException`` — see ``gbserver.api.utils`` for the
    exception→status mapping. The message carried here is surfaced verbatim as
    the HTTP detail, so it must not leak paths/identities the caller shouldn't
    see (the existing messages are already generic, e.g. ``"path not found"``).
    """


class RemoteFileBadRequest(RemoteFileError):
    """Invalid request (bad args / illegal pattern / bad range) → HTTP 400."""


class RemoteFileNotFound(RemoteFileError):
    """The requested remote path does not exist → HTTP 404."""


class RemoteFileOpFailed(RemoteFileError):
    """A remote command failed unexpectedly → HTTP 500."""


# --------------------------------------------------------------------- models


class GrepHit(BaseModel):
    path: str
    """Path of the matching file, relative to the root."""
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
    """Path of the entry, relative to the root."""
    type: str
    """One of ``file``, ``dir``, ``symlink``, ``other``."""
    size: int
    """File size in bytes; 0 for directories."""
    mtime: int
    """Mtime as Unix epoch seconds."""


# --------------------------------------------------------------------- helpers


def reject_pattern_control_chars(pattern: str) -> None:
    """Reject patterns with chars that break shell quoting or grep -F semantics.

    `shlex.quote` makes the pattern safe for the shell, and `grep -F`
    treats it as a literal — but newlines split into separate patterns
    and NULs terminate strings in C-level libraries, so we still 400 on
    those.
    """
    if any(c in pattern for c in ("\x00", "\n", "\r")):
        raise RemoteFileBadRequest("pattern contains illegal characters")


def _no_match_or_500(rc: int, stdout: str, stderr: str, what: str) -> List[str]:
    """Translate a `... | grep ... | head` exit code into hits or a domain error.

    grep exits 1 when there are no matches — that's not an error here,
    return []. rc=141 is SIGPIPE: head closed its stdin after the cap was
    reached and the producer died with EPIPE; the truncated stdout is
    still the result we want. rc>=2 is a real failure (or a stage before
    grep failed under pipefail).
    """
    if rc in (0, 141):
        # Split on '\n' only — not str.splitlines(), which also breaks on
        # embedded '\r'. Source lines from tqdm progress bars and other
        # \r-heavy output must stay intact so the parser sees the whole
        # record, not fragments.
        return [ln for ln in (stdout or "").split("\n") if ln]
    if rc == 1 and not stdout and not stderr:
        # grep's "no matches" contract: rc=1 with empty stdout AND empty
        # stderr. Any pipeline-stage failure under `set -o pipefail` (head
        # crash, I/O error, permission denied not caught by the substring
        # heuristics below) writes something to stderr — fall through so
        # those surface as 500 instead of being masked as "no hits."
        return []
    err = (stderr or "").lower()
    if "no such file" in err or "cannot access" in err:
        raise RemoteFileNotFound("path not found")
    raise RemoteFileOpFailed(f"{what} failed: {stderr.strip() or 'unknown error'}")


# ---------------------------------------------------------------- search / grep


# grep -Z output: <abs_path>\0<lineno><sep><text>, where sep is ':' for
# match lines and '-' for context lines (when -A/-B is set). The NUL
# byte unambiguously delimits the path from the rest, so filenames may
# contain ':' or '-<digits>-' and matched text may contain ':<digits>:'
# without confusing the parser.
_GREP_Z_RE = re.compile(r"^(?P<lineno>\d+)(?P<sep>[:\-])(?P<text>.*)$")


def _parse_grep_line(
    ln: str, root: PurePosixPath
) -> Optional[Tuple[str, int, str, bool]]:
    """Parse one line of ``grep -Z -n`` output into (rel_path, lineno, text, is_match).

    Format: ``<abs_path>\\0<lineno><sep><text>``. ``sep`` is ``':'`` for
    match lines and ``'-'`` for context lines (when ``-A``/``-B`` is set).
    Returns None for lines that don't fit the format (including grep's
    ``--`` group separator, which has no NUL).
    """
    nul = ln.find("\x00")
    if nul < 0:
        return None
    abs_path = ln[:nul]
    m = _GREP_Z_RE.match(ln[nul + 1 :])
    if m is None:
        return None
    lineno = int(m.group("lineno"))
    is_match = m.group("sep") == ":"
    try:
        rel = str(PurePosixPath(abs_path).relative_to(root))
    except ValueError:
        return None
    return rel, lineno, m.group("text"), is_match


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


async def run_search(
    tunnel,
    root: PurePosixPath,
    real: PurePosixPath,
    *,
    pattern: str,
    ignore_case: bool,
    regex: bool,
    before: int,
    after: int,
    stat: bool,
) -> List[GrepHit]:
    """Recursively grep for ``pattern`` under the already-resolved ``real``.

    ``root`` is used only to re-anchor matched absolute paths to paths
    relative to the root. Callers must have resolved+containment-checked
    ``real`` first. See the /files/search endpoint for parameter semantics.
    """
    # -Z emits a NUL after the filename instead of the ':' / '-'
    # separator, so embedded ':<digits>:' or '-<digits>-' in the
    # matched/context text can't masquerade as the record boundary.
    flags = "-r -n -I -H -Z"
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
        parsed = _parse_grep_line(ln, root)
        if parsed is None:
            logger.debug("[remote-files] dropped unparseable grep line: %r", ln)
            continue
        rel, lineno, text, is_match = parsed
        # Replace embedded '\r' (e.g. from tqdm progress bars) with a
        # space so the rendered text is readable, then apply the byte
        # cap on the cleaned-up string.
        if "\r" in text:
            text = text.replace("\r", " ")
        if len(text) > BUILD_FILES_GREP_LINE_MAX_BYTES:
            text = text[:BUILD_FILES_GREP_LINE_MAX_BYTES]
        hits.append(GrepHit(path=rel, line=lineno, text=text, is_match=is_match))

    if stat and hits:
        distinct_rels: List[str] = []
        seen: set[str] = set()
        for h in hits:
            if h.path not in seen:
                seen.add(h.path)
                distinct_rels.append(h.path)
        # Cap stat batch — surplus files keep size/mtime as None.
        batched = distinct_rels[:BUILD_FILES_STAT_BATCH_MAX]
        abs_paths = [root / r for r in batched]
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


# ---------------------------------------------------------------------- listing


_FIND_TYPE_MAP = {"f": "file", "d": "dir", "l": "symlink"}


def _parse_find_printf(line: str, real: PurePosixPath) -> Optional[FileEntry]:
    """Parse one ``find -printf '%P\\t%y\\t%s\\t%T@\\n'`` line into a FileEntry.

    ``%P`` is the path with the search root stripped, so we re-anchor it
    at ``real``'s relpath under the root. ``%T@`` is float epoch.
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


async def run_list(
    tunnel,
    root: PurePosixPath,
    real: PurePosixPath,
    *,
    recursive: bool,
    pattern: Optional[str],
    regex: bool,
    stat: bool,
) -> Union[List[str], List[FileEntry]]:
    """List entries under the already-resolved ``real``.

    Returns paths relative to ``root`` (sorted), or ``FileEntry`` objects
    when ``stat=true``. Callers must have resolved+containment-checked
    ``real`` first. See the /files endpoint for parameter semantics.
    """
    if stat:
        return await _list_files_stat(tunnel, root, real, recursive, pattern, regex)

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
        # rc=141 is SIGPIPE under pipefail: head closed stdin after the
        # cap; the truncated stdout is still the result we want.
        if rc not in (0, 141):
            err = (stderr or "").lower()
            if "no such file" in err or "cannot access" in err:
                raise RemoteFileNotFound("path not found")
            raise RemoteFileOpFailed(
                f"listing failed: {stderr.strip() or 'unknown error'}"
            )
        lines = [ln for ln in (stdout or "").splitlines() if ln]

    if recursive:
        # find emits absolute paths.
        rels = [str(PurePosixPath(ln).relative_to(root)) for ln in lines]
    else:
        # ls -1A emits bare names rooted at `real`.
        rels = [str((real / name).relative_to(root)) for name in lines]
    rels.sort()
    return rels


async def _list_files_stat(
    tunnel,
    root: PurePosixPath,
    real: PurePosixPath,
    recursive: bool,
    pattern: Optional[str],
    regex: bool,
) -> List[FileEntry]:
    """``stat=true`` branch of run_list: ``find -printf`` → FileEntry list.

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
    # rc=141 is SIGPIPE under pipefail: head closed stdin after the cap;
    # the truncated stdout is still the result we want.
    if rc not in (0, 141):
        err = (stderr or "").lower()
        if "no such file" in err or "cannot access" in err:
            raise RemoteFileNotFound("path not found")
        raise RemoteFileOpFailed(f"listing failed: {stderr.strip() or 'unknown error'}")

    entries: List[FileEntry] = []
    for ln in (stdout or "").splitlines():
        if not ln:
            continue
        entry = _parse_find_printf(ln, real)
        if entry is None:
            continue
        # Re-anchor entry.path to be relative to root (currently
        # absolute because we passed `real` which is absolute).
        try:
            entry.path = str(PurePosixPath(entry.path).relative_to(root))
        except ValueError:
            continue
        entries.append(entry)

    if pattern is not None:
        if regex:
            try:
                rx = re.compile(pattern)
            except re.error as e:
                raise RemoteFileBadRequest(f"invalid regex: {e}") from e
            entries = [e for e in entries if rx.search(e.path)]
        else:
            entries = [e for e in entries if pattern in e.path]

    entries.sort(key=lambda e: e.path)
    return entries


# ------------------------------------------------------------- download / peek


async def stream_sftp_file(
    tunnel, remote_path: str, max_bytes: int
) -> AsyncIterator[bytes]:
    """Yield up to ``max_bytes`` of a remote file via SFTP.

    Caps the streamed length so it matches the ``Content-Length`` derived
    from a prior stat: a file appended-to during the stream won't push
    bytes past the declared length, and a file truncated mid-stream just
    yields what's there.
    """
    chunk_size = 256 * 1024
    sftp = None
    yielded = 0
    try:
        sftp = await tunnel.start_sftp()
        async with sftp.open(remote_path, "rb", encoding=None) as fh:
            while yielded < max_bytes:
                chunk = await fh.read(min(chunk_size, max_bytes - yielded))
                if not chunk:
                    return
                yielded += len(chunk)
                yield chunk
    finally:
        if sftp is not None:
            sftp.exit()


def content_disposition(filename: str) -> str:
    """RFC 5987 Content-Disposition value with an ASCII fallback + UTF-8 form."""
    ascii_fallback = (
        filename.encode("ascii", "replace").decode("ascii").replace('"', "_")
    ) or "download.bin"
    return (
        f'attachment; filename="{ascii_fallback}"; '
        f"filename*=UTF-8''{quote(filename, safe='')}"
    )


def validate_peek_args(
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
        raise RemoteFileBadRequest("head, tail, and range are mutually exclusive")
    if head is not None:
        return "head", (head,)
    if tail is not None:
        return "tail", (tail,)
    assert range_ is not None
    try:
        start_s, end_s = range_.split("-", 1)
        start, end = int(start_s), int(end_s)
    except ValueError as e:
        raise RemoteFileBadRequest(
            "range must be of the form start-end (1-indexed line numbers)"
        ) from e
    if start < 1 or end < start:
        raise RemoteFileBadRequest("range requires 1 <= start <= end")
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
    cmd = f"set -o pipefail; {producer} | head -c {BUILD_FILES_PEEK_MAX_BYTES}"
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    # head -c truncating its input causes the producer to die with
    # SIGPIPE → rc=141 under pipefail; that's success here.
    if rc not in (0, 141):
        err = (stderr or "").lower()
        if "no such file" in err or "cannot access" in err:
            raise RemoteFileNotFound("path not found")
        raise RemoteFileOpFailed(f"peek failed: {stderr.strip() or 'unknown error'}")
    if isinstance(stdout, bytes):
        return stdout.decode("utf-8", errors="replace")
    return stdout or ""


async def peek_file(
    tunnel,
    real: PurePosixPath,
    peek: Tuple[str, Tuple[int, ...]],
) -> str:
    """Peek at the already-resolved ``real`` (head/tail/range) and return text.

    Returns the decoded slice as a ``str``; the caller wraps it in a
    ``text/plain; charset=utf-8`` HTTP response (this module stays framework-
    free). Raises ``RemoteFileBadRequest`` for a directory target. Callers must
    have resolved+containment-checked ``real`` first.
    """
    # Reject directories explicitly — head/tail on a directory would error
    # from the shell, but the message is clearer here.
    _size, is_dir = await remote_stat(tunnel, real)
    if is_dir:
        raise RemoteFileBadRequest("peek endpoint requires a file, not a directory")
    mode, args = peek
    return await _peek_text(tunnel, real, mode, args)


async def remote_stat(tunnel, target: PurePosixPath) -> tuple[int, bool]:
    """Return (size, is_dir) for `target`. 404 if missing, 500 otherwise."""
    cmd = f"stat -c '%s\t%F' -- {shlex.quote(str(target))}"  # literal TAB
    rc, stdout, stderr = await tunnel.run_remote(cmd, raise_on_error=False)
    if rc != 0:
        err = (stderr or "").strip().lower()
        if "no such file" in err or "cannot stat" in err:
            raise RemoteFileNotFound("path not found")
        raise RemoteFileOpFailed(f"stat failed: {stderr.strip() or 'unknown error'}")
    first = (stdout or "").splitlines()[0] if stdout else ""
    parts = first.split("\t")
    if len(parts) < 2:
        raise RemoteFileOpFailed(f"unexpected stat output: {first!r}")
    try:
        size = int(parts[0])
    except ValueError as e:
        raise RemoteFileOpFailed(f"unexpected stat size: {parts[0]!r}") from e
    return size, parts[1].startswith("directory")


__all__ = [
    "FileEntry",
    "GrepHit",
    "RemoteFileBadRequest",
    "RemoteFileError",
    "RemoteFileNotFound",
    "RemoteFileOpFailed",
    "content_disposition",
    "peek_file",
    "reject_pattern_control_chars",
    "remote_stat",
    "run_list",
    "run_search",
    "stream_sftp_file",
    "validate_peek_args",
]
