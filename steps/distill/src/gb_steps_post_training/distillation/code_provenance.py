"""Record which code a training run actually executed.

WHY THIS EXISTS. The lmbda sweep's premise is that its three arms differ in lmbda and in nothing else,
and a companion config check asserts exactly that -- about the CONFIGS. It says nothing about the
code, and the arms do not run at the same time: on `preemptable` they dispatch hours apart, from a repo
that is still being committed to. Arm 1 already demonstrated the failure mode: it ran to
completion, and 13 commits landed in distillation/ between that dispatch and the next arm's, including
the masking contract moving out of the configs and the row manifest. Its curve is
real and its comparison to arms 2 and 3 would not have been, and nothing in the run would have said so.

So the config invariant is checked and the code invariant was merely hoped for. This closes that by
writing the code's identity into every run's log, where the arms can be compared after the fact instead
of assumed comparable.

WHAT IT RECORDS, AND WHY NOT JUST THE SHA. A SHA describes a commit, not a working tree, and this
project runs uncommitted trees routinely -- every measurement in this repo's history was taken from one
at some point. `dirty` therefore matters more than the SHA: two arms at the same SHA with different
uncommitted edits are not comparable, and a reader who sees only the SHA would conclude they were. The
digest over distillation/*.py is what makes a dirty tree comparable at all -- it is the same number for
two runs of the same working tree and a different one otherwise, whatever git thinks.

DESIGN CONSTRAINT: THIS MUST NEVER END A RUN. It is provenance, not a precondition. Every failure mode
-- no git binary, not a repo, a detached HEAD, an unreadable file, a git call that hangs -- resolves to
a recorded "unknown" with the reason attached, never to an exception. A run that dies because it could
not describe itself would be a strictly worse outcome than one that cannot be compared later.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

SRC_GLOB = "*.py"
GIT_TIMEOUT_S = 10


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ("git", "-C", str(root), *args),
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def source_digest(src_dir: Path) -> tuple[str | None, int, str | None]:
    """sha256 over the sorted (name, bytes) of distillation/*.py. Identity of the tree as RUN."""
    try:
        files = sorted(p for p in src_dir.glob(SRC_GLOB) if p.is_file())
    except OSError as exc:
        return None, 0, f"could not list {src_dir}: {exc}"
    if not files:
        return None, 0, f"no {SRC_GLOB} under {src_dir}"
    h = hashlib.sha256()
    for p in files:
        try:
            data = p.read_bytes()
        except OSError as exc:
            return None, len(files), f"could not read {p.name}: {exc}"
        h.update(p.name.encode())
        h.update(b"\0")
        h.update(data)
    return h.hexdigest()[:16], len(files), None


def describe(src_dir: Path | str | None = None) -> dict:
    """Everything known about the code being run. Never raises."""
    src = Path(src_dir) if src_dir else Path(__file__).resolve().parent
    info: dict = {"src_dir": str(src)}

    info["tree_digest"], info["n_files"], digest_note = source_digest(src)
    if digest_note:
        info["tree_digest_note"] = digest_note

    root = _git(src, "rev-parse", "--show-toplevel")
    if not root:
        info["git"] = "unavailable"
        info["git_note"] = "no git binary, or this source tree is not in a repository"
        return info

    info["repo"] = root
    info["sha"] = _git(src, "rev-parse", "HEAD") or "unknown"
    info["branch"] = _git(src, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _git(src, "status", "--porcelain")
    if status is None:
        info["dirty"] = "unknown"
        info["git_note"] = "git status failed, so a dirty tree cannot be ruled out"
    else:
        changed = [l for l in status.splitlines() if l.strip()]
        info["dirty"] = bool(changed)
        info["n_changed"] = len(changed)
    return info


def format_block(info: dict) -> str:
    """A few lines for the run log. ClearML captures the console, so this lands in the tracker too."""
    lines = ["=== code provenance ==="]
    sha = info.get("sha", "unknown")
    short = sha[:12] if sha and sha != "unknown" else "unknown"
    dirty = info.get("dirty", "unknown")
    if dirty is True:
        state = f"DIRTY ({info.get('n_changed', '?')} paths uncommitted)"
    elif dirty is False:
        state = "clean"
    else:
        state = "dirty state UNKNOWN"
    lines.append(f"commit   : {short}  on {info.get('branch', 'unknown')}  [{state}]")
    lines.append(f"src      : {info.get('src_dir')}")
    lines.append(f"digest   : {info.get('tree_digest') or 'unknown'}"
                 f"  (sha256 over {info.get('n_files', 0)} distillation/*.py, first 16 hex)")
    lines.append("           two runs are comparable when this DIGEST matches; the commit alone is")
    lines.append("           not enough, because this project runs uncommitted trees routinely.")
    for key in ("git_note", "tree_digest_note"):
        if info.get(key):
            lines.append(f"note     : {info[key]}")
    return "\n".join(lines)


def report(src_dir: Path | str | None = None) -> dict:
    """describe() + print. Swallows everything: provenance must never end a run."""
    try:
        info = describe(src_dir)
        print(format_block(info), flush=True)
        return info
    except Exception as exc:                                          # noqa: BLE001
        print(f"=== code provenance ===\nunavailable: {type(exc).__name__}: {exc}", flush=True)
        return {"error": f"{type(exc).__name__}: {exc}"}
