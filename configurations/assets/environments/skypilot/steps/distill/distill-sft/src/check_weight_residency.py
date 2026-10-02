"""Refuse to hold an allocation waiting for weights that live on tape.

BYTE-IDENTICAL across every step that reads model weights (distill-gold,
distill-sft, distill-logit-precompute) and asserted so by distill-gold's
test_weight_residency_contract.py, for the same reason the source-delivery region is
asserted by test_source_contract.py: three copies that drift are worse than one copy that
is checked. distill-gold is the reference, and its test_check_weight_residency.py is where
this module's behaviour is pinned -- including the fixtures from the incident below.

WHY THIS EXISTS. /proj on BlueVela is tape-backed GPFS. A migrated file is fully
readable -- `open()` succeeds, the bytes are correct -- and reading it blocks on a recall
that can run to tens of GB. So every cheap gate passes and the expensive one pays: on
2026-09-21 a teacher checkpoint cleared a tokenizer-load gate, a config-key gate and a
dry-run gate, because all three ask questions a migrated file answers correctly. Its nine
shards were symlinks into another account's archive on fileset `data-eng-cos`, eight of
them with ONE 512-byte block allocated against ~5 GB logical each: ~44 GB of tape recall
away. The first read would have been the trainer's, with the GPUs already held, in the
PREEMPTABLE queue, killable before step 1 and paying the recall again on requeue.

THE INSTRUMENT, AND WHY THE OBVIOUS ONE IS NOT ENOUGH. `mmlsattr -L` reports GPFS's own
answer in `Misc attributes:`, and OFFLINE there is authoritative. The tempting substitute
is arithmetic on `stat`: a migrated file reports full logical size with almost no blocks
allocated, so `st_blocks * 512 << st_size` looks sufficient. It is not, and the
counterexample was measured rather than imagined -- mid-recall, 2026-09-21::

    model-00007-of-00009.safetensors  size=4686230768  blocks=8388608  attrs=[ARCHIVE OFFLINE]

4.29 GB allocated against 4.69 GB logical: 91.6%, which sails through any sensible ratio
while GPFS still calls the file OFFLINE. A PARTIAL RECALL reads as resident to the
heuristic and as offline to the filesystem, so the two are not interchangeable::

    mmlsattr      may CLEAR a file or CONDEMN it.  Authoritative, both directions.
    st_blocks     may CONDEMN a file only.         It can never clear one.

NOTE ON `ARCHIVE`. Every file on this cluster carries ARCHIVE, resident or not. The token
that means non-resident is OFFLINE; a check matching ARCHIVE would condemn the filesystem.

IT MUST NOT ITSELF CAUSE A RECALL -- the one way such a check is worse than none: reading
a shard to learn whether it is resident makes it resident, at full cost, while reporting a
problem it just fixed by accident. Nothing here opens a weight file. `os.stat`, `os.lstat`
and `mmlsattr -L` are metadata-only operations on the inode.

EXIT CODES DIFFER FROM THE STANDALONE AUDIT SCRIPT THIS WAS PORTED FROM, deliberately.
There, UNMEASURED is rc 2 -- a reviewer reads it. Here a non-zero rc aborts the step, so
only OFFLINE -- or a weight path the step cannot stat at all -- earns rc 1. Unmeasured
warns and returns 0: a host without mmlsattr, or a non-GPFS path, is the normal case off
this cluster and must not stop a run that would have been fine. Warn-only otherwise, with an override that announces itself.

WHAT IT DOES NOT CHECK. Not correctness, completeness or shape of the weights -- a
resident shard can still be truncated. Not datasets: the same migration applies, but a
corpus is read in megabyte pieces as training proceeds rather than as a blocking
prerequisite to step 1, so it degrades throughput instead of idling an allocation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")

# Below this, allocation arithmetic says nothing useful: GPFS block granularity, inlined
# data and metadata replication all swamp the signal, and no file this small is a recall
# worth gating on. Shards are gigabytes; config files are kilobytes.
MIN_INTERESTING = 16 * 1024 * 1024
# The heuristic's accusation threshold. Only ever used to CONDEMN, so a generous value
# costs nothing: a migrated file sits at ~0.00002 of its size, not at 0.4.
ALLOC_ACCUSE_BELOW = 0.5

OFFLINE_TOKENS = {"OFFLINE", "MIGRATED"}


def find_mmlsattr() -> str | None:
    """mmlsattr's absolute path, or None.

    Not just shutil.which: the GPFS admin tools live in /usr/lpp/mmfs/bin, which an
    interactive login shell has on PATH and a batch job does not necessarily get. A check
    that silently degraded to the weaker instrument depending on how the step was launched
    would be the worst of both.
    """
    found = shutil.which("mmlsattr")
    if found:
        return found
    fallback = "/usr/lpp/mmfs/bin/mmlsattr"
    return fallback if os.access(fallback, os.X_OK) else None


def mmlsattr_env(mm: str) -> dict[str, str]:
    """The environment to run `mm` in: this one, with GPFS's own lib/ on LD_LIBRARY_PATH.

    mmlsattr links libgpfs.so, which a GPFS host resolves through /lib64/libgpfs.so -- a
    symlink into /usr/lpp/mmfs/lib that exists on the HOST only. Inside the step's container,
    with /usr/lpp/mmfs bind-mounted, the tool is found and then fails to load (rc 127), and
    every file silently falls back to the weaker instrument. Measured on BlueVela,
    2026-09-25: the same bind mount answers `Misc attributes: ARCHIVE` once the lib/ beside
    the tool's bin/ is on the path. Derived from where the tool actually resolves, not
    hardcoded, and added only if it exists -- elsewhere this is a no-op.
    """
    env = dict(os.environ)
    lib = Path(os.path.realpath(mm)).parent.parent / "lib"
    if lib.is_dir():
        prior = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib}:{prior}" if prior else str(lib)
    return env


def parse_misc_attributes(text: str) -> set[str] | None:
    """The tokens of `mmlsattr -L`'s `Misc attributes:` field, or None if absent.

    None and set() are different answers and must not be conflated: None means the field
    was not there (another mmlsattr version, an error, a non-GPFS path) and the file is
    UNMEASURED; an empty set is a real "no attributes" reading.
    """
    for line in text.splitlines():
        if line.strip().startswith("Misc attributes:"):
            return set(line.split(":", 1)[1].split())
    return None


def attrs_of(real: str, mm: str | None) -> tuple[set[str] | None, str]:
    """(tokens, detail). tokens is None when mmlsattr could not answer."""
    if mm is None:
        return None, "mmlsattr not available"
    try:
        # Metadata only: -L reads the inode's attributes and does not stage data in.
        out = subprocess.run(
            [mm, "-L", real],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env=mmlsattr_env(mm),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"mmlsattr failed: {type(exc).__name__}"
    toks = parse_misc_attributes(out.stdout)
    if toks is None:
        first = (out.stderr or out.stdout).strip().split("\n")[0][:120]
        return None, (
            f"no Misc attributes field ({first})"
            if first
            else "no Misc attributes field"
        )
    return toks, " ".join(sorted(toks)) or "(none)"


def weight_files(d: Path) -> list[Path]:
    """Weight files under a model dir.

    rglob rather than glob because a sharded checkpoint is sometimes one level down. Dot
    directories are skipped, and `.git` is the case that matters rather than tidiness: the
    granite checkpoints under /proj are git-lfs clones, so each carries a `.git` holding a
    SECOND copy of every shard. Walking it would double every count and report residency
    for files no trainer will ever open.
    """
    out: list[Path] = []
    for p in sorted(d.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(d).parts[:-1]):
            continue
        if p.suffix in WEIGHT_SUFFIXES and p.is_file():
            out.append(p)
    return out


def probe(f: Path, mm: str | None, seen: dict[str, tuple[str, str]]) -> tuple[str, str]:
    """(verdict, detail) for one file: "ok" | "offline" | "unmeasured" | "unreadable".

    Resolved through symlinks, because THE SYMLINK IS THE POINT here: a model mirror
    commonly places weights as links at their realpath, so a teacher dir can be nine links
    into another account's archive. Asking about the link tells you nothing; asking about
    its target is the question. `seen` is keyed on the realpath so two roles sharing a
    checkpoint cost one mmlsattr call.
    """
    real = os.path.realpath(f)
    if real in seen:
        return seen[real]
    try:
        st = os.stat(real)
    except OSError as exc:
        # Refused, like OFFLINE -- a weight file the step cannot stat is not one it can
        # load -- but kept apart from it, because a permissions or dangling-symlink problem
        # answered with "stage it in from tape" sends whoever reads the log the wrong way.
        res = ("unreadable", f"cannot stat: {exc.strerror}")
        seen[real] = res
        return res
    size = st.st_size
    alloc = st.st_blocks * 512
    ratio = (alloc / size) if size else 1.0
    toks, detail = attrs_of(real, mm)

    if toks is not None:
        if toks & OFFLINE_TOKENS:
            res = (
                "offline",
                f"{detail}, {size / 2**30:.2f} GiB logical, {ratio:.1%} allocated",
            )
        else:
            # Authoritative in this direction too, which is why a 91.6%-allocated file is
            # not second-guessed when GPFS says it is here.
            res = ("ok", f"{detail}, {size / 2**30:.2f} GiB, {ratio:.0%} allocated")
    elif size >= MIN_INTERESTING and ratio < ALLOC_ACCUSE_BELOW:
        res = (
            "offline",
            f"{ratio:.2%} of {size / 2**30:.2f} GiB allocated ({detail}) -- the allocation "
            "heuristic condemns it; mmlsattr would say so authoritatively",
        )
    else:
        # The weak instrument cannot clear a file, because a partial recall looks like this
        # too (the measured shard: 91.6% allocated, still OFFLINE).
        res = (
            "unmeasured",
            f"{detail}, {ratio:.0%} of {size / 2**30:.2f} GiB allocated -- looks resident, "
            "but allocation cannot rule out a partial recall",
        )
    seen[real] = res
    return res


def looks_like_a_local_path(value: str) -> bool:
    """Whether to treat a role's value as a filesystem path at all.

    A hub identifier (`ibm-granite/granite-4.0-tiny`) is a perfectly valid value for every
    flag this guards, and it is not a path: `stat` fails on it, which `probe` reports as
    offline because for a real path that is the right reading. Without this, the preflight
    would abort every recipe that names a model by repo id -- a check whose failure mode is
    breaking correct runs is worse than no check. An absolute or explicitly relative value
    is a path; a bare `org/name` that does not exist on disk is not. A `~` value is a
    path too, and audit() expands it before looking.
    """
    if value.startswith(("/", "./", "../", "~")):
        return True
    return os.path.exists(value)


def audit(
    roles: list[tuple[str, str]], mm: str | None
) -> tuple[list[str], list[str], list[str], int]:
    """(offline, unreadable, notes, files_measured) over `role=value` pairs.

    Both of the first two refuse the run. They are separate lists only so the refusal can
    name the right remedy: tape recall for the first, the path or its permissions for the
    second.
    """
    problems: list[str] = []
    unreadable: list[str] = []
    notes: list[str] = []
    seen: dict[str, tuple[str, str]] = {}
    measured = 0

    for role, value in roles:
        if not value:
            notes.append(f"{role}: empty, skipped")
            continue
        if not looks_like_a_local_path(value):
            notes.append(f"{role}: {value!r} is not a local path (hub id?), skipped")
            continue
        p = Path(value).expanduser()
        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = weight_files(p)
        else:
            unreadable.append(f"{role}: {value} does not exist")
            continue
        if not files:
            notes.append(
                f"{role}: {value} holds no weight files (tokenizer-only overlay?)"
            )
            continue
        for f in files:
            verdict, detail = probe(f, mm, seen)
            measured += 1
            line = f"{role}: {f.name}: {detail}"
            if verdict == "offline":
                problems.append(line)
            elif verdict == "unreadable":
                unreadable.append(line)
            elif verdict == "unmeasured":
                notes.append(line)

    return problems, unreadable, notes, measured


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "roles",
        nargs="*",
        metavar="ROLE=PATH",
        help="e.g. student=/proj/kd/x teacher=/proj/kd/y; an empty PATH is skipped",
    )
    ap.add_argument(
        "--allow-offline",
        action="store_true",
        help="proceed even when GPFS says a weight file is OFFLINE (announces itself)",
    )
    args = ap.parse_args()

    parsed: list[tuple[str, str]] = []
    for item in args.roles:
        role, _, value = item.partition("=")
        if not _:
            print(f"[residency] not a ROLE=PATH pair: {item!r}", file=sys.stderr)
            return 2
        parsed.append((role, value))

    mm = find_mmlsattr()
    if mm is None:
        print(
            "[residency] WARN: mmlsattr not found; residency cannot be established here"
        )
    problems, unreadable, notes, measured = audit(parsed, mm)

    for n in notes:
        print(f"[residency] note: {n}")
    print(
        f"[residency] measured {measured} weight file(s) across {len(parsed)} role(s)"
    )

    if unreadable:
        # Not overridable by --allow-offline: that flag accepts a slow first read, and there
        # is no first read to accept for a file the step cannot even stat.
        for p in unreadable:
            print(f"[residency] PROBLEM: {p}", file=sys.stderr)
        print(
            "[residency] REFUSING to start: the weight path(s) above are missing or cannot "
            "be read by this user. This is NOT a tape recall -- check the path, a dangling "
            "symlink, or the permissions on it and its parent directories.",
            file=sys.stderr,
        )
        return 1

    if not problems:
        print("[residency] OK: no weight file is migrated")
        return 0

    for p in problems:
        print(f"[residency] PROBLEM: {p}", file=sys.stderr)
    if args.allow_offline:
        # Announces itself: the override is the interesting fact in the log when this run
        # later spends an hour in a recall nobody expected.
        print(
            f"[residency] OVERRIDDEN: {len(problems)} migrated weight file(s), proceeding "
            "because --allow-offline was given. The first read will block on a tape recall "
            "while this allocation is held.",
            file=sys.stderr,
        )
        return 0
    print(
        "[residency] REFUSING to start: the weights above are on the COS tier, so the "
        "first read blocks on a tape recall with the GPUs already allocated. Stage them "
        "in (dd to /dev/null on a login node, or mmrestripefile), or pass --allow-offline.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
