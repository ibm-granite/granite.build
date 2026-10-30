"""Rootless stand-ins for enroot's two privileged image-import helpers.

``enroot import`` extracts each image layer into a numbered directory, then
calls ``enroot-aufs2ovlfs`` to turn aufs whiteouts into overlayfs ones
(needs CAP_MKNOD / CAP_SYS_ADMIN) and ``enroot-mksquashovlfs`` to mount the
layers as an overlay and squash it (needs a mount). Some compute nodes
grant neither, on the host or inside a container, so the import fails.

Here the whiteouts stay in aufs form and ``mksquashovlfs`` is replaced by a
plain merge: the layer directories are moved into one tree bottom-up,
applying ``.wh.<name>`` (delete) and ``.wh..wh..opq`` (opaque directory) as
they go, and the tree is squashed with ``mksquashfs``. No privileges, no FUSE.

enroot finds its helpers on PATH, so :func:`helper_dir` writes two wrappers
and ``enroot import`` is run with that directory first on PATH.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

WHITEOUT = ".wh."
OPAQUE = ".wh..wh..opq"


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _is_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _strip_whiteouts(tree: Path) -> None:
    """Drop whiteout markers from a directory that covers nothing below it."""
    for root, dirs, files in os.walk(tree):
        for name in files + dirs:
            if name.startswith(WHITEOUT):
                _remove(Path(root) / name)
        dirs[:] = [d for d in dirs if not d.startswith(WHITEOUT)]


def _clear(directory: Path) -> None:
    for entry in directory.iterdir():
        _remove(entry)


def merge_layer(layer: Path, dest: Path) -> None:
    """Move ``layer`` onto ``dest`` with aufs whiteout semantics."""
    if (layer / OPAQUE).exists():
        _clear(dest)
    for entry in sorted(layer.iterdir()):
        name = entry.name
        if name == OPAQUE:
            continue
        if name.startswith(WHITEOUT):
            target = dest / name[len(WHITEOUT) :]
            if target.exists() or target.is_symlink():
                _remove(target)
            continue
        target = dest / name
        if _is_dir(entry) and _is_dir(target):
            merge_layer(entry, target)
            shutil.copystat(entry, target, follow_symlinks=False)
            continue
        if target.exists() or target.is_symlink():
            _remove(target)
        os.rename(entry, target)
        if _is_dir(target):
            _strip_whiteouts(target)


def mksquashovlfs(layers: str, output: str, args: list[str]) -> None:
    """``enroot-mksquashovlfs LAYERS OUTPUT [mksquashfs args]``. LAYERS is
    colon-separated, top layer first (enroot passes ``0:1:...:N``)."""
    dirs = [Path(d) for d in layers.split(":")]
    rootfs = Path(os.environ.get("MOUNTPOINT") or tempfile.mkdtemp(prefix="rootfs-", dir="."))
    rootfs.mkdir(parents=True, exist_ok=True)
    for layer in reversed(dirs):
        merge_layer(layer, rootfs)
    subprocess.run(["mksquashfs", str(rootfs), output, *args], check=True)


def helper_dir() -> Path:
    """A directory holding ``enroot-aufs2ovlfs`` (a no-op: the merge applies
    aufs whiteouts itself) and ``enroot-mksquashovlfs`` (this module)."""
    d = Path(tempfile.mkdtemp(prefix="granite-enroot-helpers-"))
    scripts = {
        "enroot-aufs2ovlfs": "#!/bin/sh\nexit 0\n",
        "enroot-mksquashovlfs": f'#!/bin/sh\nexec "{sys.executable}" -m granite_evals.sandbox.ovlfs "$@"\n',
    }
    for name, body in scripts.items():
        path = d / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return d


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python -m granite_evals.sandbox.ovlfs LAYERS OUTPUT [mksquashfs args]")
    mksquashovlfs(sys.argv[1], sys.argv[2], sys.argv[3:])
