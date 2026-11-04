import os
import shutil
import subprocess

import pytest

from granite_evals.sandbox import ovlfs


def _tree(root):
    out = {}
    for dirpath, dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        for name in files:
            path = os.path.join(dirpath, name)
            key = os.path.normpath(os.path.join(rel, name))
            out[key] = "->" + os.readlink(path) if os.path.islink(path) else open(path).read()
        for name in dirs:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                out[os.path.normpath(os.path.join(rel, name))] = "->" + os.readlink(path)
            else:
                out[os.path.normpath(os.path.join(rel, name)) + "/"] = ""
    return out


def _layer(root, files):
    root.mkdir()
    for rel, content in files.items():
        path = root / rel
        if rel.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if content.startswith("->"):
            path.symlink_to(content[2:])
        else:
            path.write_text(content)
    return root


def _merge(tmp_path, *layers):
    """Layers bottom-up, as a Dockerfile writes them; merged the way enroot
    numbers them (1 = newest) and passes them (top first)."""
    tmp_path.joinpath("work").mkdir()
    names = []
    for i, files in enumerate(reversed(layers), start=1):
        _layer(tmp_path / "work" / str(i), files)
        names.append(str(i))
    rootfs = tmp_path / "work" / "rootfs"
    rootfs.mkdir()
    for layer in reversed(names):
        ovlfs.merge_layer(tmp_path / "work" / layer, rootfs)
    return _tree(rootfs)


def test_newer_layer_wins(tmp_path):
    tree = _merge(tmp_path, {"etc/a": "base", "etc/b": "base"}, {"etc/a": "new"})
    assert tree == {"etc/": "", "etc/a": "new", "etc/b": "base"}


def test_whiteout_deletes_file_and_directory(tmp_path):
    tree = _merge(
        tmp_path,
        {"opt/x/f": "1", "opt/y": "1", "opt/keep": "1"},
        {"opt/.wh.x": "", "opt/.wh.y": ""},
    )
    assert tree == {"opt/": "", "opt/keep": "1"}


def test_opaque_directory_hides_lower_contents(tmp_path):
    tree = _merge(
        tmp_path,
        {"opt/app/old": "1", "opt/other": "1"},
        {"opt/app/.wh..wh..opq": "", "opt/app/new": "2"},
    )
    assert tree == {"opt/": "", "opt/other": "1", "opt/app/": "", "opt/app/new": "2"}


def test_new_directory_drops_stray_markers(tmp_path):
    tree = _merge(tmp_path, {"bin/sh": "sh"}, {"srv/.wh..wh..opq": "", "srv/d/.wh.gone": "", "srv/d/f": "1"})
    assert tree == {"bin/": "", "bin/sh": "sh", "srv/": "", "srv/d/": "", "srv/d/f": "1"}


def test_file_replaces_directory_and_symlink_replaces_file(tmp_path):
    tree = _merge(tmp_path, {"a/f": "1", "b": "file"}, {"a": "now a file", "b": "->/elsewhere"})
    assert tree == {"a": "now a file", "b": "->/elsewhere"}


def test_readded_after_whiteout(tmp_path):
    tree = _merge(tmp_path, {"f": "1"}, {".wh.f": ""}, {"f": "3"})
    assert tree == {"f": "3"}


@pytest.mark.skipif(not shutil.which("mksquashfs") or not shutil.which("unsquashfs"), reason="squashfs-tools")
def test_helper_scripts_build_a_flat_squashfs(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    _layer(work / "0", {"etc/environment": "PATH=/venv/bin"})
    _layer(work / "1", {"etc/.wh.gone": "", "app/new": "new"})
    _layer(work / "2", {"etc/environment": "", "etc/gone": "x", "app/new": "old"})
    (work / "rootfs").mkdir()
    helpers = ovlfs.helper_dir()
    env = {**os.environ, "PATH": f"{helpers}:{os.environ['PATH']}", "MOUNTPOINT": str(work / "rootfs")}
    subprocess.run(["enroot-aufs2ovlfs", "1"], cwd=work, env=env, check=True)
    subprocess.run(["enroot-mksquashovlfs", "0:1:2", "out.sqsh", "-quiet", "-no-xattrs"], cwd=work, env=env, check=True)
    subprocess.run(["unsquashfs", "-q", "-d", str(tmp_path / "x"), str(work / "out.sqsh")], check=True, capture_output=True)
    assert _tree(tmp_path / "x") == {
        "etc/": "",
        "etc/environment": "PATH=/venv/bin",
        "app/": "",
        "app/new": "new",
    }
