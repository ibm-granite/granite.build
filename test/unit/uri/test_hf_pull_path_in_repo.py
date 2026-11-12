#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

"""An hf:// repo URI with a path in the repo pulls only that path.

``hf:///datasets/org/repo/main/data/train`` downloads ``data/train`` (a file or
a directory) and nothing else, keeps the repo layout under the pull
destination, and binds ``<dest>/data/train``. Every pull path does this: the
Python ``HfURI.pull`` (bash, docker, k8s), the skypilot and LSF hfpull shell
scripts, and the skypilot inline download line.
"""

import asyncio
import os
import shutil
import stat
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from jinja2 import Environment

import gbserver
from gbcommon.types.testing import ENV_VAR_GBTEST_MOCK_HF
from gbcommon.uri.hf import HfType, HfURI, _hfpull_lock_path, hf_path_patterns
from gbserver.asset.hfstore import Hfstore
from gbserver.environment.skypilot import _inline_hf_download_cmd

_URI = "hf:///datasets/org/repo/main/training_data/answerability/parquet"
_SUB = "training_data/answerability/parquet"
_STEPS = Path(gbserver.__file__).parent / "builtins" / "steps"


@pytest.fixture(autouse=True)
def _disable_hf_op_mocking(monkeypatch):
    """Exercise the real pull paths, as test_hf.py does.

    A suite-level GBTEST_MOCK_HF (the default in mock mode, as in CI) makes
    HfURI.pull and the hfpull shell scripts return before they download, so
    the patched snapshot_download and the stub `hf` would never be called.
    Clear it here; monkeypatch restores it after each test.
    """
    monkeypatch.delenv(ENV_VAR_GBTEST_MOCK_HF, raising=False)


# ── HfURI ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("", None),
        ("a/b", ["a/b", "a/b/*"]),
        ("/a/b/", ["a/b", "a/b/*"]),
        ("config.json", ["config.json", "config.json/*"]),
    ],
)
def test_hf_path_patterns(path, expected):
    assert hf_path_patterns(path) == expected


def test_local_path():
    dest = Path("/cache/org/repo/main")
    assert HfURI.parse(_URI).local_path(dest) == dest / _SUB
    assert HfURI.parse("hf:///datasets/org/repo").local_path(dest) == dest
    # A bucket syncs the path's contents into dest itself.
    bucket = HfURI.parse("hf:///buckets/org/b/some/dir")
    assert bucket.get_hf_type() == HfType.BUCKET
    assert bucket.local_path(dest) == dest


def test_pull_filters_to_the_path_and_keeps_the_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def fake_download(**kw):
        # snapshot_download keeps the repo layout under local_dir.
        (Path(kw["local_dir"]) / _SUB).mkdir(parents=True)
        (Path(kw["local_dir"]) / _SUB / "train.parquet").write_text("x")

    uri = HfURI.parse(_URI)
    with patch(
        "gbcommon.uri.hf.snapshot_download", side_effect=fake_download
    ) as mock_dl:
        assert uri.pull(tmp_path) is True
    kw = mock_dl.call_args.kwargs
    assert kw["repo_id"] == "org/repo"
    assert kw["repo_type"] == "dataset"
    assert kw["revision"] == "main"
    assert kw["allow_patterns"] == [_SUB, f"{_SUB}/*"]
    assert (uri.local_path(tmp_path) / "train.parquet").exists()


def test_pull_of_a_path_that_matches_nothing_fails(tmp_path):
    with patch("gbcommon.uri.hf.snapshot_download"):
        assert HfURI.parse(_URI).pull(tmp_path) is False


def test_whole_repo_pull_has_no_filter(tmp_path):
    with patch("gbcommon.uri.hf.snapshot_download") as mock_dl:
        assert HfURI.parse("hf:///datasets/org/repo").pull(tmp_path) is True
    assert mock_dl.call_args.kwargs["allow_patterns"] is None


def test_local_pull_binds_the_path(tmp_path):
    from gbserver.environment.local_assets import pull_asset_hfstore

    def fake_download(**kw):
        (Path(kw["local_dir"]) / _SUB).mkdir(parents=True)

    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        bound = pull_asset_hfstore(_URI, None, None, dest=tmp_path / "d")
    assert bound == tmp_path / "d" / _SUB


def test_pull_evicted_before_the_path_lands_rewaits(tmp_path):
    """Losing the lock before the path is downloaded is a re-wait, not a miss.

    A peer that reclaims our lock is still writing the tree, so the path may not
    be there yet. The pull must re-wait and retry (the #320 design), not report
    that the path matched nothing.
    """
    dest = tmp_path / "d"
    calls = []

    def fake_download(**kw):
        calls.append(kw)
        if len(calls) == 1:
            # A peer reclaims our lock before the path has landed.
            shutil.rmtree(_hfpull_lock_path(dest), ignore_errors=True)
            return
        (Path(kw["local_dir"]) / _SUB).mkdir(parents=True)

    with patch("gbcommon.uri.hf.snapshot_download", side_effect=fake_download):
        assert HfURI.parse(_URI).pull(dest) is True
    assert len(calls) == 2


# ── hfpull step configs and bindings ─────────────────────────────────────────


def test_hfpull_step_config_carries_the_path():
    cfg = Hfstore.build_hfpull_step_config(HfURI.parse(_URI), "/cache/x")
    assert cfg["path_in_repo"] == _SUB
    assert cfg["path"] == "/cache/x"
    whole = Hfstore.build_hfpull_step_config(HfURI.parse("hf:///org/m"), "/c")
    assert whole["path_in_repo"] == ""


def test_hfpull_step_config_has_no_path_for_a_bucket():
    """A bucket is bound at dest (local_path), so the shell steps get no filter."""
    uri = HfURI.parse("hf:///buckets/org/b/some/dir")
    cfg = Hfstore.build_hfpull_step_config(uri, "/cache/x")
    assert cfg["path_in_repo"] == ""


def _hfstore():
    store = MagicMock(spec=Hfstore)
    store.resolve_token.return_value = "tok"
    return store


def _storeload(**config):
    sl = MagicMock()
    sl.mode = "default"
    sl.config = {"cache_path": "/data/cache", **config}
    return sl


@pytest.fixture
def skypilot_env():
    from gbserver.environment.skypilot import Skypilot
    from gbserver.types.environmentconfig import EnvironmentConfig

    config = EnvironmentConfig(
        name="test-skypilot",
        type="Skypilot",
        config={"default_cloud": "k8s", "idle_minutes_to_autostop": 0},
    )
    return Skypilot(event_q=asyncio.Queue(), environment_config=config)


@pytest.mark.asyncio
async def test_skypilot_binds_the_path_under_the_download_dir(skypilot_env):
    binding, step = await skypilot_env.pullasset_hfstore(
        uri=HfURI.parse(_URI), assetstore=_hfstore(), storeload_config=_storeload()
    )
    dest = "/data/cache/org/repo/main"
    assert binding["binding"]["path"] == f"{dest}/{_SUB}"
    hfp = step.config["hfpull_config"]
    assert hfp["path"] == dest
    assert hfp["path_in_repo"] == _SUB


@pytest.mark.asyncio
async def test_skypilot_inline_pull_carries_the_path(skypilot_env):
    binding, step = await skypilot_env.pullasset_hfstore(
        uri=HfURI.parse(_URI),
        assetstore=_hfstore(),
        storeload_config=_storeload(inline=True),
    )
    assert step is None
    assert binding["_hfpull"]["path_in_repo"] == _SUB
    assert binding["binding"]["path"].endswith(f"/main/{_SUB}")


@pytest.mark.asyncio
async def test_skypilot_inline_bucket_pull_has_no_path(skypilot_env):
    binding, _ = await skypilot_env.pullasset_hfstore(
        uri=HfURI.parse("hf:///buckets/org/b/some/dir"),
        assetstore=_hfstore(),
        storeload_config=_storeload(inline=True),
    )
    assert binding["_hfpull"]["path_in_repo"] == ""
    assert binding["binding"]["path"] == binding["_hfpull"]["path"]


@pytest.mark.asyncio
async def test_lsf_binds_the_path_under_the_download_dir():
    from gbserver.environment.lsf import Lsf

    env = Lsf.__new__(Lsf)
    uri = HfURI.parse(_URI)
    with (
        patch("gbserver.environment.lsf.Asset") as asset,
        patch.object(
            Lsf, "_load_builtin_hf_lsf_section", return_value=({}, {}), create=True
        ),
        patch.object(Lsf, "_warn_non_default_mode", create=True),
    ):
        asset.return_value.get_metadata.return_value = {"token_secretname": "HF_TOKEN"}
        binding, step = await env.pullasset_hfstore(
            uri=uri, assetstore=_hfstore(), storeload_config=_storeload()
        )
    dest = f"/data/cache/org/repo/{uri.hash()}"
    assert binding["binding"]["path"] == f"{dest}/{_SUB}"
    assert step.config["hfpull_config"]["path"] == dest


# ── shell paths ──────────────────────────────────────────────────────────────


def _stub_hf(tmp_path, creates):
    """A stub `hf` that records its argv and makes ``--local-dir/<creates>``."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    hf = bindir / "hf"
    hf.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$HF_ARGV"\n'
        'while [ $# -gt 0 ]; do [ "$1" = --local-dir ] && dir="$2"; shift; done\n'
        f'[ -z "{creates}" ] || mkdir -p "$dir/{creates}"\n'
    )
    hf.chmod(hf.stat().st_mode | stat.S_IEXEC)
    return bindir


def _run(tmp_path, script, creates):
    env = {
        **os.environ,
        "PATH": f"{_stub_hf(tmp_path, creates)}:{os.environ['PATH']}",
        "HF_ARGV": str(tmp_path / "argv"),
        "GB_HFPULL_LOCK_TIMEOUT": "5",
        "LLMB_LSF_LAUNCH_ID": "test",
    }
    env.pop("GBTEST_MOCK_HF", None)
    proc = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    argv = (tmp_path / "argv").read_text().splitlines()
    return proc, argv


def test_inline_line_includes_only_the_path(tmp_path):
    cmd = _inline_hf_download_cmd(
        {
            "repo": "org/repo",
            "path": str(tmp_path / "dest"),
            "revision": "main",
            "type": "dataset",
            "path_in_repo": f"/{_SUB}/",
        }
    )
    proc, argv = _run(tmp_path, cmd, _SUB)
    assert proc.returncode == 0, proc.stderr
    assert argv[argv.index("--include") + 1] == _SUB
    assert argv.count("--include") == 2
    assert f"{_SUB}/*" in argv


def test_inline_line_fails_when_the_path_matches_nothing(tmp_path):
    cmd = _inline_hf_download_cmd(
        {
            "repo": "org/repo",
            "path": str(tmp_path / "dest"),
            "revision": "main",
            "type": "dataset",
            "path_in_repo": _SUB,
        }
    )
    # Run without `set -e`, so only the explicit exit can stop the setup.
    proc, _ = _run(tmp_path, f"{cmd}\necho setup-continued", "")
    assert proc.returncode != 0
    assert "matched nothing" in proc.stderr
    assert "setup-continued" not in proc.stdout


def _render(text, path_in_repo):
    hfp = {
        "path": "DEST_PLACEHOLDER",
        "uri": _URI,
        "endpoint": "https://huggingface.co",
        "owner": "org",
        "repo": "repo",
        "revision": "main",
        "path_in_repo": path_in_repo,
        "hf": {"type": "dataset"},
    }
    return Environment().from_string(text).render(config={"hfpull_config": hfp})


def _lsf_script():
    return (
        _STEPS / "lsf" / "hfpull" / "lsf_scripts" / "hfpull" / "command.sh"
    ).read_text()


def _skypilot_script():
    import yaml

    step = yaml.safe_load((_STEPS / "skypilot" / "hfpull" / "step.yaml").read_text())
    return step["environment_configs"]["Skypilot"]["launchers"]["hfpull"]["config"][
        "run"
    ]


def _pull_slice(script, path_in_repo, dest):
    """The rendered script's HF_* variables, args and post-download check.

    The whole worker script needs bash 4+ and the LSF/SkyPilot job env, so the
    lock and self-heal are swapped for one plain `hf download "${HF_ARGS[@]}"`;
    the lines this test is about (argument building and the post-download
    check) run as they are.
    """
    text = _render(script(), path_in_repo).replace("DEST_PLACEHOLDER", str(dest))
    lines = [ln.strip() for ln in text.splitlines()]
    head = [
        ln
        for ln in lines
        if ln.startswith(
            (
                "HF_DEST=",
                "HF_URI=",
                "HF_REPO=",
                "HF_REVISION=",
                "HF_TYPE=",
                "HF_PATH_IN_REPO=",
            )
        )
    ]
    first = lines.index('HF_ARGS=("${HF_REPO}" --local-dir "${HF_DEST}")')
    last = next(
        i for i in range(first, len(lines)) if lines[i].startswith("while :; do")
    )
    args = lines[first:last]
    check_start = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith('if [[ -n "${HF_PATH_IN_REPO}" && ! -e')
    )
    check = lines[check_start : check_start + 4]
    assert check[-1] == "fi", check
    return "\n".join(["set -eu", *head, *args, 'hf download "${HF_ARGS[@]}"', *check])


_SCRIPTS = pytest.mark.parametrize(
    "script", [_lsf_script, _skypilot_script], ids=["lsf", "sky"]
)


@_SCRIPTS
def test_hfpull_script_includes_only_the_path(tmp_path, script):
    dest = tmp_path / "dest"
    proc, argv = _run(tmp_path, _pull_slice(script, _SUB, dest), _SUB)
    assert proc.returncode == 0, proc.stderr
    assert argv[:3] == ["download", "org/repo", "--local-dir"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--include"] == [
        _SUB,
        f"{_SUB}/*",
    ]
    assert (dest / _SUB).is_dir()


@_SCRIPTS
def test_hfpull_script_strips_slashes_from_the_path(tmp_path, script):
    proc, argv = _run(tmp_path, _pull_slice(script, f"/{_SUB}/", tmp_path / "d"), _SUB)
    assert proc.returncode == 0, proc.stderr
    assert argv[argv.index("--include") + 1] == _SUB


@_SCRIPTS
def test_hfpull_script_fails_when_the_path_matches_nothing(tmp_path, script):
    proc, _ = _run(tmp_path, _pull_slice(script, _SUB, tmp_path / "d"), "")
    assert proc.returncode != 0
    assert "matched nothing" in proc.stderr


@_SCRIPTS
def test_hfpull_script_whole_repo_has_no_include(tmp_path, script):
    proc, argv = _run(tmp_path, _pull_slice(script, "", tmp_path / "d"), "")
    assert proc.returncode == 0, proc.stderr
    assert "--include" not in argv
