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

"""The inline hfpull ``hf download`` line quotes every value it interpolates."""

import os
import stat
import subprocess
from pathlib import Path

import pytest

from gbserver.environment.skypilot import _inline_hf_download_cmd

_NASTY = 'x"; touch PWNED; echo "$(touch PWNED2)`touch PWNED3`'


def _run_with_stub_hf(tmp_path, cmd):
    """Run ``cmd`` under ``sh -eu`` with a stub ``hf`` that records its argv."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "hf"
    stub.write_text(
        '#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done > argv.txt\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    subprocess.run(["sh", "-euc", cmd], cwd=tmp_path, env=env, check=True)
    return (tmp_path / "argv.txt").read_text().splitlines()


def test_plain_values():
    cmd = _inline_hf_download_cmd(
        {"repo": "ibm-research/vira", "path": "/shared/hf_cache/x", "type": "dataset"}
    )
    assert cmd == (
        "hf download ibm-research/vira --local-dir /shared/hf_cache/x "
        "--repo-type dataset"
    )


@pytest.mark.parametrize("field", ["repo", "path", "revision", "type"])
def test_metacharacters_reach_hf_literally(tmp_path, field):
    info = {"repo": "o/r", "path": "/p", "revision": "main", "type": "model"}
    info[field] = _NASTY
    argv = _run_with_stub_hf(tmp_path, _inline_hf_download_cmd(info))
    flag = {"repo": None, "path": "--local-dir", "revision": "--revision"}.get(
        field, "--repo-type"
    )
    value = argv[1] if flag is None else argv[argv.index(flag) + 1]
    assert value == _NASTY
    assert not list(tmp_path.glob("PWNED*")), "a value was executed by the shell"


def test_optional_flags_omitted_when_empty(tmp_path):
    argv = _run_with_stub_hf(
        tmp_path,
        _inline_hf_download_cmd(
            {"repo": "o/r", "path": "/p", "revision": "", "type": None}
        ),
    )
    assert argv == ["download", "o/r", "--local-dir", "/p"]


class TestInlineCachePathCheck:
    """pullasset_hfstore refuses an inline pull whose cache path has a ``$``."""

    @pytest.fixture
    def env(self):
        import asyncio

        from gbserver.environment.skypilot import Skypilot
        from gbserver.types.environmentconfig import EnvironmentConfig

        cfg = EnvironmentConfig(
            name="t", type="Skypilot", config={"default_cloud": "aws"}
        )
        return Skypilot(event_q=asyncio.Queue(), environment_config=cfg)

    @staticmethod
    def _store():
        from unittest.mock import MagicMock

        from gbserver.asset.hfstore import Hfstore

        store = MagicMock(spec=Hfstore)
        store.resolve_token.return_value = "tok"
        return store

    async def _pull(self, env, config):
        from gbserver.types.environmentconfig import StoreLoad

        return await env.pullasset_hfstore(
            uri="hf:///datasets/o/r",
            assetstore=self._store(),
            storeload_config=StoreLoad(config=config),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["$HOME/hf_cache", "/data/${USER}/hf"])
    async def test_dollar_rejected_inline(self, env, path):
        with pytest.raises(ValueError, match="contains '\\$'"):
            await self._pull(env, {"cache_path": path, "inline": True})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/tmp/hf_cache", "./hf_cache", "hf_cache"])
    async def test_literal_paths_still_allowed_inline(self, env, path):
        binding, step = await self._pull(env, {"cache_path": path, "inline": True})
        assert binding["_hfpull"]["path"] == str(Path(path) / "o" / "r" / "main")
        assert step is None

    @pytest.mark.asyncio
    async def test_dollar_not_checked_without_inline(self, env):
        binding, step = await self._pull(env, {"cache_path": "$HOME/hf_cache"})
        assert "_hfpull" not in binding
        assert step is not None
