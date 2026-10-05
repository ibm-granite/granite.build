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

"""Executes the inline hf pull block that gbserver prepends to a step's setup.

Inline pulls (skypilot/aws, skypilot/kubernetes) download inside the CONSUMING
step — inside its image when it sets one — so the block must find an ``hf``
client whatever the image ships. These tests run the real generated shell under
the step prologue's ``set -eu``, with PATH narrowed to stubs, and assert which
route it took:

* pip present (SkyPilot's default images, conda/py images) -> pip install, then
  ``hf`` directly: the ORIGINAL behaviour, unchanged (backwards compatibility).
* hf preinstalled, no pip -> that ``hf``.
* no pip and no hf (plain Ubuntu), or a pip that fails (PEP 668) -> uv
  bootstrapped with curl, client run via ``uvx --from <pinned spec>``.
* nothing usable -> fail naming what is missing, before any download.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

from gbserver.environment.skypilot import (
    _INLINE_HF_CLI_SPEC,
    _inline_hfpull_setup_block,
)

_PULLS = {
    "docs": {
        "repo": "ibm-research/vira-intents-live",
        "path": "/tmp/hf_cache/ibm-research/vira-intents-live/main",
        "type": "dataset",
    },
    "model": {
        "repo": "org/model",
        "path": "/tmp/hf_cache/org/model/abc123",
        "revision": "abc123",
        "type": "model",
    },
}
_BASH = shutil.which("bash") or "/bin/bash"
# Real tools the block (and the fake uv installer) need; everything else on PATH
# is a stub, so `command -v pip/hf/curl` sees only what each test provides.
_REAL_TOOLS = ("sh", "env", "mkdir", "chmod", "cat")


def _stub(bin_dir: pathlib.Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!{_BASH}\n{body}\n")
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path):
    """A bin dir holding only real essentials; tests add stubs to it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in _REAL_TOOLS:
        real = shutil.which(tool)
        assert real, f"test needs {tool}"
        (bin_dir / tool).symlink_to(real)
    trace = tmp_path / "trace"
    trace.touch()
    return bin_dir, trace, tmp_path


def _add_hf(bin_dir, trace):
    _stub(bin_dir, "hf", f'echo "hf $*" >> {trace}')


def _add_pip(bin_dir, trace, works=True):
    # A working pip "installs" hf by dropping the stub on PATH, as a real install
    # into an on-PATH bin dir would.
    install = (
        f"printf '#!{_BASH}\\necho \"hf $*\" >> {trace}\\n' > {bin_dir}/hf; "
        f"chmod +x {bin_dir}/hf"
    )
    _stub(
        bin_dir,
        "pip",
        f'echo "pip $*" >> {trace}; '
        + (install if works else "echo externally-managed-environment >&2; exit 1"),
    )


def _add_curl(bin_dir, trace, tmp_path, layout="flat", works=True):
    # Emits a fake uv installer script; the block pipes it into `sh`, which
    # creates uvx under $UV_INSTALL_DIR (flat) or $UV_INSTALL_DIR/bin.
    sub = "" if layout == "flat" else "/bin"
    installer = tmp_path / "install.sh"
    installer.write_text(
        f'mkdir -p "$UV_INSTALL_DIR{sub}"\n'
        f"printf '#!{_BASH}\\necho \"uvx $*\" >> {trace}\\n'"
        f' > "$UV_INSTALL_DIR{sub}/uvx"\n'
        f'chmod +x "$UV_INSTALL_DIR{sub}/uvx"\n'
    )
    _stub(
        bin_dir,
        "curl",
        f'echo "curl $*" >> {trace}; ' + (f"cat {installer}" if works else "exit 22"),
    )


def _run(bin_dir, tmp_path, pulls=_PULLS):
    script = "set -eu\n" + _inline_hfpull_setup_block(pulls)
    env = {"PATH": str(bin_dir), "TMPDIR": str(tmp_path), "HOME": str(tmp_path)}
    return subprocess.run(
        [_BASH, "-c", script], capture_output=True, text=True, env=env
    )


def _calls(trace):
    return trace.read_text().splitlines()


def _expected_download_args():
    return [
        "download ibm-research/vira-intents-live --local-dir "
        "/tmp/hf_cache/ibm-research/vira-intents-live/main --repo-type dataset",
        "download org/model --local-dir /tmp/hf_cache/org/model/abc123 "
        "--revision abc123 --repo-type model",
    ]


class TestBlockIsValidShell:
    def test_parses_under_set_eu(self):
        script = "set -eu\n" + _inline_hfpull_setup_block(_PULLS)
        assert subprocess.run([_BASH, "-n"], input=script, text=True).returncode == 0

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="no shellcheck")
    def test_shellcheck_is_clean(self, tmp_path):
        f = tmp_path / "block.sh"
        f.write_text("set -eu\n" + _inline_hfpull_setup_block(_PULLS))
        assert subprocess.run(["shellcheck", "-s", "bash", str(f)]).returncode == 0

    def test_keeps_the_pinned_client_spec(self):
        assert f"gb_hf_spec='{_INLINE_HF_CLI_SPEC}'" in _inline_hfpull_setup_block(
            _PULLS
        )


class TestPipRouteIsUnchanged:
    """Backwards compatibility: images with pip behave exactly as before."""

    def test_pip_installs_pinned_client_then_hf_downloads(self, sandbox):
        bin_dir, trace, tmp = sandbox
        _add_pip(bin_dir, trace)
        _add_curl(bin_dir, trace, tmp)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        calls = _calls(trace)
        assert calls[0] == f"pip install --no-cache-dir {_INLINE_HF_CLI_SPEC}"
        assert calls[1:] == [f"hf {a}" for a in _expected_download_args()]
        assert not any(c.startswith("curl") for c in calls), "must not bootstrap uv"


class TestPreinstalledHf:
    def test_hf_without_pip_is_used_directly(self, sandbox):
        bin_dir, trace, tmp = sandbox
        _add_hf(bin_dir, trace)
        _add_curl(bin_dir, trace, tmp)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        assert _calls(trace) == [f"hf {a}" for a in _expected_download_args()]


class TestUvBootstrap:
    @pytest.mark.parametrize("layout", ["flat", "bin"])
    def test_no_pip_no_hf_bootstraps_uv(self, sandbox, layout):
        # e.g. plain Ubuntu: python3 but no pip.
        bin_dir, trace, tmp = sandbox
        _add_curl(bin_dir, trace, tmp, layout=layout)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        calls = _calls(trace)
        assert calls[0].startswith("curl -LsSf https://astral.sh/uv/install.sh")
        assert calls[1:] == [
            f"uvx --from {_INLINE_HF_CLI_SPEC} hf {a}"
            for a in _expected_download_args()
        ]
        assert "bootstrapping one with uv" in proc.stderr

    def test_failing_pip_falls_back_to_uv(self, sandbox):
        # e.g. a PEP 668 "externally managed" interpreter.
        bin_dir, trace, tmp = sandbox
        _add_pip(bin_dir, trace, works=False)
        _add_curl(bin_dir, trace, tmp)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        assert "pip install of" in proc.stderr and "failed" in proc.stderr
        assert any(c.startswith("uvx --from") for c in _calls(trace))

    def test_unreachable_installer_fails_clearly(self, sandbox):
        bin_dir, trace, tmp = sandbox
        _add_curl(bin_dir, trace, tmp, works=False)
        proc = _run(bin_dir, tmp)
        assert proc.returncode != 0
        assert "could not bootstrap uv" in proc.stderr
        assert not any("download" in c for c in _calls(trace))


class TestNothingUsable:
    def test_fails_naming_what_is_missing_before_any_download(self, sandbox):
        bin_dir, trace, tmp = sandbox
        proc = _run(bin_dir, tmp)
        assert proc.returncode != 0
        assert "no 'hf', no working 'pip' and no 'curl'" in proc.stderr
        assert _calls(trace) == []


class TestDownloadFailure:
    def test_a_failed_download_fails_setup(self, sandbox):
        bin_dir, trace, tmp = sandbox
        _stub(bin_dir, "hf", "exit 3")
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 3


class TestInjectedIntoTheLaunchedSetup:
    """The block reaches sky.Task's setup, after the `set -eu` prologue."""

    @pytest.mark.asyncio
    async def test_inline_binding_is_prepended_to_the_step_setup(self):
        from unittest.mock import patch

        from libgbtest.environments.skypilot_mocks import _make_env, _mock_sky

        env = _make_env({"default_cloud": "kubernetes"})
        mock_sky = _mock_sky()
        bindings = {
            "docs": {
                "binding": {"path": _PULLS["docs"]["path"]},
                "_hfpull": _PULLS["docs"],
            }
        }
        with (
            patch("gbserver.environment.skypilot.sky", mock_sky),
            patch("gbserver.environment.skypilot.HAS_SKYPILOT", True),
        ):
            env._get_launch_ready_event("hf-1")
            await env.launch_skypilot(
                launch_id="hf-1",
                launcher_config={"run": "true", "setup": "echo step-setup"},
                config={},
                bindings=bindings,
            )
        setup = mock_sky.Task.call_args.kwargs["setup"]
        assert setup.startswith("set -eu\n")
        assert setup.index("# -- gbserver: inline hfpull for inputs --") < setup.index(
            "echo step-setup"
        )
        assert (
            'gb_hf download "ibm-research/vira-intents-live"' in setup
        ), "download must use the resolved client"
