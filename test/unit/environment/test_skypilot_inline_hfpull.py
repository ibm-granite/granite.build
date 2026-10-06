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
* no pip and no hf (plain Ubuntu), or a pip that fails (PEP 668) -> the pinned
  uv release fetched with curl, sha256-verified, unpacked, and the client run via
  ``uvx --from <pinned spec>``. A wrong checksum, a missing checksum tool, an
  unknown CPU architecture or a failed download all refuse rather than run it.
* nothing usable -> fail naming what is missing, before any download.

The fake curl delivers a REAL ``.tar.gz`` (built here) holding a stub ``uvx``, and
the pinned checksum is patched to match it, so the block's own sha256 check,
``tar`` unpack and arch detection are what is exercised.
"""

import hashlib
import io
import pathlib
import shutil
import subprocess
import tarfile

import pytest

from gbserver.environment import skypilot as sky_env
from gbserver.environment.skypilot import (
    _INLINE_HF_CLI_SPEC,
    _INLINE_HF_UV_VERSION,
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
# Real tools the block needs; everything else on PATH is a stub, so
# `command -v pip/hf/curl` sees only what each test provides. `uname` is stubbed
# per test to control the detected architecture.
_REAL_TOOLS = ("sh", "env", "mkdir", "chmod", "cat", "cut", "tar", "gzip")
_SHA_TOOLS = ("sha256sum", "shasum")
# The fake curl only gets the narrowed PATH, which has no cp, so it calls it by
# absolute path.
_CP = shutil.which("cp") or "/bin/cp"


def _stub(bin_dir: pathlib.Path, name: str, body: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!{_BASH}\n{body}\n")
    path.chmod(0o755)


def _link_real(bin_dir: pathlib.Path, tools) -> list[str]:
    linked = []
    for tool in tools:
        real = shutil.which(tool)
        if real:
            (bin_dir / tool).symlink_to(real)
            linked.append(tool)
    return linked


@pytest.fixture
def sandbox(tmp_path):
    """A bin dir holding only real essentials (incl. a sha256 tool); tests add stubs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    missing = [
        t
        for t in ("sh", "env", "mkdir", "chmod", "cat", "cut", "tar")
        if not shutil.which(t)
    ]
    assert not missing, f"test needs {missing}"
    _link_real(bin_dir, _REAL_TOOLS)
    assert _link_real(bin_dir, _SHA_TOOLS), "test needs sha256sum or shasum"
    _stub(bin_dir, "uname", "echo x86_64")
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


def _uv_tarball(tmp_path: pathlib.Path, arch: str, trace: pathlib.Path) -> bytes:
    """A real uv-<arch>-unknown-linux-musl.tar.gz whose uvx records its argv."""
    uvx = f'#!{_BASH}\necho "uvx $*" >> {trace}\n'.encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(f"uv-{arch}-unknown-linux-musl/uvx")
        info.size, info.mode = len(uvx), 0o755
        tf.addfile(info, io.BytesIO(uvx))
    data = buf.getvalue()
    (tmp_path / f"uv-{arch}.tar.gz").write_bytes(data)
    return data


def _add_curl(
    bin_dir, trace, tmp_path, monkeypatch, arch="x86_64", works=True, pin_matches=True
):
    """curl -fsSLo <out> <url>: copy the fake release tarball to <out>."""
    data = _uv_tarball(tmp_path, arch, trace)
    if pin_matches:
        monkeypatch.setitem(
            sky_env._INLINE_HF_UV_SHA256, arch, hashlib.sha256(data).hexdigest()
        )
    _stub(
        bin_dir,
        "curl",
        f'echo "curl $3" >> {trace}; '
        + (f'{_CP} {tmp_path}/uv-{arch}.tar.gz "$2"' if works else "exit 22"),
    )


def _run(bin_dir, tmp_path, pulls=None):
    script = "set -eu\n" + _inline_hfpull_setup_block(pulls or _PULLS)
    env = {"PATH": str(bin_dir), "TMPDIR": str(tmp_path), "HOME": str(tmp_path)}
    return subprocess.run(
        [_BASH, "-c", script], capture_output=True, text=True, env=env, check=False
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


def _uv_url(arch):
    return (
        f"https://github.com/astral-sh/uv/releases/download/{_INLINE_HF_UV_VERSION}"
        f"/uv-{arch}-unknown-linux-musl.tar.gz"
    )


class TestBlockIsValidShell:
    def test_parses_under_set_eu(self):
        script = "set -eu\n" + _inline_hfpull_setup_block(_PULLS)
        proc = subprocess.run([_BASH, "-n"], input=script, text=True, check=False)
        assert proc.returncode == 0

    @pytest.mark.skipif(shutil.which("shellcheck") is None, reason="no shellcheck")
    def test_shellcheck_is_clean(self, tmp_path):
        f = tmp_path / "block.sh"
        f.write_text("set -eu\n" + _inline_hfpull_setup_block(_PULLS))
        proc = subprocess.run(["shellcheck", "-s", "bash", str(f)], check=False)
        assert proc.returncode == 0

    def test_keeps_the_pinned_client_spec(self):
        assert f"gb_hf_spec='{_INLINE_HF_CLI_SPEC}'" in _inline_hfpull_setup_block(
            _PULLS
        )

    def test_never_pipes_a_remote_script_into_a_shell(self):
        block = _inline_hfpull_setup_block(_PULLS)
        assert "| sh" not in block and "install.sh" not in block
        for sha in sky_env._INLINE_HF_UV_SHA256.values():
            assert sha in block, "every pinned checksum must be in the block"


class TestPipRouteIsUnchanged:
    """Backwards compatibility: images with pip behave exactly as before."""

    def test_pip_installs_pinned_client_then_hf_downloads(self, sandbox, monkeypatch):
        bin_dir, trace, tmp = sandbox
        _add_pip(bin_dir, trace)
        _add_curl(bin_dir, trace, tmp, monkeypatch)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        calls = _calls(trace)
        assert calls[0] == f"pip install --no-cache-dir {_INLINE_HF_CLI_SPEC}"
        assert calls[1:] == [f"hf {a}" for a in _expected_download_args()]
        assert not any(c.startswith("curl") for c in calls), "must not fetch uv"


class TestPreinstalledHf:
    def test_hf_without_pip_is_used_directly(self, sandbox, monkeypatch):
        bin_dir, trace, tmp = sandbox
        _add_hf(bin_dir, trace)
        _add_curl(bin_dir, trace, tmp, monkeypatch)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        assert _calls(trace) == [f"hf {a}" for a in _expected_download_args()]


class TestPinnedUvRoute:
    @pytest.mark.parametrize(
        "uname,arch",
        [
            ("x86_64", "x86_64"),
            ("amd64", "x86_64"),
            ("aarch64", "aarch64"),
            ("arm64", "aarch64"),
        ],
    )
    def test_no_pip_no_hf_fetches_verified_pinned_uv(
        self, sandbox, monkeypatch, uname, arch
    ):
        # e.g. plain Ubuntu: python3 but no pip.
        bin_dir, trace, tmp = sandbox
        _stub(bin_dir, "uname", f"echo {uname}")
        _add_curl(bin_dir, trace, tmp, monkeypatch, arch=arch)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        calls = _calls(trace)
        assert calls[0] == f"curl {_uv_url(arch)}"
        assert calls[1:] == [
            f"uvx --from {_INLINE_HF_CLI_SPEC} hf {a}"
            for a in _expected_download_args()
        ]
        assert f"fetching pinned uv {_INLINE_HF_UV_VERSION}" in proc.stderr

    def test_failing_pip_falls_back_to_uv(self, sandbox, monkeypatch):
        # e.g. a PEP 668 "externally managed" interpreter.
        bin_dir, trace, tmp = sandbox
        _add_pip(bin_dir, trace, works=False)
        _add_curl(bin_dir, trace, tmp, monkeypatch)
        proc = _run(bin_dir, tmp)
        assert proc.returncode == 0, proc.stderr
        assert "pip install of" in proc.stderr and "failed" in proc.stderr
        assert any(c.startswith("uvx --from") for c in _calls(trace))

    def test_checksum_mismatch_refuses_to_run_it(self, sandbox, monkeypatch):
        bin_dir, trace, tmp = sandbox
        _add_curl(bin_dir, trace, tmp, monkeypatch, pin_matches=False)
        proc = _run(bin_dir, tmp)
        assert proc.returncode != 0
        assert "failed sha256 verification" in proc.stderr
        # Only the curl fetch ran; the unverified uvx never did.
        assert not any(c.startswith(("uvx", "hf ")) for c in _calls(trace))

    def test_no_checksum_tool_refuses_to_run_it(self, sandbox, monkeypatch):
        bin_dir, trace, tmp = sandbox
        for tool in _SHA_TOOLS:
            (bin_dir / tool).unlink(missing_ok=True)
        _add_curl(bin_dir, trace, tmp, monkeypatch)
        proc = _run(bin_dir, tmp)
        assert proc.returncode != 0
        assert "no sha256 tool" in proc.stderr
        assert not any(c.startswith("uvx") for c in _calls(trace))

    def test_unsupported_arch_fails_before_any_download(self, sandbox, monkeypatch):
        bin_dir, trace, tmp = sandbox
        _stub(bin_dir, "uname", "echo ppc64le")
        _add_curl(bin_dir, trace, tmp, monkeypatch)
        proc = _run(bin_dir, tmp)
        assert proc.returncode != 0
        assert "no pinned uv build for ppc64le" in proc.stderr
        assert _calls(trace) == []

    def test_failed_download_fails_clearly(self, sandbox, monkeypatch):
        bin_dir, trace, tmp = sandbox
        _add_curl(bin_dir, trace, tmp, monkeypatch, works=False)
        proc = _run(bin_dir, tmp)
        assert proc.returncode != 0
        assert "could not download" in proc.stderr
        assert not any(c.startswith("uvx") for c in _calls(trace))


class TestNothingUsable:
    def test_fails_naming_what_is_missing_before_any_download(self, sandbox):
        bin_dir, trace, tmp = sandbox
        proc = _run(bin_dir, tmp)
        assert proc.returncode != 0
        assert "no 'hf', no working 'pip' and no 'curl'" in proc.stderr
        assert _calls(trace) == []


class TestQuoting:
    def test_values_with_shell_metacharacters_arrive_as_literal_args(self, sandbox):
        # Each value is shlex-quoted: a space stays one word, `$`/backticks never
        # expand. The stub records one argument per line.
        bin_dir, trace, tmp = sandbox
        _stub(bin_dir, "hf", f'for a in "$@"; do echo "ARG:$a"; done >> {trace}')
        odd = {
            "x": {
                "repo": "org/re po",
                "path": "/tmp/a b/$HOME/`id`",
                "revision": "rev$x",
                "type": "dataset",
            }
        }
        proc = _run(bin_dir, tmp, pulls=odd)
        assert proc.returncode == 0, proc.stderr
        assert _calls(trace) == [
            "ARG:download",
            "ARG:org/re po",
            "ARG:--local-dir",
            "ARG:/tmp/a b/$HOME/`id`",
            "ARG:--revision",
            "ARG:rev$x",
            "ARG:--repo-type",
            "ARG:dataset",
        ]


class TestDownloadFailure:
    def test_a_failed_download_fails_setup(self, sandbox):
        bin_dir, _, tmp = sandbox
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
            "gb_hf download ibm-research/vira-intents-live" in setup
        ), "download must use the resolved client"
