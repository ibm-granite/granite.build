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

"""Shared harnesses for this step's tests: run.sh against a stub mergekit-yaml, and
mergekit_prepare.main(). A plain module, not a conftest.py, so `make test-mergekit`
(which runs with --noconftest) imports it too.
"""

import json
import os
import pathlib
import subprocess
import sys

import mergekit_prepare as mp
import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_RUN = _SRC / "run.sh"

_RECIPE = {
    "merge_method": "linear",
    "models": [
        {"model": "model_1", "parameters": {"weight": 0.5}},
        {"model": "base_model", "parameters": {"weight": 0.5}},
    ],
}

# Records argv, then writes what a merge writes, unless told to fail or write nothing.
_STUB = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$STUB_ARGV"
[ -n "${STUB_STDERR:-}" ] && echo "$STUB_STDERR" >&2
[ -n "${STUB_STDOUT:-}" ] && echo "$STUB_STDOUT"
[ -n "${STUB_FD3_PROBE:-}" ] && { [ -e /dev/fd/3 ] && echo fd3-open >&2 || echo fd3-closed >&2; }
[ -n "${STUB_FAIL:-}" ] && { echo "stub: merge failed" >&2; exit 3; }
out="$2"
[ -n "${STUB_EMPTY:-}" ] && { echo '{}' > "$out/config.json"; exit 0; }
"$STUB_PYTHON" - "$out" <<'PY'
import json, struct, sys
hdr = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
with open(sys.argv[1] + "/model.safetensors", "wb") as f:
    f.write(struct.pack("<Q", len(hdr)) + hdr + b"\\0\\0\\0\\0")
open(sys.argv[1] + "/config.json", "w").write("{}")
PY
# A tokenizer, as a successful --copy-tokenizer leaves one, unless told not to (MergeKit
# swallows tokenizer errors); a chat template only when asked for.
if [ -z "${STUB_NO_TOKENIZER:-}" ]; then
  [ -z "${STUB_NO_TOK_CONFIG:-}" ] && echo '{}' > "$out/tokenizer_config.json"
  if [ -n "${STUB_VOCAB:-}" ]; then echo x > "$out/$STUB_VOCAB"
  elif [ -n "${STUB_SPM:-}" ]; then echo x > "$out/tokenizer.model"; else echo '{}' > "$out/tokenizer.json"; fi
  [ -n "${STUB_TEMPLATE:-}" ] && printf '%s' "$STUB_TEMPLATE" > "$out/chat_template.jinja"
fi
exit 0
"""


def _real_env() -> dict:
    """The env for tests that run real mergekit-yaml: this interpreter's bin dir first.

    So the mergekit-yaml and python run.sh finds are the ones whose mergekit these tests
    imported, not another install earlier on PATH (a pipx mergekit, a system python).
    """
    return {
        **os.environ,
        "PATH": f"{pathlib.Path(sys.executable).parent}:{os.environ['PATH']}",
        "PYTHONPATH": str(_SRC),
    }


def _model(path: pathlib.Path) -> str:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return str(path)


@pytest.fixture(name="env")
def env_fixture(tmp_path):
    """A workdir, two staged models, and a PATH whose mergekit-yaml is the stub."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "mergekit-yaml"
    stub.write_text(_STUB)
    stub.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    # run.sh calls `python`: put the test interpreter's own bin dir next on PATH (not a
    # symlink to it, which would lose its venv's site-packages and so PyYAML).
    pybin = pathlib.Path(sys.executable).parent
    return {
        "tmp": tmp_path,
        "work": work,
        "m1": _model(tmp_path / "staged" / "m1"),
        "base": _model(tmp_path / "staged" / "base"),
        "argv_file": tmp_path / "argv.txt",
        "env": {
            **os.environ,
            "PATH": f"{bindir}:{pybin}:{os.environ['PATH']}",
            "STUB_ARGV": str(tmp_path / "argv.txt"),
            "STUB_PYTHON": sys.executable,
            "PYTHONPATH": str(_SRC),
        },
    }


def _run_argv(env, output="./out/model", recipe=None, args=None, **over) -> list[str]:
    """run.sh's argv (without the leading ``bash``), with defaults for every option."""
    opts = {
        "venv": "",
        "input-model_1": env["m1"],
        "input-model_2": "",
        "input-model_3": "",
        "input-model_4": "",
        "input-base_model": env["base"],
        "merge-config-json": json.dumps(_RECIPE if recipe is None else recipe),
        "args-json": json.dumps(args or {}),
        "copy-tokenizer": "true",
        "lazy-unpickle": "true",
        "allow-crimes": "false",
        "trust-remote-code": "false",
        "output-model": output,
        **over,
    }
    argv = [str(_RUN)]
    for k, v in opts.items():
        argv += [f"--{k}", v]
    return argv


def _run(env, output="./out/model", recipe=None, args=None, extra_env=None, **over):
    return subprocess.run(
        ["bash", *_run_argv(env, output, recipe, args, **over)],
        cwd=env["work"],
        env={**env["env"], **(extra_env or {})},
        capture_output=True,
        text=True,
        check=False,
    )


def _markers(proc) -> list[str]:
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("GB_")]


def _stub_argv(env) -> list[str]:
    return env["argv_file"].read_text().splitlines()


def _refused(proc, message):
    assert proc.returncode != 0
    assert message in proc.stderr, proc.stderr
    assert not any(ln.startswith("GB_ARTIFACT") for ln in proc.stdout.splitlines())


_STAGED = {"model_1": "/staged/m1", "base_model": "/staged/base"}


def _main(tmp_path, merge_config, args=None, bound=None, **flags):
    """Run main(). ``bound`` defaults to whichever of model_1 / base_model the recipe's
    text mentions (a plain text search, not the code under test), since a bound input
    the recipe never names is refused."""
    recipe = tmp_path / "merge.yaml"
    text = json.dumps(merge_config)
    if bound is None:
        bound = {n: p for n, p in _STAGED.items() if n in text}
    argv = [
        "--merge-config-json",
        text,
        "--args-json",
        json.dumps(args or {}),
        "--recipe-out",
        str(recipe),
        "--expect-out",
        str(tmp_path / "expect"),
    ]
    for name, path in bound.items():
        argv += [f"--input-{name}", path]
    defaults = {
        "copy-tokenizer": "true",
        "lazy-unpickle": "true",
        "allow-crimes": "false",
        "trust-remote-code": "false",
    }
    for k, v in {**defaults, **flags}.items():
        argv += [f"--{k}", v]
    return mp.main(argv), recipe
