import importlib.util
import os
import socket
import stat
import subprocess
from pathlib import Path

import pytest

STEP_DIR = Path(__file__).resolve().parents[1]
SCRIPT = STEP_DIR / "src" / "run-bfcl.sh"

# Most of what this file asserts is pure shell -- argument parsing, the server lifecycle,
# and which subcommands get invoked -- and runs against stub `bfcl`/`vllm` binaries with
# nothing installed. Three paths are different: run-bfcl.sh delegates id selection to
# sample_test_ids.py, shard_test_ids.py and resolve_test_categories.py, all of which
# import the bfcl_eval harness to enumerate the real corpus. The harness is a dependency
# of this step's IMAGE, not of this repository (see conftest.py), so in a plain checkout
# those three helpers exit 1 and the script with them. Skip rather than stub: a stub
# helper would assert that run-bfcl.sh calls something, which is not the interesting part,
# while hiding whether the real category logic still works.
needs_harness = pytest.mark.skipif(
    importlib.util.find_spec("bfcl_eval") is None,
    reason="run-bfcl.sh's id-selection helpers import bfcl_eval, an image dependency",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_stub_bfcl(bin_dir: Path, calls_file: Path) -> None:
    stub = bin_dir / "bfcl"
    stub.write_text("#!/usr/bin/env bash\n" f'echo "$@" >> "{calls_file}"\n' "exit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


def _write_stub_vllm(path: Path, calls_file: Path) -> None:
    # A real `vllm serve` binary, standing in for the one baked into
    # /opt/vllm-serve-venv by the Dockerfile: records its argv, then answers
    # 200 on any GET so run-bfcl.sh's readiness poll against /v1/models
    # succeeds, and keeps serving until run-bfcl.sh's cleanup trap kills it.
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "\n"
        f"with open({str(calls_file)!r}, 'a') as f:\n"
        "    f.write(' '.join(sys.argv[1:]) + '\\n')\n"
        "\n"
        "args = sys.argv[1:]\n"
        "port = int(args[args.index('--port') + 1])\n"
        "\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200)\n"
        "        self.end_headers()\n"
        "\n"
        "    def log_message(self, *a):\n"
        "        pass\n"
        "\n"
        "HTTPServer(('127.0.0.1', port), Handler).serve_forever()\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_run_bfcl_starts_server_then_invokes_generate_then_evaluate(tmp_path):
    bfcl_calls_file = tmp_path / "bfcl_calls.log"
    bfcl_calls_file.write_text("")
    _write_stub_bfcl(tmp_path, bfcl_calls_file)

    vllm_calls_file = tmp_path / "vllm_calls.log"
    vllm_calls_file.write_text("")
    vllm_stub = tmp_path / "vllm-stub"
    _write_stub_vllm(vllm_stub, vllm_calls_file)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["VLLM_SERVE_BIN"] = str(vllm_stub)
    env["BFCL_BIN"] = str(tmp_path / "bfcl")

    output_dir = tmp_path / "out"
    port = _free_port()
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/model",
            "--model-id",
            "ibm-granite/granite-4",
            "--test-categories",
            "simple",
            "--num-gpus-generate",
            "1",
            "--num-gpus-evaluate",
            "1",
            "--gpu-memory-utilization",
            "0.5",
            "--output-dir",
            str(output_dir),
            "--vllm-port",
            str(port),
        ],
        env=env,
        check=True,
        timeout=30,
    )

    vllm_calls = vllm_calls_file.read_text().splitlines()
    assert len(vllm_calls) == 1
    assert vllm_calls[0].startswith("serve /model")
    assert f"--port {port}" in vllm_calls[0]
    assert "--tensor-parallel-size 1" in vllm_calls[0]

    bfcl_calls = bfcl_calls_file.read_text().splitlines()
    assert len(bfcl_calls) == 2
    assert bfcl_calls[0].startswith("generate")
    assert "--local-model-path /model" in bfcl_calls[0]
    assert "--skip-server-setup" in bfcl_calls[0]
    assert "--backend" not in bfcl_calls[0]
    assert bfcl_calls[1].startswith("evaluate")
    assert "--model ibm-granite/granite-4" in bfcl_calls[1]
    assert output_dir.is_dir()


@needs_harness
def test_run_bfcl_sample_fraction_uses_run_ids_and_partial_eval(tmp_path):
    bfcl_calls_file = tmp_path / "bfcl_calls.log"
    bfcl_calls_file.write_text("")
    _write_stub_bfcl(tmp_path, bfcl_calls_file)

    vllm_calls_file = tmp_path / "vllm_calls.log"
    vllm_calls_file.write_text("")
    vllm_stub = tmp_path / "vllm-stub"
    _write_stub_vllm(vllm_stub, vllm_calls_file)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["VLLM_SERVE_BIN"] = str(vllm_stub)
    env["BFCL_BIN"] = str(tmp_path / "bfcl")

    output_dir = tmp_path / "out"
    port = _free_port()
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/model",
            "--model-id",
            "ibm-granite/granite-4",
            "--test-categories",
            "all",
            "--output-dir",
            str(output_dir),
            "--vllm-port",
            str(port),
            "--sample-fraction",
            "0.25",
            "--sample-seed",
            "7",
        ],
        env=env,
        check=True,
        timeout=60,
    )

    assert (output_dir / "test_case_ids_to_generate.json").is_file()

    bfcl_calls = bfcl_calls_file.read_text().splitlines()
    assert len(bfcl_calls) == 2
    assert "--run-ids" in bfcl_calls[0]
    assert "evaluate" in bfcl_calls[1] and "--partial-eval" in bfcl_calls[1]


@needs_harness
def test_run_bfcl_shard_generates_with_run_ids_and_skips_evaluate(tmp_path):
    bfcl_calls_file = tmp_path / "bfcl_calls.log"
    bfcl_calls_file.write_text("")
    _write_stub_bfcl(tmp_path, bfcl_calls_file)

    vllm_calls_file = tmp_path / "vllm_calls.log"
    vllm_calls_file.write_text("")
    vllm_stub = tmp_path / "vllm-stub"
    _write_stub_vllm(vllm_stub, vllm_calls_file)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["VLLM_SERVE_BIN"] = str(vllm_stub)
    env["BFCL_BIN"] = str(tmp_path / "bfcl")

    output_dir = tmp_path / "out"
    port = _free_port()
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/model",
            "--model-id",
            "ibm-granite/granite-4",
            "--output-dir",
            str(output_dir),
            "--vllm-port",
            str(port),
            "--num-shards",
            "8",
            "--shard-index",
            "3",
            "--exclude-categories",
            "web_search",
        ],
        env=env,
        check=True,
        timeout=60,
    )

    assert (output_dir / "test_case_ids_to_generate.json").is_file()

    bfcl_calls = bfcl_calls_file.read_text().splitlines()
    # --num-shards implies --skip-evaluate: only the generate call happens.
    assert len(bfcl_calls) == 1
    assert bfcl_calls[0].startswith("generate")
    assert "--run-ids" in bfcl_calls[0]


def test_run_bfcl_rejects_sample_fraction_with_num_shards():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/model",
            "--model-id",
            "x",
            "--output-dir",
            "/tmp/bfcl-test-mutex",
            "--sample-fraction",
            "0.25",
            "--num-shards",
            "8",
            "--shard-index",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr


def test_run_bfcl_rejects_num_shards_without_shard_index():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-path",
            "/model",
            "--model-id",
            "x",
            "--output-dir",
            "/tmp/bfcl-test-missing-shard-index",
            "--num-shards",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "shard-index" in result.stderr


def test_run_bfcl_rejects_evaluate_only_with_num_shards():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-id",
            "x",
            "--output-dir",
            "/tmp/bfcl-test-evaluate-only-mutex",
            "--evaluate-only",
            "--num-shards",
            "8",
            "--shard-index",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "cannot be combined" in result.stderr


@needs_harness
def test_run_bfcl_evaluate_only_skips_server_and_generate_and_excludes_categories(
    tmp_path,
):
    bfcl_calls_file = tmp_path / "bfcl_calls.log"
    bfcl_calls_file.write_text("")
    _write_stub_bfcl(tmp_path, bfcl_calls_file)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["BFCL_BIN"] = str(tmp_path / "bfcl")

    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-id",
            "ibm-granite/granite-4",
            "--output-dir",
            str(output_dir),
            "--evaluate-only",
            "--exclude-categories",
            "web_search",
        ],
        env=env,
        check=True,
        timeout=30,
    )

    bfcl_calls = bfcl_calls_file.read_text().splitlines()
    assert len(bfcl_calls) == 1
    assert bfcl_calls[0].startswith("evaluate")
    assert "web_search_base" not in bfcl_calls[0]
    assert "web_search_no_snippet" not in bfcl_calls[0]
    assert "simple" in bfcl_calls[0]  # a real, non-excluded category made it through

    # bfcl_eval's `evaluate --test-category` is a typer Option that consumes
    # exactly ONE token per occurrence -- passing multiple categories as
    # separate space-separated words after a single --test-category (rather
    # than one comma-joined value) fails at the real CLI with "unexpected
    # extra argument(s)" for every category after the first. This stub can't
    # catch that itself (it just logs argv), so assert the structure directly:
    # exactly one "--test-category" token, followed by exactly one value
    # token containing every resolved category comma-joined.
    argv = bfcl_calls[0].split()
    test_category_positions = [
        i for i, tok in enumerate(argv) if tok == "--test-category"
    ]
    assert len(test_category_positions) == 1
    value = argv[test_category_positions[0] + 1]
    assert "," in value  # more than one category survived exclusion
    assert " " not in value


def test_run_bfcl_evaluate_only_does_not_require_model_path(tmp_path):
    bfcl_calls_file = tmp_path / "bfcl_calls.log"
    bfcl_calls_file.write_text("")
    _write_stub_bfcl(tmp_path, bfcl_calls_file)

    env = dict(os.environ)
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env["BFCL_BIN"] = str(tmp_path / "bfcl")

    output_dir = tmp_path / "merged"
    output_dir.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-id",
            "ibm-granite/granite-4",
            "--output-dir",
            str(output_dir),
            "--evaluate-only",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_run_bfcl_requires_model_path():
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--model-id",
            "x",
            "--output-dir",
            "/tmp/bfcl-test-missing-model-path",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "model-path" in result.stderr


def test_run_bfcl_rejects_unknown_flag():
    result = subprocess.run(
        ["bash", str(SCRIPT), "--not-a-real-flag", "x"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unknown argument" in result.stderr
