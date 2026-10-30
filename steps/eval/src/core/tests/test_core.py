import json

import pytest
from granite_evals import cli, data, registry, suites
from granite_evals.registry import Benchmark, RunConfig
from granite_evals.results import write_results
from granite_evals.sandbox import enroot
from granite_evals.sandbox.enroot import enroot_uri
from granite_evals.serving import ServerConfig, VLLMServer, server_env


@pytest.mark.parametrize(
    "image,uri",
    [
        (
            "swebench/sweb.eval.x86_64.a_1776_b-1:latest",
            "docker://swebench/sweb.eval.x86_64.a_1776_b-1:latest",
        ),
        ("docker.io/swebench/x:latest", "docker://swebench/x:latest"),
        ("us.icr.io/ns/img:1.0", "docker://us.icr.io#ns/img:1.0"),
        ("localhost:5000/img", "docker://localhost:5000#img"),
    ],
)
def test_enroot_uri(image, uri):
    assert enroot_uri(image) == uri


def test_enroot_sandbox_keeps_tmp_between_commands(tmp_path, monkeypatch):
    # enroot mounts a fresh tmpfs on /tmp per `start`; the sandbox binds its own dir.
    monkeypatch.setenv("ENROOT_TEMP_PATH", str(tmp_path))
    monkeypatch.setattr(enroot, "ensure_squashfs", lambda image: tmp_path / "x.sqsh")
    monkeypatch.setattr(enroot.subprocess, "run", lambda *a, **k: None)
    sb = enroot.EnrootSandbox("img:latest")
    sb.start()
    argv = sb._exec_argv("true", "/testbed", {})
    assert argv[argv.index("--mount") + 1] == f"{sb.tmp}:/tmp"
    assert sb.tmp.parent == tmp_path and sb.tmp.is_dir()
    assert sb._run_env()["NVIDIA_VISIBLE_DEVICES"] == "void"
    sb.close()
    assert not sb.tmp.exists()


def test_enroot_sandbox_localhost_is_ipv4(tmp_path, monkeypatch):
    # Shared host network: localhost -> ::1 reaches nothing a task bound on IPv4.
    monkeypatch.setenv("ENROOT_TEMP_PATH", str(tmp_path))
    monkeypatch.setattr(enroot, "ensure_squashfs", lambda image: tmp_path / "x.sqsh")
    monkeypatch.setattr(enroot.subprocess, "run", lambda *a, **k: None)
    sb = enroot.EnrootSandbox("img:latest")
    sb.start()
    argv = sb._exec_argv("true", "/", {})
    mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "--mount"]
    assert mounts == [f"{sb.tmp}:/tmp", f"{sb.hosts}:/etc/hosts"]
    assert "127.0.0.1" in sb.hosts.read_text()
    sb.close()
    assert not sb.hosts.exists()


def test_ipv4_localhost():
    bluevela = (
        "127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4\n"
        "::1         localhost localhost.localdomain localhost6 localhost6.localdomain6\n"
        "10.0.0.5 node5 # compute\n"
    )
    assert enroot.ipv4_localhost(bluevela) == (
        "127.0.0.1   localhost localhost.localdomain localhost4 localhost4.localdomain4\n"
        "::1 localhost.localdomain localhost6 localhost6.localdomain6\n"
        "10.0.0.5 node5 # compute\n"
    )
    assert enroot.ipv4_localhost("::1 localhost\n") == "127.0.0.1 localhost\n"
    assert enroot.ipv4_localhost("") == "127.0.0.1 localhost\n"


def test_enroot_sandbox_kills_its_leftovers_before_remove(tmp_path, monkeypatch):
    # No PID namespace: a daemon that rewrites its environment (redis-server) would
    # outlive the sandbox and answer the next task on its port.
    monkeypatch.setenv("ENROOT_DATA_PATH", str(tmp_path / "data"))
    calls = []
    monkeypatch.setattr(
        enroot, "_kill_rooted", lambda rootfs: calls.append(("kill", rootfs)) or 1
    )
    monkeypatch.setattr(
        enroot.subprocess, "run", lambda argv, **k: calls.append(("run", argv[1]))
    )
    sb = enroot.EnrootSandbox("img:latest")
    sb.close()
    assert calls == [("kill", str(tmp_path / "data" / sb.name)), ("run", "remove")]


def test_enroot_sandbox_gets_no_secrets(monkeypatch):
    # enroot start passes its environment into the sandbox, where model-written code runs.
    for name in (
        "GRANITE_EVALS_JUDGE_API_KEY",
        "GRANITE_EVALS_USER_API_KEY",
        "HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
    ):
        monkeypatch.setenv(name, "s3cret")
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", "/tmp/ledger.jsonl")
    sb = enroot.EnrootSandbox("img:latest", env={"HF_TOKEN": "explicit"})
    env = sb._run_env()
    assert (
        "s3cret" not in env.values()
        and env["PATH"]
        and env["GRANITE_EVALS_SPEND_LEDGER"]
    )
    argv = sb._exec_argv("true", "/", sb.env)
    assert (
        argv[argv.index("--env") + 1] == "HF_TOKEN=explicit"
    )  # explicit env still reaches it


def test_take_is_stable_and_limited():
    rows = [{"id": "b"}, {"id": "c"}, {"id": "a"}]
    assert [r["id"] for r in data.take(rows, 2, key="id")] == ["a", "b"]
    assert len(data.take(rows, None, key="id")) == 3


def test_dataset_source_pin_and_override(tmp_path):
    cls = registry.get("hmmt-feb25")
    pinned = cls(RunConfig(model="m", output_dir=tmp_path))
    assert pinned.dataset_source() == (cls.dataset, cls.dataset_revision)
    assert len(cls.dataset_revision) == 40
    other = cls(RunConfig(model="m", output_dir=tmp_path, dataset="/local/ds"))
    assert other.dataset_source() == ("/local/ds", None)


def test_every_benchmark_pins_its_dataset():
    for bid, cls in registry.all_benchmarks().items():
        assert cls.dataset, bid
        assert cls.dataset_revision or cls.dataset.startswith(("http", "git+")), bid


def test_suites_reference_unique_ids():
    for name in suites.names():
        ids = [b["id"] for b in suites.load(name)["benchmarks"]]
        assert len(ids) == len(set(ids)), name


def test_every_implemented_benchmark_is_in_a_suite():
    listed = {
        b["id"] for name in suites.names() for b in suites.load(name)["benchmarks"]
    }
    assert set(registry.all_benchmarks()) <= listed


class _Dummy(Benchmark):
    id = "dummy"
    metric = "accuracy"

    def run(self, base_url, served_model_name):
        return {"value": 0.5, "n": 2, "extra": 1}


def test_results_schema(tmp_path):
    b = _Dummy(RunConfig(model="m", output_dir=tmp_path, limit=2))
    path = write_results(b, b.run("", "m"), started=0.0, served_model_name="m")
    record = json.loads(path.read_text())
    assert record["benchmark"] == "dummy"
    assert record["value"] == 0.5 and record["n"] == 2
    assert record["smoke"] is True and record["details"] == {"extra": 1}


def _granite42(d):
    # Excerpts of ibm-granite/granite-4.2-3b's chat_template.jinja and plugin.
    d.mkdir()
    (d / "chat_template.jinja").write_text(
        "<tool_call>\n<function={{ tool_call.name }}>\n<parameter="
    )
    (d / "granite_thinking_parser.py").write_text(
        '@ReasoningParserManager.register_module("granite_thinking_parser")\nclass GraniteThinkingParser: ...'
    )
    return d


def test_vllm_command_resolves_granite42_parsers(tmp_path):
    m = _granite42(tmp_path / "m")
    cmd = VLLMServer(
        ServerConfig(model=str(m), served_model_name="g"), tmp_path / "log"
    ).command()
    assert cmd[:3] == ["vllm", "serve", str(m)]
    assert "--enable-auto-tool-choice" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "qwen3_coder"
    assert cmd[cmd.index("--reasoning-parser") + 1] == "granite_thinking_parser"
    assert cmd[cmd.index("--reasoning-parser-plugin") + 1] == str(
        m / "granite_thinking_parser.py"
    )


def test_vllm_command_parsers_explicit_and_undetected(tmp_path):
    m = _granite42(tmp_path / "m")
    cfg = ServerConfig(
        model=str(m),
        served_model_name="g",
        tool_call_parser="hermes",
        reasoning_parser="",
    )
    cmd = VLLMServer(cfg, tmp_path / "log").command()
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert "--reasoning-parser" not in cmd
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "tokenizer_config.json").write_text(
        '{"chat_template": "{{ messages }}"}'
    )
    cmd = VLLMServer(
        ServerConfig(model=str(tmp_path / "plain"), served_model_name="g"),
        tmp_path / "log",
    ).command()
    assert "--tool-call-parser" not in cmd and "--reasoning-parser" not in cmd


def test_vllm_env_avoids_jit_sampler(monkeypatch):
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    assert server_env()["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    assert server_env()["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_cli_list(capsys):
    assert cli.main(["list", "--suite", "granite42"]) == 0
    out = capsys.readouterr().out
    # aime25 lives in the shared core, so it is registered in every family image
    assert "[x] aime25" in out


def test_dockerfile_extras_run_without_project_uv_config():
    """A family's extras.sh runs from the project dir; UV_NO_CONFIG keeps a separate
    env they build from inheriting this project's [tool.uv] overrides."""
    from pathlib import Path

    root = Path(__file__).parents[3]
    dockerfiles = sorted(root.glob("*/Dockerfile.*"))
    if not dockerfiles:  # the image build runs the tests without the family dirs
        pytest.skip("no family Dockerfiles (inside the image)")
    for dockerfile in dockerfiles:
        # Re-join continuation lines so a RUN and its command read as one string.
        runs = []
        for line in dockerfile.read_text().splitlines():
            if line.startswith("RUN"):
                runs.append(line)
            elif runs and runs[-1].endswith("\\"):
                runs[-1] += "\n" + line
        for run in runs:
            if "/tmp/extras/" in run and ".sh" in run:
                assert "UV_NO_CONFIG=1 bash" in run, dockerfile.name
