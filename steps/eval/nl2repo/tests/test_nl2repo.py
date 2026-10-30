"""NL2RepoBench adapter tests with a fake sandbox (no containers, no model, no network)."""

import base64
import io
import json
import tarfile
from pathlib import Path

import pytest
from granite_evals.benchmarks import nl2repo as nr
from granite_evals.registry import RunConfig, get
from granite_evals.sandbox import ExecResult

TASK_A_COMMANDS = ["pip install -e .", "pytest --continue-on-collection-errors tests/a"]
TASK_B_COMMANDS = ["pip install -e .", "pytest --continue-on-collection-errors tests/b"]


# -- fixtures and fakes ---------------------------------------------------------


def make_task(
    root: Path,
    name="a",
    count=100,
    commands=None,
    test_files=("test_a.py",),
    start="# spec\n",
) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "start.md").write_text(start)
    (d / "test_case_count.txt").write_text(str(count))
    (d / "test_commands.json").write_text(json.dumps(commands or TASK_A_COMMANDS))
    (d / "test_files.json").write_text(json.dumps(list(test_files)))
    return d


def two_tasks(root: Path) -> Path:
    make_task(
        root, name="a", count=100, commands=TASK_A_COMMANDS, test_files=("tests/a",)
    )
    make_task(
        root, name="b", count=50, commands=TASK_B_COMMANDS, test_files=("tests/b",)
    )
    (root / "task_difficulty.csv").write_text("a,b\n")  # not a dir: never a task
    return root


def tar_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def ws_tar_bytes(files: dict[str, bytes]) -> bytes:
    """A staged-out generation tar: members under a top-level workspace/ dir."""
    return tar_bytes({f"workspace/{name}": data for name, data in files.items()})


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def tar_members(data: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        return sorted(m.name for m in tar if m.isfile())


def make_tarball(files: dict[str, bytes]) -> bytes:
    """A GitHub-archive-shaped tarball: everything under <repo>-<commit>/."""
    prefix = f"NL2RepoBench-{nr.COMMIT}"
    return tar_bytes({f"{prefix}/{name}": data for name, data in files.items()})


STAGED_WS_B64 = b64(
    ws_tar_bytes(
        {
            "pkg/impl.py": b"impl\n",
            "pyproject.toml": b"[project]\n",
            "test_six.py": b"# gold\n",
        }
    )
)

RUN_OUTPUTS = {
    "openhands.core.main": ExecResult("agent log\n", 0),
    "workspace 2>/dev/null | base64": ExecResult(STAGED_WS_B64, 0),
    "trajectories 2>/dev/null | base64": ExecResult(
        b64(tar_bytes({"trajectories/t.json": b"{}"})), 0
    ),
    "tests/a": ExecResult("40 passed", 0),
    "tests/b": ExecResult("25 passed", 0),
}


class FakeSandbox:
    """Answers execute() from a needle -> ExecResult table; records everything."""

    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.commands, self.cwd, self.files = [], [], {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def write_file(self, path, content):
        self.files[path] = content

    def execute(self, command, *, cwd="/", timeout=None, env=None, stdin=None):
        self.commands.append(command)
        self.cwd.append(cwd)
        for needle, result in self.outputs.items():
            if needle in command:
                return result
        return ExecResult("", 0)


class SandboxFactory:
    """A make_sandbox stand-in that hands out FakeSandbox instances."""

    def __init__(self, outputs=None):
        self.outputs = outputs or {}
        self.sandboxes, self.images, self.envs = [], [], []

    def __call__(self, image, *, backend="", env=None):
        sb = FakeSandbox(self.outputs)
        self.sandboxes.append(sb)
        self.images.append(image)
        self.envs.append(env)
        return sb


@pytest.fixture
def bench(tmp_path):
    return nr.NL2RepoBench(RunConfig(model="m", output_dir=tmp_path))


# -- registration, options and pins ---------------------------------------------


def test_registered_with_its_pins():
    assert get("nl2repo") is nr.NL2RepoBench
    assert nr.NL2RepoBench.id == "nl2repo"
    assert nr.NL2RepoBench.splittable is True
    assert nr.NL2RepoBench.metric == "test pass rate"
    assert nr.NL2RepoBench.dataset == nr.REPO
    assert nr.NL2RepoBench.dataset_revision == nr.COMMIT
    assert nr.COMMIT == "781a1da1ee41fb8edb0bed22f586d69111610edf"
    assert nr.N_TASKS == 104
    assert (
        nr.TASKS_SHA256
        == "1de36960a0fc944c560dcb012adc4a874c22327d7bfe1de62e3c76afdea0197d"
    )


def test_gold_mode_needs_no_server(tmp_path):
    plain = nr.NL2RepoBench(RunConfig(model="m", output_dir=tmp_path))
    gold = nr.NL2RepoBench(
        RunConfig(model="m", output_dir=tmp_path, options={"agent": "gold"})
    )
    assert plain.needs_server() and not plain.gold
    assert gold.gold and not gold.needs_server()


def test_option_parsing(bench):
    assert bench.opt("agent", "model") == "model"
    assert bench.opt("max_iterations", 500) == 500
    bench.config.options["enable_browsing"] = "false"
    assert bench.opt("enable_browsing", True) is False
    bench.config.options["enable_browsing"] = "0"
    assert bench.opt("enable_browsing", True) is False
    bench.config.options["eval_timeout"] = "60"
    assert bench.opt("eval_timeout", 1800) == 60


# -- tree digest and Task -------------------------------------------------------


def test_tree_digest_is_order_independent_and_content_sensitive(tmp_path):
    for name in ("one", "two"):
        d = tmp_path / name
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "b.txt").write_text("b")
        (d / "a.txt").write_text("a")
    assert nr.tree_digest(tmp_path / "one") == nr.tree_digest(tmp_path / "two")
    (tmp_path / "two" / "a.txt").write_text("A")
    assert nr.tree_digest(tmp_path / "one") != nr.tree_digest(tmp_path / "two")


def test_task_reads_the_four_canonical_files(tmp_path):
    d = make_task(
        tmp_path,
        count=42,
        commands=["pip install -e .", "pytest t"],
        test_files=("t.py",),
    )
    task = nr.Task(d)
    assert task.name == "a"
    assert task.start_md == "# spec\n"
    assert task.test_case_count == 42
    assert task.commands == ["pip install -e .", "pytest t"]
    assert task.test_files == ["t.py"]
    assert task.image("ghcr.io/x", "1.0") == "ghcr.io/x/a:1.0"


def test_task_rejects_a_dir_with_missing_files(tmp_path):
    d = tmp_path / "broken"
    d.mkdir()
    (d / "start.md").write_text("x")
    with pytest.raises(RuntimeError, match="test_case_count"):
        nr.Task(d)


def test_task_rejects_json_that_is_not_a_list_of_strings(tmp_path):
    d = make_task(tmp_path)
    (d / "test_commands.json").write_text('{"pip": "install"}')
    with pytest.raises(RuntimeError, match="list of strings"):
        nr.Task(d).commands


# -- TaskSet: the launch-time fetch ----------------------------------------------

CANONICAL_FOUR = {
    "test_files/a/start.md": b"# spec\n",
    "test_files/a/test_case_count.txt": b"7",
    "test_files/a/test_commands.json": b'["pytest t"]',
    "test_files/a/test_files.json": b'["t.py"]',
}


def _pin_single_task(tmp_path, monkeypatch, files):
    """Pin N_TASKS/TASKS_SHA256 at a one-task tree of the canonical four files."""
    extracted = tmp_path / "expected"
    for name, data in files.items():
        out = extracted / name[len("test_files/") :]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    monkeypatch.setattr(nr, "N_TASKS", 1)
    monkeypatch.setattr(nr, "TASKS_SHA256", nr.tree_digest(extracted))


def test_taskset_fetches_verifies_and_caches(tmp_path, monkeypatch):
    files = {
        "README.md": b"unlicensed upstream readme",  # outside test_files: never extracted
        "test_files/.DS_Store": b"junk",
        "test_files/task_difficulty.csv": b"difficulty\n",
        **CANONICAL_FOUR,
    }
    _pin_single_task(tmp_path, monkeypatch, CANONICAL_FOUR)
    fetches = []
    ts = nr.TaskSet(
        root=tmp_path / "cache", fetch=lambda: fetches.append(1) or make_tarball(files)
    )
    got = ts.ensure()
    assert got == tmp_path / "cache" / nr.COMMIT / "test_files"
    assert sorted(str(p.relative_to(got)) for p in got.rglob("*")) == [
        "a",
        "a/start.md",
        "a/test_case_count.txt",
        "a/test_commands.json",
        "a/test_files.json",
    ]
    ts.ensure()  # cached now: the fetch is not repeated
    assert len(fetches) == 1


def test_taskset_skips_symlinks_and_traversal_names(tmp_path, monkeypatch):
    _pin_single_task(tmp_path, monkeypatch, CANONICAL_FOUR)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:

        def add(name: str, data: bytes, *, symlink=False):
            info = tarfile.TarInfo(f"NL2RepoBench-{nr.COMMIT}/{name}")
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
            else:
                info.size = len(data)
            tar.addfile(info, None if symlink else io.BytesIO(data))

        for name, data in CANONICAL_FOUR.items():
            add(name, data)
        add("test_files/a/link", b"", symlink=True)
        add("test_files/../escape.py", b"evil")
        add("openhands/post_processor.py", b"upstream harness code")
    got = nr.TaskSet(root=tmp_path / "cache", fetch=lambda: buf.getvalue()).ensure()
    assert not (got / "a" / "link").exists()
    assert not (tmp_path / "cache" / nr.COMMIT / "escape.py").exists()
    assert not list(tmp_path.joinpath("cache").rglob("post_processor.py"))
    assert sorted(p.name for p in (got / "a").iterdir()) == sorted(
        ["start.md", "test_case_count.txt", "test_commands.json", "test_files.json"]
    )


def test_taskset_refuses_a_fetch_that_does_not_match_the_pin(tmp_path, monkeypatch):
    _pin_single_task(tmp_path, monkeypatch, CANONICAL_FOUR)
    wrong = {k: v + b"x" if k.endswith(".md") else v for k, v in CANONICAL_FOUR.items()}
    ts = nr.TaskSet(root=tmp_path / "cache", fetch=lambda: make_tarball(wrong))
    with pytest.raises(RuntimeError, match="does not match the pinned tree"):
        ts.ensure()
    assert not (tmp_path / "cache" / nr.COMMIT / "test_files").exists()
    assert not (tmp_path / "cache" / nr.COMMIT / "test_files.partial").exists()


def test_taskset_refuses_a_fetch_with_the_wrong_task_count(tmp_path, monkeypatch):
    files = {
        **CANONICAL_FOUR,
        "test_files/b/start.md": b"# other\n",
        "test_files/b/test_case_count.txt": b"1",
        "test_files/b/test_commands.json": b"[]",
        "test_files/b/test_files.json": b"[]",
    }
    extracted = tmp_path / "expected"
    for name, data in files.items():
        out = extracted / name[len("test_files/") :]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    monkeypatch.setattr(nr, "N_TASKS", 1)
    monkeypatch.setattr(nr, "TASKS_SHA256", nr.tree_digest(extracted))
    ts = nr.TaskSet(root=tmp_path / "cache", fetch=lambda: make_tarball(files))
    with pytest.raises(RuntimeError, match="does not match the pinned tree"):
        ts.ensure()


def test_a_failed_fetch_never_touches_the_cached_tree(tmp_path, monkeypatch):
    _pin_single_task(tmp_path, monkeypatch, CANONICAL_FOUR)
    good = nr.TaskSet(
        root=tmp_path / "cache", fetch=lambda: make_tarball(CANONICAL_FOUR)
    ).ensure()
    (good / "a" / "start.md").write_text("tampered")  # the cache stops verifying...
    wrong = {k: v + b"x" if k.endswith(".md") else v for k, v in CANONICAL_FOUR.items()}
    ts = nr.TaskSet(root=tmp_path / "cache", fetch=lambda: make_tarball(wrong))
    with pytest.raises(RuntimeError, match="does not match the pinned tree"):
        ts.ensure()
    assert (good / "a" / "start.md").read_text() == "tampered"  # ...but is not replaced
    assert not (tmp_path / "cache" / nr.COMMIT / "test_files.partial").exists()


# -- strip and pytest parsing ----------------------------------------------------


def test_strip_workspace_removes_package_files_and_gold_tests(tmp_path):
    ws = tmp_path / "ws"
    (ws / "pkg" / "tests").mkdir(parents=True)
    (ws / "pyproject.toml").write_text("")
    (ws / "pkg" / "setup.py").write_text("")
    (ws / "pkg" / "impl.py").write_text("impl")
    (ws / "pkg" / "tests" / "test_x.py").write_text("gold")
    (ws / "test_top.py").write_text("gold")
    stripped = nr.strip_workspace(ws, ["pkg/tests", "test_top.py", "missing.py"])
    assert stripped == {
        "package_files": ["pkg/setup.py", "pyproject.toml"],
        "test_files": ["pkg/tests", "test_top.py"],
        "test_files_missing": ["missing.py"],
    }
    assert not (ws / "pkg" / "tests").exists()
    assert not (ws / "test_top.py").exists()
    assert (ws / "pkg" / "impl.py").read_text() == "impl"


def test_strip_workspace_rejects_paths_that_escape(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(ValueError, match="escapes the workspace"):
        nr.strip_workspace(ws, ["../outside"])
    with pytest.raises(ValueError, match="escapes the workspace"):
        nr.strip_workspace(ws, ["/etc/passwd"])


def _result(command, output):
    return {"command": command, "output": output}


def test_parse_pytest_results_sums_over_pytest_commands():
    out = nr.parse_pytest_results(
        [
            _result("pip install -e .", "3 passed"),  # not a pytest command: ignored
            _result("pytest a", "10 passed"),
            _result("pytest b", "5 passed, 2 failed, 1 error"),
        ],
        100,
    )
    assert out == {
        "passed": 15,
        "failed": 2,
        "errors": 1,
        "total": 100,
        "success_rate": 0.15,
    }


def test_parse_pytest_results_caps_the_rate_at_one():
    out = nr.parse_pytest_results([_result("pytest", "50 passed")], 10)
    assert out["success_rate"] == 1.0 and out["passed"] == 50


def test_parse_pytest_results_with_no_cases_scores_zero():
    assert (
        nr.parse_pytest_results([_result("pytest", "1 passed")], 0)["success_rate"]
        == 0.0
    )


# -- config.toml -----------------------------------------------------------------


def test_render_config_has_upstreams_effective_settings():
    cfg = nr.render_config(
        workspace="/tmp/nl2repo-run/workspace",
        llm_key="granite_4_2",
        model="granite-4.2",
        base_url="http://llm:8000/v1",
        max_iterations=500,
        enable_browsing=True,
    )
    assert 'runtime = "local"' in cfg
    assert "max_iterations = 500" in cfg
    assert 'save_trajectory_path = "./trajectories"' in cfg
    assert "[llm.granite_4_2]" in cfg
    assert 'model = "openai/granite-4.2"' in cfg
    assert 'api_key = "EMPTY"' in cfg
    assert 'base_url = "http://llm:8000/v1"' in cfg
    assert "enable_browsing = true" in cfg
    assert "enable_llm_editor = false" in cfg
    assert 'llm_config = "granite_4_2"' in cfg
    assert '[condenser]\ntype = "noop"' in cfg
    assert 'volumes = "/tmp/nl2repo-run/workspace:/workspace:rw"' in cfg
    assert (
        "temperature" not in cfg
        and "top_p" not in cfg
        and "max_output_tokens" not in cfg
    )


def test_render_config_toggles_browsing_and_sampling():
    cfg = nr.render_config(
        workspace="/w",
        llm_key="k",
        model="m",
        base_url="http://x/v1",
        max_iterations=50,
        enable_browsing=False,
        temperature=0.1,
        top_p=0.9,
        max_tokens=2048,
    )
    assert "enable_browsing = false" in cfg
    assert "temperature = 0.1" in cfg
    assert "top_p = 0.9" in cfg
    assert "max_output_tokens = 2048" in cfg


def test_llm_key_sanitizes_for_toml():
    assert nr.llm_key("granite-4.2/a:b") == "granite-4_2_a_b"


# -- generate --------------------------------------------------------------------


def test_generate_runs_openheads_with_the_task_and_stages_out(
    bench, tmp_path, monkeypatch
):
    task = nr.Task(make_task(tmp_path / "tasks", start="# the spec\n"))
    idir = tmp_path / "out" / "repeat-0" / "a"
    idir.mkdir(parents=True)
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    bench._generate(task, idir, "http://llm:8000/v1", "granite-4.2", 0)

    assert factory.images == [nr.OPENHANDS_IMAGE]
    env = factory.envs[0]
    assert env["RUNTIME"] == "local"
    assert env["SKIP_DEPENDENCY_CHECK"] == "1"
    assert env["CONFIG_FILE"] == f"{nr.RUN_DIR}/config.toml"
    assert env["AGENT_LLM_CONFIG"] == nr.llm_key("granite-4.2")
    assert env["OH_TASK"] == nr.TASK_PROMPT
    assert "SANDBOX_RUNTIME_CONTAINER_IMAGE" not in env  # inert under the local runtime

    sb = factory.sandboxes[0]
    assert sb.files[f"{nr.RUN_DIR}/workspace/start.md"] == "# the spec\n"
    cfg = sb.files[f"{nr.RUN_DIR}/config.toml"]
    assert 'model = "openai/granite-4.2"' in cfg
    assert 'base_url = "http://llm:8000/v1"' in cfg
    main = [c for c in sb.commands if "openhands.core.main" in c][0]
    assert f'--config-file={nr.RUN_DIR}/config.toml -t "$OH_TASK"' in main
    assert sb.cwd[sb.commands.index(main)] == nr.RUN_DIR

    assert tar_members((idir / "workspace.tar.gz").read_bytes()) == [
        "workspace/pkg/impl.py",
        "workspace/pyproject.toml",
        "workspace/test_six.py",
    ]
    assert tar_members((idir / "trajectory.tar.gz").read_bytes()) == [
        "trajectories/t.json"
    ]
    assert (idir / "openhands.log").read_text() == "agent log\n"
    meta = json.loads((idir / "meta.json").read_text())
    assert meta["image"] == nr.OPENHANDS_IMAGE
    assert meta["model"] == "openai/granite-4.2"
    assert meta["exit_code"] == 0 and meta["timed_out"] is False


def test_a_failed_stage_out_still_grades_an_empty_workspace(
    bench, tmp_path, monkeypatch
):
    task = nr.Task(make_task(tmp_path / "tasks"))
    idir = tmp_path / "out" / "repeat-0" / "a"
    idir.mkdir(parents=True)
    factory = SandboxFactory(
        {
            "openhands.core.main": ExecResult("log", 0),
            "workspace 2>/dev/null | base64": ExecResult("", 1),  # staging failed
        }
    )
    monkeypatch.setattr(nr, "make_sandbox", factory)
    bench._generate(task, idir, "http://llm:8000/v1", "m", 0)
    assert tar_members((idir / "workspace.tar.gz").read_bytes()) == []


def test_generate_without_an_endpoint_is_an_error(bench, tmp_path):
    task = nr.Task(make_task(tmp_path / "tasks"))
    with pytest.raises(RuntimeError, match="no model endpoint"):
        bench._generate(task, tmp_path, "", "m", 0)


def test_gold_workspace_is_start_md_only(bench, tmp_path):
    task = nr.Task(make_task(tmp_path / "tasks", start="# spec\n"))
    ws_tar = tmp_path / "workspace.tar.gz"
    bench._gold_workspace(task, ws_tar)
    assert tar_members(ws_tar.read_bytes()) == ["workspace/start.md"]
    assert json.loads((tmp_path / "meta.json").read_text()) == {"gold": True}


# -- grade -----------------------------------------------------------------------


def _grade_setup(
    tmp_path, *, count=100, commands=None, test_files=("tests/a",), ws_files=None
):
    task = nr.Task(
        make_task(
            tmp_path / "tasks", count=count, commands=commands, test_files=test_files
        )
    )
    idir = tmp_path / "out" / "repeat-0" / "a"
    idir.mkdir(parents=True)
    (idir / "workspace.tar.gz").write_bytes(
        ws_tar_bytes(
            ws_files
            or {
                "pyproject.toml": b"[project]\n",
                "tests/a/test_x.py": b"gold",
                "pkg/impl.py": b"impl",
            }
        )
    )
    return task, idir


def test_grade_merges_the_stripped_workspace_and_runs_the_commands(
    bench, tmp_path, monkeypatch
):
    task, idir = _grade_setup(
        tmp_path,
        commands=[
            "SETUPTOOLS_SCM_PRETEND_VERSION_1.0=1 pip install -e . && pytest tests/a"
        ],
    )
    factory = SandboxFactory({"pytest": ExecResult("40 passed, 2 failed, 1 error", 0)})
    monkeypatch.setattr(nr, "make_sandbox", factory)
    report = bench._grade(task, idir)

    assert report["status"] == "graded"
    assert report["rate"] == 0.4
    assert report["passed"] == 40 and report["failed"] == 2 and report["errors"] == 1
    assert report["total"] == 100
    assert report["image"] == f"{nr.IMAGE_PREFIX}/a:{nr.IMAGE_TAG}"
    assert report["stripped"]["package_files"] == ["pyproject.toml"]
    assert report["stripped"]["test_files"] == ["tests/a"]

    assert factory.images == [report["image"]]
    assert factory.envs == [{"PYTHONPATH": "/workspace"}]
    sb = factory.sandboxes[0]
    assert (
        sb.commands[0]
        == f"mkdir -p /workspace && base64 -d {nr.WS_B64} | tar -xzpf - --no-same-owner -C /workspace"
    )
    # The shlex fix: a command with && and a VAR=... prefix runs verbatim, not shlex-split.
    assert (
        sb.commands[1]
        == "SETUPTOOLS_SCM_PRETEND_VERSION_1.0=1 pip install -e . && pytest tests/a"
    )
    assert sb.cwd[1] == "/workspace"
    # What was staged in is the stripped tree, not the agent's.
    assert tar_members(base64.b64decode(sb.files[nr.WS_B64])) == ["pkg/impl.py"]
    assert (
        (idir / "test_output.txt")
        .read_text()
        .startswith(
            "$ SETUPTOOLS_SCM_PRETEND_VERSION_1.0=1 pip install -e . && pytest tests/a"
        )
    )


def test_a_timed_out_command_is_persisted_with_its_partial_score(
    bench, tmp_path, monkeypatch
):
    task, idir = _grade_setup(tmp_path, count=10, commands=["pytest tests/a"])
    factory = SandboxFactory({"pytest": ExecResult("7 passed", -1, timed_out=True)})
    monkeypatch.setattr(nr, "make_sandbox", factory)
    report = bench._grade(task, idir)
    assert report["status"] == "eval_timeout"
    assert report["rate"] == 0.7 and report["passed"] == 7
    assert report["commands"][0]["timed_out"] is True


def test_an_oversized_workspace_grades_zero_without_a_sandbox(
    bench, tmp_path, monkeypatch
):
    task, idir = _grade_setup(tmp_path, commands=["pytest t"], ws_files={"x": b"x"})
    bench.config.options["ws_max_mb"] = "0"
    monkeypatch.setattr(
        nr,
        "make_sandbox",
        lambda *a, **k: pytest.fail("no sandbox for an oversized workspace"),
    )
    report = bench._grade(task, idir)
    assert report["status"] == "workspace_too_large"
    assert report["rate"] == 0.0 and report["passed"] == 0


# -- run(): task set selection ----------------------------------------------------


def test_tasks_root_from_dataset_or_env(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(model="m", output_dir=tmp_path, dataset=str(tmp_path / "tasks"))
    )
    assert bench._tasks_root() == (tmp_path / "tasks", False)
    monkeypatch.delenv(nr.CACHE_ENV, raising=False)
    monkeypatch.setenv(nr.CACHE_ENV, str(tmp_path / "tasks"))
    bench = nr.NL2RepoBench(RunConfig(model="m", output_dir=tmp_path))
    assert bench._tasks_root() == (tmp_path / "tasks", False)


def test_tasks_root_must_point_at_a_directory(tmp_path, monkeypatch):
    bench = nr.NL2RepoBench(
        RunConfig(model="m", output_dir=tmp_path, dataset=str(tmp_path / "nope"))
    )
    with pytest.raises(SystemExit, match="not a directory"):
        bench._tasks_root()
    monkeypatch.setenv(nr.CACHE_ENV, str(tmp_path / "nope"))
    bench = nr.NL2RepoBench(RunConfig(model="m", output_dir=tmp_path))
    with pytest.raises(SystemExit, match="not a directory"):
        bench._tasks_root()


def test_selection_filters_by_regex_and_limit(tmp_path):
    two_tasks(tmp_path / "tasks")
    make_task(tmp_path / "tasks", name="extra")
    bench = nr.NL2RepoBench(
        RunConfig(model="m", output_dir=tmp_path, dataset=str(tmp_path / "tasks"))
    )
    assert [t.name for t in bench._select(tmp_path / "tasks")] == ["a", "b", "extra"]
    bench.config.options["tasks"] = "^[ab]$"
    assert [t.name for t in bench._select(tmp_path / "tasks")] == ["a", "b"]
    bench.config.options.pop("tasks")
    bench.config.limit = 2
    assert [t.name for t in bench._select(tmp_path / "tasks")] == ["a", "b"]


# -- run(): aggregation, phases, resume -------------------------------------------


def test_run_aggregates_the_test_pass_rate(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m", output_dir=tmp_path / "out", dataset=str(tmp_path / "tasks")
        )
    )
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    out = bench.run("http://llm:8000/v1", "granite-4.2")

    assert out["value"] == pytest.approx(
        (0.4 + 0.5) / 2
    )  # mean over tasks of passed/count
    assert out["n"] == 2
    assert out["task_failed"] == 0 and out["incomplete"] is False
    assert out["pass_at_k"]["k"] == 1
    assert out["pass_at_k"]["pass_at_1"] == pytest.approx(0.45)
    assert out["per_repeat"][0]["statuses"] == {"graded": 2}
    assert (
        out["per_repeat"][0]["passed"] == 65
        and out["per_repeat"][0]["test_cases"] == 150
    )
    assert out["task_set"] == {
        "root": str(tmp_path / "tasks"),
        "tasks": 2,
        "tasks_sha256": None,
    }
    assert out["dataset"] == str(tmp_path / "tasks") and out["dataset_revision"] is None
    assert out["tasks"] == ["a", "b"]
    assert out["images"]["grade_prefix"] == nr.IMAGE_PREFIX
    assert any("bash -c" in d for d in out["deviations"])
    # One agent sandbox and one grade sandbox per task.
    assert factory.images.count(nr.OPENHANDS_IMAGE) == 2
    assert sorted(set(factory.images) - {nr.OPENHANDS_IMAGE}) == [
        f"{nr.IMAGE_PREFIX}/a:{nr.IMAGE_TAG}",
        f"{nr.IMAGE_PREFIX}/b:{nr.IMAGE_TAG}",
    ]
    r0 = tmp_path / "out" / "repeat-0"
    for t in ("a", "b"):
        for f in (
            "workspace.tar.gz",
            "openhands.log",
            "trajectory.tar.gz",
            "meta.json",
            "report.json",
            "test_output.txt",
        ):
            assert (r0 / t / f).is_file(), f"{t}/{f}"


def test_the_fetched_pin_is_recorded_as_verified(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")

    class FakeTaskSet:
        def __init__(self, root=None):
            pass

        def ensure(self):
            return tmp_path / "tasks"

    monkeypatch.setattr(nr, "TaskSet", FakeTaskSet)
    bench = nr.NL2RepoBench(RunConfig(model="m", output_dir=tmp_path / "out"))
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    out = bench.run("http://llm:8000/v1", "granite-4.2")
    assert out["task_set"]["tasks_sha256"] == nr.TASKS_SHA256
    assert out["dataset"] == nr.REPO and out["dataset_revision"] == nr.COMMIT


def test_rerun_skips_finished_tasks(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    out_dir = tmp_path / "out"
    bench = nr.NL2RepoBench(
        RunConfig(model="m", output_dir=out_dir, dataset=str(tmp_path / "tasks"))
    )
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    first = bench.run("http://llm:8000/v1", "granite-4.2")
    assert len(factory.sandboxes) == 4
    factory2 = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory2)
    second = bench.run("http://llm:8000/v1", "granite-4.2")
    assert factory2.sandboxes == []  # every report.json short-circuits the task
    assert second["value"] == first["value"]


def test_repeats_feed_pass_at_k(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            dataset=str(tmp_path / "tasks"),
            repeats=2,
        )
    )
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    out = bench.run("http://llm:8000/v1", "granite-4.2")
    assert out["pass_at_k"]["k"] == 2
    assert out["pass_at_k"]["pass_at_1"] == pytest.approx(0.45)
    assert out["pass_at_k"]["pass_at_k"] == pytest.approx(
        0.45
    )  # the best of the two runs
    assert len(out["per_repeat"]) == 2
    assert (tmp_path / "out" / "repeat-1" / "a" / "report.json").is_file()


def test_generate_phase_writes_workspaces_and_grades_nothing(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            dataset=str(tmp_path / "tasks"),
            phase="generate",
        )
    )
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    out = bench.run("http://llm:8000/v1", "granite-4.2")
    assert out["n"] == 2 and "value" not in out
    assert out["per_repeat"] == [{"repeat": 0, "generated": 2, "failed": []}]
    assert factory.images == [nr.OPENHANDS_IMAGE, nr.OPENHANDS_IMAGE]
    assert not (tmp_path / "out" / "repeat-0" / "a" / "report.json").exists()
    assert (tmp_path / "out" / "repeat-0" / "b" / "workspace.tar.gz").is_file()


def test_score_phase_grades_saved_workspaces_without_the_model(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    cfg = {
        "model": "m",
        "output_dir": tmp_path / "out",
        "dataset": str(tmp_path / "tasks"),
    }
    generate = nr.NL2RepoBench(RunConfig(**cfg, phase="generate"))
    monkeypatch.setattr(nr, "make_sandbox", SandboxFactory(RUN_OUTPUTS))
    generate.run("http://llm:8000/v1", "granite-4.2")

    score = nr.NL2RepoBench(RunConfig(**cfg, phase="score"))
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    out = score.run("", "")  # no endpoint: the score phase never talks to a model
    assert out["value"] == pytest.approx(0.45)
    assert nr.OPENHANDS_IMAGE not in factory.images
    assert sorted(factory.images) == [
        f"{nr.IMAGE_PREFIX}/a:{nr.IMAGE_TAG}",
        f"{nr.IMAGE_PREFIX}/b:{nr.IMAGE_TAG}",
    ]


def test_score_phase_without_a_generation_records_not_generated(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            dataset=str(tmp_path / "tasks"),
            phase="score",
        )
    )
    monkeypatch.setattr(
        nr, "make_sandbox", lambda *a, **k: pytest.fail("score phase never generates")
    )
    out = bench.run("", "")
    assert out["value"] == 0.0 and out["n"] == 2
    assert out["per_repeat"][0]["statuses"] == {"not_generated": 2}
    assert not (
        tmp_path / "out" / "repeat-0" / "a" / "report.json"
    ).exists()  # left for a generate run


def test_infrastructure_failures_fail_the_run_over_the_allowed_fraction(
    tmp_path, monkeypatch
):
    two_tasks(tmp_path / "tasks")

    def broken(*a, **k):
        raise ConnectionError("image pull failed")

    monkeypatch.setattr(nr, "make_sandbox", broken)
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m", output_dir=tmp_path / "out", dataset=str(tmp_path / "tasks")
        )
    )
    with pytest.raises(SystemExit, match="task_failed 2/2"):
        bench.run("http://llm:8000/v1", "granite-4.2")


def test_infrastructure_failures_within_the_allowed_fraction_are_reported(
    tmp_path, monkeypatch
):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            dataset=str(tmp_path / "tasks"),
            options={"max_failed_frac": "1"},
        )
    )
    factory = SandboxFactory(RUN_OUTPUTS)

    def flaky(image, *, backend="", env=None):
        if image.endswith("/a:1.0"):
            raise ConnectionError("pull failed")
        return factory(image, backend=backend, env=env)

    monkeypatch.setattr(nr, "make_sandbox", flaky)
    out = bench.run("http://llm:8000/v1", "granite-4.2")
    assert out["task_failed"] == 1 and out["incomplete"] is True
    assert out["per_repeat"][0]["statuses"] == {"error:ConnectionError": 1, "graded": 1}
    assert out["value"] == pytest.approx(
        0.25
    )  # the failed task scores 0, never a model loss in pass@k terms


def test_generate_phase_leaves_failed_tasks_for_a_rerun(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")

    def broken(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr(nr, "make_sandbox", broken)
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            dataset=str(tmp_path / "tasks"),
            phase="generate",
        )
    )
    out = bench.run("http://llm:8000/v1", "granite-4.2")
    assert out["n"] == 0  # no SystemExit: rerunning generate retries them
    assert sorted(out["per_repeat"][0]["failed"]) == ["a", "b"]


# -- run(): gold ------------------------------------------------------------------


def test_gold_run_grades_task_images_without_a_model(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            dataset=str(tmp_path / "tasks"),
            options={"agent": "gold"},
        )
    )
    factory = SandboxFactory(RUN_OUTPUTS)
    monkeypatch.setattr(nr, "make_sandbox", factory)
    out = bench.run("", "")
    assert nr.OPENHANDS_IMAGE not in factory.images  # no agent sandbox at all
    assert factory.images  # ...but every task image is graded
    # The graded workspace is the untouched one: start.md only.
    sb = factory.sandboxes[0]
    assert tar_members(base64.b64decode(sb.files[nr.WS_B64])) == ["start.md"]
    assert json.loads(
        (tmp_path / "out" / "repeat-0" / "a" / "meta.json").read_text()
    ) == {"gold": True}
    assert out["per_repeat"][0]["statuses"] == {"graded": 2}


def test_gold_generate_phase_writes_workspaces_without_a_sandbox(tmp_path, monkeypatch):
    two_tasks(tmp_path / "tasks")
    bench = nr.NL2RepoBench(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            dataset=str(tmp_path / "tasks"),
            phase="generate",
            options={"agent": "gold"},
        )
    )
    monkeypatch.setattr(
        nr,
        "make_sandbox",
        lambda *a, **k: pytest.fail("gold generate touches no sandbox"),
    )
    out = bench.run("", "")
    assert out["n"] == 2
    for t in ("a", "b"):
        assert tar_members(
            (tmp_path / "out" / "repeat-0" / t / "workspace.tar.gz").read_bytes()
        ) == ["workspace/start.md"]
