"""MCPMark adapter tests with a fake harness (no network, no DB, no browser)."""

import json
import re
import subprocess
from pathlib import Path

import pytest
from granite_evals.benchmarks import mcpmark as mm
from granite_evals.registry import RunConfig, get
from granite_evals.sandbox.nodelock import NodeLock

# -- fixtures and fakes ---------------------------------------------------------


def make_checkout(root: Path, service="filesystem", suite="standard", per_category=(("cat_a", 2), ("cat_b", 1))):
    """A fake MCPMark checkout: pipeline.py, the commit marker, and a
    tasks/<service>/<suite>/<category>/<task>/ tree (one meta.json per task)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pipeline.py").write_text("# upstream pipeline\n")
    (root / ".granite-commit").write_text(f"{mm.MCPMARK_COMMIT}\n")
    for category, n in per_category:
        for i in range(n):
            task = root / "tasks" / service / suite / category / f"task_{i}"
            task.mkdir(parents=True)
            (task / "meta.json").write_text(json.dumps({"task_name": f"{category}/task_{i}"}))
    return root


@pytest.fixture
def baked(tmp_path, monkeypatch):
    """A fake /opt/mcpmark/<commit>: the source tree (filesystem and postgres
    standard tasks) plus a dummy venv python. N_TASKS is patched to the FAKE
    tree's counts so run() proceeds past the pin check -- the real counts are
    asserted in test_registered_with_its_pins, and the check itself in
    test_task_count_at_the_pin_is_enforced on an unpatched fake."""
    home = make_checkout(tmp_path / "opt-mcpmark")
    make_checkout(home, service="postgres", per_category=(("cat_a", 2),))
    venv_py = home / ".venv" / "bin" / "python"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("#!/bin/sh\n")
    monkeypatch.setattr(mm, "MCPMARK_HOME", home)
    monkeypatch.setattr(mm, "MCPMARK_PY", venv_py)
    monkeypatch.setattr(mm, "N_TASKS", {**mm.N_TASKS, ("filesystem", "standard"): 3, ("postgres", "standard"): 2})
    return home


@pytest.fixture
def lock_dir(tmp_path, monkeypatch):
    """A GRANITE_EVALS_LOCK_DIR that exists (the flock fallback opens a file
    in it; the abstract-socket path on Linux never looks at it)."""
    d = tmp_path / "locks"
    d.mkdir()
    monkeypatch.setenv("GRANITE_EVALS_LOCK_DIR", str(d))
    return d


def mkbench(service, tmp_path, **kw):
    cls = {
        "filesystem": mm.McpmarkFilesystem,
        "postgres": mm.McpmarkPostgres,
        "playwright": mm.McpmarkPlaywright,
    }[service]
    config = {"model": "m", "output_dir": tmp_path / "out", "workers": 1}
    config.update(kw)
    return cls(RunConfig(**config))


class FakeRunner:
    """A subprocess.run stand-in that records every call and, for pipeline
    invocations, writes each task's meta.json the way the real harness does:
    under <output-dir>/<exp>/<model-slug>__<service-dir>/run-1/<cat>__<task>/
    (the slug is the --models string with dots replaced; the service dir
    carries a -<suite> suffix unless standard)."""

    def __init__(self, outcomes=None, service="filesystem"):
        self.outcomes = outcomes or {}  # "category/task" -> meta.json content
        self.service = service
        self.calls = []

    def _meta(self, task_id):
        if (outcome := self.outcomes.get(task_id)) is None:
            return {"execution_result": {"success": True}}
        if callable(outcome):
            return outcome()
        return {"execution_result": outcome}

    def __call__(self, argv, *, cwd=None, env=None, check=False, capture_output=False, text=False):
        self.calls.append({"argv": list(argv), "cwd": cwd, "env": env})
        if "pipeline" in argv[:3]:
            opt = argv.index
            service_dir = self.service + (
                "" if argv[opt("--task-suite") + 1] == "standard" else f"-{argv[opt('--task-suite') + 1]}"
            )
            run_dir = (
                Path(argv[opt("--output-dir") + 1])
                / argv[opt("--exp-name") + 1]
                / f"{argv[opt('--models') + 1].replace('.', '-')}__{service_dir}"
                / "run-1"
            )
            task_id = argv[opt("--tasks") + 1]
            task_dir = run_dir / task_id.replace("/", "__")
            task_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "meta.json").write_text(json.dumps(self._meta(task_id)))
        return subprocess.CompletedProcess(argv, 0, "", "")


PASS = {"success": True}
FAIL = {"success": False}  # ran, failed verification: a scored 0
FAIL_MESSAGE = {"success": False, "error_message": "AssertionError: expected output.txt"}
RATE_LIMITED = {"success": False, "error_message": "RateLimitError: 429 slow down"}


# -- registration, options and pins ---------------------------------------------


def test_registered_with_its_pins():
    assert get("mcpmark-filesystem") is mm.McpmarkFilesystem
    assert get("mcpmark-postgres") is mm.McpmarkPostgres
    assert get("mcpmark-playwright") is mm.McpmarkPlaywright
    for cls, service in (
        (mm.McpmarkFilesystem, "filesystem"),
        (mm.McpmarkPostgres, "postgres"),
        (mm.McpmarkPlaywright, "playwright"),
    ):
        assert cls.metric == "pass@1"
        assert cls.extra == "mcpmark"
        assert cls.service == service
        assert cls.splittable is False  # verify.py runs inline per task
        assert cls.dataset == mm.REPO == "eval-sys/mcpmark"
        assert cls.dataset_revision == mm.MCPMARK_COMMIT
    assert mm.MCPMARK_COMMIT == "cd45b7f57923b9b3985467f5139927575f83141c"
    assert len(mm.MCPMARK_COMMIT) == 40
    for pin in (mm.TARBALL_SHA256, mm.TREE_SHA256):
        assert re.fullmatch(r"[0-9a-f]{64}", pin), pin
    # Task counts from the pinned commit's git tree.
    assert mm.N_TASKS == {
        ("filesystem", "standard"): 30,
        ("filesystem", "easy"): 10,
        ("postgres", "standard"): 21,
        ("postgres", "easy"): 10,
        ("playwright", "standard"): 4,
        ("playwright", "easy"): 0,
    }
    # errors.py at the pin: substring patterns, timeouts deliberately absent.
    assert mm.RETRYABLE_PATTERNS == (
        "ratelimit",
        "agent execution failed",
        "unavailable",
        "internal server error",
        "network error",
        "quota",
        "account balance",
        "mcp network error",
        "state duplication error",
        "thought_signature",
        "overloaded.",
    )


def test_the_benchmarks_are_in_the_extended_suite():
    from granite_evals import suites

    listed = {b["id"] for b in suites.load("extended")["benchmarks"]}
    assert {"mcpmark-filesystem", "mcpmark-postgres", "mcpmark-playwright"} <= listed


def _extras_sh():
    """mcpmark/extras.sh when running from a repo checkout; the assembled test
    trees (make test-src, the image) do not carry it."""
    for parent in Path(__file__).resolve().parents[:3]:
        if (parent / "extras.sh").is_file() and (parent / "Dockerfile.gpu").is_file():
            return parent / "extras.sh"
    return None


def test_the_pins_appear_in_extras_sh():
    extras = _extras_sh()
    if extras is None:
        pytest.skip("mcpmark/extras.sh is not beside this tree (assembled test run)")
    text = extras.read_text()
    for pin in (mm.MCPMARK_COMMIT, mm.TARBALL_SHA256, mm.TREE_SHA256):
        assert pin in text, pin


def test_option_parsing(tmp_path):
    bench = mkbench("filesystem", tmp_path)
    assert bench.opt("task_suite", "standard") == "standard"
    bench.config.options["timeout"] = "120"
    assert bench.opt("timeout", 3600) == 120
    bench.config.options["max_failed_frac"] = "0.5"
    assert bench.opt("max_failed_frac", 0.05) == 0.5


def test_task_suite_is_validated(tmp_path):
    bench = mkbench("filesystem", tmp_path, options={"task_suite": "hard"})
    with pytest.raises(SystemExit, match="task_suite must be one of standard, easy"):
        bench.task_suite


# -- helpers: slug, retryable, tree digest ---------------------------------------


def test_model_slug_matches_upstream():
    # evaluator.py at the pin: the --models string ("openai/<served>") with
    # every "." replaced; the slash stays a directory separator.
    assert mm.model_slug("granite-4.1-3b") == "openai/granite-4-1-3b"
    assert mm.model_slug("m") == "openai/m"


def test_is_retryable_matches_upstreams_patterns():
    assert mm.is_retryable("RateLimitError: 429")
    assert mm.is_retryable("MCP Network Error while talking to the server")
    assert mm.is_retryable("State Duplication Error: could not restore the task state")
    assert mm.is_retryable("The service is temporarily overloaded.")
    assert not mm.is_retryable("Operation timed out after 3600 seconds")  # scored 0
    assert not mm.is_retryable("AssertionError: expected output.txt")  # scored 0
    assert not mm.is_retryable(None)
    assert not mm.is_retryable("")


def test_retry_tag_names_the_matched_pattern():
    assert mm.retry_tag("RateLimitError: 429") == "ratelimit"
    assert mm.retry_tag("Connection closed") == ""
    assert mm.retry_tag(None) == ""


def test_tree_digest_is_order_independent_and_content_sensitive(tmp_path):
    for name in ("one", "two"):
        d = tmp_path / name
        (d / "sub").mkdir(parents=True)
        (d / "sub" / "b.zip").write_bytes(b"b")
        (d / "a.zip").write_bytes(b"a")
    assert mm.tree_digest(tmp_path / "one") == mm.tree_digest(tmp_path / "two")
    (tmp_path / "two" / "a.zip").write_bytes(b"A")
    assert mm.tree_digest(tmp_path / "one") != mm.tree_digest(tmp_path / "two")


# -- the writable checkout --------------------------------------------------------


def test_checkout_copies_the_baked_tree_without_the_venv(baked, tmp_path):
    bench = mkbench("filesystem", tmp_path)
    checkout = bench._checkout()
    assert checkout == tmp_path / "out" / "mcpmark-checkout"
    assert (checkout / "pipeline.py").is_file()
    assert (checkout / ".granite-commit").read_text().strip() == mm.MCPMARK_COMMIT
    assert (checkout / "tasks" / "filesystem" / "standard" / "cat_a" / "task_0").is_dir()
    assert not (checkout / ".venv").exists()  # the ~1 GB venv stays baked
    assert bench._checkout() == checkout  # stable: no re-copy
    assert not (tmp_path / "out" / "mcpmark-checkout.partial").exists()


def test_checkout_refuses_a_mismatched_commit_marker(baked, tmp_path):
    out = tmp_path / "out"
    dest = out / "mcpmark-checkout"
    (dest / "pipeline.py").parent.mkdir(parents=True)
    (dest / "pipeline.py").write_text("# tampered\n")
    (dest / ".granite-commit").write_text("deadbeef\n")
    bench = mkbench("filesystem", tmp_path)
    with pytest.raises(SystemExit, match="is not MCPMark"):
        bench._checkout()


def test_checkout_dataset_override_is_trusted_as_is(tmp_path):
    checkout = make_checkout(tmp_path / "custom")
    bench = mkbench("filesystem", tmp_path, dataset=str(checkout))
    assert bench._checkout() == checkout
    bench = mkbench("filesystem", tmp_path, dataset=str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="not an MCPMark checkout"):
        bench._checkout()


# -- task ids and limit ------------------------------------------------------------


def test_task_ids_are_sorted_category_and_suite_scoped(tmp_path):
    checkout = make_checkout(tmp_path / "c", per_category=(("cat_b", 1), ("cat_a", 2)))
    (checkout / "tasks" / "filesystem" / "standard" / "notes.txt").write_text("not a dir")
    make_checkout(checkout, suite="easy", per_category=(("cat_a", 1),))
    bench = mkbench("filesystem", tmp_path, dataset=str(checkout))
    assert bench._task_ids(checkout) == ["cat_a/task_0", "cat_a/task_1", "cat_b/task_0"]
    bench.config.options["task_suite"] = "easy"
    assert bench._task_ids(checkout) == ["cat_a/task_0"]


def test_task_count_at_the_pin_is_enforced(tmp_path, monkeypatch):
    home = make_checkout(tmp_path / "opt-mcpmark")  # 3 fake tasks, pin says 30
    monkeypatch.setattr(mm, "MCPMARK_HOME", home)  # N_TASKS left real
    monkeypatch.setattr(mm, "MCPMARK_PY", home / ".venv" / "bin" / "python")
    bench = mkbench("filesystem", tmp_path)
    with pytest.raises(SystemExit, match=r"3 filesystem/standard tasks at .* not 30"):
        bench.run("http://llm:8000/v1", "granite-4.1-3b")
    # A --dataset checkout is trusted: no count enforcement (recorded as such).
    checkout = make_checkout(tmp_path / "custom")
    bench = mkbench("filesystem", tmp_path, dataset=str(checkout))
    monkeypatch.setattr(mm.subprocess, "run", FakeRunner())
    out = bench.run("http://llm:8000/v1", "granite-4.1-3b")
    assert out["n"] == 3


def test_an_empty_suite_fails_loudly(baked, tmp_path):
    # playwright has no easy tasks at the pin; a LIMIT smoke against it must
    # not produce a nonsense 0-scored run.
    bench = mkbench("playwright", tmp_path, options={"task_suite": "easy"})
    with pytest.raises(SystemExit, match="no playwright tasks in the easy suite"):
        bench.run("http://llm:8000/v1", "granite-4.1-3b")


def test_limit_takes_the_first_n_sorted_tasks(baked, tmp_path, monkeypatch):
    bench = mkbench("filesystem", tmp_path, limit=2)
    runner = FakeRunner()
    monkeypatch.setattr(mm.subprocess, "run", runner)
    bench.run("http://llm:8000/v1", "granite-4.1-3b")
    invoked = [c["argv"][c["argv"].index("--tasks") + 1] for c in runner.calls]
    assert invoked == ["cat_a/task_0", "cat_a/task_1"]


def test_repeats_and_workers_are_validated(baked, tmp_path):
    bench = mkbench("filesystem", tmp_path, repeats=2)
    with pytest.raises(SystemExit, match="one agent run per task"):
        bench.run("http://llm:8000/v1", "granite-4.1-3b")
    bench = mkbench("filesystem", tmp_path, workers=4)
    with pytest.raises(SystemExit, match="serially per service"):
        bench.run("http://llm:8000/v1", "granite-4.1-3b")


# -- pipeline env and argv ---------------------------------------------------------


def _checkout_for(bench, tmp_path):
    return bench._checkout()


def test_pipeline_env_wiring(baked, tmp_path):
    checkout = make_checkout(tmp_path / "c")
    fs = mkbench("filesystem", tmp_path)
    env = fs._pipeline_env(checkout, "http://llm:8000/v1")
    assert env["OPENAI_BASE_URL"] == "http://llm:8000/v1"
    assert ";" not in env["OPENAI_BASE_URL"]  # never a joined multi-endpoint URL
    assert env["OPENAI_API_KEY"] == "EMPTY"
    assert env["PYTHONPATH"] == str(checkout)
    assert env["FILESYSTEM_TEST_ROOT"] == str(tmp_path / "out" / "test_environments")
    assert env["UV_CONSTRAINT"] == str(mm.MCP_CONSTRAINTS)
    assert env["PIP_CONSTRAINT"] == str(mm.MCP_CONSTRAINTS)
    assert env["PATH"].startswith(f"{mm.MCPMARK_HOME}/.venv/bin:")

    pg = mkbench("postgres", tmp_path)
    env = pg._pipeline_env(
        checkout,
        "http://llm:8000/v1",
        {"POSTGRES_HOST": "localhost", "POSTGRES_PORT": "15433", "POSTGRES_USERNAME": "postgres", "POSTGRES_PASSWORD": "mcpmark"},
    )
    assert env["POSTGRES_PORT"] == "15433" and env["POSTGRES_HOST"] == "localhost"
    assert "FILESYSTEM_TEST_ROOT" not in env

    pw = mkbench("playwright", tmp_path)
    env = pw._pipeline_env(checkout, "http://llm:8000/v1")
    assert "FILESYSTEM_TEST_ROOT" not in env and "POSTGRES_PORT" not in env


def test_pipeline_argv_is_upstreams_contract(baked, tmp_path, monkeypatch):
    bench = mkbench("filesystem", tmp_path)
    checkout = bench._checkout()
    runner = FakeRunner()
    monkeypatch.setattr(mm.subprocess, "run", runner)
    bench._run_pipeline(checkout, bench._pipeline_env(checkout, "http://llm:8000/v1"), "granite-4.1-3b", "cat_a/task_0")
    call = runner.calls[0]
    assert call["argv"] == [
        str(mm.MCPMARK_PY),
        "-m",
        "pipeline",
        "--mcp",
        "filesystem",
        "--models",
        "openai/granite-4.1-3b",
        "--exp-name",
        "run",
        "--task-suite",
        "standard",
        "--tasks",
        "cat_a/task_0",
        "--k",
        "1",
        "--timeout",
        "3600",
        "--output-dir",
        str(tmp_path / "out" / "mcpmark"),
    ]
    assert call["cwd"] == checkout
    assert call["env"]["OPENAI_BASE_URL"] == "http://llm:8000/v1"


def test_pipeline_argv_carries_the_options(baked, tmp_path, monkeypatch):
    bench = mkbench(
        "filesystem",
        tmp_path,
        options={"task_suite": "easy", "timeout": "120", "compaction_token": "8000", "reasoning_effort": "low"},
    )
    checkout = bench._checkout()
    runner = FakeRunner()
    monkeypatch.setattr(mm.subprocess, "run", runner)
    bench._run_pipeline(checkout, bench._pipeline_env(checkout, "http://llm:8000/v1"), "m", "cat_a/task_0")
    argv = runner.calls[0]["argv"]
    assert argv[argv.index("--task-suite") + 1] == "easy"
    assert argv[argv.index("--timeout") + 1] == "120"
    assert argv[argv.index("--compaction-token") + 1] == "8000"
    assert argv[argv.index("--reasoning-effort") + 1] == "low"


# -- aggregation -------------------------------------------------------------------


def _write_meta(run_dir, task_id, content):
    task_dir = run_dir / task_id.replace("/", "__")
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "meta.json").write_text(json.dumps({"execution_result": content}))


def test_aggregate_scores_passes_and_fails_leaving_infra_out(tmp_path):
    checkout = make_checkout(tmp_path / "c")
    # max_failed_frac 0.5: the two infra rows (2/4) stay below the fatal bar --
    # this test is about the scoring, test_infrastructure_failures... is about the bar.
    bench = mkbench("filesystem", tmp_path, options={"task_suite": "easy", "max_failed_frac": "0.5"})
    run_dir = bench._run_dir("granite-4.1-3b")
    assert run_dir == (
        tmp_path / "out" / "mcpmark" / "run" / "openai" / "granite-4-1-3b__filesystem-easy" / "run-1"
    )
    _write_meta(run_dir, "cat_a/task_0", PASS)
    _write_meta(run_dir, "cat_a/task_1", FAIL_MESSAGE)  # ran, failed: scored 0
    _write_meta(run_dir, "cat_b/task_0", RATE_LIMITED)  # infra: excluded
    # cat_b/task_1: no meta.json (never produced): infra, excluded
    ids = ["cat_a/task_0", "cat_a/task_1", "cat_b/task_0", "cat_b/task_1"]
    out = bench._aggregate(checkout, run_dir, ids, mm.REPO, mm.MCPMARK_COMMIT)
    assert out["value"] == 0.5  # 1 pass over 2 scored
    assert out["n"] == 2 and out["n_total"] == 4
    assert out["task_failed"] == 2 and out["task_total"] == 4 and out["incomplete"] is True
    assert out["pass_at_k"]["k"] == 1 and out["pass_at_k"]["pass_at_1"] == 0.5
    assert out["per_task"] == {"cat_a/task_0": 1.0, "cat_a/task_1": 0.0}
    assert out["per_task_status"] == {
        "cat_a/task_0": "pass",
        "cat_a/task_1": "fail",
        "cat_b/task_0": "retryable:ratelimit",
        "cat_b/task_1": "missing",
    }
    assert out["mcpmark_commit"] == mm.MCPMARK_COMMIT
    assert out["service"] == "filesystem" and out["task_suite"] == "easy"
    assert out["state_digest"] == {}  # nothing downloaded in this run


def test_aggregate_records_state_payload_digests(tmp_path):
    checkout = make_checkout(tmp_path / "c")
    bench = mkbench("postgres", tmp_path)
    (checkout / "postgres_state").mkdir()
    (checkout / "postgres_state" / "employees.backup").write_bytes(b"employees")
    (tmp_path / "out" / "test_environments").mkdir(parents=True)  # empty: skipped
    digests = bench._state_digests(checkout)
    assert list(digests) == ["postgres_state"]
    assert digests["postgres_state"] == mm.tree_digest(checkout / "postgres_state")


def test_infrastructure_failures_over_the_allowed_fraction_fail_the_run(tmp_path):
    checkout = make_checkout(tmp_path / "c")
    bench = mkbench("filesystem", tmp_path, options={"max_failed_frac": "0"})
    run_dir = bench._run_dir("m")
    _write_meta(run_dir, "cat_a/task_0", PASS)
    ids = ["cat_a/task_0", "cat_a/task_1"]  # task_1: missing meta.json
    with pytest.raises(SystemExit, match="task_failed 1/2"):
        bench._aggregate(checkout, run_dir, ids, mm.REPO, mm.MCPMARK_COMMIT)


# -- resume ------------------------------------------------------------------------


def test_done_mirrors_upstreams_resume_rule(tmp_path):
    run_dir = tmp_path / "run-1"
    _write_meta(run_dir, "cat/pass", PASS)
    _write_meta(run_dir, "cat/fail", FAIL)
    _write_meta(run_dir, "cat/fail_message", FAIL_MESSAGE)
    _write_meta(run_dir, "cat/ratelimited", RATE_LIMITED)
    (run_dir / "cat__partial").mkdir(parents=True)
    (run_dir / "cat__partial" / "meta.json").write_text("{ truncated")
    for task, done in (
        ("cat/pass", True),
        ("cat/fail", True),  # failed verification: finished, scored 0
        ("cat/fail_message", True),  # a non-retryable error: finished, scored 0
        ("cat/ratelimited", False),  # retryable: the pipeline wipes and retries
        ("cat/partial", False),  # a partial write: re-run
        ("cat/missing", False),  # never produced
    ):
        assert mm.McpmarkFilesystem(RunConfig(model="m", output_dir=tmp_path))._done(run_dir, task) is done, task


def test_run_skips_tasks_the_harness_already_finished(baked, tmp_path, monkeypatch):
    # max_failed_frac 1: the rate-limited task must not fail the run itself.
    bench = mkbench("filesystem", tmp_path, options={"max_failed_frac": "1"})
    first = FakeRunner(
        outcomes={
            "cat_a/task_0": PASS,
            "cat_a/task_1": RATE_LIMITED,  # retried on the next run
            "cat_b/task_0": FAIL,
        }
    )
    monkeypatch.setattr(mm.subprocess, "run", first)
    out1 = bench.run("http://llm:8000/v1", "granite-4.1-3b")
    assert out1["value"] == 0.5 and out1["n"] == 2  # the rate-limited task is out
    assert out1["task_failed"] == 1 and out1["incomplete"] is True

    second = FakeRunner(outcomes={"cat_a/task_1": PASS})
    monkeypatch.setattr(mm.subprocess, "run", second)
    out2 = bench.run("http://llm:8000/v1", "granite-4.1-3b")
    invoked = [c["argv"][c["argv"].index("--tasks") + 1] for c in second.calls]
    assert invoked == ["cat_a/task_1"]  # only the retryable task re-ran
    # cat_b/task_0 stays a scored 0 (a finished failure is never re-run): 2/3.
    assert out2["value"] == pytest.approx(2 / 3) and out2["n"] == 3 and out2["incomplete"] is False


# -- the postgres cluster -----------------------------------------------------------


def test_postgres_cluster_starts_on_a_free_node_locked_port(baked, tmp_path, monkeypatch, lock_dir):
    bench = mkbench("postgres", tmp_path)
    runner = FakeRunner(service="postgres")
    monkeypatch.setattr(mm.subprocess, "run", runner)
    with bench._postgres_server() as pg:
        assert pg == {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "15432",
            "POSTGRES_USERNAME": "postgres",
            "POSTGRES_PASSWORD": "mcpmark",
        }
        argvs = [c["argv"] for c in runner.calls]
        initdb = next(a for a in argvs if a[0].endswith("/initdb"))
        assert initdb[:5] == [str(mm.PG_BINDIR / "initdb"), "-D", initdb[2], "-U", "postgres"]
        assert initdb[5:] == ["--auth=trust"]
        start = next(a for a in argvs if a[0].endswith("/pg_ctl") and a[-1] == "start")
        assert start[0] == str(mm.PG_BINDIR / "pg_ctl")
        assert start[start.index("-o") + 1].startswith("-p 15432 -k ")  # port + unix socket dir
        assert "-w" in start
        data_dir = Path(initdb[2])
        assert data_dir.parent != tmp_path  # node-local scratch, not the output dir
    stop = next(a for a in (c["argv"] for c in runner.calls) if a[0].endswith("/pg_ctl") and a[-1] == "stop")
    assert "-m" in stop and stop[stop.index("-m") + 1] == "immediate"
    assert not data_dir.exists()  # cleaned up
    # The lock is free again.
    assert NodeLock("mcpmark-postgres-15432").try_acquire()


def test_postgres_cluster_skips_a_port_another_job_holds(baked, tmp_path, monkeypatch, lock_dir):
    holder = NodeLock("mcpmark-postgres-15432")
    assert holder.try_acquire()  # another job on this node
    bench = mkbench("postgres", tmp_path)
    runner = FakeRunner(service="postgres")
    monkeypatch.setattr(mm.subprocess, "run", runner)
    with bench._postgres_server() as pg:
        assert pg["POSTGRES_PORT"] == "15433"
        argvs = [c["argv"] for c in runner.calls]
        start = next(a for a in argvs if a[0].endswith("/pg_ctl") and a[-1] == "start")
        assert start[start.index("-o") + 1].startswith("-p 15433 -k ")
    holder.release()


def test_postgres_cluster_fails_loudly_when_no_port_is_free(baked, tmp_path, monkeypatch, lock_dir):
    held = [NodeLock(f"mcpmark-postgres-{port}") for port in mm.PG_PORTS]
    assert all(lock.try_acquire() for lock in held)
    bench = mkbench("postgres", tmp_path)
    monkeypatch.setattr(mm.subprocess, "run", FakeRunner(service="postgres"))
    with pytest.raises(SystemExit, match="no free postgres port"):
        with bench._postgres_server():
            pass


def test_postgres_preflight_runs_the_pipx_server_and_gates_on_it(baked, tmp_path, monkeypatch):
    bench = mkbench("postgres", tmp_path)
    env = bench._pipeline_env(bench._checkout(), "http://llm:8000/v1")
    runner = FakeRunner(service="postgres")
    monkeypatch.setattr(mm.subprocess, "run", runner)
    bench._postgres_preflight(env)
    assert runner.calls[0]["argv"] == ["pipx", "run", mm.POSTGRES_MCP, "--help"]
    assert runner.calls[0]["env"] is env

    def broken(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 3, "", "Connection closed")

    monkeypatch.setattr(mm.subprocess, "run", broken)
    with pytest.raises(SystemExit, match="preflight failed"):
        bench._postgres_preflight(env)


def test_postgres_run_starts_the_cluster_and_passes_it_to_the_pipeline(baked, tmp_path, monkeypatch, lock_dir):
    bench = mkbench("postgres", tmp_path)
    runner = FakeRunner(service="postgres", outcomes={"cat_a/task_0": PASS, "cat_a/task_1": FAIL})
    monkeypatch.setattr(mm.subprocess, "run", runner)
    out = bench.run("http://llm:8000/v1", "granite-4.1-3b")
    assert out["value"] == 0.5 and out["n"] == 2
    pipeline_calls = [c for c in runner.calls if "pipeline" in c["argv"][:3]]
    assert [c["argv"][c["argv"].index("--mcp") + 1] for c in pipeline_calls] == ["postgres"] * 2
    assert {c["env"]["POSTGRES_PORT"] for c in pipeline_calls} == {"15432"}
    assert pipeline_calls[0]["env"]["OPENAI_BASE_URL"] == "http://llm:8000/v1"
    # The cluster was stopped after the last task.
    assert any(
        a[-1] == "stop" for a in (c["argv"] for c in runner.calls) if a[0].endswith("/pg_ctl")
    )


# -- run(): the end-to-end shape ------------------------------------------------------


def test_run_writes_one_pipeline_invocation_per_task(baked, tmp_path, monkeypatch):
    bench = mkbench("filesystem", tmp_path, options={"max_failed_frac": "1"})
    runner = FakeRunner(
        outcomes={"cat_a/task_0": PASS, "cat_a/task_1": FAIL, "cat_b/task_0": RATE_LIMITED}
    )
    monkeypatch.setattr(mm.subprocess, "run", runner)
    out = bench.run("http://llm:8000/v1", "granite-4.1-3b")
    invoked = [c["argv"][c["argv"].index("--tasks") + 1] for c in runner.calls]
    assert invoked == ["cat_a/task_0", "cat_a/task_1", "cat_b/task_0"]  # serial, sorted
    assert out["value"] == 0.5 and out["n"] == 2 and out["n_total"] == 3
    assert out["dataset"] == mm.REPO and out["dataset_revision"] == mm.MCPMARK_COMMIT
    assert out["tasks"] == ["cat_a/task_0", "cat_a/task_1", "cat_b/task_0"]
    # The writable checkout is the pipeline's cwd and PYTHONPATH.
    checkout = tmp_path / "out" / "mcpmark-checkout"
    assert all(c["cwd"] == checkout for c in runner.calls)
    assert all(c["env"]["PYTHONPATH"] == str(checkout) for c in runner.calls)


def test_run_overrides_the_checkout_with_dataset(baked, tmp_path, monkeypatch):
    checkout = make_checkout(tmp_path / "custom", per_category=(("only", 1),))
    bench = mkbench("filesystem", tmp_path, dataset=str(checkout))
    runner = FakeRunner()
    monkeypatch.setattr(mm.subprocess, "run", runner)
    out = bench.run("http://llm:8000/v1", "granite-4.1-3b")
    assert out["n"] == 1
    assert runner.calls[0]["cwd"] == checkout
    assert not (tmp_path / "out" / "mcpmark-checkout").exists()  # no copy made
    assert out["dataset"] == str(checkout) and out["dataset_revision"] is None
