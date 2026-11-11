"""SWE-bench adapter tests with a fake sandbox (no containers, no model)."""

import os
from pathlib import Path

import pytest

pytest.importorskip("swebench")
pytest.importorskip("minisweagent")

from granite_evals.benchmarks import swebench as sb_mod  # noqa: E402
from granite_evals.registry import RunConfig  # noqa: E402
from granite_evals.sandbox import ExecResult  # noqa: E402

INSTANCE = {
    "instance_id": "org__repo-1",
    "image": "swebench/sweb.eval.x86_64.org_1776_repo-1:latest",
    "repo": "org/repo",
    "version": "1.0",
    "problem_statement": "fix it",
    "FAIL_TO_PASS": '["test_a.py::test_fixed"]',
    "PASS_TO_PASS": '["test_a.py::test_kept"]',
    "log_parser": "parse_log_pytest",
    "eval_type": "pass_and_fail",
    "eval_script": "cd /testbed\npytest test_a.py",
}

PYTEST_LOG = """>>>>> Start Test Output
PASSED test_a.py::test_fixed
PASSED test_a.py::test_kept
>>>>> End Test Output
"""


class FakeSandbox:
    def __init__(self, apply_ok=True, log=PYTEST_LOG):
        self.apply_ok, self.log, self.commands, self.files = apply_ok, log, [], {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def write_file(self, path, content):
        self.files[path] = content

    def execute(self, command, *, cwd="/", timeout=None, env=None, stdin=None):
        self.commands.append(command)
        if command.startswith(("git apply", "patch ")):
            return ExecResult("", 0 if self.apply_ok else 1)
        if command == "/bin/bash /eval.sh":
            return ExecResult(self.log, 0)
        return ExecResult("", 0)


@pytest.fixture
def bench(tmp_path):
    return sb_mod.SWEBenchVerified(RunConfig(model="m", output_dir=tmp_path))


def test_empty_patch_is_unresolved_without_a_sandbox(bench, tmp_path, monkeypatch):
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("no sandbox for empty patch"))
    assert bench._grade(INSTANCE, "  \n", tmp_path)["status"] == "empty_patch"


def test_resolved_when_tests_pass(bench, tmp_path, monkeypatch):
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    report = bench._grade(INSTANCE, "diff --git a/x b/x\n", tmp_path)
    assert report["status"] == "graded" and report["resolved"] is True
    assert sb_mod.PATCH_FILE in fake.files and "/eval.sh" in fake.files
    assert (tmp_path / "test_output.txt").read_text() == PYTEST_LOG


def test_unresolved_when_fail_to_pass_fails(bench, tmp_path, monkeypatch):
    log = PYTEST_LOG.replace("PASSED test_a.py::test_fixed", "FAILED test_a.py::test_fixed")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox(log=log))
    assert bench._grade(INSTANCE, "diff\n", tmp_path)["resolved"] is False


def test_patch_that_never_applies(bench, tmp_path, monkeypatch):
    fake = FakeSandbox(apply_ok=False)
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    assert bench._grade(INSTANCE, "diff\n", tmp_path)["status"] == "patch_failed"
    assert "/bin/bash /eval.sh" not in fake.commands


MIRROR = "https://mirror.example/maven2?a=1&b=2"


def test_no_maven_mirror_unless_configured(bench, tmp_path, monkeypatch):
    monkeypatch.delenv(sb_mod.MAVEN_MIRROR_ENV, raising=False)
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    bench._grade(INSTANCE, "diff --git a/x b/x\n", tmp_path)
    assert sb_mod.MAVEN_SETTINGS_STAGED not in fake.files
    assert sb_mod.MAVEN_SETTINGS_INSTALL not in fake.commands


def test_maven_mirror_is_installed_before_the_eval_script(bench, tmp_path, monkeypatch):
    monkeypatch.setenv(sb_mod.MAVEN_MIRROR_ENV, MIRROR)
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    assert bench._grade(INSTANCE, "diff --git a/x b/x\n", tmp_path)["resolved"] is True
    assert fake.files[sb_mod.MAVEN_SETTINGS_STAGED] == sb_mod.maven_settings(MIRROR)
    assert fake.commands.index(sb_mod.MAVEN_SETTINGS_INSTALL) < fake.commands.index("/bin/bash /eval.sh")


def test_maven_settings_mirror_only_central():
    import xml.etree.ElementTree as ET

    ns = {"s": "http://maven.apache.org/SETTINGS/1.2.0"}
    mirrors = ET.fromstring(sb_mod.maven_settings(MIRROR)).findall("s:mirrors/s:mirror", ns)
    assert len(mirrors) == 1
    assert mirrors[0].findtext("s:mirrorOf", namespaces=ns) == "central"
    assert mirrors[0].findtext("s:url", namespaces=ns) == MIRROR


@pytest.mark.parametrize("shipped", [False, True])
def test_maven_mirror_install_keeps_a_settings_xml_the_image_ships(tmp_path, shipped):
    import subprocess

    staged = tmp_path / "staged.xml"
    staged.write_text("<settings/>")
    home = tmp_path / "home"
    if shipped:
        (home / ".m2").mkdir(parents=True)
        (home / ".m2" / "settings.xml").write_text("<image/>")
    cmd = sb_mod.MAVEN_SETTINGS_INSTALL.replace(sb_mod.MAVEN_SETTINGS_STAGED, str(staged))
    subprocess.run(["bash", "-c", cmd], env={"HOME": str(home), "PATH": "/usr/bin:/bin"}, check=True)
    assert (home / ".m2" / "settings.xml").read_text() == ("<image/>" if shipped else "<settings/>")
    assert not staged.exists()


def test_agent_sandbox_gets_the_maven_mirror_too(bench, tmp_path, monkeypatch):
    from granite_evals.sandbox import minisweagent_env

    monkeypatch.setenv(sb_mod.MAVEN_MIRROR_ENV, MIRROR)
    box = ProSandbox()
    monkeypatch.setattr(minisweagent_env, "make_sandbox", lambda *a, **k: box)
    monkeypatch.setattr(sb_mod, "_kill_marked", lambda marker: None)
    _patch_agent(monkeypatch, RaisingAgent)
    bench._generate(INSTANCE, tmp_path, "http://h/v1", "m", 0)
    assert box.files[sb_mod.MAVEN_SETTINGS_STAGED] == sb_mod.maven_settings(MIRROR)
    assert sb_mod.MAVEN_SETTINGS_INSTALL in [c for c, _ in box.commands]
    assert sb_mod.GRADLE_INIT_INSTALL in [c for c, _ in box.commands]


def test_gradle_init_script_is_installed_with_the_maven_settings(bench, tmp_path, monkeypatch):
    monkeypatch.setenv(sb_mod.MAVEN_MIRROR_ENV, MIRROR)
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    bench._grade(INSTANCE, "diff --git a/x b/x\n", tmp_path)
    assert fake.files[sb_mod.GRADLE_INIT_STAGED] == sb_mod.gradle_init_script(MIRROR)
    assert fake.commands.index(sb_mod.GRADLE_INIT_INSTALL) < fake.commands.index("/bin/bash /eval.sh")


def test_no_gradle_init_script_unless_configured(bench, tmp_path, monkeypatch):
    monkeypatch.delenv(sb_mod.MAVEN_MIRROR_ENV, raising=False)
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    bench._grade(INSTANCE, "diff --git a/x b/x\n", tmp_path)
    assert sb_mod.GRADLE_INIT_STAGED not in fake.files


def test_gradle_init_script_quotes_the_mirror_and_matches_central_by_host():
    script = sb_mod.gradle_init_script("https://m.example/maven2?a='1'\\x")
    assert "def graniteMirror = 'https://m.example/maven2?a=\\'1\\'\\\\x'" in script
    for host in sb_mod.MAVEN_CENTRAL_HOSTS:
        assert f"'{host}'" in script
    assert "buildscript.repositories" in script and "dependencyResolutionManagement" in script


@pytest.mark.parametrize("gradle_user_home", [None, "gh"])
@pytest.mark.parametrize("shipped", [False, True])
def test_gradle_init_install_keeps_an_image_script_and_honours_gradle_user_home(tmp_path, shipped, gradle_user_home):
    import subprocess

    staged = tmp_path / "staged.gradle"
    staged.write_text("// granite")
    home = tmp_path / "home"
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    initd = home / ".gradle" / "init.d"
    if gradle_user_home:
        env["GRADLE_USER_HOME"] = str(tmp_path / gradle_user_home)
        initd = tmp_path / gradle_user_home / "init.d"
    if shipped:
        initd.mkdir(parents=True)
        (initd / "granite-maven-mirror.gradle").write_text("// image")
    cmd = sb_mod.GRADLE_INIT_INSTALL.replace(sb_mod.GRADLE_INIT_STAGED, str(staged))
    subprocess.run(["bash", "-c", cmd], env=env, check=True)
    assert (initd / "granite-maven-mirror.gradle").read_text() == ("// image" if shipped else "// granite")
    assert not staged.exists()


def test_grade_kills_what_its_sandbox_left_running(bench, tmp_path, monkeypatch):
    # e.g. an eval_script's mvn after its timeout: no PID namespace under enroot.
    envs, killed = [], []
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: envs.append(k.get("env")) or FakeSandbox())
    monkeypatch.setattr(sb_mod, "_kill_marked", killed.append)
    assert bench._grade(INSTANCE, "diff --git a/x b/x\n", tmp_path)["resolved"] is True
    assert killed == [envs[0][sb_mod.PRO_MARKER]] and INSTANCE["instance_id"] in killed[0]


class RaisingAgent:
    def __init__(self, model, env, **kwargs):
        pass

    def run(self, task, **kwargs):
        import litellm

        raise litellm.exceptions.ContextWindowExceededError("context length exceeded", "m", "hosted_vllm")

    def save(self, path, *extra):
        import json

        path.write_text(json.dumps(extra[0] if extra else {}))


def _patch_agent(monkeypatch, agent_cls):
    import minisweagent.agents.default as default_mod
    import minisweagent.models as models_mod

    monkeypatch.setattr(default_mod, "DefaultAgent", agent_cls)
    monkeypatch.setattr(models_mod, "get_model", lambda config: object())


def test_context_window_overflow_ends_the_run_with_an_empty_patch(bench, tmp_path, monkeypatch):
    # As mini-swe-agent's swebench runner records it: unresolved, persisted, not an error to retry.
    from granite_evals.sandbox import minisweagent_env

    sandbox_env, killed = {}, []
    monkeypatch.setattr(minisweagent_env, "make_sandbox", lambda *a, **k: sandbox_env.update(k.get("env") or {}) or ProSandbox())
    monkeypatch.setattr(sb_mod, "_kill_marked", killed.append)
    _patch_agent(monkeypatch, RaisingAgent)
    assert bench._generate(INSTANCE, tmp_path, "http://h/v1", "m", 0) == ""
    import json

    assert json.loads((tmp_path / "traj.json").read_text())["info"]["exit_status"] == "ContextWindowExceededError"
    assert killed == [sandbox_env[sb_mod.PRO_MARKER]]
    report = bench._instance(INSTANCE, tmp_path / "rep", "http://h/v1", "m", 0)
    assert report["status"] == "empty_patch" and (tmp_path / "rep" / INSTANCE["instance_id"] / "report.json").exists()


def test_other_agent_errors_still_propagate(bench, tmp_path, monkeypatch):
    from granite_evals.sandbox import minisweagent_env

    class Down(RaisingAgent):
        def run(self, task, **kwargs):
            raise ConnectionError("server down")

    monkeypatch.setattr(minisweagent_env, "make_sandbox", lambda *a, **k: ProSandbox())
    _patch_agent(monkeypatch, Down)
    report = bench._instance(INSTANCE, tmp_path, "http://h/v1", "m", 0)
    assert report["status"] == "error:ConnectionError"
    assert not (tmp_path / INSTANCE["instance_id"] / "report.json").exists()  # retried on resume


def test_run_aggregates_repeats_and_resumes(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchVerified(RunConfig(model="m", output_dir=tmp_path, repeats=2, limit=1, workers=1))
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [INSTANCE, {**INSTANCE, "instance_id": "z"}])
    calls = []
    monkeypatch.setattr(bench, "_generate", lambda inst, idir, url, served, k: calls.append(k) or ("diff\n" if k == 0 else ""))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox())
    out = bench.run("http://x/v1", "m")
    assert out["n"] == 1 and out["instances"] == ["org__repo-1"]
    assert [r["resolved"] for r in out["per_repeat"]] == [1, 0]
    assert out["value"] == 0.5
    # pass@2: resolved in one of the two runs
    assert out["pass_at_k"] == {"k": 2, "pass_at_1": 0.5, "pass_at_k": 1.0, "n": 1, "how": "resolved in any of the k runs"}
    bench.run("http://x/v1", "m")  # resume: nothing regenerated
    assert calls == [0, 1]


def test_agent_config_points_at_served_model(bench):
    config = bench._agent_config("http://h:1/v1", "granite", 2)
    assert config["model"]["model_name"] == "hosted_vllm/granite"
    assert config["model"]["model_kwargs"]["api_base"] == "http://h:1/v1"
    assert config["model"]["model_kwargs"]["seed"] == 2
    assert config["environment"]["cwd"] == "/testbed"


def test_gold_mode_grades_reference_patch_without_a_model(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchVerified(
        RunConfig(model="m", output_dir=tmp_path, repeats=1, workers=1, options={"patch": "gold"})
    )
    assert bench.needs_server() is False
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [{**INSTANCE, "patch": "gold-diff\n"}])
    monkeypatch.setattr(bench, "_generate", lambda *a: pytest.fail("gold mode must not call the agent"))
    fake = FakeSandbox()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    assert bench.run("", "m")["value"] == 1.0
    assert fake.files[sb_mod.PATCH_FILE] == "gold-diff\n"


# -- --phase generate / score ----------------------------------------------------


def _phased(tmp_path, monkeypatch, phase, calls, **options):
    bench = sb_mod.SWEBenchVerified(
        RunConfig(model="m", output_dir=tmp_path, repeats=2, workers=1, phase=phase, options=options)
    )
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [INSTANCE])
    monkeypatch.setattr(bench, "_generate", lambda inst, idir, url, served, k: calls.append(k) or ("diff\n" if k == 0 else ""))
    return bench


def test_generate_phase_writes_patches_and_grades_nothing(tmp_path, monkeypatch):
    calls = []
    bench = _phased(tmp_path, monkeypatch, "generate", calls)
    assert bench.splittable and bench.needs_server()
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("generate phase grades nothing"))
    out = bench.run("http://x/v1", "m")
    assert "value" not in out and out["n"] == 1 and calls == [0, 1]
    assert [r["generated"] for r in out["per_repeat"]] == [1, 1]
    idir = tmp_path / "repeat-0" / INSTANCE["instance_id"]
    assert (idir / "patch.diff").read_text() == "diff\n" and not (idir / "report.json").exists()
    bench.run("http://x/v1", "m")  # resume: nothing regenerated
    assert calls == [0, 1]


def test_generate_phase_leaves_failed_instances_for_a_rerun(tmp_path, monkeypatch):
    bench = _phased(tmp_path, monkeypatch, "generate", [])
    monkeypatch.setattr(bench, "_generate", lambda *a: (_ for _ in ()).throw(ConnectionError("down")))
    out = bench.run("http://x/v1", "m")
    assert out["n"] == 0 and out["per_repeat"][0]["failed"] == [INSTANCE["instance_id"]]
    assert not (tmp_path / "repeat-0" / INSTANCE["instance_id"] / "patch.diff").exists()


def test_score_phase_grades_saved_patches_without_the_model(tmp_path, monkeypatch):
    calls = []
    _phased(tmp_path, monkeypatch, "generate", calls).run("http://x/v1", "m")
    bench = _phased(tmp_path, monkeypatch, "score", calls)
    monkeypatch.setattr(bench, "_generate", lambda *a: pytest.fail("score phase must not call the agent"))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox())
    out = bench.run("", "m")
    assert out["value"] == 0.5 and [r["statuses"] for r in out["per_repeat"]] == [{"graded": 1}, {"empty_patch": 1}]
    assert (tmp_path / "repeat-0" / INSTANCE["instance_id"] / "report.json").exists()
    # Rerunnable: graded instances are skipped.
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("already graded"))
    assert bench.run("", "m")["value"] == 0.5
    # Same as --phase all on the same generations.
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox())
    both = _phased(tmp_path / "all", monkeypatch, "all", []).run("http://x/v1", "m")
    assert both["value"] == 0.5 and both["per_repeat"] == out["per_repeat"]


def test_score_phase_with_a_missing_patch_is_unresolved_and_never_generates(tmp_path, monkeypatch):
    calls = []
    _phased(tmp_path, monkeypatch, "generate", calls).run("http://x/v1", "m")
    (tmp_path / "repeat-0" / INSTANCE["instance_id"] / "patch.diff").unlink()
    bench = _phased(tmp_path, monkeypatch, "score", calls)
    monkeypatch.setattr(bench, "_generate", lambda *a: pytest.fail("score phase must not call the agent"))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox())
    out = bench.run("", "m")
    assert out["per_repeat"][0]["statuses"] == {"not_generated": 1} and out["value"] == 0.0
    assert not (tmp_path / "repeat-0" / INSTANCE["instance_id"] / "report.json").exists()  # a later score retries it


def test_gold_generate_phase_writes_the_reference_patch_without_a_server(tmp_path, monkeypatch):
    bench = _phased(tmp_path, monkeypatch, "generate", [], patch="gold")
    assert bench.needs_server() is False and bench.serves() is False
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [{**INSTANCE, "patch": "gold-diff\n"}])
    monkeypatch.setattr(bench, "_generate", lambda *a: pytest.fail("gold mode must not call the agent"))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("generate phase grades nothing"))
    assert bench.run("", "m")["n"] == 1
    assert (tmp_path / "repeat-1" / INSTANCE["instance_id"] / "patch.diff").read_text() == "gold-diff\n"


def test_check_mode_checks_in_the_score_phase_only(tmp_path, monkeypatch):
    def bench(phase):
        return sb_mod.SWEBenchMultilingual(
            RunConfig(model="none", output_dir=tmp_path, phase=phase, options={"check": "data"})
        )

    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [ML_INSTANCE])
    monkeypatch.setattr(sb_mod, "probe_image", lambda ref: pytest.fail("generate phase checks nothing"))
    out = bench("generate").run("", "none")
    assert out["n"] == 1 and "value" not in out and not (tmp_path / "check.json").exists()
    monkeypatch.setattr(sb_mod, "probe_image", lambda ref: {"status": 200, "digest": None})
    assert bench("score").run("", "none")["value"] == 1.0 and (tmp_path / "check.json").exists()


# -- SWE-bench Multilingual ----------------------------------------------------

ML_INSTANCE = {
    **INSTANCE,
    "instance_id": "tokio-rs__tokio-1",
    "image": "swebench/sweb.eval.x86_64.tokio-rs_1776_tokio-1:latest",
    "repo": "tokio-rs/tokio",
    "FAIL_TO_PASS": '["time::test_fixed"]',
    "PASS_TO_PASS": '["time::test_kept"]',
    "log_parser": "parse_log_cargo",
    "eval_script": "cd /testbed\ncargo test",
    "patch": "gold-diff\n",
}

CARGO_LOG = """>>>>> Start Test Output
test time::test_fixed ... ok
test time::test_kept ... ok
>>>>> End Test Output
"""


def test_multilingual_is_registered_like_verified(tmp_path):
    from granite_evals import registry

    cls = registry.get("swebench-multilingual")
    assert cls is sb_mod.SWEBenchMultilingual
    assert cls.metric == "pass@1 resolve rate" and cls.default_repeats == 1
    assert cls.dataset == "SWE-bench/SWE-bench_Multilingual" and len(cls.dataset_revision) == 40
    bench = cls(RunConfig(model="m", output_dir=tmp_path))
    assert bench.dataset_config() is None
    assert bench._agent_config("http://h/v1", "g", 0)["environment"]["cwd"] == "/testbed"


def test_multilingual_grades_with_the_instance_log_parser(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchMultilingual(RunConfig(model="m", output_dir=tmp_path))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox(log=CARGO_LOG))
    assert bench._grade(ML_INSTANCE, "diff\n", tmp_path)["resolved"] is True
    failing = CARGO_LOG.replace("test_fixed ... ok", "test_fixed ... FAILED")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: FakeSandbox(log=failing))
    assert bench._grade(ML_INSTANCE, "diff\n", tmp_path)["resolved"] is False


# -- SWE-bench Pro -------------------------------------------------------------

import hashlib  # noqa: E402

PRO_IID = "instance_org__repo-abc-vnan"
PRO_INSTANCE = {
    "instance_id": PRO_IID,
    "docker_image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{PRO_IID}",
    "patch": "hf-gold\n",
    "problem_statement": "not what the agent sees",
}
PRO_CONFIG_YAML = """agent:
  system_template: 'sys'
  instance_template: 'Please solve this issue: {{task}} on {{system}} {{machine}}'
  step_limit: 0
  cost_limit: 3.0
environment:
  env:
    PAGER: cat
model:
  model_kwargs:
    drop_params: true
    reasoning_effort: high
  model_class: litellm
"""


def pro_files(iid=PRO_IID):
    t = f"tasks/{iid}/"
    return {
        t + "instruction.md": b"A code repository is available in the `/app` directory. Do X.",
        t + "solution/gold_patch.diff": b"gold-diff\n",
        t + "tests/test.sh": b"#!/bin/bash\necho verifier\n",
        t + "tests/run_script.sh": b"#!/bin/bash\n",
        t + "tests/parser.py": b"print(1)\n",
        t + "tests/config.json": b'{"fail_to_pass": "[]"}',
        t + "tests/test_patch.patch": b"diff --git a/t b/t\n",
        sb_mod.PRO_AGENT_CONFIG: PRO_CONFIG_YAML.encode(),
    }


def make_fetch(files, calls=None):
    sums = "".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k, v in files.items()).encode()

    def fetch(path):
        if calls is not None:
            calls.append(path)
        return sums if path == "SHA256SUMS" else files[path]

    return fetch


@pytest.fixture
def pro(tmp_path):
    bench = sb_mod.SWEBenchPro(RunConfig(model="m", output_dir=tmp_path, repeats=1, workers=1))
    # A different commit skips the pinned SHA256SUMS digest, so fake sums verify.
    bench.tasks = sb_mod.ProTasks(tmp_path / "tasks", commit="test", fetch=make_fetch(pro_files()))
    return bench


class ProSandbox:
    """Fake of a V2 image: /app exists, test.sh writes the given reward."""

    def __init__(self, reward="1", apply_rc=0, patch="", pid_ns=True):
        self.reward, self.apply_rc, self.patch, self.pid_ns = reward, apply_rc, patch, pid_ns
        self.commands, self.files = [], {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def start(self):
        pass

    def close(self):
        pass

    def write_file(self, path, content):
        self.files[path] = content

    def execute(self, command, *, cwd="/", timeout=None, env=None, stdin=None):
        self.commands.append((command, cwd))
        if command == sb_mod.PRO_WORKDIR:
            return ExecResult("/app\n", 0)
        if command == sb_mod.PRO_APPLY:
            return ExecResult("Applied patch", self.apply_rc)
        if command == sb_mod.PRO_PIDNS_PROBE:
            return ExecResult("", 0) if self.pid_ns else ExecResult("unshare: unshare failed: Operation not permitted", 1)
        if command in (sb_mod.PRO_VERIFY_PIDNS_CMD, sb_mod.PRO_VERIFY_PLAIN_CMD):
            return ExecResult("", 0 if self.reward == "1" else 1)
        if command == sb_mod.PRO_LISTENING:
            return ExecResult("0100007F:18EB\n", 0)
        if command.startswith("tail -c ") and command.endswith("/logs/verifier/run-script-stdout.txt"):
            return ExecResult('{"stats": {}}\n', 0)
        if command.startswith("tail -c "):
            return ExecResult("tail: cannot open", 1)
        if command == "cat /logs/verifier/reward.txt":
            return ExecResult(f"{self.reward}\n", 0) if self.reward is not None else ExecResult("No such file", 1)
        if command == "cat /logs/verifier/test-stdout.txt":
            return ExecResult("RESULT: PASSED\n", 0)
        if command == "cat /logs/verifier/output.json":
            return ExecResult('{"tests": []}', 0)
        if command.startswith("uname"):
            return ExecResult("Linux\n5.14\n#1 SMP\nx86_64\n", 0)
        if command == "cat /tmp/model.patch":
            return ExecResult(self.patch, 0)
        return ExecResult("", 0)


def test_pro_is_registered_with_its_pins():
    from granite_evals import registry

    cls = registry.get("swebench-pro")
    assert cls.metric == "pass@1 resolve rate" and cls.default_repeats == 1
    assert cls.dataset == "ScaleAI/SWE-bench_Pro" and len(cls.dataset_revision) == 40
    assert len(sb_mod.PRO_COMMIT) == 40 and len(sb_mod.PRO_SHA256SUMS_SHA256) == 64


def test_pro_subset_selects_the_hf_config(tmp_path):
    cfg = lambda **o: sb_mod.SWEBenchPro(RunConfig(model="m", output_dir=tmp_path, options=o))  # noqa: E731
    assert cfg().dataset_config() == "default"
    assert cfg(subset="hard").dataset_config() == "hard"
    with pytest.raises(SystemExit):
        cfg(subset="v1").dataset_config()


def test_pro_task_files_are_verified_and_cached(tmp_path):
    files, calls = pro_files(), []
    tasks = sb_mod.ProTasks(tmp_path, commit="test", fetch=make_fetch(files, calls))
    assert set(tasks.tests(PRO_IID)) == {"test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch"}
    n = len(calls)
    again = sb_mod.ProTasks(tmp_path, commit="test", fetch=make_fetch(files, calls))
    assert again.tests(PRO_IID)["test.sh"].startswith("#!/bin/bash")
    assert len(calls) == n  # all served from the cache
    cached = tmp_path / "test" / f"tasks/{PRO_IID}/tests/test.sh"
    cached.write_text("corrupted")
    fresh = sb_mod.ProTasks(tmp_path, commit="test", fetch=make_fetch(files, calls))
    assert fresh.tests(PRO_IID)["test.sh"].startswith("#!/bin/bash")
    assert calls[n:] == [f"tasks/{PRO_IID}/tests/test.sh"]  # a bad cached file is fetched again

    sums_fetch = make_fetch(files)
    bad = sb_mod.ProTasks(tmp_path / "bad", commit="test", fetch=lambda p: sums_fetch(p) if p == "SHA256SUMS" else b"evil")
    with pytest.raises(RuntimeError, match="sha256"):
        bad.task_file(PRO_IID, "instruction.md")
    with pytest.raises(KeyError):
        sb_mod.ProTasks(tmp_path / "x", commit="test", fetch=make_fetch(files)).tests("instance_missing")


def test_pinned_sha256sums_digest_is_enforced(tmp_path):
    tasks = sb_mod.ProTasks(tmp_path, fetch=lambda p: b"not the pinned file")
    with pytest.raises(RuntimeError, match="sha256"):
        tasks.sums()


def test_pro_grade_runs_the_task_verifier_in_a_fresh_sandbox(pro, tmp_path, monkeypatch):
    fake = ProSandbox(reward="1")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda image, **k: fake if image == PRO_INSTANCE["docker_image"] else None)
    report = pro._grade(PRO_INSTANCE, "diff --git a/x b/x\n", tmp_path)
    assert report["status"] == "graded" and report["resolved"] is True and report["reward"] == 1.0
    assert fake.files["/tmp/replay.patch"] == "diff --git a/x b/x\n"
    assert set(fake.files) >= {f"/tests/{n}" for n in ("test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch")}
    commands = [c for c, _ in fake.commands]
    assert commands.index(sb_mod.PRO_APPLY) < commands.index(sb_mod.PRO_VERIFY_PIDNS_CMD)
    assert (sb_mod.PRO_VERIFY_PIDNS_CMD, "/app") in fake.commands
    assert sb_mod.PRO_VERIFY_PLAIN_CMD not in commands
    assert report["verifier_pid_ns"] is True
    assert "node_locks" not in report  # the fake task's scripts use no fixed resource
    assert (tmp_path / "test_output.txt").read_text() == "RESULT: PASSED\n"
    assert (tmp_path / "output.json").exists()
    # The verifier's raw logs and the ports already listening are kept for debugging.
    assert (tmp_path / "run-script-stdout.txt").read_text() == '{"stats": {}}\n'
    assert not (tmp_path / "run-script-stderr.txt").exists()
    assert (tmp_path / "listening.txt").read_text() == "0100007F:18EB\n"
    assert commands.index(sb_mod.PRO_LISTENING) < commands.index(sb_mod.PRO_VERIFY_PIDNS_CMD)


def test_pro_verifier_runs_in_its_own_pid_namespace_when_it_can():
    # Services test.sh starts (redis-server, Xvfb) must die with it, as with Harbor's container.
    assert sb_mod.PRO_VERIFY_PIDNS_CMD.startswith("unshare --pid --fork --mount-proc --kill-child /tests/test.sh ")
    assert sb_mod.PRO_PIDNS_PROBE == "unshare --pid --fork --mount-proc --kill-child true"
    for cmd in (sb_mod.PRO_VERIFY_PIDNS_CMD, sb_mod.PRO_VERIFY_PLAIN_CMD):
        assert cmd.endswith("> /logs/verifier/test-stdout.txt 2>&1")


def test_pro_verifier_without_pid_namespace_is_recorded_not_silent(pro, tmp_path, monkeypatch, caplog):
    fake = ProSandbox(reward="1", pid_ns=False)
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    with caplog.at_level("WARNING"):
        report = pro._grade(PRO_INSTANCE, "diff\n", tmp_path)
    commands = [c for c, _ in fake.commands]
    assert sb_mod.PRO_VERIFY_PLAIN_CMD in commands and sb_mod.PRO_VERIFY_PIDNS_CMD not in commands
    assert report["verifier_pid_ns"] is False and report["resolved"] is True
    assert "no PID namespace for the verifier" in caplog.text and "Operation not permitted" in caplog.text


def test_pro_grade_unresolved_on_zero_reward_or_missing_reward(pro, tmp_path, monkeypatch):
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: ProSandbox(reward="0"))
    assert pro._grade(PRO_INSTANCE, "diff\n", tmp_path) == {
        "instance_id": PRO_IID, "resolved": False, "status": "graded", "reward": 0.0, "apply_rc": 0, "verifier_rc": 1,
        "verifier_pid_ns": True,
    }
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: ProSandbox(reward=None))
    assert pro._grade(PRO_INSTANCE, "diff\n", tmp_path)["status"] == "no_reward"


def test_pro_grade_still_runs_the_verifier_when_the_patch_does_not_apply(pro, tmp_path, monkeypatch):
    # patch_replay grades whatever applied; the verifier decides.
    fake = ProSandbox(reward="0", apply_rc=1)
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    report = pro._grade(PRO_INSTANCE, "diff\n", tmp_path)
    assert report["apply_rc"] == 1 and report["status"] == "graded" and not report["resolved"]
    assert any(c == sb_mod.PRO_VERIFY_PIDNS_CMD for c, _ in fake.commands)


def test_pro_empty_patch_needs_no_sandbox(pro, tmp_path, monkeypatch):
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("no sandbox for an empty patch"))
    assert pro._grade(PRO_INSTANCE, "\n", tmp_path)["status"] == "empty_patch"


def test_pro_agent_config_layers_the_protocol_file_over_mini(pro):
    config = pro._agent_config("http://h:1/v1", "granite", 1)
    assert config["agent"]["instance_template"].startswith("Please solve this issue: {{task}}")
    assert config["agent"]["step_limit"] == 0 and config["agent"]["cost_limit"] == 0
    assert config["agent"]["wall_time_limit_seconds"] == sb_mod.PRO_AGENT_BUDGET_S
    assert config["model"]["model_name"] == "hosted_vllm/granite"
    assert config["model"]["model_kwargs"]["reasoning_effort"] == "high"
    assert config["model"]["model_kwargs"]["api_base"] == "http://h:1/v1"
    assert config["model"]["model_kwargs"]["seed"] == 1
    assert config["environment"]["env"]["PAGER"] == "cat"


def test_pro_generate_gives_the_instruction_and_captures_the_diff(pro, tmp_path, monkeypatch):
    fake = ProSandbox(patch="diff --git a/f b/f\n+fix\n")
    from granite_evals.sandbox import minisweagent_env

    import contextlib

    sandbox_env, killed, locks = {}, [], []

    @contextlib.contextmanager
    def fake_locks(keys, **kw):
        locks.append(("lock", tuple(keys)))
        yield 0.0
        locks.append(("unlock", len(killed)))  # released after the kill

    monkeypatch.setattr(sb_mod, "node_locks", fake_locks)
    monkeypatch.setattr(sb_mod, "pro_fixed_resources", lambda tests: ["x99"])
    monkeypatch.setattr(minisweagent_env, "make_sandbox", lambda *a, **k: sandbox_env.update(k.get("env") or {}) or fake)
    monkeypatch.setattr(sb_mod, "_kill_marked", killed.append)
    seen = {}

    class FakeAgent:
        def __init__(self, model, env, **kwargs):
            seen["agent"], seen["cwd"] = kwargs, env.config.cwd

        def run(self, task, **kwargs):
            seen["task"], seen["vars"] = task, kwargs
            return {"exit_status": "TimeExceeded", "submission": ""}  # no submit: the diff still counts

        def save(self, path, *extra):
            path.write_text("{}")

    import minisweagent.agents.default as default_mod
    import minisweagent.models as models_mod

    monkeypatch.setattr(default_mod, "DefaultAgent", FakeAgent)
    monkeypatch.setattr(models_mod, "get_model", lambda config: object())
    patch = pro._generate(PRO_INSTANCE, tmp_path, "http://h/v1", "granite", 0)
    assert patch == "diff --git a/f b/f\n+fix\n"
    assert seen["task"].startswith("A code repository is available in the `/app` directory")
    assert seen["cwd"] == "/app" and seen["vars"] == {"system": "Linux", "release": "5.14", "version": "#1 SMP", "machine": "x86_64"}
    assert (sb_mod.PRO_CAPTURE, "/app") in fake.commands
    assert (tmp_path / "traj.json").exists()
    # Every agent command carries the marker; what they left running is killed afterwards.
    assert killed == [sandbox_env[sb_mod.PRO_MARKER]] and PRO_IID in killed[0]
    assert locks == [("lock", ("x99",)), ("unlock", 1)]


def test_pro_grade_kills_what_its_sandbox_left_running(pro, tmp_path, monkeypatch):
    envs, killed = [], []
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: envs.append(k.get("env")) or ProSandbox(reward="1"))
    monkeypatch.setattr(sb_mod, "_kill_marked", killed.append)
    pro._grade(PRO_INSTANCE, "diff\n", tmp_path)
    assert killed == [envs[0][sb_mod.PRO_MARKER]]


def test_kill_marked_kills_only_processes_with_the_marker(tmp_path, monkeypatch):
    import signal

    proc = tmp_path / "proc"
    for pid, environ in {"101": b"A=1\0GRANITE_EVALS_SANDBOX_ID=t-1\0", "102": b"GRANITE_EVALS_SANDBOX_ID=t-10\0", "103": b"B=2\0"}.items():
        (proc / pid).mkdir(parents=True)
        (proc / pid / "environ").write_bytes(environ)
    (proc / "self").mkdir()
    real_listdir, real_isdir = os.listdir, os.path.isdir
    monkeypatch.setattr(sb_mod.os, "listdir", lambda p: real_listdir(proc) if p == "/proc" else real_listdir(p))
    monkeypatch.setattr(sb_mod.os.path, "isdir", lambda p: True if p == "/proc" else real_isdir(p))
    monkeypatch.setattr(sb_mod, "Path", lambda p: proc if p == "/proc" else Path(p))
    sent = []
    monkeypatch.setattr(sb_mod.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert sb_mod._kill_marked("t-1") == 1
    assert sent == [(101, signal.SIGKILL)]


# -- fixed resources and node locks ---------------------------------------------

NODEBB_RUN_SCRIPT = """#!/bin/bash
prepare_test_environment() {
  redis-server --daemonize yes --protected-mode no --appendonly yes
  while ! redis-cli ping; do sleep 1; done
  echo '{"url":"http://localhost:4568","secret":"s","database":"redis","redis":{"host":"127.0.0.1","port":6379},"port":"4568"}' > config.json
}
"""
QUTE_RUN_SCRIPT = """#!/bin/bash
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
pytest "$@"
"""


def test_pro_fixed_resources_come_from_the_scripts_that_run():
    assert sb_mod.pro_fixed_resources({"run_script.sh": NODEBB_RUN_SCRIPT}) == ["port-4568", "port-6379"]
    assert sb_mod.pro_fixed_resources({"run_script.sh": QUTE_RUN_SCRIPT}) == ["x99"]
    assert sb_mod.pro_fixed_resources({"test.sh": "redis-server --port 7000 &\n"}) == ["port-7000"]
    assert sb_mod.pro_fixed_resources({"run_script.sh": "redis-server --port 0 --unixsocket /tmp/r.sock\n"}) == []
    assert sb_mod.pro_fixed_resources({"run_script.sh": "Xvfb :1 &\nexport DISPLAY=:1\n"}) == ["x1"]
    # Port literals in the test code (test_patch) are test data, not services.
    assert sb_mod.pro_fixed_resources({"run_script.sh": "#!/bin/bash\ngo test ./...\n",
                                       "test_patch.patch": "+ addr := \"localhost:5432\"\n"}) == []


@pytest.mark.parametrize("abstract", [False, pytest.param(True, marks=pytest.mark.skipif(
    not __import__("sys").platform.startswith("linux"), reason="abstract unix sockets are Linux-only"))])
def test_node_lock_excludes_other_holders_until_released(tmp_path, monkeypatch, abstract):
    import uuid

    monkeypatch.setenv("GRANITE_EVALS_LOCK_DIR", str(tmp_path))
    key = f"test-{uuid.uuid4().hex[:8]}"
    first, second = sb_mod.NodeLock(key, abstract=abstract), sb_mod.NodeLock(key, abstract=abstract)
    first.acquire()
    assert second._try() is None  # held by another holder (another thread or job)
    assert sb_mod.NodeLock(f"{key}-other", abstract=abstract)._try() is not None  # other keys are free
    first.release()
    assert second.acquire() < 1.0
    second.release()


def test_node_locks_wait_for_the_holder(tmp_path, monkeypatch):
    import threading
    import time
    import uuid

    monkeypatch.setenv("GRANITE_EVALS_LOCK_DIR", str(tmp_path))
    keys = [f"port-{uuid.uuid4().hex[:6]}", f"x{uuid.uuid4().hex[:6]}"]
    events, holding = [], threading.Event()

    def holder():
        with sb_mod.node_locks(keys, poll_s=0.01):
            events.append("a-in")
            holding.set()
            time.sleep(0.2)
            events.append("a-out")

    t = threading.Thread(target=holder)
    t.start()
    holding.wait()
    with sb_mod.node_locks(list(reversed(keys)), poll_s=0.01) as waited:  # any order: taken sorted
        events.append("b-in")
    t.join()
    assert events == ["a-in", "a-out", "b-in"] and waited > 0.1
    with sb_mod.node_locks([]) as waited:
        assert waited == 0.0


def test_pro_grade_holds_its_locks_until_the_marker_kill(pro, tmp_path, monkeypatch):
    import contextlib

    events = []

    @contextlib.contextmanager
    def fake_locks(keys, **kw):
        events.append(("lock", tuple(keys)))
        yield 1.5
        events.append(("unlock", tuple(keys)))

    monkeypatch.setattr(sb_mod, "pro_fixed_resources", lambda tests: ["port-6379"])
    monkeypatch.setattr(sb_mod, "node_locks", fake_locks)
    monkeypatch.setattr(sb_mod, "_kill_marked", lambda marker: events.append(("kill",)))
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: ProSandbox(reward="1"))
    report = pro._grade(PRO_INSTANCE, "diff\n", tmp_path)
    assert events == [("lock", ("port-6379",)), ("kill",), ("unlock", ("port-6379",))]
    assert report["node_locks"] == ["port-6379"] and report["lock_wait_s"] == 1.5 and report["resolved"]


def test_pro_gold_run_uses_the_task_reference_patch(pro, tmp_path, monkeypatch):
    pro.config.options["patch"] = "gold"
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [PRO_INSTANCE])
    monkeypatch.setattr(pro, "_generate", lambda *a: pytest.fail("gold mode must not call the agent"))
    fake = ProSandbox(reward="1")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    out = pro.run("", "m")
    assert out["value"] == 1.0 and out["subset"] == "default"
    assert out["per_repeat"][0]["verifier_pid_ns"] == {"true": 1} and out["verifier_pid_ns"] == {"true": 1}
    assert out["verifier"]["commit"] == sb_mod.PRO_COMMIT
    assert fake.files["/tmp/replay.patch"] == "gold-diff\n"


def test_pro_generate_then_score(pro, tmp_path, monkeypatch):
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [PRO_INSTANCE])
    monkeypatch.setattr(pro, "_generate", lambda *a: "diff\n")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("generate phase grades nothing"))
    pro.config.phase = "generate"
    out = pro.run("http://x/v1", "m")
    assert out["n"] == 1 and "value" not in out and "verifier_pid_ns" not in out
    pro.config.phase = "score"
    monkeypatch.setattr(pro, "_generate", lambda *a: pytest.fail("score phase must not call the agent"))
    fake = ProSandbox(reward="1")
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: fake)
    out = pro.run("", "m")
    assert out["value"] == 1.0 and out["verifier_pid_ns"] == {"true": 1}
    assert fake.files["/tmp/replay.patch"] == "diff\n"


# -- check=data ----------------------------------------------------------------


def test_parse_image_ref():
    assert sb_mod.parse_image_ref("swebench/sweb.eval.x86_64.a_1776_b-1:latest") == (
        "registry-1.docker.io",
        "swebench/sweb.eval.x86_64.a_1776_b-1",
        "latest",
    )
    assert sb_mod.parse_image_ref(f"ghcr.io/scaleapi/swe-bench_pro-v2:{PRO_IID}") == (
        "ghcr.io",
        "scaleapi/swe-bench_pro-v2",
        PRO_IID,
    )
    assert sb_mod.parse_image_ref("ubuntu") == ("registry-1.docker.io", "library/ubuntu", "latest")


def test_probe_image_gets_an_anonymous_token():
    import httpx

    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), request.headers.get("authorization")))
        if request.url.path == "/token":
            return httpx.Response(200, json={"token": "t0k"})
        if request.headers.get("authorization") != "Bearer t0k":
            return httpx.Response(401, headers={"www-authenticate": 'Bearer realm="https://ghcr.io/token",service="ghcr.io"'})
        if request.url.path.endswith("/missing"):
            return httpx.Response(404)
        return httpx.Response(200, headers={"docker-content-digest": "sha256:abc"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert sb_mod.probe_image("ghcr.io/o/r:tag", client=client) == {"status": 200, "digest": "sha256:abc"}
    assert seen[1] == ("GET", "https://ghcr.io/token?service=ghcr.io&scope=repository%3Ao%2Fr%3Apull", None)
    assert seen[2][0] == "HEAD" and seen[2][2] == "Bearer t0k"
    assert sb_mod.probe_image("ghcr.io/o/r:missing", client=client)["status"] == 404


def test_check_mode_needs_no_model_and_reports_failures(tmp_path, monkeypatch):
    bench = sb_mod.SWEBenchMultilingual(
        RunConfig(model="none", output_dir=tmp_path, workers=2, options={"check": "data"})
    )
    assert bench.needs_server() is False
    bad = {**ML_INSTANCE, "instance_id": "bad-2", "log_parser": "parse_log_nope", "image": "x/missing:latest"}
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [ML_INSTANCE, bad])
    monkeypatch.setattr(sb_mod, "make_sandbox", lambda *a, **k: pytest.fail("check mode runs no sandbox"))
    monkeypatch.setattr(sb_mod, "probe_image", lambda ref: {"status": 404 if "missing" in ref else 200, "digest": None})
    out = bench.run("", "none")
    assert out["mode"] == "check=data" and out["n"] == 2 and out["ok"] == 1 and out["value"] == 0.5
    (failure,) = out["failures"]
    assert failure["instance_id"] == "bad-2"
    assert any("parse_log_nope" in p for p in failure["problems"])
    assert "image manifest: HTTP 404" in failure["problems"]
    assert (tmp_path / "check.json").exists()


def test_pro_check_mode_verifies_every_task_file(pro, tmp_path, monkeypatch):
    pro.config.options["check"] = "data"
    missing = {**PRO_INSTANCE, "instance_id": "instance_gone"}
    monkeypatch.setattr(sb_mod.data, "load_split", lambda *a, **k: [PRO_INSTANCE, missing])
    monkeypatch.setattr(sb_mod, "probe_image", lambda ref: {"status": 200, "digest": "d"})
    out = pro.run("", "none")
    assert out["ok"] == 1 and out["subset"] == "default"
    assert out["failures"][0]["instance_id"] == "instance_gone"
    assert "KeyError" in out["failures"][0]["problems"][0]
