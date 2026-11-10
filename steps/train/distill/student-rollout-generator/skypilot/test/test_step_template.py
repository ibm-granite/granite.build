"""Contract tests for the vllm-server step-template.yaml.

These tests pin the CONTRACT — the readiness gate, the two markers, the monitor
overlay. Whether the trainer's NCCL weight-sync group can span two LSF allocations is
not something a test can answer; build 9973e766 answered it on BlueVela.

The failures they do catch are the ones that cost a queue slot and produce no
diagnostic: a shell syntax error, a marker that never fires so the consumer target
never dispatches, or a monitor overlay that gbserver rejects at config time.
"""

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

_STEP = Path(__file__).resolve().parent.parent / "step-template.yaml"
# The run script as the template rendered it before the trainer stop file existed,
# keyed by case. A flat file, not a directory: publish-step copies every directory
# under test/ as a per-cluster build test.
_GOLDEN = Path(__file__).resolve().parent / "golden_run_scripts.json"


@pytest.fixture(scope="module")
def step():
    return yaml.safe_load(_STEP.read_text())


@pytest.fixture(scope="module")
def launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["vllm"]["config"]


@pytest.fixture(scope="module")
def run_script(launcher):
    return launcher["run"]


@pytest.fixture(scope="module")
def monitor(step):
    return step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]


def _as_shell(script):
    """Approximate what fill_objtemplate leaves behind, for a syntax check."""
    script = re.sub(r"\{%.*?%\}", "", script, flags=re.S)
    return re.sub(r"\{\{.*?\}\}", "X", script, flags=re.S)


class TestShape:
    def test_it_is_an_lsf_skypilot_service(self, step):
        assert step["name"] == "student-rollout-generator"
        assert step["type"] == "SERVICE"
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf", "slurm"]

    def test_it_declares_both_outputs(self, step):
        """The URL for the trainer, the cluster name for teardown. An undeclared
        output makes the resolver drop the NEWARTIFACT event silently."""
        assert set(step["outputs"]["optional"]) == {"vllm_url", "cluster_name"}

    def test_resources_are_left_to_the_build(self, launcher):
        """Accelerators and queue are site values; hardcoding them here is what
        makes a step unportable."""
        assert launcher["resources"] == {}

    def test_the_model_is_a_required_input(self, step):
        """Serving the wrong model is not a failure, it is a different algorithm
        that runs. For on-policy GOLD this must be the STUDENT."""
        assert set(step["inputs"]["required"]) == {"model"}
        assert "model_path" not in step["config"]["vllm_config"]


class TestGenerationFloorIsNotAServerSetting:
    """The floor travels with every request from the trainer (logit-distill's
    gold_config.min_completion_length). The server-side key and its env export were
    removed in #664; a recipe that still sets the key is refused, not ignored."""

    def test_the_key_is_gone_from_the_defaults(self, step):
        assert "min_completion_length" not in step["config"]["vllm_config"]

    def test_nothing_exports_a_server_side_floor(self, run_script):
        assert "GOLD_MIN_TOKENS" not in run_script

    def test_a_recipe_that_still_sets_it_is_refused(self, run_script):
        guard = "{% if config.vllm_config.min_completion_length is defined %}"
        assert guard in run_script
        block = run_script[run_script.index(guard) :]
        block = block[: block.index("{% endif %}")]
        assert "FATAL: vllm_config.min_completion_length is no longer read" in block
        assert "gold_config.min_completion_length" in block
        assert "exit 1" in block


class TestRunScriptIsValidShell:
    def test_it_parses(self, run_script):
        result = subprocess.run(
            ["bash", "-n"], input=_as_shell(run_script), capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_it_uses_the_container_interpreter(self, run_script):
        assert "/stage/" not in run_script
        assert 'PYBIN="python"' in run_script


class TestReadinessGate:
    """The whole reason the server is its own target: publishing the URL only when
    the server answers makes the dependency graph the health gate."""

    def test_the_server_is_backgrounded_not_execd(self, run_script):
        """An exec'd process cannot echo afterwards, so it could only publish the
        URL before the model had loaded — and the trainer would dispatch into a
        multi-minute load and fail to connect."""
        assert "SERVER_PID=$!" in run_script
        assert "exec " not in run_script

    def test_health_is_polled_before_publishing(self, run_script):
        health = run_script.index("/health/")
        marker = run_script.index("GB_ARTIFACT_ID:vllm_url")
        assert health < marker, "the URL is published before /health is checked"

    def test_a_dead_server_fails_rather_than_waiting_out_the_timeout(self, run_script):
        """SERVER_PID is a direct child here, unlike the reference launcher's local
        blaunch pid, so kill -0 is authoritative and needs no log heuristic."""
        assert 'kill -0 "$SERVER_PID"' in run_script
        assert "before becoming healthy" in run_script

    def test_the_timeout_is_bounded_and_generous(self, step):
        """A cold multi-GB load plus CUDA graph capture is minutes, and the first
        read off a shared filesystem is the slow one."""
        assert step["config"]["vllm_config"]["health_timeout_seconds"] >= 600

    def test_it_holds_the_allocation_afterwards(self, run_script):
        """A service that returns ends its own step, and the trainer would lose the
        server mid-run.

        Asserted as a PROPERTY of the hold section rather than as the script's last
        token: the section waits on the server exactly once, whether or not the
        lifetime cap is armed, and everything after that wait only runs once the
        server is gone -- and fails the target.
        """
        # Scoped to the hold section: an earlier wait in the readiness gate reaps a
        # server that died during startup, and it is not the one counted here.
        tail = run_script[run_script.index("# Hold the allocation") :].rstrip()
        assert tail.count('wait "$SERVER_PID"') == 1
        after = tail[tail.index('wait "$SERVER_PID"') :]
        # The step must not fall off the end of the script, and must not exit with
        # a bare server code that could be 0.
        assert after.rstrip().endswith('exit "$RC"')
        assert '[ "$RC" -ne 0 ] || RC=1' in after

    def test_the_lifetime_cap_defaults_off(self, step):
        """Every recipe that does not set it runs for hours and is torn down by
        its teardown target; a default cap would kill those servers mid-run."""
        assert step["config"]["vllm_config"]["max_lifetime_seconds"] == 0


class TestMarkers:
    def test_both_use_the_shipped_generic_artifact_rule(self, run_script):
        """ARTIFACT_STATE, not ARTIFACT_PATH: the mem:// store passes
        binding["state"] through verbatim, while a path would be run through
        filesystem normalisation and mangle http://host:8001 to /http:/host:8001.
        """
        for marker in ("GB_ARTIFACT_ID:vllm_url", "GB_ARTIFACT_ID:cluster_name"):
            assert marker in run_script
        assert "GB_ARTIFACT_PATH" not in run_script

    def test_each_marker_is_on_its_own_line(self, run_script):
        """get_events_from_log_line reassigns log_line after a matching config, so
        two markers sharing a line makes the second see a truncated one."""
        lines = [l for l in run_script.splitlines() if "GB_ARTIFACT_ID:" in l]
        assert len(lines) == 2
        for line in lines:
            assert line.count("GB_ARTIFACT_ID:") == 1

    def test_the_advertised_address_is_resolved_not_a_bare_hostname(self, run_script):
        """The consumer is in a DIFFERENT allocation, so the address has to resolve
        from there. Resolved the way distill-gold resolves MASTER_ADDR, because
        that is the validated path on this cluster."""
        assert "/etc/hosts" in run_script
        assert "getent ahostsv4" in run_script
        assert "GB_ARTIFACT_STATE:http://${ADDR}:${PORT}" in run_script


class TestMonitor:
    def test_it_extends_rather_than_replaces_the_shipped_rules(self, monitor):
        """resolve_monitor_config REJECTS an overlay that sets event_configs on a
        ref: lists replace wholesale, so it would drop the generic artifact rules
        both markers depend on. This is a config-time error, not a runtime one,
        but it costs a submission either way."""
        assert monitor["ref"] == "space://monitors/skypilot"
        assert "event_configs" not in monitor["config"]
        assert "extra_event_configs" in monitor["config"]

    def test_log_retrieval_is_periodic_not_a_startup_window(self, monitor):
        """A startup_window has to be guessed, and once it closes the scrape never
        fires again — so a load slower than the guess means the URL never
        publishes and the consumer never dispatches, with no error anywhere.
        """
        assert "'periodic'" in monitor["config"]["log_retrieval"]["mode"]

    def test_the_status_rule_shares_no_line_with_a_marker(self, monitor):
        rule = monitor["config"]["extra_event_configs"][0]
        assert "GB_ARTIFACT_ID" not in rule["line_regex"]


class TestVllmWorkarounds:
    """Carried over from the reference launcher, which is the only configuration
    that has served this model successfully."""

    def test_the_rpc_socket_path_is_kept_short(self, run_script):
        """vLLM binds an AF_UNIX socket under VLLM_RPC_BASE_PATH; the path is
        capped at 108 bytes and ZeroMQ refuses over 107. A default TMPDIR measured
        111 on this cluster and failed AFTER the engine started loading."""
        assert "VLLM_RPC_BASE_PATH=" in run_script
        assert "/tmp/vllm-" in run_script

    def test_the_attention_backend_is_pinned(self, launcher):
        assert launcher["envs"]["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"

    def test_data_parallel_defaults_to_the_allocation(self, step, run_script):
        """From the allocation rather than a parameter, so it always describes the
        node it got."""
        assert step["config"]["vllm_config"]["data_parallel_size"] == ""
        assert 'DP="$GPUS"' in run_script

    def test_the_gpu_count_comes_from_what_skypilot_sets(self, run_script):
        """SKYPILOT_NUM_GPUS_PER_NODE is set on every backend; NUM_GPUS_PER_NODE is
        set by nothing at all. Reading only the latter made a 1-GPU allocation start
        8 data-parallel ranks, which vLLM then failed to place ("Error setting
        CUDA_VISIBLE_DEVICES: local range: [5, 6) base value: '0'" — smoke build
        04be7abd, target vllm-server)."""
        assert (
            'GPUS="${SKYPILOT_NUM_GPUS_PER_NODE:-${NUM_GPUS_PER_NODE:-8}}"'
            in run_script
        )


def _jinja(text, values):
    """Render text (the template's run block, or a section of it) with the step's
    vllm_config defaults overridden by values."""
    from jinja2 import StrictUndefined, Template

    defaults = yaml.safe_load(_STEP.read_text())["config"]["vllm_config"]
    config = {"vllm_config": dict(defaults, **values)}
    return Template(text, undefined=StrictUndefined).render(
        config=config, bindings={"model": {"binding": {"path": "/models/student"}}}
    )


def _section(run_script, begin, end=None):
    """One section of the run block, cut at its own comment headers."""
    start = run_script.index(begin)
    return run_script[start : run_script.index(end, start) if end else None]


def _run_under_launcher(tmp_path, body, server, values, send=None, on_hold=None):
    """Run `body` the way the launcher does -- with `set -eu` in front of it -- against a
    stand-in server process, and return (exit code, stderr).

    `send`, if given, is a signal delivered to the script itself (not its process
    group) once it logs that it is holding the allocation, i.e. once its trap is armed.
    `on_hold`, if given, is called at that same point, before any signal.

    The output goes to files, not pipes: the watchdog's `sleep` outlives its killed
    subshell and would hold a pipe open until it finished. The whole session is killed
    afterwards for the same reason.
    """
    body = _jinja(body, values)
    assert "{{" not in body, "a template value the test did not supply"
    script = tmp_path / "run.sh"
    script.write_text(f"set -eu\n({server}) &\nSERVER_PID=$!\n{body}\n")
    out, err = tmp_path / "out", tmp_path / "err"
    with out.open("w") as o, err.open("w") as e:
        proc = subprocess.Popen(
            ["bash", str(script)], stdout=o, stderr=e, start_new_session=True
        )
        try:
            if send is not None or on_hold is not None:
                deadline = time.monotonic() + 20
                while "holding the allocation" not in out.read_text():
                    assert proc.poll() is None, err.read_text()
                    assert time.monotonic() < deadline, "the hold never started"
                    time.sleep(0.05)
                if on_hold is not None:
                    on_hold()
                if send is not None:
                    proc.send_signal(send)
            rc = proc.wait(timeout=30)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return rc, err.read_text()


class TestFailuresAreReportedUnderTheLauncherPrefix:
    """The launcher prepends `set -eu` to every run block, so a bare `wait` on a dead
    server ends the script before the line that reports why. These run the template's
    own sections under that prefix."""

    def test_a_server_that_dies_while_starting_says_so(self, run_script, tmp_path):
        gate = _section(run_script, "# Wait for readiness", "# Healthy.")
        rc, err = _run_under_launcher(
            tmp_path,
            f'PYBIN=python3\nPORT=9\n{gate}\necho "unreachable: marked healthy" >&2',
            server="exit 7",
            values={"health_timeout_seconds": 30, "health_poll_seconds": 0.1},
        )
        assert "server exited with 7 before becoming healthy" in err
        assert rc == 1

    def test_a_server_the_lifetime_cap_kills_says_so(self, run_script, tmp_path):
        hold = _section(run_script, "MAX_LIFETIME=")
        rc, err = _run_under_launcher(
            tmp_path, hold, server="sleep 20", values={"max_lifetime_seconds": 1}
        )
        assert "lifetime cap of 1s reached" in err
        assert rc == 1

    def test_a_server_that_dies_under_the_cap_still_fails_the_step(
        self, run_script, tmp_path
    ):
        """Capturing the exit code must not swallow it: a crash that is not the
        watchdog's doing has to fail the step with the server's own code."""
        hold = _section(run_script, "MAX_LIFETIME=")
        rc, err = _run_under_launcher(
            tmp_path, hold, server="exit 3", values={"max_lifetime_seconds": 60}
        )
        assert "lifetime cap" not in err.replace("lifetime cap 60s armed", "")
        assert rc == 3


_EARLY_DEATH = "the vLLM server exited (rc={}) before the teardown target ended it"


class TestADeadServerFailsItsTarget:
    """A server that dies before the teardown target ends it must leave its target
    FAILED, never SUCCESS. On a build retry with target reuse, gbserver reuses SUCCESS
    targets of the same build, so a dead server recorded as SUCCESS would hand the
    retried trainer a dead URL; a FAILED one is relaunched alongside it.

    A normal teardown is not decided here: launch_skypilot_teardown records the
    cluster as intentionally torn down before it downs it, and the server's monitor
    then ends as SUCCESS whatever the job's exit status was."""

    @pytest.mark.parametrize("code", [0, 3, 137])
    def test_any_uncapped_server_exit_fails_the_step(self, run_script, tmp_path, code):
        """0 included: vLLM exits cleanly on a SIGTERM aimed at it, e.g. an idle
        kill, and a clean exit is what used to end the target SUCCESS."""
        hold = _section(run_script, "MAX_LIFETIME=")
        rc, err = _run_under_launcher(
            tmp_path, hold, server=f"exit {code}", values={"max_lifetime_seconds": 0}
        )
        assert "FATAL: " + _EARLY_DEATH.format(code) in err
        assert rc == (code or 1)

    def test_a_clean_exit_under_the_cap_fails_the_step(self, run_script, tmp_path):
        hold = _section(run_script, "MAX_LIFETIME=")
        rc, err = _run_under_launcher(
            tmp_path, hold, server="exit 0", values={"max_lifetime_seconds": 60}
        )
        assert "FATAL: " + _EARLY_DEATH.format(0) in err
        assert rc == 1

    def test_the_cap_still_reports_itself_and_exits_1(self, run_script, tmp_path):
        hold = _section(run_script, "MAX_LIFETIME=")
        rc, err = _run_under_launcher(
            tmp_path, hold, server="sleep 20", values={"max_lifetime_seconds": 1}
        )
        assert "lifetime cap of 1s reached" in err
        assert _EARLY_DEATH.format(143) not in err
        assert rc == 1

    @pytest.mark.parametrize(
        "sig, name, code", [(signal.SIGTERM, "SIGTERM", 143), (signal.SIGINT, "SIGINT", 130)]
    )
    @pytest.mark.parametrize("cap", [0, 60])
    def test_a_signal_while_serving_is_trapped(
        self, run_script, tmp_path, sig, name, code, cap
    ):
        """The script cannot tell a teardown's signal from a scheduler's, so the trap
        keeps the status a signal-killed shell reports today (128 + signo) and says
        why; gbserver, not this code, is what turns a teardown into SUCCESS."""
        hold = _section(run_script, "MAX_LIFETIME=")
        rc, err = _run_under_launcher(
            tmp_path,
            hold,
            server="sleep 20",
            values={"max_lifetime_seconds": cap},
            send=sig,
        )
        assert f"FATAL: received {name} while serving" in err
        assert rc == code

    def test_the_trap_is_armed_only_once_serving(self, run_script):
        """Before the URL is published nothing depends on this target, and the
        readiness gate's own failures already exit 1."""
        trap = run_script.index("trap ")
        assert trap > run_script.index("GB_ARTIFACT_ID:vllm_url")


class TestSlurmExecutorEnv:
    def test_the_stripped_env_is_restored(self, launcher):
        assert launcher["envs"]["PYTHONPATH"] == "/opt/distill/src"
        assert launcher["envs"]["VIRTUAL_ENV"] == "/stage/.venv"


_STOP_FATAL = "FATAL: the trainer failed ({}); stopping the rollout server"


class TestTrainerStopFile:
    """vllm_config.stop_file: logit-distill's rank 0 writes it when the trainer exits
    non-zero (its rollout_stop_file), and the server stops at once and fails its
    target, instead of holding the allocation until max_lifetime_seconds because the
    teardown target, gated on the trainer's checkpoint, can never run."""

    @pytest.mark.parametrize(
        "golden,values", [("defaults", {}), ("lifetime_cap", {"max_lifetime_seconds": 7200})]
    )
    def test_empty_renders_byte_identical(self, run_script, golden, values):
        """The goldens were rendered from the template before stop_file existed, so a
        recipe that does not set it runs exactly the script it ran before."""
        expected = json.loads(_GOLDEN.read_text())[golden]
        assert _jinja(run_script, values) == expected
        assert _jinja(run_script, dict(values, stop_file="")) == expected

    def test_the_keys_and_their_defaults(self, step):
        assert step["config"]["vllm_config"]["stop_file"] == ""
        assert step["config"]["vllm_config"]["stop_poll_seconds"] == 30

    def test_the_rendered_script_is_valid_shell(self, run_script):
        result = subprocess.run(
            ["bash", "-n"],
            input=_jinja(run_script, {"stop_file": "/w/run/b1/rollout.stop"}),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_the_stale_file_is_removed_before_the_url_is_published(self, run_script):
        rendered = _jinja(run_script, {"stop_file": "/w/run/b1/rollout.stop"})
        assert rendered.index('rm -f "$STOP_FILE"') < rendered.index(
            "GB_ARTIFACT_ID:vllm_url"
        )

    def test_one_wait_point(self, run_script):
        rendered = _jinja(run_script, {"stop_file": "/w/s", "max_lifetime_seconds": 60})
        tail = rendered[rendered.index("# Hold the allocation") :]
        assert tail.count('wait "$SERVER_PID"') == 1

    @staticmethod
    def _hold(
        tmp_path, run_script, server, stale=False, on_hold=None, send=None, **values
    ):
        """The script from the stale-file removal on, against a stand-in server."""
        stop = tmp_path / "shared" / "rollout.stop"
        if stale:
            stop.parent.mkdir(parents=True)
            stop.write_text("tr-old\nan earlier attempt's trainer failed\n")
        values = dict(stop_file=str(stop), stop_poll_seconds=0.1, **values)
        hold = _section(run_script, "# Healthy.")
        body = "DEADLINE=0\nADDR=127.0.0.1\nPORT=9\n" + hold
        rc, err = _run_under_launcher(
            tmp_path, body, server=server, values=values, on_hold=on_hold, send=send
        )
        return rc, err, stop

    @pytest.mark.parametrize("cap", [0, 60])
    def test_the_marker_stops_the_server_and_fails_the_step(
        self, run_script, tmp_path, cap
    ):
        stop = tmp_path / "shared" / "rollout.stop"

        def trainer_fails():
            stop.parent.mkdir(parents=True, exist_ok=True)
            stop.write_text("tr-7\nlogit-distill rank 0 exited with status 1\n")

        started = time.monotonic()
        rc, err, _ = self._hold(
            tmp_path,
            run_script,
            server="sleep 20",
            on_hold=trainer_fails,
            max_lifetime_seconds=cap,
        )
        assert rc == 1
        assert (
            _STOP_FATAL.format("tr-7; logit-distill rank 0 exited with status 1")
            in err
        )
        # Not the cap's message, nor the early-death one: the server was stopped.
        assert "lifetime cap of" not in err
        assert "before the teardown target ended it" not in err
        assert time.monotonic() - started < 15, "the server ran out its own sleep"

    def test_a_stale_marker_is_removed_and_not_acted_on(self, run_script, tmp_path):
        rc, err, stop = self._hold(
            tmp_path, run_script, server="sleep 1; exit 0", stale=True
        )
        assert not stop.exists()
        assert "the trainer failed" not in err
        # The server ended on its own, which still fails the target.
        assert "FATAL: " + _EARLY_DEATH.format(0) in err
        assert rc == 1

    def test_a_signal_while_polling_is_still_trapped(self, run_script, tmp_path):
        rc, err, _ = self._hold(
            tmp_path, run_script, server="sleep 20", send=signal.SIGTERM
        )
        assert "FATAL: received SIGTERM while serving" in err
        assert "the trainer failed" not in err
        assert rc == 143
