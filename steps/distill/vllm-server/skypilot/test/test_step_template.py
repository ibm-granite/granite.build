"""Contract tests for the vllm-server step-template.yaml.

These tests pin the CONTRACT — the readiness gate, the two markers, the monitor
overlay. Whether the trainer's NCCL weight-sync group can span two LSF allocations is
not something a test can answer; build 9973e766 answered it on BlueVela.

The failures they do catch are the ones that cost a queue slot and produce no
diagnostic: a shell syntax error, a marker that never fires so the consumer target
never dispatches, or a monitor overlay that gbserver rejects at config time.
"""

import os
import re
import signal
import subprocess
from pathlib import Path

import pytest
import yaml

_STEP = Path(__file__).resolve().parent.parent / "step-template.yaml"


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
        assert step["name"] == "vllm-server"
        assert step["type"] == "SERVICE"
        assert step["environment_configs"]["Skypilot"]["subtypes"] == ["lsf"]

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


class TestRunScriptIsValidShell:
    def test_it_parses(self, run_script):
        result = subprocess.run(
            ["bash", "-n"], input=_as_shell(run_script), capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_it_uses_the_container_interpreter(self, run_script):
        assert "PATH=/stage/.venv/bin:$PATH" in run_script
        assert "PYBIN=/stage/.venv/bin/python" in run_script


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

        Asserted as a PROPERTY of every path rather than as the script's last token.
        This test used to read `endswith('wait "$SERVER_PID"')`, which was true only
        while the tail was a single unconditional wait; once max_lifetime_seconds
        added a watchdog the script ends in `fi` and the waits moved inside two
        branches. The literal form would have failed a correct step -- and it did,
        the moment the authoring template caught up with the published asset.
        """
        # Scoped to the hold section: an earlier wait in the readiness gate reaps a
        # server that died during startup, and it is not one of the two counted here.
        tail = run_script[run_script.index("# Hold the allocation") :].rstrip()
        # One wait per branch: with the cap armed, and without it.
        assert tail.count('wait "$SERVER_PID"') == 2
        # Nothing after the waits may run in the normal path except the watchdog
        # cleanup, the reaped-flag check and the exit with the server's code, all of
        # which only follow a returned server. In particular the step must not end by falling off the end of the
        # script with the server still up.
        assert tail.endswith("fi")
        for branch in ("$MAX_LIFETIME", "else"):
            assert branch in tail

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


def _section(run_script, begin, end=None):
    """One section of the run block, cut at its own comment headers."""
    start = run_script.index(begin)
    return run_script[start : run_script.index(end, start) if end else None]


def _run_under_launcher(tmp_path, body, server, values):
    """Run `body` the way the launcher does -- with `set -eu` in front of it -- against a
    stand-in server process, and return (exit code, stderr).

    The output goes to files, not pipes: the watchdog's `sleep` outlives its killed
    subshell and would hold a pipe open until it finished. The whole session is killed
    afterwards for the same reason.
    """
    for key, value in values.items():
        body = body.replace("{{ config.vllm_config.%s }}" % key, str(value))
    assert "{{" not in body, "a template value the test did not supply"
    script = tmp_path / "run.sh"
    script.write_text(f"set -eu\n({server}) &\nSERVER_PID=$!\n{body}\n")
    out, err = tmp_path / "out", tmp_path / "err"
    with out.open("w") as o, err.open("w") as e:
        proc = subprocess.Popen(
            ["bash", str(script)], stdout=o, stderr=e, start_new_session=True
        )
        try:
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
