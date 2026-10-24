"""Tests for autotune.slurm — host-list parsing, CPU sizing, srun argv and teardown."""

import subprocess
import sys

import pytest

from autotune.slurm import ray_down, ray_up_srun
from autotune.slurm.ray_up_srun import (
    _build_srun_cmd,
    _cpus_per_node,
    _expand_hostlist,
    _read_slurm_nodelist,
    start_multinode_ray_cluster_srun,
)


class TestExpandHostlist:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("node1", ["node1"]),
            ("a,b", ["a", "b"]),
            ("p1-r01-n[1-3]", ["p1-r01-n1", "p1-r01-n2", "p1-r01-n3"]),
            ("p1-r01-n[1,3-4],p2-r03-n2", ["p1-r01-n1", "p1-r01-n3", "p1-r01-n4", "p2-r03-n2"]),
            ("n[08-10]", ["n08", "n09", "n10"]),
            ("n[1-2]-ib", ["n1-ib", "n2-ib"]),
        ],
    )
    def test_expansion(self, expr, expected):
        assert _expand_hostlist(expr) == expected


class TestReadSlurmNodelist:
    def test_raises_outside_allocation(self, monkeypatch):
        monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
        monkeypatch.delenv("SLURM_NODELIST", raising=False)
        with pytest.raises(RuntimeError, match="SLURM_JOB_NODELIST"):
            _read_slurm_nodelist()

    def test_uses_scontrol(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_NODELIST", "h[1-2]")

        def fake_run(cmd, **kw):
            assert cmd == ["scontrol", "show", "hostnames", "h[1-2]"]
            return subprocess.CompletedProcess(cmd, 0, stdout="h1\nh2\nh1\n", stderr="")

        monkeypatch.setattr(ray_up_srun.subprocess, "run", fake_run)
        assert _read_slurm_nodelist() == ["h1", "h2"]  # deduped, ordered

    def test_falls_back_without_scontrol(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_NODELIST", "h[1-2]")

        def missing(cmd, **kw):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(ray_up_srun.subprocess, "run", missing)
        assert _read_slurm_nodelist() == ["h1", "h2"]


class TestCpusPerNode:
    def test_run_length_encoding(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "96(x2),48")
        assert _cpus_per_node(["a", "b", "c"]) == {"a": 96, "b": 96, "c": 48}

    def test_count_mismatch_falls_back_to_cpus_on_node(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "96")
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "64")
        assert _cpus_per_node(["a", "b"]) == {"a": 64, "b": 64}

    def test_malformed_falls_back(self, monkeypatch):
        monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "garbage")
        monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
        assert _cpus_per_node(["a"]) == {"a": 8}


class TestCpusFollowNodelistOrder:
    """The head is rotated to the front of the worker list; CPU counts are not.

    SLURM_JOB_CPUS_PER_NODE is in SLURM_JOB_NODELIST order. Mapping it onto the
    rotated list gave each host another host's count, so on a mixed-size
    allocation srun asked a 48-CPU node for 96.
    """

    def _spawned_cpus(self, monkeypatch, tmp_path, *, nodelist, head, cpus_per_node):
        monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", cpus_per_node)
        monkeypatch.setattr(ray_up_srun, "_read_slurm_nodelist", lambda: list(nodelist))
        monkeypatch.setattr(ray_up_srun.socket, "gethostname", lambda: head)
        monkeypatch.setattr(ray_up_srun, "dump_env_fingerprint", lambda: None)
        monkeypatch.setattr(ray_up_srun, "resolve_ray_temp_dir", lambda suffix: str(tmp_path / suffix))
        monkeypatch.setattr(ray_up_srun, "_start_head", lambda d: ("10.0.0.1", 6379, 8265))
        monkeypatch.setattr(ray_up_srun, "_ray_init_with_timeout", lambda **kw: True)
        monkeypatch.setattr(ray_up_srun, "_wait_for_workers_blaunch", lambda **kw: None)
        monkeypatch.setattr(ray_up_srun, "summarize_ray_nodes", lambda: None)
        monkeypatch.setattr(ray_up_srun.ray, "cluster_resources", lambda: {"GPU": 8 * len(nodelist)})

        spawned = {}

        def fake_popen(argv, log_path, env):
            # The head's worker runs locally (no srun, so no --nodelist).
            host = next((a.split("=", 1)[1] for a in argv if a.startswith("--nodelist=")), head.split(".")[0])
            spawned[host] = int(argv[argv.index("--num_cpus") + 1])
            return subprocess.Popen(["true"])

        monkeypatch.setattr(ray_up_srun, "_popen_worker", fake_popen)
        start_multinode_ray_cluster_srun(num_workers=len(nodelist), gpus_per_worker=8, log_dir=str(tmp_path))
        return spawned

    def test_rotated_head_keeps_each_hosts_own_count(self, monkeypatch, tmp_path):
        spawned = self._spawned_cpus(
            monkeypatch, tmp_path, nodelist=["h1", "h2", "h3"], head="h3", cpus_per_node="96(x2),48"
        )
        assert spawned == {"h1": 96, "h2": 96, "h3": 48}

    def test_fqdn_head_still_finds_its_count(self, monkeypatch, tmp_path):
        spawned = self._spawned_cpus(
            monkeypatch, tmp_path, nodelist=["h1", "h2"], head="h2.cluster.example", cpus_per_node="96,48"
        )
        assert spawned == {"h1": 96, "h2": 48}


class TestBuildSrunCmd:
    def _argv(self):
        return _build_srun_cmd(host="h2", head_address="10.0.0.1:6379", gpus_per_worker=8, cpus=96)

    def test_one_step_pinned_to_the_host(self):
        argv = self._argv()
        assert argv[0] == "srun"
        assert "--nodes=1" in argv and "--ntasks=1" in argv
        assert "--nodelist=h2" in argv

    def test_step_gets_the_hosts_cpus_not_one(self):
        # Without --cpus-per-task, task/cgroup confines the raylet to 1 core.
        assert "--cpus-per-task=96" in self._argv()

    def test_overlap_mem_and_no_mpi(self):
        argv = self._argv()
        assert "--overlap" in argv
        assert "--mem=0" in argv
        assert "--mpi=none" in argv

    def test_no_gres_flag(self):
        # The step inherits the job's GRES; naming one clashes with --gpus jobs.
        assert not any(a.startswith(("--gres", "--gpus")) for a in self._argv())

    def test_runs_worker_entry_with_current_interpreter_no_shell(self):
        argv = self._argv()
        i = argv.index(sys.executable)
        assert argv[i : i + 3] == [sys.executable, "-m", "autotune.lsf.worker_entry"]
        tail = argv[i + 3 :]
        assert tail == ["--head_address", "10.0.0.1:6379", "--num_gpus", "8", "--num_cpus", "96"]
        assert "bash" not in argv and "conda" not in " ".join(argv)


class TestRequiredArgs:
    def test_missing_log_dir_raises(self):
        with pytest.raises(ValueError, match="log_dir is required"):
            start_multinode_ray_cluster_srun(num_workers=1, gpus_per_worker=1)

    def test_non_positive_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            start_multinode_ray_cluster_srun(num_workers=0, gpus_per_worker=1, log_dir="/tmp/x")

    def test_node_count_mismatch_raises_before_ray_starts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SLURM_JOB_NODELIST", "h[1-2]")
        monkeypatch.setattr(ray_up_srun, "_read_slurm_nodelist", lambda: ["h1", "h2"])
        monkeypatch.setattr(ray_up_srun.socket, "gethostname", lambda: "h1")
        monkeypatch.setattr(ray_up_srun, "dump_env_fingerprint", lambda: None)

        def no_head(*a, **kw):
            raise AssertionError("head must not start on a mismatched allocation")

        monkeypatch.setattr(ray_up_srun, "_start_head", no_head)
        with pytest.raises(RuntimeError, match="num_workers=3"):
            start_multinode_ray_cluster_srun(num_workers=3, gpus_per_worker=8, log_dir=str(tmp_path))


class TestTeardown:
    def test_empty_handle_is_noop(self, monkeypatch):
        monkeypatch.setattr(ray_down, "_ray_shutdown_with_timeout", lambda **kw: pytest.fail("should not run"))
        ray_down.stop_multinode_ray_cluster_srun({})

    def test_signals_workers_and_stops_head(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ray_down, "_ray_shutdown_with_timeout", lambda **kw: calls.append("shutdown"))
        monkeypatch.setattr(
            ray_down, "_kill_blaunch_pids", lambda pids, timeout_s: calls.append(("kill", pids, timeout_s))
        )
        monkeypatch.setattr(ray_down, "stop_ray_cluster", lambda d: calls.append(("stop", d)))
        ray_down.stop_multinode_ray_cluster_srun({"scheduler": "slurm", "worker_pids": [11, 12], "temp_dir": "/t"})
        assert calls == ["shutdown", ("kill", [11, 12], ray_down.SRUN_TERM_GRACE_S), ("stop", "/t")]

    def test_kills_real_process_group(self, tmp_path):
        # End-to-end on the reused LSF helper: a sleeping child in its own
        # session (as _popen_worker spawns it) is gone after teardown.
        proc = ray_up_srun._popen_worker(["sleep", "60"], str(tmp_path / "w.log"), env={"PATH": "/usr/bin:/bin"})
        ray_down._kill_blaunch_pids([proc.pid], timeout_s=5)
        assert proc.poll() is not None


class TestMainWiring:
    def test_bringup_timeout_exits_non_zero(self):
        # A cluster that never came up ran no HPO and saved no model, so exiting
        # 0 let a Granite.build step record it as a success. Both scheduler
        # paths must exit with the dedicated, non-zero bring-up code.
        src = open("main.py").read()
        assert "EXIT_BRINGUP_TIMEOUT = 3" in src
        assert src.count("sys.exit(EXIT_BRINGUP_TIMEOUT)") == 2
        assert "sys.exit(0)" not in src

    def test_scheduler_flag_and_dispatch(self):
        src = open("main.py").read()
        assert 'choices=["lsf", "slurm"]' in src
        assert 'args.scheduler == "slurm"' in src
        assert "stop_multinode_ray_cluster_srun(multinode_info)" in src
