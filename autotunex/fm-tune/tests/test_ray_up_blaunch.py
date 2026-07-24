"""Tests for autotune.lsf.ray_up_blaunch — host-list parsing and argv shape."""

import pytest
from autotune.lsf.ray_up_blaunch import (
    _build_blaunch_cmd,
    _build_local_cmd,
    _partition_hosts,
    _read_lsf_hostfile,
    _short_hostname,
    start_multinode_ray_cluster_blaunch,
)


class TestReadLsfHostfile:
    def test_reads_djob_hostfile_dedupes_preserves_order(self, tmp_path, monkeypatch):
        hf = tmp_path / "hosts"
        hf.write_text("h1\nh2\nh1\nh3\nh2\n")
        monkeypatch.setenv("LSB_DJOB_HOSTFILE", str(hf))
        monkeypatch.delenv("LSB_HOSTS", raising=False)
        assert _read_lsf_hostfile() == ["h1", "h2", "h3"]

    def test_falls_back_to_lsb_hosts(self, monkeypatch):
        monkeypatch.delenv("LSB_DJOB_HOSTFILE", raising=False)
        monkeypatch.setenv("LSB_HOSTS", "h1 h1 h1 h2 h2 h2 h3")
        assert _read_lsf_hostfile() == ["h1", "h2", "h3"]

    def test_djob_hostfile_takes_precedence(self, tmp_path, monkeypatch):
        hf = tmp_path / "hosts"
        hf.write_text("ha\nhb\n")
        monkeypatch.setenv("LSB_DJOB_HOSTFILE", str(hf))
        monkeypatch.setenv("LSB_HOSTS", "hx hy")
        assert _read_lsf_hostfile() == ["ha", "hb"]

    def test_falls_back_when_djob_hostfile_missing_on_disk(self, tmp_path, monkeypatch):
        # File-not-found should silently fall back to LSB_HOSTS rather than fail.
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("LSB_DJOB_HOSTFILE", str(missing))
        monkeypatch.setenv("LSB_HOSTS", "h1 h2")
        assert _read_lsf_hostfile() == ["h1", "h2"]

    def test_raises_when_neither_set(self, monkeypatch):
        monkeypatch.delenv("LSB_DJOB_HOSTFILE", raising=False)
        monkeypatch.delenv("LSB_HOSTS", raising=False)
        with pytest.raises(RuntimeError, match="LSB_DJOB_HOSTFILE.*LSB_HOSTS"):
            _read_lsf_hostfile()

    def test_raises_when_both_empty(self, monkeypatch):
        monkeypatch.delenv("LSB_DJOB_HOSTFILE", raising=False)
        monkeypatch.setenv("LSB_HOSTS", "   ")
        with pytest.raises(RuntimeError):
            _read_lsf_hostfile()


class TestPartitionHosts:
    def test_head_first_no_rotation(self):
        head, all_workers, remote = _partition_hosts(
            ["h1", "h2", "h3"], "h1", num_workers=3
        )
        assert head == "h1"
        assert all_workers == ["h1", "h2", "h3"]
        assert remote == ["h2", "h3"]

    def test_rotates_when_head_not_first(self):
        head, all_workers, remote = _partition_hosts(
            ["h2", "h3", "h1"], "h1", num_workers=3
        )
        assert head == "h1"
        assert all_workers == ["h1", "h2", "h3"]
        assert remote == ["h2", "h3"]

    def test_single_host(self):
        head, all_workers, remote = _partition_hosts(["h1"], "h1", num_workers=1)
        assert head == "h1"
        assert all_workers == ["h1"]
        assert remote == []

    def test_raises_when_head_not_in_list(self):
        with pytest.raises(RuntimeError, match="not in LSF host list"):
            _partition_hosts(["h2", "h3"], "h1", num_workers=2)

    def test_raises_on_count_mismatch(self):
        with pytest.raises(RuntimeError, match="num_workers=3"):
            _partition_hosts(["h1", "h2"], "h1", num_workers=3)

    def test_fqdn_head_matches_short_lsf_list(self):
        # BlueVela case: socket.gethostname() returns FQDN, LSF gives short
        # names. Should match on the short form and return short names.
        head, all_workers, remote = _partition_hosts(
            ["p3-r31-n3", "p2-r23-n4"],
            "p3-r31-n3.banana.rmf.example.com",
            num_workers=2,
        )
        assert head == "p3-r31-n3"
        assert all_workers == ["p3-r31-n3", "p2-r23-n4"]
        assert remote == ["p2-r23-n4"]

    def test_fqdn_head_when_not_first_rotates(self):
        head, all_workers, remote = _partition_hosts(
            ["p2-r23-n4", "p3-r31-n3"],
            "p3-r31-n3.banana.rmf.example.com",
            num_workers=2,
        )
        assert head == "p3-r31-n3"
        assert all_workers == ["p3-r31-n3", "p2-r23-n4"]
        assert remote == ["p2-r23-n4"]


class TestShortHostname:
    def test_strips_domain(self):
        assert _short_hostname("p3-r31-n3.banana.rmf.example.com") == "p3-r31-n3"

    def test_no_domain_returns_input(self):
        assert _short_hostname("cccxc602") == "cccxc602"

    def test_strips_only_first_dot(self):
        # We only care about the leading label.
        assert _short_hostname("a.b.c.d") == "a"


class TestBuildBlaunchCmd:
    def _common_kwargs(self):
        return dict(
            head_address="10.0.0.1:6379",
            gpus_per_worker=4,
            cores_per_worker=32,
            conda_env="/u/me/envs/autotune",
            rdma_env={"NCCL_IB_HCA": "mlx5_0", "NCCL_DEBUG": "WARN"},
        )

    def test_argv_starts_with_blaunch_z_host(self):
        argv = _build_blaunch_cmd(host="h2", **self._common_kwargs())
        assert argv[:3] == ["blaunch", "-z", "h2"]
        assert argv[3:5] == ["bash", "-lc"]

    def test_inner_contains_worker_entry_module_and_args(self):
        argv = _build_blaunch_cmd(host="h2", **self._common_kwargs())
        inner = argv[-1]
        assert "python -m autotune.lsf.worker_entry" in inner
        assert "--head_address 10.0.0.1:6379" in inner
        assert "--num_gpus 4" in inner
        assert "--num_cpus 32" in inner

    def test_inner_uses_exec_so_bash_replaces_itself(self):
        # `exec` replaces bash with the python process so SIGTERM goes
        # straight to python (no bash-as-middleman) and we have one fewer
        # process per worker holding LSF log fds.
        argv = _build_blaunch_cmd(host="h2", **self._common_kwargs())
        inner = argv[-1]
        assert "exec env" in inner

    def test_inner_does_not_redirect_inline(self):
        # Redirection happens at Popen time (stdout/stderr=log file fd),
        # NOT in the inner string.  This keeps the children from inheriting
        # the driver's fd 1/2, which is what kept LSF from reaping the job.
        argv = _build_blaunch_cmd(host="h2", **self._common_kwargs())
        inner = argv[-1]
        assert "2>&1" not in inner
        assert ".log" not in inner

    def test_inner_activates_conda_env(self):
        argv = _build_blaunch_cmd(host="h2", **self._common_kwargs())
        inner = argv[-1]
        assert "conda activate /u/me/envs/autotune" in inner

    def test_inner_includes_rdma_env(self):
        argv = _build_blaunch_cmd(host="h2", **self._common_kwargs())
        inner = argv[-1]
        assert "NCCL_IB_HCA=mlx5_0" in inner
        assert "NCCL_DEBUG=WARN" in inner


class TestBuildLocalCmd:
    """Local (head-host) worker uses the same inner string but no blaunch wrapping."""

    def test_argv_is_bash_lc_only(self):
        argv = _build_local_cmd(
            head_address="10.0.0.1:6379",
            gpus_per_worker=4,
            cores_per_worker=32,
            conda_env="/u/me/envs/autotune",
            rdma_env={"NCCL_IB_HCA": "mlx5_0"},
        )
        assert argv[:2] == ["bash", "-lc"]
        assert "blaunch" not in argv
        assert "python -m autotune.lsf.worker_entry" in argv[-1]
        # No inline redirection — Popen handles stdout/stderr.
        assert "2>&1" not in argv[-1]


class TestRequiredArgs:
    """Bring-up must reject silently-broken call sites."""

    def test_missing_log_dir_raises(self):
        # log_dir was previously a default (relative "logs") — easy to forget,
        # which sent worker logs to CWD instead of <output_dir>/logs/. Now
        # required so cleanup() can find and wipe them.
        with pytest.raises(ValueError, match="log_dir is required"):
            start_multinode_ray_cluster_blaunch(
                num_workers=1,
                gpus_per_worker=1,
                conda_env="/u/me/envs/autotune",
                # no log_dir
            )

    def test_empty_log_dir_raises(self):
        with pytest.raises(ValueError, match="log_dir is required"):
            start_multinode_ray_cluster_blaunch(
                num_workers=1,
                gpus_per_worker=1,
                conda_env="/u/me/envs/autotune",
                log_dir="",
            )
