"""Tests for autotune.cluster — port reservation and GPU isolation."""

import os
import socket

import autotune.cluster as cluster
from autotune.cluster import (
    compute_ray_data_sizing,
    ensure_gpu_isolation,
    ray_data_block_target,
    release_sockets,
    reserve_ports,
)


class TestReservePorts:
    def test_zero_returns_empty(self):
        ports, sockets = reserve_ports(0)
        assert ports == []
        assert sockets == []

    def test_returns_n_ports(self):
        ports, sockets = reserve_ports(5)
        try:
            assert len(ports) == 5
            assert len(sockets) == 5
        finally:
            release_sockets(sockets)

    def test_ports_are_distinct(self):
        ports, sockets = reserve_ports(10)
        try:
            assert len(set(ports)) == 10
        finally:
            release_sockets(sockets)

    def test_ports_are_valid(self):
        ports, sockets = reserve_ports(3)
        try:
            for p in ports:
                assert 1 <= p <= 65535
        finally:
            release_sockets(sockets)

    def test_sockets_lack_so_reuseaddr(self):
        # reserve_ports deliberately leaves SO_REUSEADDR unset so the port
        # reservation is exclusive while the socket is held (see reserve_ports
        # docstring — required for the multi-node Ray head bind).
        _, sockets = reserve_ports(2)
        try:
            for s in sockets:
                assert s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) == 0
        finally:
            release_sockets(sockets)


class TestReleaseSockets:
    def test_empty_list(self):
        # Should be a no-op
        release_sockets([])

    def test_idempotent_on_already_closed(self):
        _, sockets = reserve_ports(2)
        release_sockets(sockets)
        # Calling again on already-closed sockets should not raise
        release_sockets(sockets)

    def test_handles_mix_of_open_closed(self):
        _, sockets = reserve_ports(3)
        sockets[0].close()  # close one externally
        release_sockets(sockets)  # should not raise


class TestEnsureGpuIsolation:
    def test_sets_when_unset(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        ensure_gpu_isolation(2)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_respects_existing(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
        ensure_gpu_isolation(2)
        # Existing value preserved
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3,4"

    def test_warns_when_existing_exceeds_requested(self, monkeypatch, caplog):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5")
        # Even with too-many existing, it does not modify the env
        ensure_gpu_isolation(2)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5"

    def test_zero_gpus_sets_empty(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        ensure_gpu_isolation(0)
        # range(0) is empty → ""
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    def test_one_gpu(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        ensure_gpu_isolation(1)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    def test_empty_string_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "  ")
        ensure_gpu_isolation(2)
        # Whitespace-only is treated as unset → overwritten
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


class TestRayDataBlockTarget:
    def test_clamps_to_row_count(self):
        # Fewer rows than concurrency → don't over-partition.
        assert ray_data_block_target(64, 10) == 10

    def test_uses_concurrency_when_rows_plentiful(self):
        assert ray_data_block_target(8, 100000) == 8

    def test_never_below_one(self):
        assert ray_data_block_target(0, 0) == 1
        assert ray_data_block_target(-5, 100) == 1

    def test_single_row(self):
        assert ray_data_block_target(16, 1) == 1


class TestComputeRayDataSizing:
    def _patch_cpus(self, monkeypatch, total):
        monkeypatch.setattr(cluster.ray, "cluster_resources", lambda: {"CPU": total, "GPU": 0})

    def test_auto_reserves_workers(self, monkeypatch):
        self._patch_cpus(monkeypatch, 64)
        concurrency, num_cpus = compute_ray_data_sizing(8, None, None)
        assert concurrency == 56  # 64 - 8
        assert num_cpus == 1.0

    def test_override_takes_precedence(self, monkeypatch):
        self._patch_cpus(monkeypatch, 64)
        concurrency, num_cpus = compute_ray_data_sizing(8, 4, 0.5)
        assert concurrency == 4
        assert num_cpus == 0.5

    def test_auto_clamped_to_one(self, monkeypatch):
        # More workers than CPUs → still at least 1.
        self._patch_cpus(monkeypatch, 2)
        concurrency, _ = compute_ray_data_sizing(8, None, None)
        assert concurrency == 1

    def test_floors_fractional_cpus(self, monkeypatch):
        self._patch_cpus(monkeypatch, 7.9)
        concurrency, _ = compute_ray_data_sizing(2, None, None)
        assert concurrency == 5  # floor(7.9)=7, 7-2

    def test_falls_back_when_resources_unavailable(self, monkeypatch):
        def _raise():
            raise RuntimeError("no cluster")

        monkeypatch.setattr(cluster.ray, "cluster_resources", _raise)
        concurrency, num_cpus = compute_ray_data_sizing(4, None, None)
        assert concurrency == 1  # total_cpus=0 → max(1, 0-4)
        assert num_cpus == 1.0


class TestResolveLocalRayGpus:
    def test_non_cuda_returns_zero(self, monkeypatch):
        from autotune import cluster
        from autotune.device import Accelerator

        monkeypatch.setattr(cluster, "detect_accelerator", lambda: Accelerator("mps", 1, False, False, False, True))
        assert cluster.resolve_local_ray_gpus() == 0

    def test_cuda_honours_visible_devices(self, monkeypatch):
        from autotune import cluster
        from autotune.device import Accelerator

        monkeypatch.setattr(cluster, "detect_accelerator", lambda: Accelerator("cuda", 8, True, True, True, True))
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
        assert cluster.resolve_local_ray_gpus() == 3
