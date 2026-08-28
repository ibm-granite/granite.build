"""Tests for autotune.lsf.ray_up_blaunch._default_ib_hca / _rdma_env."""

import pytest

from autotune.lsf.ray_up_blaunch import (
    _DEFAULT_IB_HCA_BY_FLEET,
    _GPU_MODEL_BY_FLEET,
    DEFAULT_FLEET,
    _default_ib_hca,
    _gpu_model_for_fleet,
    _rdma_env,
)


class TestDefaultIbHca:
    def test_a100_single_rail(self):
        assert _default_ib_hca("a100") == "mlx5_0"

    def test_h100_eight_compute_rails_in_order(self):
        rails = _default_ib_hca("h100").split(",")
        assert rails == [f"mlx5_{i}" for i in range(8)]
        # Storage/management rails must NOT be present.
        assert "mlx5_8" not in rails
        assert "mlx5_9" not in rails

    def test_unknown_fleet_falls_back_to_default(self, caplog):
        result = _default_ib_hca("not-a-real-fleet")
        assert result == _DEFAULT_IB_HCA_BY_FLEET[DEFAULT_FLEET]

    def test_unknown_fleet_logs_warning(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="autotune.lsf.ray_up_blaunch")
        _default_ib_hca("typo-fleet")
        assert any("unknown fleet" in r.message and "typo-fleet" in r.message for r in caplog.records)

    def test_default_fleet_is_in_table(self):
        # Sanity: the fallback target must itself be a known fleet.
        assert DEFAULT_FLEET in _DEFAULT_IB_HCA_BY_FLEET


class TestRdmaEnv:
    def test_single_rail_passthrough(self):
        env = _rdma_env(ib_hca="mlx5_0")
        assert env["NCCL_IB_HCA"] == "mlx5_0"

    def test_multi_rail_passthrough(self):
        # _rdma_env should echo the rail list verbatim — no parsing, no munging.
        rails = "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
        env = _rdma_env(ib_hca=rails)
        assert env["NCCL_IB_HCA"] == rails

    def test_throughput_tunings_are_set(self):
        # These were specifically chosen to be safe single-rail and beneficial
        # multi-rail; they should be present regardless of which fleet's HCA
        # list we passed.
        env = _rdma_env(ib_hca="mlx5_0,mlx5_1")
        assert env["NCCL_IB_QPS_PER_CONNECTION"] == "4"
        assert env["NCCL_IB_SPLIT_DATA_ON_QPS"] == "1"
        assert env["NCCL_BUFFSIZE"] == "8388608"
        assert env["NCCL_MIN_NCHANNELS"] == "4"

    def test_gdr_tunings_are_set(self):
        env = _rdma_env(ib_hca="mlx5_0")
        assert env["NCCL_NET_GDR_LEVEL"] == "PXB"
        assert env["NCCL_IB_CUDA_SUPPORT"] == "1"
        assert env["NCCL_IB_GDR_LEVEL"] == "5"
        assert env["NCCL_IB_DISABLE"] == "0"

    def test_socket_ifname_unset_by_default(self):
        env = _rdma_env(ib_hca="mlx5_0")
        assert "NCCL_SOCKET_IFNAME" not in env

    def test_socket_ifname_when_provided(self):
        env = _rdma_env(ib_hca="mlx5_0", ib_ifname="ib0")
        assert env["NCCL_SOCKET_IFNAME"] == "ib0"


@pytest.mark.parametrize("fleet", ["a100", "h100"])
def test_known_fleet_default_resolves(fleet):
    """Every advertised fleet must produce a non-empty rail list."""
    rails = _default_ib_hca(fleet)
    assert rails
    for r in rails.split(","):
        assert r.startswith("mlx5_"), f"unexpected rail name: {r!r}"


class TestGpuModelForFleet:
    def test_a100(self):
        assert _gpu_model_for_fleet("a100") == "NVIDIAA100_SXM4_80GB"

    def test_h100(self):
        assert _gpu_model_for_fleet("h100") == "NVIDIAH10080GBHBM3"

    def test_unknown_fleet_falls_back_to_default(self):
        assert _gpu_model_for_fleet("not-a-real-fleet") == _GPU_MODEL_BY_FLEET[DEFAULT_FLEET]

    def test_default_fleet_is_in_table(self):
        assert DEFAULT_FLEET in _GPU_MODEL_BY_FLEET
