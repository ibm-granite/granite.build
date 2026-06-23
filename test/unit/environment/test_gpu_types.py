import pytest

from gbserver.environment.runpod import (
    RUNPOD_GPU_MAP,
    UnknownGPUType,
    resolve_runpod_gpu_type,
)


class TestGPUTypeMapping:
    def test_resolve_known_type(self):
        assert resolve_runpod_gpu_type("A100-80GB") == "NVIDIA A100 80GB PCIe"

    def test_resolve_known_type_case_insensitive(self):
        assert resolve_runpod_gpu_type("a100-80gb") == "NVIDIA A100 80GB PCIe"

    def test_resolve_h100(self):
        assert resolve_runpod_gpu_type("H100-80GB") == "NVIDIA H100 80GB HBM3"

    def test_resolve_l40s(self):
        assert resolve_runpod_gpu_type("L40S") == "NVIDIA L40S"

    def test_resolve_rtx4090(self):
        assert resolve_runpod_gpu_type("RTX-4090") == "NVIDIA GeForce RTX 4090"

    def test_resolve_passthrough_native_id(self):
        assert (
            resolve_runpod_gpu_type("NVIDIA A100 80GB PCIe") == "NVIDIA A100 80GB PCIe"
        )

    def test_resolve_unknown_type_raises(self):
        with pytest.raises(UnknownGPUType, match="FAKE-GPU"):
            resolve_runpod_gpu_type("FAKE-GPU")

    def test_map_has_expected_entries(self):
        assert "A100-80GB" in RUNPOD_GPU_MAP
        assert "A100-40GB" in RUNPOD_GPU_MAP
        assert "H100-80GB" in RUNPOD_GPU_MAP
        assert "L40S" in RUNPOD_GPU_MAP
        assert "RTX-4090" in RUNPOD_GPU_MAP
