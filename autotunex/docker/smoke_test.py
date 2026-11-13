# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU smoke test for the CUDA kernel base (docker/cuda-base/Dockerfile).

Imports flash-attn, mamba-ssm and causal-conv1d, runs one tiny forward pass through
each compiled kernel, and checks it against the package's own PyTorch reference (or
torch's SDPA for flash-attn). Prints the torch, CUDA and GPU versions first. Exits
non-zero on the first failure, so a byoc build fails with it.

Run it on a GPU host inside the image:

    python docker/smoke_test.py
"""

from __future__ import annotations

import sys

import torch


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, tol: float) -> None:
    # The error relative to the reference's largest value, so the tolerance does not
    # depend on the random inputs' scale.
    err = ((out.float() - ref.float()).abs().max() / ref.float().abs().max().clamp(min=1)).item()
    print(f"{name}: max error vs reference {err:.1e} of its range (tolerance {tol:.0e})")
    if not err <= tol:  # also catches NaN
        sys.exit(f"FAIL: {name} differs from its reference")


def main() -> None:
    """Print the versions, then run and check each kernel once; exit non-zero on failure."""
    print(f"python {sys.version.split()[0]}")
    print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("FAIL: torch sees no CUDA device")
    props = torch.cuda.get_device_properties(0)
    # flash-attn is compiled for sm80 and sm90 only (FLASH_ATTN_CUDA_ARCHS in fm-tune).
    print(f"GPU {props.name}, sm_{props.major}{props.minor}, {torch.cuda.device_count()} visible")

    import causal_conv1d
    import flash_attn
    import mamba_ssm
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref
    from flash_attn import flash_attn_func
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    for package in (flash_attn, mamba_ssm, causal_conv1d):
        print(f"{package.__name__} {package.__version__}")

    torch.manual_seed(0)
    dev = "cuda"

    # flash-attn: (batch, seqlen, heads, headdim) in bf16, against torch's SDPA.
    q, k, v = (torch.randn(2, 64, 4, 64, device=dev, dtype=torch.bfloat16) for _ in range(3))
    out = flash_attn_func(q, k, v, causal=True)
    ref = torch.nn.functional.scaled_dot_product_attention(
        *(t.float().transpose(1, 2) for t in (q, k, v)), is_causal=True
    ).transpose(1, 2)
    _check("flash-attn flash_attn_func", out, ref, tol=2e-2)

    # causal-conv1d: x (batch, dim, seqlen), weight (dim, width).
    x = torch.randn(2, 64, 128, device=dev)
    weight = torch.randn(64, 4, device=dev)
    bias = torch.randn(64, device=dev)
    out = causal_conv1d_fn(x, weight, bias, activation="silu")
    _check(
        "causal-conv1d causal_conv1d_fn",
        out,
        causal_conv1d_ref(x, weight, bias, activation="silu"),
        tol=1e-3,
    )

    # mamba-ssm: u, delta (batch, dim, seqlen); A (dim, dstate); B, C (batch, dstate, seqlen).
    batch, dim, dstate, seqlen = 2, 32, 16, 64
    u = torch.randn(batch, dim, seqlen, device=dev)
    delta = torch.rand(batch, dim, seqlen, device=dev) * 0.1
    a = -torch.rand(dim, dstate, device=dev)
    b = torch.randn(batch, dstate, seqlen, device=dev)
    c = torch.randn(batch, dstate, seqlen, device=dev)
    d = torch.randn(dim, device=dev)
    out = selective_scan_fn(u, delta, a, b, c, D=d, delta_softplus=True)
    ref = selective_scan_ref(u, delta, a, b, c, D=d, delta_softplus=True)
    _check("mamba-ssm selective_scan_fn", out, ref, tol=1e-3)

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
