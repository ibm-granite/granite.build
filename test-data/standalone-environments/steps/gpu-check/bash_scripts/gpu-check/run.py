#!/usr/bin/env python3
"""GPU availability check."""

import os
import sys


def main():
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"GPU count: {device_count}")
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                print(f"GPU {i}: {name} ({mem:.1f} GB)")
            print("GPU_CHECK_SUCCESS")
        else:
            print("GPU_CHECK_NO_GPU")
    except ImportError:
        print("GPU_CHECK_NO_TORCH: torch not installed")
        sys.exit(1)


if __name__ == "__main__":
    main()
