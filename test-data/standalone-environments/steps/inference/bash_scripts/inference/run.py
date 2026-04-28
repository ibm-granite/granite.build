#!/usr/bin/env python3
"""Verify HF model download: check model directory exists with expected files."""

import os
import sys

EXPECTED_FILES = ["config.json"]


def main():
    # The bash environment exports bindings as LLMB_BASH_INPUT_<NAME>
    model_path = os.environ.get("LLMB_BASH_INPUT_MODEL", "")
    if not model_path and len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not model_path:
        print("ERROR: No model path provided (expected LLMB_BASH_INPUT_MODEL env var)")
        sys.exit(1)

    print(f"Checking model at: {model_path}")

    if not os.path.isdir(model_path):
        print(f"ERROR: Model path does not exist or is not a directory: {model_path}")
        sys.exit(1)

    files = os.listdir(model_path)
    print(f"Model directory contains {len(files)} items: {sorted(files)}")

    for expected in EXPECTED_FILES:
        if expected not in files:
            print(f"ERROR: Expected file '{expected}' not found in model directory")
            sys.exit(1)

    print("INFERENCE_SUCCESS: Model directory verified")


if __name__ == "__main__":
    main()
