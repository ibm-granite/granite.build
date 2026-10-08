#!/usr/bin/env python3
"""Work around published HF checkpoints that ship an llm-compressor
recipe.yaml (scheme: FP8_DYNAMIC, etc.) but no matching config.json
quantization_config -- vLLM can't autodetect a loader for those, and
forcing --quantization fp8 picks the wrong (native, not compressed-tensors)
loader, which fails opaquely on the checkpoint's real weight_scale tensors.
See steps/bfcl-eval/README.md for the full failure mode this works around.

Usage: patch_quant_config.py <model_path> <output_dir>
Prints the model path to actually load (staged copy if patched, otherwise
the original model_path unchanged) on stdout. Exit 0 with the original
path when there's nothing to patch (no config.json, config.json already
has a quantization_config, no recipe.yaml, or recipe.yaml names no
quantization scheme). Exit 1 if recipe.yaml names a scheme this script
doesn't know how to translate.
"""

from __future__ import annotations

import json
import os
import sys

import yaml

# Each entry is the compressed-tensors quantization_config that reproduces
# the given llm-compressor QuantizationModifier scheme. Confirmed against a
# real checkpoint (ibm-granite/granite-4.2-30b-fp8) by cross-checking its
# model.safetensors.index.json: FP8_DYNAMIC's weights are per-channel
# static fp8 (weight + weight_scale, no input_scale tensor), matching
# strategy="channel"/dynamic=False for weights and dynamic=True for
# activations below.
KNOWN_SCHEMES = {
    "FP8_DYNAMIC": {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            }
        },
        "format": "float-quantized",
        "quantization_status": "compressed",
    },
}


def find_scheme(recipe: dict) -> str | None:
    for stage in recipe.values():
        if not isinstance(stage, dict):
            continue
        modifiers = stage.get("default_modifiers", stage)
        if not isinstance(modifiers, dict):
            continue
        for key, mod in modifiers.items():
            if (
                isinstance(mod, dict)
                and "QuantizationModifier" in key
                and mod.get("scheme")
            ):
                return mod["scheme"]
    return None


def main() -> int:
    model_path, output_dir = sys.argv[1], sys.argv[2]

    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        print(model_path)
        return 0

    with open(config_path) as f:
        config = json.load(f)
    if config.get("quantization_config"):
        print(model_path)
        return 0

    recipe_path = os.path.join(model_path, "recipe.yaml")
    if not os.path.isfile(recipe_path):
        print(model_path)
        return 0

    with open(recipe_path) as f:
        recipe = yaml.safe_load(f)
    scheme = find_scheme(recipe) if isinstance(recipe, dict) else None
    if not scheme:
        print(model_path)
        return 0

    if scheme not in KNOWN_SCHEMES:
        print(
            f"patch_quant_config: {model_path}'s recipe.yaml records "
            f"quantization scheme {scheme!r}, which has no known "
            "compressed-tensors mapping in this script (known schemes: "
            f"{sorted(KNOWN_SCHEMES)}). config.json has no "
            "quantization_config, so vLLM will likely fail to load this "
            "checkpoint's quantized weights. Add a mapping for this scheme "
            "to KNOWN_SCHEMES in src/patch_quant_config.py.",
            file=sys.stderr,
        )
        return 1

    staging_dir = os.path.join(output_dir, ".quant-config-patch")
    os.makedirs(staging_dir, exist_ok=True)
    # A symlink's target resolves relative to the link's own directory, not
    # cwd, so a relative model_path would leave every staged link dangling.
    source_dir = os.path.abspath(model_path)
    for name in os.listdir(source_dir):
        if name == "config.json":
            continue
        link_path = os.path.join(staging_dir, name)
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(os.path.join(source_dir, name), link_path)

    config["quantization_config"] = KNOWN_SCHEMES[scheme]
    with open(os.path.join(staging_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(
        f"patch_quant_config: {model_path} has no quantization_config but "
        f"recipe.yaml records scheme {scheme!r}; staged a patched copy at "
        f"{staging_dir}",
        file=sys.stderr,
    )
    print(staging_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
