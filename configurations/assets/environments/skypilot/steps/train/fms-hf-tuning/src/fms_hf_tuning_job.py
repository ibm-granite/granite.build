# SPDX-License-Identifier: Apache-2.0
"""Job-file builder and output checks for the fms-hf-tuning step.

Baked into the image at /opt/fms-hf-tuning/ and called by fms_hf_tuning_run.sh:

  build         merge the user's args with the step-owned keys, check every key
                and value against fms-hf-tuning's own parser, write the job JSON
                (and the data config, when one is given)
  check-output  refuse a saved model or adapter with no weights, rewrite an
                adapter's base_model_name_or_path, print the versions

fms-hf-tuning reads the job JSON with HfArgumentParser.parse_dict(...,
allow_extra_keys=True), which ignores a misspelled key and does no type
conversion (a YAML `1e-5` arrives as the string "1e-5"). `build` closes both
gaps before any GPU time: each value goes through the argparse action's own
type and choices, as on the command line, then parse_dict runs with extra keys
refused. Only `tuning` (fms-hf-tuning) is imported lazily, so the pure helpers
import without it.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
from importlib import metadata
from typing import Any, Callable, cast

# Keys the step sets itself. step-template.yaml's `owned` list is the same set
# (test_fms_hf_tuning_job.py pins it).
OWNED = frozenset(
    [
        "model_name_or_path",
        "tokenizer_name_or_path",
        "output_dir",
        "save_model_dir",
        "training_data_path",
        "validation_data_path",
        "data_config_path",
        "peft_method",
        "quantization_method",
        "trackers",
        "report_to",
        "experiment",
        "clearml_project",
        "clearml_task",
        "run_uri_export_path",
        "push_to_hub",
        "hub_model_id",
        "hub_strategy",
        "hub_token",
        "hub_private_repo",
        "resume_from_checkpoint",
        "accelerate_launch_args",
        "fsdp",
        "fsdp_config",
        "deepspeed",
        "log_level_replica",
        "do_dataprocessing_only",
    ]
)

# Arguments sft_trainer's parser adds outside its dataclasses; parse_dict does
# not know them, so they are checked by type only.
EXTRA_ARGS = frozenset(
    ["peft_method", "quantization_method", "exp_metadata", "alora_invocation_string"]
)

# Step defaults a user's args may override: no mid-run checkpoints (only the
# final model is registered), and per-step logging so the `Step:` progress line
# appears (upstream logs once per epoch by default).
DEFAULTS = {"save_strategy": "no", "logging_strategy": "steps", "logging_steps": 10}


class StepError(Exception):
    """A user-facing error: printed with the step prefix, exit 1."""


def decode_b64_json(text: str, what: str) -> Any:
    """Decode the template's `| json_dumps | b64encode` value.

    :param text: The base64 text.
    :param what: The config field, for the message.
    :returns: The decoded value.
    :raises StepError: When it is not base64 JSON.
    """
    try:
        return json.loads(base64.b64decode(text, validate=True).decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise StepError(f"{what} is not base64-encoded JSON ({e}).") from e


def _convert(act: argparse.Action, key: str, value: Any) -> Any:
    """Convert one value as argparse would from the command line."""
    if isinstance(value, bool) and act.type is not str:
        text = "true" if value else "false"
    else:
        text = value if isinstance(value, str) else str(value)
    convert = cast(Callable[[str], Any], act.type)  # coerce_args skips type=None
    try:
        out = convert(text)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as e:
        raise StepError(f"args.{key}: {value!r} is not a valid value ({e}).") from e
    if act.choices and out not in act.choices:
        raise StepError(
            f"args.{key}: {value!r} is not one of {', '.join(map(str, act.choices))}."
        )
    return out


def coerce_args(parser: argparse.ArgumentParser, args: dict) -> dict:
    """Check every key exists and convert every value with its argparse type.

    Mappings (e.g. gradient_checkpointing_kwargs) and nulls pass through as is;
    parse_dict takes them directly.

    :param parser: fms-hf-tuning's parser (or any argparse parser in tests).
    :param args: The user's args.
    :returns: The converted args.
    :raises StepError: On an unknown key, a list for a single-value argument,
        or a value the argument's type or choices reject.
    """
    # Keep the first action per dest: a default-True bool is followed by an
    # untyped `--no_<field>` store_false twin with the same dest.
    actions: dict[str, argparse.Action] = {}
    for a in parser._actions:  # pylint: disable=protected-access
        actions.setdefault(a.dest, a)
    out = {}
    for key, value in args.items():
        act = actions.get(key)
        if act is None or key == "help":
            raise StepError(
                f"args.{key} is not an fms-hf-tuning argument (check the spelling; "
                "names are fms-hf-tuning's, e.g. r, lora_alpha, target_modules)."
            )
        if value is None or isinstance(value, dict) or act.type is None:
            out[key] = value
            continue
        many = act.nargs in ("+", "*")
        if isinstance(value, list):
            if not many:
                raise StepError(f"args.{key} takes one value, got a list.")
            if not value:
                raise StepError(f"args.{key} is an empty list; remove it instead.")
            out[key] = [_convert(act, key, v) for v in value]
        else:
            out[key] = (
                [_convert(act, key, value)] if many else _convert(act, key, value)
            )
    return out


def merge_job(  # pylint: disable=too-many-arguments
    args: dict,
    *,
    base_model: str,
    output_dir: str,
    save_model_dir: str,
    peft_method: str,
    data_path: str,
    data_config_path: str,
    clearml: bool,
    clearml_project: str,
    clearml_task: str,
) -> dict:
    """The job dict: step defaults, then the user's args, then the owned keys.

    :raises StepError: When args sets an owned key.
    """
    owned = sorted(set(args) & OWNED)
    if owned:
        raise StepError(
            f"args.{owned[0]} is set by the step; use the inputs, dataset_name, "
            "data_config, peft_method, output_path or clearml_* instead."
        )
    job = {**DEFAULTS, **args}
    job.update(
        model_name_or_path=base_model,
        output_dir=output_dir,
        save_model_dir=save_model_dir,
        report_to="none",
    )
    if peft_method != "none":
        job["peft_method"] = peft_method
    if data_config_path:
        job["data_config_path"] = data_config_path
    else:
        job["training_data_path"] = data_path
    if clearml:
        job.update(
            trackers=["clearml"],
            clearml_project=clearml_project or "fms-hf-tuning",
            clearml_task=clearml_task or f"sft-{peft_method}",
        )
    return job


def strict_check(job: dict) -> dict:
    """Convert the job with fms-hf-tuning's parser and parse it, extra keys refused.

    :returns: The converted job, as written to the job JSON.
    :raises StepError: On an unknown key, a bad value, or a parse failure.
    """
    from tuning.sft_trainer import (  # pylint: disable=import-outside-toplevel
        get_parser,
    )

    parser = get_parser()
    job = coerce_args(parser, job)
    fields = {k: v for k, v in job.items() if k not in EXTRA_ARGS}
    try:
        parser.parse_dict(fields, allow_extra_keys=False)
    except (TypeError, ValueError) as e:
        raise StepError(f"fms-hf-tuning rejects the job config: {e}") from e
    return job


def tensor_count(path: str) -> int:
    """Number of tensors in a safetensors file, from its header (no torch needed).

    :raises StepError: When the file is unreadable or its header malformed.
    """
    try:
        with open(path, "rb") as f:
            (length,) = struct.unpack("<Q", f.read(8))
            if length > 100_000_000:  # safetensors caps the header at 100 MB
                raise ValueError(f"header length {length} is out of range")
            header = json.loads(f.read(length))
    except (OSError, struct.error, ValueError) as e:
        raise StepError(f"{path} is not a readable safetensors file ({e}).") from e
    if not isinstance(header, dict):
        raise StepError(
            f"{path} is not a readable safetensors file (header is not a map)."
        )
    return sum(1 for k in header if k != "__metadata__")


def check_output(  # pylint: disable=too-many-branches
    out_dir: str, peft_method: str
) -> int:
    """Refuse a saved output that is not a model with weights.

    :param out_dir: The save_model_dir.
    :param peft_method: none (a full model) or lora/alora (an adapter).
    :returns: The tensor count.
    :raises StepError: When files are missing or there are no weights.
    """
    if peft_method == "none":
        if not os.path.isfile(os.path.join(out_dir, "config.json")):
            raise StepError(
                f"training finished but {out_dir} has no config.json; nothing registered."
            )
        index = os.path.join(out_dir, "model.safetensors.index.json")
        if os.path.isfile(index):
            try:
                with open(index, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                raise StepError(f"{index} is not readable JSON ({e}).") from e
            if not isinstance(data, dict):
                raise StepError(f"{index} is not a JSON mapping.")
            weight_map = data.get("weight_map")
            if weight_map is None:
                weight_map = {}
            if not isinstance(weight_map, dict):
                raise StepError(f"{index}: weight_map is not a mapping.")
            for shard in sorted(set(weight_map.values())):
                if not os.path.isfile(os.path.join(out_dir, shard)):
                    raise StepError(
                        f"{index} names {shard}, which is missing; nothing registered."
                    )
            count = len(weight_map)
        else:
            shards = sorted(
                n for n in os.listdir(out_dir) if n.endswith(".safetensors")
            )
            if not shards:
                raise StepError(
                    f"training finished but {out_dir} has no *.safetensors weights; "
                    "nothing registered."
                )
            count = sum(tensor_count(os.path.join(out_dir, n)) for n in shards)
        what = "model"
    else:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not os.path.isfile(os.path.join(out_dir, name)):
                raise StepError(
                    f"training finished but {out_dir} has no {name}; nothing registered."
                )
        count = tensor_count(os.path.join(out_dir, "adapter_model.safetensors"))
        what = "adapter"
    if count == 0:
        raise StepError(
            f"the saved {what} in {out_dir} has no weights (0 tensors); training ran but "
            "nothing was saved. Nothing registered."
        )
    return count


def rewrite_base_model(out_dir: str, base_model_id: str) -> None:
    """Record a portable base id in adapter_config.json (it holds the staged path)."""
    path = os.path.join(out_dir, "adapter_config.json")
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        raise StepError(f"{path} is not readable JSON ({e}).") from e
    if not isinstance(cfg, dict):
        raise StepError(f"{path} is not a JSON mapping.")
    cfg["base_model_name_or_path"] = base_model_id
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError as e:
        raise StepError(f"cannot write {path} ({e}).") from e


def _write_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except OSError as e:
        raise StepError(f"cannot write {path} ({e}).") from e


def _version(dist: str) -> str:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="fms_hf_tuning_job.py", allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", allow_abbrev=False)
    for opt in (
        "--args-b64",
        "--data-config-b64",
        "--data-path",
        "--base-model",
        "--peft-method",
        "--output-dir",
        "--save-model-dir",
        "--clearml",
        "--clearml-project",
        "--clearml-task",
        "--job-out",
        "--data-config-out",
    ):
        b.add_argument(opt, required=True)
    c = sub.add_parser("check-output", allow_abbrev=False)
    c.add_argument("out_dir")
    c.add_argument("peft_method")
    c.add_argument("--base-model-id", default="")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry; a StepError exits 1 with the step's error prefix."""
    ns = _parse(sys.argv[1:] if argv is None else argv)
    try:
        if ns.cmd == "build":
            args = decode_b64_json(ns.args_b64, "fms_hf_tuning_config.args")
            data_config = decode_b64_json(
                ns.data_config_b64, "fms_hf_tuning_config.data_config"
            )
            if not isinstance(args, dict) or not isinstance(data_config, dict):
                raise StepError("args and data_config must be mappings.")
            dc_path = ""
            if data_config:
                _write_json(ns.data_config_out, data_config)
                dc_path = os.path.abspath(ns.data_config_out)
            elif not ns.data_path:
                raise StepError("no training data path and no data_config.")
            job = merge_job(
                args,
                base_model=ns.base_model,
                output_dir=ns.output_dir,
                save_model_dir=ns.save_model_dir,
                peft_method=ns.peft_method,
                data_path=ns.data_path,
                data_config_path=dc_path,
                clearml=ns.clearml == "on",
                clearml_project=ns.clearml_project,
                clearml_task=ns.clearml_task,
            )
            job = strict_check(job)
            _write_json(ns.job_out, job)
        else:
            count = check_output(ns.out_dir, ns.peft_method)
            if ns.base_model_id and ns.peft_method != "none":
                rewrite_base_model(ns.out_dir, ns.base_model_id)
            print(f"fms-hf-tuning-output: ok {count}", flush=True)
            print(
                "fms-hf-tuning-versions:",
                _version("fms-hf-tuning"),
                _version("transformers"),
                flush=True,
            )
    except StepError as e:
        print(f"fms-hf-tuning: ERROR {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
