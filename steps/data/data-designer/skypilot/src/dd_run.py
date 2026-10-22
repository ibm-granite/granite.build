#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one NeMo Data Designer generation for the data-designer step.

Invoked from the generated step.yaml's ``run:`` block (after ``dd_guard.sh``) and
shipped to the node by ``file_mounts: {src: src}``.

WHY THE PYTHON API AND NOT ``data-designer create``
The CLI reads model providers from ``$DATA_DESIGNER_HOME/model_providers.yaml``, so
using it would mean writing a credential to a file on the node. Some endpoints need a
custom auth header rather than a bearer token — some gateways take ``<Header>: <key>`` —
and Data Designer's ``extra_headers`` are literal strings, not secret references.
Building the providers here lets a header value come from the environment at run time
(``extra_headers_from_env``) and never touch disk, the build.yaml or the persisted
step config.

CONTRACT
  dd_run.py --output-path <dir> --artifact-id <id> --num-records <n> --resume <mode>
            [--export-format <fmt>] [--min-records <n>] [--seed-path <dir>]
            [--seed-glob <glob>]

  Everything else arrives in $GB_DD_SPEC_B64: base64 of the JSON object
  {"config": {...}, "model_providers": [...], "run_config": {...}, "local_model": {...}}.

ORDER, and why it matters
  1. Everything that cannot depend on the work is checked first (marker-safe id and
     path, a non-empty output dir under resume=never, the spec, credentials, the seed
     override, the providers — reported without their secrets), so a bad build fails
     in seconds rather than after the generation.
  2. Data Designer's validate() checks the config with no model calls. Then a previous
     run in output_path is checked the way Data Designer would check it (config,
     buffer_size, preserve_dropped_columns, num_records): resume=always refuses a
     mismatch, resume=if_possible clears it and starts fresh in place.
  3. create() — which health-checks each referenced model first.
  4. The row count is checked against --min-records, then the optional export.
  5. Only then is the artifact marker printed, so a failed or short run registers
     nothing.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

SPEC_ENV = "GB_DD_SPEC_B64"
RESUME_MODES = ("never", "always", "if_possible")
EXPORT_FORMATS = ("jsonl", "csv", "parquet")
# Seed source types whose `path` names a file, glob or directory on the filesystem —
# the ones a staged input can stand in for. `hf` names a Hub dataset and is left as is.
PATH_SEED_TYPES = ("local", "directory", "file_contents", "agent_rollout")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class StepError(Exception):
    """A configuration problem reported as one ERROR line, without a traceback."""


def load_spec(env: Mapping[str, str]) -> dict[str, Any]:
    """Decode and shape-check the config + providers the template passed in."""
    raw = env.get(SPEC_ENV, "")
    if not raw:
        raise StepError(f"${SPEC_ENV} is empty; the step template did not pass a spec.")
    try:
        spec = json.loads(base64.b64decode(raw, validate=True))
    except (binascii.Error, ValueError) as e:
        raise StepError(f"${SPEC_ENV} is not base64-encoded JSON: {e}") from None
    if not isinstance(spec, dict):
        raise StepError(f"${SPEC_ENV} must decode to an object, got {type(spec).__name__}.")
    config = spec.get("config")
    if not isinstance(config, dict) or not config:
        raise StepError("data_designer_config.config must be a non-empty mapping.")
    local_model = spec.get("local_model") or {}
    if not isinstance(local_model, dict):
        raise StepError("data_designer_config.local_model must be a mapping.")
    providers = spec.get("model_providers") or []
    if not isinstance(providers, list):
        raise StepError("data_designer_config.model_providers must be a list.")
    if not providers and not local_model.get("runtime"):
        raise StepError(
            "data_designer_config.model_providers must be a non-empty list, unless "
            "local_model serves a model on the node."
        )
    run_config = spec.get("run_config") or {}
    if not isinstance(run_config, dict):
        raise StepError("data_designer_config.run_config must be a mapping of RunConfig fields.")
    return {
        "config": config,
        "model_providers": providers,
        "run_config": run_config,
        "local_model": local_model,
    }


def local_model_provider(local_model: Mapping[str, Any], providers: list[Any]) -> dict[str, Any] | None:
    """The provider for a model the step serves on the node, or None when there is none.

    The server listens on 127.0.0.1, so the endpoint never leaves the node. Ollama's
    OpenAI-compatible API ignores the key, but the client sends one, so a fixed
    placeholder is used — not a secret, and nothing to declare.
    """
    runtime = local_model.get("runtime") or ""
    if not runtime:
        return None
    if runtime != "ollama":
        raise StepError(f"local_model.runtime must be empty or 'ollama', got {runtime!r}.")
    name = local_model.get("name") or "local"
    taken = {p.get("name") for p in providers if isinstance(p, dict)}
    if name in taken:
        raise StepError(
            f"local_model.name {name!r} is also a model_providers name; rename one."
        )
    return {
        "name": name,
        "endpoint": f"http://127.0.0.1:{int(local_model.get('port') or 11434)}/v1",
        "api_key": "ollama",
    }


def _require_env(env: Mapping[str, str], name: Any, where: str) -> str:
    if not isinstance(name, str) or not _ENV_NAME.match(name):
        raise StepError(f"{where} must name an environment variable, got {name!r}.")
    value = env.get(name)
    if not value:
        # Name the variable and how it gets there, never a value.
        raise StepError(
            f"{where} reads ${name}, which is not set on the node. Declare the space "
            f"secret under config.skypilot.secrets.secret_names_to_use_as_env_variable "
            f"with env_name: {name}."
        )
    return value


def build_providers(providers: list[Any], env: Mapping[str, str]) -> list[dict[str, Any]]:
    """Resolve the step-only credential keys into plain ModelProvider dicts.

    ``api_key_env: NAME`` becomes ``api_key: $NAME``; ``extra_headers_from_env:
    {Header: NAME}`` is merged into ``extra_headers`` as ``Header: $NAME``. Both fail
    loudly when the variable is missing. A plain ``api_key`` passes through untouched —
    note Data Designer reads it as an environment variable NAME first and falls back
    to the literal string, so a missing variable is sent as the key itself. Prefer
    ``api_key_env``.
    """
    resolved = []
    for i, entry in enumerate(providers):
        if not isinstance(entry, dict):
            raise StepError(f"model_providers[{i}] must be a mapping, got {type(entry).__name__}.")
        p = copy.deepcopy(entry)
        label = f"model_providers[{i}] ({p.get('name', '?')})"

        key_env = p.pop("api_key_env", None)
        if key_env is not None:
            if p.get("api_key"):
                raise StepError(f"{label} sets both api_key and api_key_env; keep one.")
            p["api_key"] = _require_env(env, key_env, f"{label}.api_key_env")

        header_env = p.pop("extra_headers_from_env", None)
        if header_env is not None:
            if not isinstance(header_env, dict):
                raise StepError(f"{label}.extra_headers_from_env must map header -> env var.")
            headers = dict(p.get("extra_headers") or {})
            for header, name in header_env.items():
                headers[header] = _require_env(
                    env, name, f"{label}.extra_headers_from_env[{header!r}]"
                )
            p["extra_headers"] = headers
        resolved.append(p)
    return resolved


def apply_seed_path(config: dict[str, Any], seed_path: str, seed_glob: str) -> dict[str, Any]:
    """Point the config's seed source at the staged `seed` input, when one is bound."""
    if not seed_path:
        return config
    config = copy.deepcopy(config)
    inner = config["data_designer"] if "data_designer" in config else config
    seed_config = inner.get("seed_config") if isinstance(inner, dict) else None
    if not isinstance(seed_config, dict) or not isinstance(seed_config.get("source"), dict):
        raise StepError(
            "the `seed` input is bound but the config has no seed_config.source to point "
            "at it; add a seed_config (e.g. seed_type: local) or unbind the input."
        )
    source = seed_config["source"]
    seed_type = source.get("seed_type")
    if seed_type not in PATH_SEED_TYPES:
        raise StepError(
            f"the `seed` input is bound but seed_config.source.seed_type is {seed_type!r}; "
            f"a staged path can only replace one of {', '.join(PATH_SEED_TYPES)}."
        )
    source["path"] = f"{seed_path.rstrip('/')}/{seed_glob}" if seed_glob else seed_path
    return config


def check_output_dir(output_path: Path, resume: str) -> None:
    """Refuse a non-empty output dir under resume=never.

    Data Designer would otherwise write to a timestamped sibling
    (``<name>_<MM-DD-YYYY_HHMMSS>``), and the artifact marker — which must match the
    declared output's path — would point at the stale data.
    """
    if resume == "never" and output_path.is_dir() and any(output_path.iterdir()):
        raise StepError(
            f"output_path {str(output_path)!r} is not empty and resume is 'never'. Use "
            f"resume: if_possible to continue or restart in place, or a fresh output_path."
        )


# What Data Designer writes into output_path. clear_previous_run removes only these,
# so anything else a user keeps there survives.
_DD_ARTIFACTS = (
    "parquet-files",
    "tmp-partial-parquet-files",
    "dropped-columns-parquet-files",
    "processors-files",
    "images",
    "metadata.json",
    "builder_config.json",
)


def resume_mismatch(
    output_path: Path,
    fingerprint: Mapping[str, Any],
    run_config: Mapping[str, Any],
    num_records: int,
) -> str | None:
    """Why the previous run in output_path can't be resumed, or None when it can.

    Mirrors the checks Data Designer 0.9.5 makes once generation has started
    (``_check_resume_config_compatibility`` and ``_load_resume_state``), so a mismatch is
    caught BEFORE create(), not after a full generation. Measured before this existed:
      - a changed buffer_size failed the run: "Cannot resume: buffer_size=4 does not match";
      - a changed preserve_dropped_columns made Data Designer regenerate everything into
        a timestamped sibling dir, which the step then refused to register.
    Also returns None when there is no previous run (no metadata.json) — nothing to
    resume or to clear. ``run_config`` is the build's RunConfig fields; omitted ones take
    Data Designer's defaults.
    """
    metadata_path = output_path / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return "its metadata.json is unreadable"
    if not isinstance(metadata, dict):
        return "its metadata.json is not an object"

    from data_designer.config.run_config import RunConfig

    current = RunConfig.model_validate(dict(run_config))
    if metadata.get("config_hash_version") != fingerprint["config_hash_version"]:
        return "it was written by a Data Designer with a different config fingerprint version"
    if metadata.get("config_hash") != fingerprint["config_hash"]:
        return "the config changed"
    if metadata.get("buffer_size") != current.buffer_size:
        return (
            f"run_config.buffer_size changed ({metadata.get('buffer_size')} -> "
            f"{current.buffer_size})"
        )
    if metadata.get("preserve_dropped_columns", True) != current.preserve_dropped_columns:
        return "run_config.preserve_dropped_columns changed"
    target = metadata.get("target_num_records")
    if isinstance(target, int) and num_records < target:
        return f"num_records ({num_records}) is below the previous run's target ({target})"
    return None


def clear_previous_run(output_path: Path) -> None:
    """Remove a previous run's Data Designer artifacts from output_path, and nothing else.

    Only the entries in ``_DD_ARTIFACTS`` and Data Designer's export file
    (``<name>.<format>``) are removed, so other files a user keeps in output_path
    survive. Called only after resume_mismatch found a metadata.json there, so
    output_path is a Data Designer artifact dir and never ``/`` or a parent of one.
    """
    import shutil

    for name in _DD_ARTIFACTS + tuple(f"{output_path.name}.{f}" for f in EXPORT_FORMATS):
        entry = output_path / name
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        elif entry.exists() or entry.is_symlink():
            entry.unlink()


def count_resumable_records(output_path: Path, resume: str) -> int:
    """Rows already checkpointed in output_path that this run will keep, not regenerate.

    Data Designer resumes from the completed batch files under ``parquet-files/``
    (measured: it reported "2 of 4 row group(s) already complete" from 2 files while
    metadata.json still said 1), so this counts the same files. Called after
    clear_previous_run, so a run that can't be resumed has already been cleared and
    counts 0.
    Recorded as step metadata, which makes a resume visible in build lineage.

    Counts only when a metadata.json is present too: without one Data Designer does not
    resume, so batch files alone would over-report (a review suspicion, kept out by
    construction).
    """
    if resume == "never" or not (output_path / "metadata.json").is_file():
        return 0
    batches = sorted((output_path / "parquet-files").glob("*.parquet"))
    if not batches:
        return 0
    import pyarrow.parquet as pq

    return sum(pq.read_metadata(b).num_rows for b in batches)


def validate_providers(resolved: list[dict[str, Any]], model_provider_cls: Any) -> list[Any]:
    """Build Data Designer ModelProvider objects without letting a credential reach an error.

    ``resolved`` already carries secrets (api_key, extra_headers values) read from the
    environment. A pydantic ValidationError prints the offending ``input_value`` — the
    whole entry, secrets included — so a typo such as ``endpont:`` used to put the key
    in the traceback (found in review, reproduced). So each entry is validated first
    with every credential value replaced by a placeholder, and errors are reported by
    field location only, as a StepError.
    """
    from pydantic import ValidationError

    objects = []
    for i, entry in enumerate(resolved):
        redacted = dict(entry)
        if redacted.get("api_key"):
            redacted["api_key"] = "<redacted>"
        if isinstance(redacted.get("extra_headers"), dict):
            redacted["extra_headers"] = {k: "<redacted>" for k in redacted["extra_headers"]}
        try:
            model_provider_cls.model_validate(redacted)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(x) for x in err['loc']) or '<entry>'}: {err['msg']}"
                for err in e.errors(include_input=False, include_url=False)
            )
            raise StepError(
                f"model_providers[{i}] ({entry.get('name', '?')}) is not a valid provider: "
                f"{problems}"
            ) from None
        objects.append(model_provider_cls.model_validate(entry))
    return objects


def check_marker_values(artifact_id: str, output_path: Path) -> None:
    """The marker is space-delimited and the monitor puts the path in a JSON string."""
    if not artifact_id or re.search(r'[\s"]', artifact_id):
        raise StepError(f"artifact id must be non-empty with no whitespace or '\"': {artifact_id!r}")
    if '"' in str(output_path):
        raise StepError(f"output path contains a double quote: {str(output_path)!r}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--artifact-id", required=True)
    ap.add_argument("--num-records", required=True, type=int)
    ap.add_argument("--resume", required=True, choices=RESUME_MODES)
    ap.add_argument("--export-format", default="", choices=("",) + EXPORT_FORMATS)
    ap.add_argument("--min-records", default="")
    ap.add_argument("--seed-path", default="")
    ap.add_argument("--seed-glob", default="")
    return ap.parse_args(argv)


def run(args: argparse.Namespace, env: Mapping[str, str]) -> int:
    output_path = Path(os.path.abspath(args.output_path))
    check_marker_values(args.artifact_id, output_path)
    check_output_dir(output_path, args.resume)
    if args.num_records < 1:
        raise StepError(f"num_records must be positive, got {args.num_records}.")
    min_records = int(args.min_records) if args.min_records != "" else 1
    if min_records > args.num_records:
        raise StepError(
            f"min_records ({min_records}) is greater than num_records ({args.num_records}); "
            f"the run could never pass."
        )

    spec = load_spec(env)
    config = apply_seed_path(spec["config"], args.seed_path, args.seed_glob)
    providers = build_providers(spec["model_providers"], env)
    local = local_model_provider(spec["local_model"], providers)
    if local:
        providers.append(local)

    # Both must be set before data_designer is imported: telemetry is read at import
    # time, and the home dir is where Data Designer writes its default settings files —
    # kept in the per-run workdir rather than a shared node's ~/.data-designer.
    os.environ.setdefault("NEMO_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("DATA_DESIGNER_HOME", os.path.abspath(".data-designer-home"))

    from importlib.metadata import version

    from data_designer.config.config_builder import DataDesignerConfigBuilder
    from data_designer.config.models import ModelProvider
    from data_designer.config.run_config import ResumeMode, RunConfig
    from data_designer.interface import DataDesigner

    builder = DataDesignerConfigBuilder.from_config(config)
    provider_objects = validate_providers(providers, ModelProvider)
    # artifact_path/dataset_name are output_path's parent/basename, so Data Designer
    # writes the dataset exactly at output_path — the path the marker registers.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    designer = DataDesigner(
        artifact_path=str(output_path.parent),
        model_providers=provider_objects,
    )
    if spec["run_config"]:
        designer.set_run_config(RunConfig.model_validate(spec["run_config"]))
    designer.validate(builder)

    # Settle what happens to a previous run in output_path BEFORE generating, so a
    # mismatch costs nothing: Data Designer only discovers it inside create().
    mismatch = resume_mismatch(
        output_path, builder.build().fingerprint(), spec["run_config"], args.num_records
    )
    if mismatch and args.resume == "always":
        raise StepError(
            f"cannot resume the run in {str(output_path)!r}: {mismatch}. Use resume: "
            f"if_possible to start fresh there, or a new output_path."
        )
    if mismatch and args.resume == "if_possible":
        print(
            f"data-designer: not resuming the run in {str(output_path)!r} because "
            f"{mismatch}; clearing it and starting fresh in place"
        )
        clear_previous_run(output_path)
    if args.resume == "always" and not (output_path / "metadata.json").is_file():
        raise StepError(
            f"resume: always needs a previous run to continue, and {str(output_path)!r} "
            f"has none. Use resume: if_possible, which starts fresh when there is "
            f"nothing to resume."
        )
    resumed = count_resumable_records(output_path, args.resume)
    if resumed:
        print(f"data-designer: resuming — {resumed} record(s) already checkpointed in {str(output_path)!r}")
    results = designer.create(
        builder,
        num_records=args.num_records,
        dataset_name=output_path.name,
        resume=ResumeMode(args.resume),
    )

    written = Path(results.artifact_storage.base_dataset_path).resolve()
    if written != output_path.resolve():
        raise StepError(
            f"Data Designer wrote to {str(written)!r}, not output_path {str(output_path)!r}; "
            f"refusing to register a path that does not hold this run's data."
        )

    count = results.count_records()
    print(f"data-designer: generated {count} of {args.num_records} requested record(s)")
    if count < min_records:
        raise StepError(
            f"only {count} record(s) were generated, below min_records={min_records}; "
            f"rows whose cells failed are dropped — see the log above for the failures."
        )

    if args.export_format:
        exported = results.export(
            output_path / f"{output_path.name}.{args.export_format}", format=args.export_format
        )
        print(f"data-designer: exported {exported}")

    # Lineage, then the marker. Both must start at the beginning of a line for the
    # monitor's regex, and the marker must be last: nothing after it can fail the run.
    print(f"GB_STEP_METADATA_KEY:data_designer_version GB_STEP_METADATA_VALUE:{version('data-designer')}")
    print(f"GB_STEP_METADATA_KEY:records_generated GB_STEP_METADATA_VALUE:{count}")
    print(f"GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:{resumed}")
    print(f"GB_ARTIFACT_ID:{args.artifact_id} GB_ARTIFACT_PATH:{output_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run(args, os.environ)
    except StepError as e:
        print(f"data-designer: ERROR {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
