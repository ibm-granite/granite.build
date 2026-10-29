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

  Everything else arrives in $GB_DD_SPEC_B64: base64 of the YAML mapping
  {"config": {...}, "model_providers": [...], "run_config": {...}, "local_model": {...}}
  (the template's to_yaml filter; dates in it are read back as ISO strings).

ORDER, and why it matters
  1. Everything that cannot depend on the work is checked first (marker-safe id and
     path, a non-empty output dir under resume=never, the spec, credentials, the seed
     override, the providers — reported without their secrets), so a bad build fails
     in seconds rather than after the generation.
  2. Data Designer's validate() checks the config with no model calls. Then a previous
     run in output_path is checked the way Data Designer would check it (config,
     buffer_size, preserve_dropped_columns). A run that matches is resumed. A mismatch
     is refused by resume=if_possible (the default) and resume=always, with nothing
     removed; only resume=restart clears it and starts fresh in place. So is Data
     Designer output left with no metadata.json. A num_records below the previous target is
     refused in every mode and nothing is cleared.
  3. create() — which health-checks each referenced model first.
  4. The row count is checked against --min-records, then the optional export.
  5. Only then is the artifact marker printed, so a failed or short run registers
     nothing.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import copy
import datetime
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

SPEC_ENV = "GB_DD_SPEC_B64"
RESUME_MODES = ("never", "always", "if_possible", "restart")
EXPORT_FORMATS = ("jsonl", "csv", "parquet")
# Seed source types whose `path` names a file, glob or directory on the filesystem —
# the ones a staged input can stand in for. `hf` names a Hub dataset and is left as is.
PATH_SEED_TYPES = ("local", "directory", "file_contents", "agent_rollout")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The key local_model_provider() sends. Ollama ignores it; it is not a secret.
LOCAL_API_KEY = "ollama"


class StepError(Exception):
    """A configuration problem reported as one ERROR line, without a traceback."""


def _isoformat_dates(value: Any) -> Any:
    """``value`` with every date/datetime turned into its ISO string, recursively.

    YAML reads an unquoted ``2024-01-01`` as a date. Data Designer wants strings
    (e.g. the datetime sampler's ``start``/``end``), and its config must stay JSON-able
    for the fingerprint and builder_config.json.
    """
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _isoformat_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_isoformat_dates(v) for v in value]
    return value


def load_spec(env: Mapping[str, str]) -> dict[str, Any]:
    """Decode and shape-check the config + providers the template passed in."""
    raw = env.get(SPEC_ENV, "")
    if not raw:
        raise StepError(f"${SPEC_ENV} is empty; the step template did not pass a spec.")
    import yaml  # a data-designer dependency, so in the step's venv

    try:
        spec = yaml.safe_load(base64.b64decode(raw, validate=True))
    except (binascii.Error, ValueError, yaml.YAMLError) as e:
        raise StepError(f"${SPEC_ENV} is not base64-encoded YAML: {e}") from None
    spec = _isoformat_dates(spec)
    if not isinstance(spec, dict):
        raise StepError(
            f"${SPEC_ENV} must decode to an object, got {type(spec).__name__}."
        )
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
        raise StepError(
            "data_designer_config.run_config must be a mapping of RunConfig fields."
        )
    return {
        "config": config,
        "model_providers": providers,
        "run_config": run_config,
        "local_model": local_model,
    }


def local_model_provider(
    local_model: Mapping[str, Any], providers: list[Any]
) -> dict[str, Any] | None:
    """The provider for a model the step serves on the node, or None when there is none.

    The server listens on 127.0.0.1, so the endpoint never leaves the node. Ollama's
    OpenAI-compatible API ignores the key, but the client sends one, so a fixed
    placeholder is used — not a secret, and nothing to declare.
    """
    runtime = local_model.get("runtime") or ""
    if not runtime:
        return None
    if runtime != "ollama":
        raise StepError(
            f"local_model.runtime must be empty or 'ollama', got {runtime!r}."
        )
    name = local_model.get("name") or "local"
    taken = {p.get("name") for p in providers if isinstance(p, dict)}
    if name in taken:
        raise StepError(
            f"local_model.name {name!r} is also a model_providers name; rename one."
        )
    return {
        "name": name,
        "endpoint": f"http://127.0.0.1:{int(local_model.get('port') or 11434)}/v1",
        "api_key": LOCAL_API_KEY,
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


def build_providers(
    providers: list[Any], env: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Resolve the step-only credential keys into plain ModelProvider dicts.

    ``api_key_env: NAME`` becomes ``api_key: $NAME``; ``extra_headers_from_env:
    {Header: NAME}`` is merged into ``extra_headers`` as ``Header: $NAME``. Both fail
    loudly when the variable is missing. A plain ``api_key`` passes through untouched —
    note Data Designer reads it as an environment variable NAME first and falls back
    to the literal string, so a missing variable is sent as the key itself. Prefer
    ``api_key_env``.
    """
    # Duplicate names are refused here, BEFORE any credential is resolved: Data
    # Designer rejects them too, but its error prints the whole provider list, secrets
    # included (third review, reproduced).
    names = [e.get("name") for e in providers if isinstance(e, dict)]
    dupes = sorted({str(n) for n in names if names.count(n) > 1})
    if dupes:
        raise StepError(
            f"model_providers names must be unique; repeated: {', '.join(dupes)}."
        )
    resolved = []
    for i, entry in enumerate(providers):
        if not isinstance(entry, dict):
            raise StepError(
                f"model_providers[{i}] must be a mapping, got {type(entry).__name__}."
            )
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
                raise StepError(
                    f"{label}.extra_headers_from_env must map header -> env var."
                )
            headers = dict(p.get("extra_headers") or {})
            for header, name in header_env.items():
                headers[header] = _require_env(
                    env, name, f"{label}.extra_headers_from_env[{header!r}]"
                )
            p["extra_headers"] = headers
        resolved.append(p)
    return resolved


def provider_secrets(
    providers: list[Mapping[str, Any]], env: Mapping[str, str]
) -> list[str]:
    """Every credential value in the resolved providers, longest first.

    Decided by where a value came from, never by its length (third review of #543: a
    length cutoff let a short real key through and still garbled long plain values):
      - ``api_key``: from ``api_key_env``, or literal. A plain api_key that names a
        variable set on the node is a NAME (Data Designer reads it as one first), so
        that variable's value is the secret and the name stays readable in errors.
      - every ``extra_headers`` value, literal or from ``extra_headers_from_env``.
        Which headers carry a credential can't be told from their names, so none is
        assumed safe.
    The local model's placeholder key is the one exception: it is not a secret, and
    redacting it turned ``dd_ollama.sh`` into ``dd_<redacted>.sh`` (re-review of #543).

    Args:
        providers: The resolved provider dicts from build_providers() (and the local
            model's provider, if any).
        env: The environment the providers were resolved from.

    Returns:
        The distinct non-empty secret values, longest first, so a secret containing
        another is replaced whole.
    """
    secrets = set()
    for p in providers:
        key = p.get("api_key")
        if isinstance(key, str):
            secrets.add(env.get(key) or key)
        headers = p.get("extra_headers")
        if isinstance(headers, dict):
            secrets.update(v for v in headers.values() if isinstance(v, str))
    secrets.discard(LOCAL_API_KEY)
    return sorted((x for x in secrets if x), key=len, reverse=True)


def redact(text: str, secrets: list[str]) -> str:
    """``text`` with every value in ``secrets`` replaced by ``<redacted>``.

    One pass over ``text``, so a ``<redacted>`` already written is never scanned again:
    replacing secrets one after another let a short value (a header such as ``2`` or
    ``e``) rewrite the markers and every word around them (independent review of the
    #543 fixes). Each secret is matched as written and as Python's repr shows it
    (escaped backslashes and quotes), since Data Designer's ModelProvider repr is
    where tracebacks print it. A value under 8 characters (``none``, ``2``, ``1.0``,
    ``en-US``) is matched only as a whole token, not inside a longer word, number or
    path, so it doesn't turn ``nonexistent``, ``212`` or ``11.05s`` into noise; it is
    still always redacted where it stands alone. Anything 8 characters or longer
    (``sk-…``, ``Bearer …``, a 32-character hex token) is matched anywhere, even glued
    to other text.

    Args:
        text: The text to clean, e.g. an error message or a formatted traceback.
        secrets: Values to remove, as provider_secrets() returns them.

    Returns:
        The cleaned text.
    """
    forms = set()
    for secret in secrets:
        forms.update({secret, repr(secret)[1:-1]})
    forms.discard("")
    if not forms:
        return text
    patterns = [
        rf"(?<![\w.\-/]){re.escape(f)}(?![\w.\-/])" if len(f) < 8 else re.escape(f)
        for f in sorted(forms, key=len, reverse=True)
    ]
    return re.sub("|".join(patterns), "<redacted>", text)


@contextlib.contextmanager
def redacted_errors(secrets: list[str], passthrough: tuple[type, ...] = ()):
    """Turn any Data Designer error into a StepError with every credential removed.

    validate() and create() raise Data Designer's own errors as tracebacks, and its
    ModelProvider prints api_key and extra_headers in clear (its repr shows them), so
    an error that includes a provider would put the key in the job log (review of
    #543). The traceback is still printed, for debugging, with each secret replaced.
    ``passthrough`` exceptions are re-raised untouched, for a caller that reports them
    itself without their values.
    """
    try:
        yield
    except (StepError, *passthrough):
        raise
    except Exception as e:
        print(redact("".join(traceback.format_exception(e)), secrets), file=sys.stderr)
        raise StepError(
            redact(f"Data Designer failed: {type(e).__name__}: {e}", secrets)
        ) from None


def apply_seed_path(
    config: dict[str, Any], seed_path: str, seed_glob: str
) -> dict[str, Any]:
    """Point the config's seed source at the staged `seed` input, when one is bound."""
    if not seed_path:
        return config
    config = copy.deepcopy(config)
    inner = config["data_designer"] if "data_designer" in config else config
    seed_config = inner.get("seed_config") if isinstance(inner, dict) else None
    if not isinstance(seed_config, dict) or not isinstance(
        seed_config.get("source"), dict
    ):
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
            f"resume: if_possible to continue the run there, resume: restart to discard "
            f"it, or a fresh output_path."
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
    "scheduler_events.jsonl",  # only with run_config.write_scheduler_events
)


# Keys every Data Designer 0.9.5 metadata.json has (checked against a real run). Together
# with builder_config.json they mark output_path as Data Designer's own artifact dir.
_DD_METADATA_KEYS = ("config_hash", "config_hash_version", "target_num_records")


def read_metadata(output_path: Path) -> dict[str, Any] | None | str:
    """Read output_path's ``metadata.json`` ONCE for every previous-run check.

    Returns None when there is none, the parsed object when it is valid JSON, or a
    string saying why it can't be used. The checks used to each read it on their own
    and disagreed on bad JSON (one raised, the others ignored it; review of #543).
    """
    metadata_path = output_path / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return "its metadata.json is not valid JSON"
    return (
        metadata if isinstance(metadata, dict) else "its metadata.json is not an object"
    )


def foreign_contents(
    output_path: Path, metadata: dict[str, Any] | None | str
) -> str | None:
    """Why output_path can't be treated as a Data Designer run's directory, or None.

    ``metadata`` is ``read_metadata(output_path)``. A ``metadata.json`` that is not
    Data Designer's must never be cleared: the third review reproduced a user's
    ``metadata.json`` being taken for a previous run, and ``images/`` plus a csv beside
    it deleted.
    """
    if metadata is None:
        return None
    if isinstance(metadata, str):
        return metadata
    if not all(k in metadata for k in _DD_METADATA_KEYS):
        return "its metadata.json is not one Data Designer wrote"
    if not (output_path / "builder_config.json").is_file():
        return "it has a metadata.json but no builder_config.json"
    return None


# Data Designer outputs that can't mix into the next run, so a leftover copy is never
# refused: it clears the partials and rewrites builder_config.json on every start. A run
# interrupted before its first row group usually leaves only these. With an image column
# it may also leave images/, and with run_config.write_scheduler_events the appended
# scheduler_events.jsonl; those WOULD carry over, so they are refused like any other
# output and resume: restart clears them (independent review of the #543 fixes).
_RESTARTABLE_LEFTOVERS = ("tmp-partial-parquet-files", "builder_config.json")


def orphaned_outputs(output_path: Path, metadata: Any) -> list[str]:
    """Data Designer outputs left in output_path without a metadata.json beside them.

    Data Designer neither resumes nor clears them: it writes the new run beside them,
    so they become part of the registered dataset. Row groups in ``parquet-files/``
    mixed two runs' rows (measured: "generated 4 of 2 requested", rows from two
    configs; review of #543); dropped columns, processor outputs and images were kept
    the same way (re-review of #543). Only ``_RESTARTABLE_LEFTOVERS``, which Data
    Designer clears or rewrites itself, is not counted. Returns the names found, empty
    when there is a metadata.json.
    """
    if metadata is not None:
        return []
    found = []
    for name in _DD_ARTIFACTS:
        entry = output_path / name
        if name in _RESTARTABLE_LEFTOVERS or name == "metadata.json":
            continue
        if entry.is_dir() and not entry.is_symlink():
            if any(entry.iterdir()):
                found.append(f"{name}/")
        elif entry.exists() or entry.is_symlink():
            found.append(name)
    found += [
        f"{output_path.name}.{f}"
        for f in EXPORT_FORMATS
        if (output_path / f"{output_path.name}.{f}").exists()
    ]
    return found


def resume_mismatch(
    metadata: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> str | None:
    """Why the previous run in output_path can't be resumed, or None when it can.

    Mirrors the checks Data Designer 0.9.5 makes once generation has started
    (``_check_resume_config_compatibility`` and ``_load_resume_state``), so a mismatch is
    caught BEFORE create(), not after a full generation. A num_records below the
    previous target is deliberately NOT a mismatch here; see num_records_below_target.
    Measured before this existed:
      - a changed buffer_size failed the run: "Cannot resume: buffer_size=4 does not match";
      - a changed preserve_dropped_columns made Data Designer regenerate everything into
        a timestamped sibling dir, which the step then refused to register.
    ``metadata`` is the previous run's metadata.json, which foreign_contents() has
    confirmed is Data Designer's. ``run_config`` is the build's RunConfig fields; omitted
    ones take Data Designer's defaults.
    """
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
    if (
        metadata.get("preserve_dropped_columns", True)
        != current.preserve_dropped_columns
    ):
        return "run_config.preserve_dropped_columns changed"
    return None


def post_processed_refusal(metadata: Mapping[str, Any], num_records: int) -> str | None:
    """Why a previous run's after-generation processing rules out this resume, or None.

    Mirrors ``_post_generation_processed_resume_result`` (Data Designer 0.9.5): once
    after-generation processors have run, or started, the parquet files may already be
    rewritten, so the dataset can't be extended. Data Designer only refuses that inside
    create(), after the step has printed "resuming" (third review); checked up front
    here. The same num_records with a finished, processed dataset is fine — nothing to do.
    """
    state = metadata.get("post_generation_state")
    if state == "started":
        return (
            "its after-generation processing started but did not finish, so its files "
            "may be part-rewritten"
        )
    processed = metadata.get("post_generation_processed", False) or state == "complete"
    target = metadata.get("target_num_records")
    if processed and isinstance(target, int) and num_records > target:
        return (
            f"after-generation processors already ran on its {target} records, so it "
            f"can't be extended to {num_records}"
        )
    return None


def num_records_below_target(
    metadata: Mapping[str, Any], num_records: int
) -> int | None:
    """The previous run's target when num_records is below it, else None.

    Not a resume mismatch: the previous run is the same dataset, and clearing it to honour
    a smaller num_records would destroy rows for no reason. Data Designer refuses this
    too; the step refuses it up front, in every resume mode, and never clears.
    """
    target = metadata.get("target_num_records")
    return target if isinstance(target, int) and num_records < target else None


def compiled_fingerprint(designer: Any, builder: Any) -> dict[str, Any]:
    """The fingerprint Data Designer records for this config: of the COMPILED config.

    Data Designer compiles the config before generating (``DatasetBuilder.__init__`` calls
    ``compile_data_designer_config``): it adds the seed's columns, and an internal row-id
    column when the config has no sampler or seed column. ``metadata.json`` stores the
    fingerprint of that compiled config, so comparing the build's raw config against it
    made every expression-only, LLM-only or seeded config look changed (found in the
    second review, reproduced): the step then cleared a resumable run and
    ``always`` never resumed. This compiles a fresh copy exactly as ``DataDesigner.validate``
    does and fingerprints that. It uses Data Designer's private ``_create_resource_provider``,
    acceptable because ``dd_version`` is pinned; ``test_dd_run.py`` covers the configs that
    exposed the bug.
    """
    from data_designer.engine.compiler import compile_data_designer_config

    # Named as DataDesigner.validate names it, not after output_path: with the real
    # dataset name, ArtifactStorage logs "will be saved to <name>_<timestamp> instead"
    # for an existing output_path, which nothing does (third review). The name doesn't
    # enter the fingerprint.
    provider = designer._create_resource_provider("validate-configuration", builder)
    return compile_data_designer_config(builder.build(), provider).fingerprint()


def clear_previous_run(output_path: Path) -> None:
    """Remove a previous run's Data Designer artifacts from output_path, and nothing else.

    Only the entries in ``_DD_ARTIFACTS`` and Data Designer's export file
    (``<name>.<format>``) are removed, so other files a user keeps in output_path
    survive. Called only once output_path is confirmed as a Data Designer artifact dir:
    by foreign_contents() (its metadata.json keys plus builder_config.json), or, for
    row groups left without a metadata.json, by its builder_config.json. So it is never
    ``/``, a parent of one, or a directory that only happens to hold a metadata.json.
    """
    import shutil

    for name in _DD_ARTIFACTS + tuple(
        f"{output_path.name}.{f}" for f in EXPORT_FORMATS
    ):
        entry = output_path / name
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        elif entry.exists() or entry.is_symlink():
            entry.unlink()


def count_resumable_records(output_path: Path, has_metadata: bool) -> int:
    """Rows already checkpointed in output_path that this run will keep, not regenerate.

    Data Designer resumes from the completed batch files under ``parquet-files/``
    (measured: it reported "2 of 4 row group(s) already complete" from 2 files while
    metadata.json still said 1), so this counts the same files. A run that can't be
    resumed has been refused or cleared by now. Recorded as step metadata, which makes
    a resume visible in build lineage.

    Counts only when the previous run's metadata.json is kept (``has_metadata``):
    without one Data Designer does not resume, so batch files alone would over-report.
    """
    if not has_metadata:
        return 0
    batches = sorted((output_path / "parquet-files").glob("*.parquet"))
    if not batches:
        return 0
    import pyarrow.parquet as pq

    return sum(pq.read_metadata(b).num_rows for b in batches)


def _provider_error(i: int, entry: Mapping[str, Any], err: Any) -> StepError:
    """A StepError naming the failing fields by location only — never their values."""
    problems = "; ".join(
        f"{'.'.join(str(x) for x in e['loc']) or '<entry>'}: {e['msg']}"
        for e in err.errors(include_input=False, include_url=False)
    )
    return StepError(
        f"model_providers[{i}] ({entry.get('name', '?')}) is not a valid provider: {problems}"
    )


def validate_providers(
    resolved: list[dict[str, Any]], model_provider_cls: Any
) -> list[Any]:
    """Build Data Designer ModelProvider objects without letting a credential reach an error.

    ``resolved`` already carries secrets (api_key, extra_headers values) read from the
    environment. A pydantic ValidationError prints the offending ``input_value`` — the
    whole entry, secrets included — so a typo such as ``endpont:`` used to put the key
    in the traceback (found in review, reproduced).

    Two passes, both reporting by field location only (``include_input=False``) and
    without chaining the original exception:
      1. with each credential replaced by a placeholder OF THE SAME TYPE CLASS (a string
         stays a string; a non-string stays as given, since its type is the error), to
         report structural problems — missing or misspelled fields — first;
      2. the real entry, so a credential of the wrong type (``api_key: 12345``) is also
         reported by location, not as a traceback quoting the value (second review).
    """
    from pydantic import ValidationError

    objects = []
    for i, entry in enumerate(resolved):
        redacted = dict(entry)
        if isinstance(redacted.get("api_key"), str) and redacted["api_key"]:
            redacted["api_key"] = "<redacted>"
        if isinstance(redacted.get("extra_headers"), dict):
            redacted["extra_headers"] = {
                k: ("<redacted>" if isinstance(v, str) else v)
                for k, v in redacted["extra_headers"].items()
            }
        for candidate in (redacted, entry):
            try:
                provider = model_provider_cls.model_validate(candidate)
            except ValidationError as e:
                raise _provider_error(i, entry, e) from None
        objects.append(provider)
    return objects


def check_marker_values(artifact_id: str, output_path: Path) -> None:
    """The marker is space-delimited and the monitor puts the path in a JSON string."""
    if not artifact_id or re.search(r'[\s"]', artifact_id):
        raise StepError(
            f"artifact id must be non-empty with no whitespace or '\"': {artifact_id!r}"
        )
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

    provider_objects = validate_providers(providers, ModelProvider)
    # From here on Data Designer raises its own errors; none may show a credential.
    secrets = provider_secrets(providers, env)
    # artifact_path/dataset_name are output_path's parent/basename, so Data Designer
    # writes the dataset exactly at output_path — the path the marker registers.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    from pydantic import ValidationError

    try:
        with redacted_errors(secrets, passthrough=(ValidationError,)):
            designer = DataDesigner(
                artifact_path=str(output_path.parent),
                model_providers=provider_objects,
            )
    except ValidationError as e:
        # Its input_value is the provider list with credentials resolved; report the
        # failing locations only.
        raise StepError(
            "model_providers were rejected: "
            + "; ".join(
                f"{'.'.join(str(x) for x in err['loc']) or '<providers>'}: {err['msg']}"
                for err in e.errors(include_input=False, include_url=False)
            )
        ) from None
    with redacted_errors(secrets):
        # The builder must come AFTER DataDesigner(): constructing DataDesigner seeds
        # Data Designer's default model configs into DATA_DESIGNER_HOME, and a config
        # that names no model_configs picks those defaults up when the builder is
        # created. Built before, the first run fingerprinted an empty model list and
        # every later run (same workdir, home now seeded) the defaults, so an unchanged
        # config looked changed (measured: sha256:d8e0… then a3e9…).
        builder = DataDesignerConfigBuilder.from_config(config)
        if spec["run_config"]:
            designer.set_run_config(RunConfig.model_validate(spec["run_config"]))
        designer.validate(builder)

        # Settle what happens to a previous run in output_path BEFORE generating, so a
        # mismatch costs nothing: Data Designer only discovers it inside create(). First
        # make sure there IS a Data Designer run there; never clear anything else.
        # resume=never skips all of it: check_output_dir() already refused a non-empty
        # output_path.
        metadata = None if args.resume == "never" else read_metadata(output_path)
        foreign = (
            None if args.resume == "never" else foreign_contents(output_path, metadata)
        )
        if foreign:
            raise StepError(
                f"output_path {str(output_path)!r} can't be resumed or reused: {foreign}. "
                f"Nothing was removed. Use an output_path of its own for this step."
            )
        orphaned = (
            [] if args.resume == "never" else orphaned_outputs(output_path, metadata)
        )
        if orphaned:
            problem = (
                f"output_path {str(output_path)!r} holds Data Designer output "
                f"({', '.join(orphaned)}) but no metadata.json, so it can't be resumed, "
                f"and Data Designer would add the new run beside it. Nothing was removed."
            )
            if args.resume != "restart":
                raise StepError(
                    f"{problem} Set resume: restart to discard it, or use a new "
                    f"output_path."
                )
            # Data Designer writes builder_config.json when a run starts, so it is what
            # still marks the directory as Data Designer's once metadata.json is gone.
            # Without it, clearing is refused even under restart, as for foreign
            # contents.
            if not (output_path / "builder_config.json").is_file():
                raise StepError(
                    f"{problem} There is no builder_config.json either, so this does not "
                    f"look like a Data Designer run and resume: restart won't clear it. Use "
                    f"an output_path of its own for this step."
                )
            print(
                f"data-designer: {str(output_path)!r} holds Data Designer output but no "
                f"metadata.json; resume: restart, so clearing it and starting fresh in "
                f"place"
            )
            clear_previous_run(output_path)
        if metadata:
            # Fingerprinted only when there is a previous run to compare with: it uses a
            # private Data Designer API, which a fresh run then never touches (review of
            # #543).
            mismatch = resume_mismatch(
                metadata, compiled_fingerprint(designer, builder), spec["run_config"]
            )
            if not mismatch:
                target = num_records_below_target(metadata, args.num_records)
                if target is not None:
                    raise StepError(
                        f"num_records ({args.num_records}) is below the {target} the run "
                        f"in {str(output_path)!r} was started with. Resume keeps or extends "
                        f"a dataset, never shrinks it: use num_records >= {target}, or a "
                        f"new output_path. Nothing was removed."
                    )
                processed = post_processed_refusal(metadata, args.num_records)
                if processed:
                    raise StepError(
                        f"cannot resume the run in {str(output_path)!r}: {processed}. Use a "
                        f"new output_path. Nothing was removed."
                    )
            # Deleting a previous run is opt-in: only resume: restart does it. The
            # default (if_possible) and always both refuse a run they can't continue, so
            # changing a config never silently costs the previous dataset.
            elif args.resume in ("always", "if_possible"):
                raise StepError(
                    f"cannot resume the run in {str(output_path)!r}: {mismatch}. Nothing was "
                    f"removed. To discard that run and start fresh there, set resume: "
                    f"restart; to keep it, use a new output_path."
                )
            else:
                print(
                    f"data-designer: not resuming the run in {str(output_path)!r} because "
                    f"{mismatch}; resume: restart, so clearing it and starting fresh in place"
                )
                clear_previous_run(output_path)
                metadata = None
        # `always` with no checkpoint yet (a first run, or one interrupted before its
        # first row group finished) starts fresh, as Data Designer documents for ALWAYS.
        # Data Designer itself raises instead when output_path doesn't exist at all
        # (measured), so it is asked for if_possible in that one case, which starts
        # fresh in place either way.
        dd_resume = "if_possible" if args.resume == "restart" else args.resume
        if args.resume == "always" and not metadata:
            print(
                f"data-designer: resume: always, but {str(output_path)!r} has no checkpoint "
                f"yet; starting fresh"
            )
            dd_resume = "if_possible"
        resumed = count_resumable_records(output_path, bool(metadata))
        if resumed:
            print(
                f"data-designer: resuming — {resumed} record(s) already checkpointed in {str(output_path)!r}"
            )
        results = designer.create(
            builder,
            num_records=args.num_records,
            dataset_name=output_path.name,
            resume=ResumeMode(dd_resume),
        )

        written = Path(results.artifact_storage.base_dataset_path).resolve()
        if written != output_path.resolve():
            raise StepError(
                f"Data Designer wrote to {str(written)!r}, not output_path {str(output_path)!r}; "
                f"refusing to register a path that does not hold this run's data."
            )

        count = results.count_records()
        print(
            f"data-designer: generated {count} of {args.num_records} requested record(s)"
        )
        if count < min_records:
            raise StepError(
                f"only {count} record(s) were generated, below min_records={min_records}; "
                f"rows whose cells failed are dropped — see the log above for the failures."
            )

        if args.export_format:
            exported = results.export(
                output_path / f"{output_path.name}.{args.export_format}",
                format=args.export_format,
            )
            print(f"data-designer: exported {exported}")

    # Lineage, then the marker. Both must start at the beginning of a line for the
    # monitor's regex, and the marker must be last: nothing after it can fail the run.
    print(
        f"GB_STEP_METADATA_KEY:data_designer_version GB_STEP_METADATA_VALUE:{version('data-designer')}"
    )
    print(f"GB_STEP_METADATA_KEY:records_generated GB_STEP_METADATA_VALUE:{count}")
    print(f"GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:{resumed}")
    print(
        f"GB_ARTIFACT_ID:{args.artifact_id} GB_ARTIFACT_PATH:{output_path}", flush=True
    )
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
