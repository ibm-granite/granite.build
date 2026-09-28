"""Build-tracking metadata for SkyPilot tasks (unmanaged and managed).

Renders a small set of identifying fields from a step's ``run_metadata`` into
two channels attached to every launched SkyPilot task, so an operator can find
the cluster job for a given build:

  * :func:`task_metadata_labels` -> ``sky.Resources(labels=...)``. SkyPilot
    applies these only on Kubernetes (pod labels) and cloud backends (instance
    tags); they are silently ignored on SLURM/LSF.
  * :func:`task_metadata_comment` / :func:`apply_slurm_comment_override` -> a
    SLURM ``#SBATCH --comment=`` directive, via a ``slurm.sbatch_options.comment``
    entry in the ``_cluster_config_overrides`` dict. SkyPilot applies this only
    on the SLURM backend; the ``slurm`` config section is schema-valid (and
    inert) on other backends, so setting it unconditionally is safe.

Both channels carry the same fields, defined once in :data:`_METADATA_FIELDS`
so a new field is a one-line addition (mirrors the
``STANDARD_STEP_ENV_FROM_RUN_METADATA`` table in ``environment.py``).

Note: an LSF backend (e.g. bluevela) receives neither channel. SkyPilot's LSF
provisioner reads ``bsub_options`` only from the operator's persistent
``~/.sky/config.yaml`` (``lsf.cluster_configs.<name>.bsub_options``), never from
per-task ``_cluster_config_overrides``. Per-task LSF metadata would require
operator config and is out of scope here.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping

# Maximum length of a Kubernetes label value (also within cloud tag limits).
_MAX_LABEL_VALUE_LEN = 63


def normalize_run_metadata(run_metadata: Any) -> Dict:
    """Coerce a launcher's ``run_metadata`` kwarg into a plain dict.

    The codebase passes ``run_metadata`` as either a ``dict`` or an
    ``EntityRunMetadata`` object, so both SkyPilot launchers normalize it before
    use. Shared here to keep that logic in one place.

    Args:
        run_metadata: The raw value (dict, EntityRunMetadata, ``None``, or other).

    Returns:
        The value unchanged if it is already a dict; its ``to_dict()`` result if
        it exposes one; otherwise ``{}`` (also for ``None``).
    """
    if isinstance(run_metadata, dict):
        return run_metadata
    if hasattr(run_metadata, "to_dict"):
        return run_metadata.to_dict()
    return {}


@dataclass(frozen=True)
class _MetaField:
    """One tracked metadata field and how it renders into each channel.

    Attributes:
        run_key: Key to read from the ``run_metadata`` mapping.
        comment_key: Key used for this field in the ``--comment`` string.
        label_key: Kubernetes/cloud label key (must be label-safe).
    """

    run_key: str
    comment_key: str
    label_key: str


# Order here is the order fields render into the comment string. Add a line to
# roll a new field out to both channels (labels + SLURM comment) at once.
_METADATA_FIELDS = (
    _MetaField("build_id", "build_id", "gb-build-id"),
    _MetaField("build_config_name", "build_name", "gb-build-name"),
    _MetaField("target_name", "target", "gb-target-name"),
    _MetaField("targetstep_uri", "step_uri", "gb-step-uri"),
    _MetaField("targetrun_id", "target_id", "gb-target-id"),
    _MetaField("targetsteprun_id", "step_id", "gb-step-id"),
)


def _field_value(run_metadata: Mapping, run_key: str) -> str:
    """Return the stripped string value for a field, or "" if absent/blank.

    Args:
        run_metadata: The step's run metadata mapping.
        run_key: Field key to read.

    Returns:
        The value coerced to a stripped string, or "" when missing/blank.
    """
    value = run_metadata.get(run_key)
    return str(value).strip() if value is not None else ""


def _label_value(value: str) -> str:
    """Coerce a raw value into a Kubernetes/cloud label-safe token.

    Lowercases, collapses each run of non-``[a-z0-9]`` characters to a single
    ``-``, trims leading/trailing ``-``, then truncates to the 63-char label
    limit (dropping any ``-`` left dangling by truncation).

    Args:
        value: The raw field value.

    Returns:
        A ``[a-z0-9-]`` token no longer than 63 chars (may be "").
    """
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:_MAX_LABEL_VALUE_LEN].rstrip("-")


def _comment_value(value: str) -> str:
    """Coerce a raw value into a single delimiter-safe comment token.

    Two hazards are neutralized so a value can never break out of its slot in
    the ``key=value;key=value`` comment:

    * **Whitespace/newlines** are collapsed to ``_``. SkyPilot emits the SLURM
      comment unquoted (``--comment=<value>``), so whitespace would split the
      directive and newlines are rejected by SkyPilot's schema.
    * **The structural delimiters ``;`` and ``=``** are replaced (``;`` -> ``,``,
      ``=`` -> ``-``). Otherwise a free-form value like ``a;target=evil`` would
      render as ``build_name=a;target=evil``, and a ``;``-splitting consumer
      would parse ``target=evil`` as a fabricated extra field (and lose the
      original value).

    Internal ``:``/``/`` are preserved so a full step uri (e.g.
    ``space://steps/foo``) survives intact.

    Args:
        value: The raw field value.

    Returns:
        The value with whitespace collapsed to ``_`` and ``;``/``=`` replaced.
    """
    collapsed = re.sub(r"\s+", "_", value)
    return collapsed.replace(";", ",").replace("=", "-")


def task_metadata_labels(run_metadata: Mapping) -> Dict[str, str]:
    """Build label-safe key/value pairs for ``sky.Resources(labels=...)``.

    Args:
        run_metadata: The step's run metadata mapping.

    Returns:
        A dict of ``label_key -> label-safe value`` for every tracked field
        with a non-empty value; ``{}`` when nothing is present (so the caller
        can pass ``labels=... or None``).
    """
    labels: Dict[str, str] = {}
    for meta in _METADATA_FIELDS:
        label = _label_value(_field_value(run_metadata, meta.run_key))
        if label:
            labels[meta.label_key] = label
    return labels


def task_metadata_comment(run_metadata: Mapping) -> str:
    """Build the single-line ``key=value;key=value`` SLURM comment string.

    Args:
        run_metadata: The step's run metadata mapping.

    Returns:
        A ``;``-joined ``key=value`` string over every tracked field with a
        non-empty value, or "" when nothing is present.
    """
    parts = []
    for meta in _METADATA_FIELDS:
        value = _comment_value(_field_value(run_metadata, meta.run_key))
        if value:
            parts.append(f"{meta.comment_key}={value}")
    return ";".join(parts)


def apply_slurm_comment_override(overrides: Dict, run_metadata: Mapping) -> None:
    """Attach a SLURM ``--comment`` directive to cluster config overrides.

    Mutates ``overrides`` in place, setting
    ``overrides["slurm"]["sbatch_options"]["comment"]`` when the rendered
    comment is non-empty. Only SkyPilot's SLURM backend consumes this; the
    ``slurm`` section is schema-valid and inert on other backends. (LSF does
    NOT read per-task overrides — see the module docstring.)

    Args:
        overrides: The ``_cluster_config_overrides`` dict to mutate.
        run_metadata: The step's run metadata mapping.
    """
    comment = task_metadata_comment(run_metadata)
    if comment:
        overrides.setdefault("slurm", {}).setdefault("sbatch_options", {})[
            "comment"
        ] = comment
