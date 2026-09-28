#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Decompose a job into flat lineage rows.

One job execution becomes a set of ``(source, job_id, target)`` triples. The
``job_id`` on every row is what keeps the decomposition lossless: a job with N
inputs and one output (or one input and M outputs) emits max(N, M) rows, and
grouping those rows back by ``job_id`` recovers the original input and output
sets. Without it the flattening would dissolve the grouping, and "each input
produced each output" would be an unrecoverable claim rather than a projection.

The guard in :func:`to_lineage_rows` keeps a job from having more than one of
both, so the pairing is never a true cartesian product.

This is deliberately generic -- the input is a plain dict, not a granite.build
type -- because the same function serves the build sink and the future importers
of Lakehouse/dmf-ng/W&B lineage. Nothing here imports storage or granite.build
models.

An endpoint is the artifact's **normalized URI**, so a producer only has to record
a URI to be understood; there is no per-source identity translation to supply.

Ported from ``prototype-lineage-py`` (``lineage/decompose.py``, in turn
``NewStatsInput.getLineageRecords``), including its N*M guard -- see
:func:`to_lineage_rows`.
"""

from typing import Optional

from gbserver.lineage.uri_normalize import normalize_uri

# Job metadata copied verbatim onto every row emitted from one job. The traversal
# never reads these; they travel so a row can be explained without a second
# lookup.
JOB_METADATA_KEYS = (
    "release_id",
    "category",
    "job_name",
    # The space/build the execution belongs to, as "<space_name>/<build_name>".
    # Load-bearing for authorization, not display: the read path splits it on the
    # first "/" to recover the space and prunes nodes the caller cannot see. A row
    # that loses it fails closed, so every node built from it disappears from every
    # graph -- which is why it must survive decomposition.
    "job_namespace",
    "job_id",
    "job_type",
    "owner",
    "job_started_at",
    "job_completed_at",
    "job_status",
    "job_input_params",
    "execution_stats",
    "job_output_stats",
    "source_code_details",
    # The originating endpoint records, when a source carries them. Lakehouse's
    # dmf.lineage keeps each endpoint as a struct whose snapshot_id, path and
    # extra fields have no column in gb_lineage_index; carrying them here is the only
    # way they survive, and they are not recoverable once Lakehouse is off.
    # Absent from every other source, so nothing else is affected.
    "source_object",
    "target_object",
)


class LineageDecomposeError(ValueError):
    """A job entry cannot be decomposed into lineage rows."""


class LineageRowDraft:
    """One decomposed row, before it becomes a stored item.

    A plain container rather than the storage model, so decomposition stays
    independent of the storage layer and testable without a database.

    ``input`` and ``output`` are the artifacts' **normalized URIs**, and ``""``
    for the terminal cases -- a creation job has no input, a deletion job no
    output. That empty string is real information, not a missing value, and the
    traversal stops there rather than chaining through it. It is also what
    ``normalize_uri`` returns for a URI it cannot identify, which collapses to the
    same thing: an endpoint with no usable identity ends a path.

    There is no separate ``source_uri``/``target_uri`` here any more. The old
    identifier scheme did not encode a URI's scheme, so the real URI had to travel
    alongside it; with the URI *as* the identity the pair would be the same string
    twice.

    Attributes:
        job_id: identity of the job execution; the same value on every row of one
            job, and what makes the decomposition regroupable.
        input: normalized URI of the input artifact, or ``""`` (creation).
        output: normalized URI of the output artifact, or ``""`` (deletion).
        input_artifact: the input artifact dict this row came from, if any.
        output_artifact: the output artifact dict this row came from, if any.
        metadata: the job metadata carried onto this row.
    """

    __slots__ = (
        "job_id",
        "input",
        "output",
        "input_artifact",
        "output_artifact",
        "metadata",
    )

    def __init__(
        self,
        job_id: str,
        input: str = "",
        output: str = "",
        input_artifact: Optional[dict] = None,
        output_artifact: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.job_id = job_id
        self.input = input
        self.output = output
        self.input_artifact = input_artifact
        self.output_artifact = output_artifact
        self.metadata = metadata if metadata is not None else {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineageRowDraft):
            return NotImplemented
        return self.key() == other.key() and self.metadata == other.metadata

    def key(self) -> tuple:
        """Return the identity tuple the storage unique constraint mirrors."""
        return (self.job_id, self.input, self.output)

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return (
            f"LineageRowDraft(job_id={self.job_id!r}, input={self.input!r}, "
            f"output={self.output!r})"
        )


def _artifact_uri(artifact: Optional[dict]) -> str:
    """Read an artifact dict's real URI.

    The URI is carried, not re-derived, because a canonical identifier encodes
    only type, namespace, name, table and revision -- never the scheme. ``lh://``
    happens to be reconstructible from those pieces; ``hf://`` and ``s3://`` are
    not, so losing the URI here loses it for good.

    Three keys are tried because the producers disagree on shape and all three are
    real. ``_artifact_to_lineage_entry`` sets a top-level ``uri`` *and* mirrors the
    same value into the facets as ``artifact_uri`` and ``gb-artifact-uri``; an
    importer supplying only a facet bag should not silently lose its URIs.

    Args:
        artifact: the artifact dict, or ``None`` for a terminal endpoint.

    Returns:
        The URI, or ``""`` when there is no artifact or it carries none. Empty is a
        real answer -- the read path falls back to a synthesized identity rather
        than treating the node as absent.
    """
    if not artifact:
        return ""
    direct = artifact.get("uri")
    if direct:
        return str(direct)
    facets = artifact.get("facets") or {}
    for key in ("artifact_uri", "gb-artifact-uri"):
        value = facets.get(key)
        if value:
            return str(value)
    return ""


def _normalized_job_keys() -> tuple:
    """The job-dict keys holding artifact lists, in endpoint order.

    Exposed so a caller reading endpoints out of a job entry (the build-graph
    seeding) uses the same keys decomposition does, rather than restating them.
    """
    return ("sources", "targets")


def _endpoint(artifact: Optional[dict]) -> str:
    """Return an artifact dict's normalized URI, or ``""`` when it has none.

    The two ways of getting ``""`` are deliberately indistinguishable here: no
    artifact at all (a terminal -- this job had no input, or produced no output),
    and an artifact whose URI carries no identity this index can key on. Both mean
    the path ends, and both are the row's terminal marker.

    The second case is worth counting rather than only ending a path, which is why
    ``normalize_uri`` logs it: a rising count means some producer is emitting a URI
    shape with no identity rule, and every such endpoint is a hole in the graph.
    """
    if not artifact:
        return ""
    return normalize_uri(_artifact_uri(artifact))


def _job_metadata(job: dict) -> dict:
    """Collect the metadata keys present on ``job``."""
    return {key: job[key] for key in JOB_METADATA_KEYS if key in job}


def to_lineage_rows(job: dict) -> list[LineageRowDraft]:
    """Decompose a job entry into flat lineage rows.

    Args:
        job: the job entry. Recognized keys: ``job_id`` (required), ``sources``
            and ``targets`` (lists of artifact dicts, either may be empty), plus
            the optional metadata in :data:`JOB_METADATA_KEYS`. An artifact dict is
            read only for its URI (see :func:`_artifact_uri`), so any producer that
            records one is supported without a per-source translation.

    Returns:
        The rows, in a deterministic order: by source then target as given. By the
        guard one side is at most one, so a job yields max(N, M) rows, all sharing
        ``job_id``.

    Raises:
        LineageDecomposeError: if ``job_id`` is missing or empty, if the job has
            neither sources nor targets (there is no lineage to record), or if it
            has more than one of both (see the guard below).

    Ported from the prototype: a job with ``len(sources) > 1 and
    len(targets) > 1`` is rejected, so every accepted job -- and therefore every
    stored row -- satisfies ``min(#sources, #targets) <= 1``.

    granite.build's own producers already satisfy this, because ``wandb_jobstats``
    emits one event per output artifact rather than one per target run. The guard
    is here for the generic entry point: an importer with genuinely N*M records
    must split them into one job per target before calling this, keeping a
    distinct job identity per piece, rather than relax the guard.

    Do NOT make this function auto-split such a job to avoid raising. That was
    tried and reverted, because it trades a visible refusal for a silently wrong
    graph: the run node is derived from ``job_id``
    (``graph_builder.py``, :func:`_run_node_id`), so giving the pieces distinct
    ids makes ONE execution render as TWO run nodes, and nothing in a row can put
    them back together. The edges survive but the execution does not, which is a
    worse failure than the caller getting an error it can act on. A caller holding
    an N*M record must decide how to attribute it -- this function cannot decide
    for it without inventing provenance.
    """
    job_id = job.get("job_id") or ""
    if not job_id:
        raise LineageDecomposeError(
            "job entry has no job_id; rows would be unattributable and "
            f"ungroupable: {job!r}"
        )

    sources: list[dict] = list(job.get("sources") or [])
    targets: list[dict] = list(job.get("targets") or [])
    if not sources and not targets:
        raise LineageDecomposeError(
            f"job {job_id!r} has neither sources nor targets; nothing to record"
        )
    if len(sources) > 1 and len(targets) > 1:
        raise LineageDecomposeError(
            f"job {job_id!r}: too many sources {len(sources)} and targets "
            f"{len(targets)}; split it into one job per target"
        )

    metadata = _job_metadata(job)

    def draft(
        source: Optional[dict],
        target: Optional[dict],
    ) -> LineageRowDraft:
        return LineageRowDraft(
            job_id=job_id,
            input=_endpoint(source),
            output=_endpoint(target),
            input_artifact=source,
            output_artifact=target,
            metadata=dict(metadata),
        )

    # Deletion terminal: inputs consumed, nothing produced. target stays None.
    if not targets:
        return [draft(source, None) for source in sources]

    # Creation terminal: an output with no recorded input. source stays None.
    if not sources:
        return [draft(None, target) for target in targets]

    # The remaining shapes: N sources x 1 target (the prototype's 2b) and
    # 1 source x M targets (its 3). The guard above rules out N*M with both
    # sides > 1, so this comprehension emits max(N, M) rows, never a true
    # cartesian product -- but it still covers both without special-casing.
    return [draft(source, target) for source in sources for target in targets]


def group_by_job(rows: list[LineageRowDraft]) -> dict[str, dict[str, set]]:
    """Recover each job's input and output sets from decomposed rows.

    The inverse of the flattening: the rows of one job still say which inputs and
    which outputs that execution had, so the flat pairing is a projection rather
    than a loss. This holds independently of the guard -- it is what makes the
    fan-out shapes the guard *does* accept (N sources x 1 target, 1 source x M
    targets) safe to store flat.

    Args:
        rows: decomposed rows, from any number of jobs.

    Returns:
        ``{job_id: {"sources": {...}, "targets": {...}}}`` with terminal endpoints
        excluded, so a creation job reports no sources and a deletion job no
        targets.

        The exclusion is by truthiness, not ``is not None``: a terminal is the empty
        string, so a ``None`` check would report ``{""}`` as an input -- a phantom
        artifact that every creation job in the set would appear to share.
    """
    grouped: dict[str, dict[str, set]] = {}
    for row in rows:
        entry = grouped.setdefault(row.job_id, {"sources": set(), "targets": set()})
        if row.input:
            entry["sources"].add(row.input)
        if row.output:
            entry["targets"].add(row.output)
    return grouped
