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

"""The shape of a lineage row's ``attributes`` blob.

The row has three columns -- ``job_id``, ``input``, ``output`` -- and everything
else lives in one JSON blob. That blob needs a written contract more than the
columns do, precisely because the database will not enforce it: it is ``Text``, so
nothing in here is queryable, indexable, or validated on write. Without a contract
each producer invents its own keys and every reader has to tolerate all of them.

Four top-level groups, by who owns the value:

``input`` / ``output``
    What each endpoint *is* -- ``kind`` and ``name`` -- plus ``alt_uris``, the
    other spellings the producer recorded for it (``hf:///org/repo`` beside the
    canonical web URL, ``lh://.../granite-dot-build`` beside the stripped form, an
    explicit alias such as the physical ``s3://`` path). The canonical URI is
    already the row's identity, so it is not repeated here. The alternatives are
    carried for display and for a future alias lookup; like the rest of the blob
    they are **not queryable**, so a walk never matches on them.

    Optionally ``produced_by`` -- ``{build_id, target_run_uuid, artifact_id}`` --
    the granite.build execution that produced the artifact, as the artifact records
    it. That is not this row's job when the row consumes it, and it is kept because
    the producing job is often missing from the index. See :data:`PRODUCED_BY`.

``job``
    The execution: name, status, owner, timestamps, namespace. Describes the run
    node the read path rebuilds by grouping rows on ``job_id``.

``origin``
    Where the row came from: which ``system`` produced it, and that system's own
    ``ids``.

``ids`` is deliberately an open sub-map and the only open part of the contract. A
build id and a target run uuid go there for granite.build; another source puts a
pipeline id, a DAG run id, or nothing. They are **not** columns -- see
:mod:`gbserver.storage.stored_lineage_row` -- so nothing can filter on them, which
is the point: they are for explaining a row, not for finding one.

**The large payloads are not in a ROW blob; they are in the JOB blob.** A job with N
inputs and M outputs decomposes into N*M rows, so carrying ``job_input_params`` (the
full step configs), ``execution_stats``, ``job_output_stats`` or
``source_code_details`` on a row would store N*M copies of the same step config and
the index would be dominated by data it never queries. They are therefore written
once per execution, to :mod:`gbserver.storage.stored_lineage_job`, under a fifth
group -- :data:`PAYLOAD` -- assembled by :func:`build_job_attributes`. A reader
looking for them on a row should look there instead.

``owner``, ``job_namespace`` and ``job_status`` are written to **both** a row blob and
the job table's columns. That is the one deliberate exception to "one value, one
home", and the reason is authorization: the access filter reads ``job_namespace`` off
the row it already holds, so moving it to a column-only home would cost a lookup per
row -- or, if dropped, would prune every node. See the note beside
``_JOB_KEY_FROM_FLAT``.
"""

from typing import Any, Dict, List, Optional

from gbserver.lineage.uri_normalize import normalize_uri

# Top-level groups. Named as constants because both the writer and the readers key
# on them, and a typo in either place is a silently empty node rather than an error.
INPUT = "input"
OUTPUT = "output"
JOB = "job"
ORIGIN = "origin"

# Fifth group, on a JOB blob only -- never on a row. One bucket rather than four
# loose top-level keys, so "the large payloads" is a single thing to reason about
# (and to exclude, should one of them ever need excluding again).
PAYLOAD = "payload"

# Endpoint detail keys, inside INPUT / OUTPUT.
KIND = "kind"
NAME = "name"
ALT_URIS = "alt_uris"
# Which granite.build execution produced the artifact, as the artifact itself
# records it -- not the job of this row, which may only have consumed it. Kept on
# the endpoint because the producing job is often absent from the index (a quarter
# of the granite.build inputs in the Lakehouse dump), and then this is the only
# place that says where the artifact came from. A map of the keys below, blanks
# omitted.
PRODUCED_BY = "produced_by"
PRODUCED_BY_BUILD_ID = "build_id"
PRODUCED_BY_TARGET_RUN_UUID = "target_run_uuid"
PRODUCED_BY_ARTIFACT_ID = "artifact_id"

# Where an artifact dict records its producer: the facet keys granite.build writes
# (``wandb_jobstats``), which Lakehouse keeps verbatim in an endpoint's ``extra``.
_PRODUCED_BY_FROM_FACET = {
    "gb-build-id": PRODUCED_BY_BUILD_ID,
    "gb-target-id": PRODUCED_BY_TARGET_RUN_UUID,
    "gb-artifact-id": PRODUCED_BY_ARTIFACT_ID,
}

# Where an artifact dict may carry a URI. The first three are the spellings the
# producers already write (see ``decompose._artifact_uri``); ``uri_aliases`` is the
# explicit list a producer fills with forms that cannot be derived from the URI
# string, such as the physical storage path behind an ``lh://`` artifact.
_URI_KEYS = ("uri",)
_FACET_URI_KEYS = ("artifact_uri", "gb-artifact-uri")
URI_ALIASES = "uri_aliases"

# Job keys. These mirror what the API handler needs to build an ArtifactRunEntry;
# the names drop the redundant ``job_`` prefix they had when the blob was flat.
JOB_NAME = "name"
JOB_TYPE = "type"
JOB_STATUS = "status"
JOB_OWNER = "owner"
JOB_NAMESPACE = "namespace"
JOB_STARTED_AT = "started_at"
JOB_COMPLETED_AT = "completed_at"
JOB_CATEGORY = "category"

# Origin keys.
ORIGIN_SYSTEM = "system"
ORIGIN_IDS = "ids"

# Payload keys, inside PAYLOAD on a job blob. Spelled as the producers spell them
# (``wandb_jobstats``, ``wandb_service``) and as the wire models read them, so the
# value passes through without a rename nobody would be able to follow.
PAYLOAD_INPUT_PARAMS = "job_input_params"
PAYLOAD_EXECUTION_STATS = "execution_stats"
PAYLOAD_OUTPUT_STATS = "job_output_stats"
PAYLOAD_SOURCE_CODE = "source_code_details"

# The flat keys that make up the PAYLOAD group.
_PAYLOAD_KEY_FROM_FLAT = {
    "job_input_params": PAYLOAD_INPUT_PARAMS,
    "execution_stats": PAYLOAD_EXECUTION_STATS,
    "job_output_stats": PAYLOAD_OUTPUT_STATS,
    "source_code_details": PAYLOAD_SOURCE_CODE,
}

# How a flat job dict (the shape every producer emits today, via
# ``decompose.JOB_METADATA_KEYS``) maps onto the ``job`` group. Kept as data rather
# than as a chain of ``.get()`` calls so the translation is inspectable and the
# dropped keys are visibly dropped.
_JOB_KEY_FROM_FLAT = {
    "job_name": JOB_NAME,
    "job_type": JOB_TYPE,
    "job_status": JOB_STATUS,
    "owner": JOB_OWNER,
    "job_namespace": JOB_NAMESPACE,
    "job_started_at": JOB_STARTED_AT,
    "job_completed_at": JOB_COMPLETED_AT,
    "category": JOB_CATEGORY,
}

# Three of those keys -- ``job_status``, ``owner`` and ``job_namespace`` -- are ALSO
# columns on the job table (:mod:`gbserver.storage.stored_lineage_job`). They are
# deliberately still written into the row blob, which looks like the
# column-vs-blob duplication this contract otherwise avoids, and is not:
#
# ``job_namespace`` on a ROW is load-bearing for authorization. ``api.lineage``
# splits it on the first "/" to recover the space name and prunes runs the caller
# cannot see, reading it from the row's blob via ``graph_builder._run_node`` -- and a
# run with neither namespace nor owner fails closed. Dropping it from the row blob
# would therefore prune every node and make a populated index look empty.
#
# The job table's copies are for *filtering* (indexed columns); the row blob's are
# for authorizing a row the reader already holds, without a second lookup per row.
# Two homes, two different jobs. If they are ever made to disagree, the row blob is
# the one the access filter trusts.

# Flat keys that are deliberately NOT carried, and why. Listed so a reader asking
# "where did this go?" finds an answer here instead of assuming an oversight.
_DROPPED_FLAT_KEYS = {
    # Large, and identical across every row of one job. NOT lost: relocated to the
    # job table's PAYLOAD group, written once per execution by
    # :func:`build_job_attributes`. See the module docstring.
    "job_input_params",
    "execution_stats",
    "job_output_stats",
    "source_code_details",
    # Duplicates: release_id IS the build id (wandb_jobstats sets it from
    # targetrun.build_id), and job_id is already a column. Two fields holding one
    # value is the pattern that diverges silently.
    "release_id",
    "job_id",
}


def build_attributes(
    job_metadata: Optional[Dict[str, Any]] = None,
    input_artifact: Optional[Dict[str, Any]] = None,
    output_artifact: Optional[Dict[str, Any]] = None,
    source_system: str = "",
    ids: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble a row's ``attributes`` blob.

    Args:
        job_metadata: the flat job dict a producer emitted; translated onto the
            ``job`` group, dropping the keys in :data:`_DROPPED_FLAT_KEYS`.
        input_artifact: the input artifact dict, read only for kind and name.
        output_artifact: the output artifact dict, likewise.
        source_system: which system produced this row.
        ids: the originating system's own identifiers.

    Returns:
        The blob. Empty groups are omitted rather than written as ``{}``, so a
        reader can tell "not recorded" from "recorded as empty" -- and a row from a
        source that knows nothing about jobs does not carry an empty ``job`` map.
    """
    attributes: Dict[str, Any] = {}

    input_detail = _endpoint_detail(input_artifact)
    if input_detail:
        attributes[INPUT] = input_detail
    output_detail = _endpoint_detail(output_artifact)
    if output_detail:
        attributes[OUTPUT] = output_detail

    job = _job_detail(job_metadata or {})
    if job:
        attributes[JOB] = job

    origin: Dict[str, Any] = {ORIGIN_SYSTEM: source_system}
    carried_ids = {key: value for key, value in (ids or {}).items() if value}
    if carried_ids:
        origin[ORIGIN_IDS] = carried_ids
    attributes[ORIGIN] = origin

    return attributes


def _endpoint_detail(artifact: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What an endpoint *is*, from the artifact dict the producer supplied.

    Kind, name, and the alternative spellings of its URI. The canonical URI is the
    row's identity and is not repeated, and anything else an artifact dict holds
    belongs to the system that produced it rather than to the graph.
    """
    if not artifact:
        return {}
    detail: Dict[str, Any] = {}
    kind = artifact.get("artifact_type") or artifact.get("type") or ""
    if kind:
        detail[KIND] = str(kind)
    name = artifact.get("name") or ""
    if name:
        detail[NAME] = str(name)
    alternatives = _alt_uris(artifact)
    if alternatives:
        detail[ALT_URIS] = alternatives
    produced_by = _produced_by(artifact)
    if produced_by:
        detail[PRODUCED_BY] = produced_by
    return detail


def _produced_by(artifact: Dict[str, Any]) -> Dict[str, str]:
    """The artifact's producing build, target run and artifact id, blanks omitted.

    Read from the ``gb-*`` facets, where granite.build writes them. An explicit
    :data:`PRODUCED_BY` map on the artifact dict wins key by key.
    """
    facets = artifact.get("facets") or {}
    produced_by = {
        key: str(facets[facet])
        for facet, key in _PRODUCED_BY_FROM_FACET.items()
        if facets.get(facet)
    }
    explicit = artifact.get(PRODUCED_BY)
    if isinstance(explicit, dict):
        produced_by.update({k: str(v) for k, v in explicit.items() if v})
    return produced_by


def _alt_uris(artifact: Dict[str, Any]) -> List[str]:
    """Every recorded spelling of the artifact's URI other than the canonical one.

    Kept verbatim, not normalized: the point is to remember the forms the identity
    rule folds away, and normalizing them would fold them back. Sorted and deduped
    so the same artifact always yields the same blob, whatever order the producer
    wrote its keys in.
    """
    facets = artifact.get("facets") or {}
    recorded = [artifact.get(key) for key in _URI_KEYS]
    recorded += [facets.get(key) for key in _FACET_URI_KEYS]
    # The row's identity is the first URI key present, normalized -- the same rule
    # decompose._artifact_uri applies -- so this drops exactly that URI.
    identity = next((str(value).strip() for value in recorded if value), "")
    canonical = normalize_uri(identity) if identity else ""

    for aliases in (artifact.get(URI_ALIASES), facets.get(URI_ALIASES)):
        if isinstance(aliases, (list, tuple)):
            recorded.extend(aliases)
    spellings = {str(value).strip() for value in recorded if value}
    return sorted(set(spellings) - {"", canonical})


def _job_detail(job_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a flat job dict onto the ``job`` group."""
    job: Dict[str, Any] = {}
    for flat_key, key in _JOB_KEY_FROM_FLAT.items():
        value = job_metadata.get(flat_key)
        if value:
            job[key] = value
    return job


def build_job_attributes(
    job_metadata: Optional[Dict[str, Any]] = None,
    source_system: str = "",
    ids: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble a JOB's ``attributes`` blob, for
    :mod:`gbserver.storage.stored_lineage_job`.

    Differs from :func:`build_attributes` in two ways, both following from "one row
    per execution" rather than one row per endpoint pair:

    - It carries the :data:`PAYLOAD` group -- the four large payloads -- because at
      one row per job they are stored once rather than N*M times.
    - It has no :data:`INPUT` / :data:`OUTPUT` groups. Those describe one endpoint
      pair, which is a property of a row; a job has many.

    The ``job`` group here carries the same keys a row's group does, including the
    three that are also columns on the job table. A reader of the group therefore does
    not have to know which fields happen to be promoted.

    Args:
        job_metadata: the flat job dict a producer emitted.
        source_system: which system produced this job.
        ids: the originating system's own identifiers.

    Returns:
        The blob. Empty groups are omitted rather than written as ``{}``, so a reader
        can tell "not recorded" from "recorded as empty".
    """
    attributes: Dict[str, Any] = {}
    metadata = job_metadata or {}

    job = _job_detail(metadata)
    if job:
        attributes[JOB] = job

    payload: Dict[str, Any] = {}
    for flat_key, key in _PAYLOAD_KEY_FROM_FLAT.items():
        value = metadata.get(flat_key)
        if value:
            payload[key] = value
    if payload:
        attributes[PAYLOAD] = payload

    origin: Dict[str, Any] = {ORIGIN_SYSTEM: source_system}
    carried_ids = {key: value for key, value in (ids or {}).items() if value}
    if carried_ids:
        origin[ORIGIN_IDS] = carried_ids
    attributes[ORIGIN] = origin

    return attributes


# -- Readers. Every one tolerates a missing group, because a row may come from a
# source that had nothing to say in it, and because rows written by an older
# importer outlive any given version of this contract.


def endpoint_kind(attributes: Optional[Dict[str, Any]], side: str) -> str:
    """The artifact type recorded for one endpoint, or ``""``.

    Args:
        attributes: the row's blob.
        side: :data:`INPUT` or :data:`OUTPUT`.
    """
    return str(((attributes or {}).get(side) or {}).get(KIND, "") or "")


def endpoint_name(attributes: Optional[Dict[str, Any]], side: str) -> str:
    """The display name recorded for one endpoint, or ``""``."""
    return str(((attributes or {}).get(side) or {}).get(NAME, "") or "")


def endpoint_produced_by(
    attributes: Optional[Dict[str, Any]], side: str
) -> Dict[str, str]:
    """The producing build / target run / artifact recorded for one endpoint, or ``{}``."""
    value = ((attributes or {}).get(side) or {}).get(PRODUCED_BY) or {}
    return (
        {str(k): str(v) for k, v in value.items() if v}
        if isinstance(value, dict)
        else {}
    )


def endpoint_alt_uris(attributes: Optional[Dict[str, Any]], side: str) -> List[str]:
    """The alternative URI spellings recorded for one endpoint, or ``[]``."""
    values = ((attributes or {}).get(side) or {}).get(ALT_URIS) or []
    return [str(value) for value in values if value]


def job_detail(attributes: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``job`` group, or an empty map."""
    return (attributes or {}).get(JOB) or {}


def payload_detail(attributes: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``payload`` group of a JOB blob, or an empty map.

    Always empty for a row blob, which never carries the group. Note that
    ``job_input_params`` in here holds full step configs: a read path must send it out
    through ``api.lineage.get_redacted_job_input_params`` rather than returning it
    raw.
    """
    return (attributes or {}).get(PAYLOAD) or {}


def origin_system(attributes: Optional[Dict[str, Any]]) -> str:
    """Which system produced the row, or ``""``."""
    return str(((attributes or {}).get(ORIGIN) or {}).get(ORIGIN_SYSTEM, "") or "")


def origin_id(attributes: Optional[Dict[str, Any]], key: str) -> str:
    """One of the originating system's identifiers, or ``""``.

    Not queryable -- this reads the blob. A caller filtering many rows on an id is
    doing a scan, and should ask whether the question belongs to this index at all.
    """
    origin = (attributes or {}).get(ORIGIN) or {}
    return str((origin.get(ORIGIN_IDS) or {}).get(key, "") or "")
