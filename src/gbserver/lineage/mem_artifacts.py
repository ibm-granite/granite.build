"""Keep ``mem://`` artifacts out of recorded lineage.

A ``mem://`` URI is not persisted data: it is an opaque key into a build's
in-memory Memstore, used to hand a value (e.g. a service URL) from one target to
the next. It does not exist outside the build, so there is nothing to version,
download or draw, and recording it as lineage is noise. That includes the edge it
used to imply between the producing and consuming targets: it is dropped too, on
purpose (#561).

A ``mem://`` artifact counts as absent. Every gate that decides whether a target
has lineage, and how many runs it should have, must read :func:`lineage_view`
rather than the target's raw dicts, or they disagree:

- ``lineage_reconciler.select_recordable_targets`` (artifact-less skip),
- ``lineage_reconciler.expected_run_count`` (run count),
- ``wandb_jobstats._build_events_for_target`` (what is emitted),
- ``indexer.TargetsLineageIndexer._index`` (artifact-less skip).

If the selector keeps a target the emitter then declines to emit for, the target
is reported unrecorded on every scan and pins the checkpoint (see the note in
``_build_events_for_target``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.singleton_storage import SingletonAdminStorage
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import MEM_URI_SCHEME


def is_mem_uri(uri: Optional[str]) -> bool:
    """Whether ``uri`` is a ``mem://`` URI.

    Compares the parsed, lowercased scheme, as URI dispatch does, so ``MEM://k``
    is caught too.
    """
    if not isinstance(uri, str) or not uri:
        return False
    try:
        scheme = urlparse(uri).scheme
    except ValueError:
        return False
    return scheme.lower() == MEM_URI_SCHEME


def is_mem_resource(resource: dict) -> bool:
    """Whether a lineage resource dict points at a ``mem://`` URI.

    Reads the same keys ``decompose._artifact_uri`` does: the top-level ``uri``
    and the ``artifact_uri`` / ``gb-artifact-uri`` facets, since producers do not
    all fill the same one.
    """
    if is_mem_uri(resource.get("uri")):
        return True
    facets = resource.get("facets") or {}
    if not isinstance(facets, dict):
        return False
    return any(is_mem_uri(facets.get(k)) for k in ("artifact_uri", "gb-artifact-uri"))


@dataclass(frozen=True)
class LineageOutput:
    """One kept output artifact.

    ``index`` is its position in the *original* output list, or -1 when that list
    held a single artifact, so the ``name[i]`` reference stays the one the build
    uses: dropping a mem:// sibling must not renumber or un-index it.
    """

    uuid: str
    index: int
    artifact: Optional[ArtifactRegistration]


@dataclass(frozen=True)
class LineageView:
    """A target's artifacts as lineage sees them: mem:// removed, each resolved once.

    ``artifact`` is ``None`` for an uuid the registry does not resolve; such an
    uuid is kept, so it still counts as it did before mem:// filtering existed.
    """

    inputs: dict[str, tuple[str, Optional[ArtifactRegistration]]]
    outputs: dict[str, list[LineageOutput]]

    @property
    def has_outputs(self) -> bool:
        return any(self.outputs.values())

    @property
    def is_artifact_less(self) -> bool:
        return not self.inputs and not self.has_outputs

    @property
    def output_count(self) -> int:
        return sum(len(kept) for kept in self.outputs.values())


def _resolve(
    storage: SingletonAdminStorage, uuid: str
) -> tuple[bool, Optional[ArtifactRegistration]]:
    """Return (keep, artifact) for ``uuid``."""
    artifact = storage.artifact_registry.get_by_uuid(uuid)
    if not isinstance(artifact, ArtifactRegistration):
        return True, None
    return not is_mem_uri(artifact.uri), artifact


def lineage_view(
    storage: SingletonAdminStorage, target: StoredTargetRun
) -> LineageView:
    """Resolve ``target``'s artifacts once and drop the mem:// ones.

    An output name whose every uuid was mem:// is dropped, so a target whose only
    outputs were mem:// reads as having no outputs: it is recorded as a no-output
    target when it has real inputs, and skipped as artifact-less when it has none.
    """
    inputs: dict[str, tuple[str, Optional[ArtifactRegistration]]] = {}
    for name, uuid in target.input_artifacts.items():
        keep, artifact = _resolve(storage, uuid)
        if keep:
            inputs[name] = (uuid, artifact)

    outputs: dict[str, list[LineageOutput]] = {}
    for name, uuids in target.output_artifacts.items():
        indexed = len(uuids) > 1
        kept = []
        for position, uuid in enumerate(uuids):
            keep, artifact = _resolve(storage, uuid)
            if keep:
                kept.append(LineageOutput(uuid, position if indexed else -1, artifact))
        # Keep a name that was already empty as-is; drop one only when this
        # filter emptied it.
        if kept or not uuids:
            outputs[name] = kept
    return LineageView(inputs=inputs, outputs=outputs)
