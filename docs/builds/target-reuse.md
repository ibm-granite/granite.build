# Design: Hash-based Target Skip Detection

## Problem

Build pipelines often re-run the same target with identical inputs. Without a skip mechanism,
every build re-executes every target from scratch, wasting compute time and money.


## Solution

Before a target runs, compute
a SHA-256 hash of its complete definition (environment, steps, step configs, and resolved input
URIs). Store this hash in the database on every successful run. On subsequent builds, look up
whether any previous successful run shares the same hash. If one exists, the target is skipped
and its previously-written artifact URIs are fetched from the registry and forwarded directly to
downstream targets.

The hash encodes *exactly the work the target performs*. If the hash matches, the outputs are
guaranteed to be identical, so re-running is unnecessary.

---

## Architecture

### Build execution model

A build consists of a DAG of targets. `BuildRun` owns the async event loop and maintains:

- `starting_targets` — targets with no upstream dependencies
- `binding_to_target_mapping` — maps an output binding to the downstream targets waiting on it
- `input_status_update_lock` — async lock protecting concurrent binding updates

Each target is dispatched via `BuildRun.__dispatch_target()`. When a target finishes (or is
skipped), its output bindings are resolved and any downstream targets with all inputs now
satisfied are dispatched in turn.

### Skip hook

`Build` (the domain model) holds an optional callback:

```python
target_already_run_fn: Optional[
    Callable[[str], Optional[tuple[str, dict[str, list[str]]]]]
]
```

`BuildRun.__dispatch_target()` calls this with the target's definition hash. The callback
returns either `None` (no prior run found → execute normally) or a tuple of:

- the UUID of the original `StoredTargetRun`
- a dict mapping `binding_id → list[artifact_uri]` (the actual URIs written by that run)

`BuildRunner` wires up the concrete implementation of this callback when constructing `Build`.
`Build` and `BuildRun` have no knowledge of storage; they only call the hook.

---

## The Two Hashes

There are two independent hashes in the system. They are easy to confuse but serve entirely
different purposes.

### 1. Definition hash — for skip detection

**Computed by**: `BuildRun.__compute_target_def_hash()`
**Stored in**: `gb_targets.target_hash`
**Format**: SHA-256, 64 hex characters

```
hash_input = env_uri
           + "|" + step_uri_1 + "|" + step_uri_2 + ...
           + "|" + step_config_1_json + "|" + step_config_2_json + ...
           + "|" + sorted_input_uri_1 + "|" + sorted_input_uri_2 + ...
```

**Scope**: Everything that determines what the target *does* and what it *receives*. The target
name is intentionally excluded — two targets with different names but identical configurations
share a hash, so either can satisfy a skip check for the other.

Input URIs are sorted to make the hash order-independent. Step configs are included as JSON to
capture parameter changes (e.g. a changed hyperparameter means a different hash → no skip).

### 2. URI hash — for deterministic output naming

**Computed by**: `BuildRun.__resolve_output_uris()`
**Used in**: `{{ target_hash }}` template in output URI configs
**Format**: 8-character short alphanumeric (base-62)

```
hash_input = step_uri_1 + "|" + step_uri_2 + ...
           + "|" + sorted_input_uri_1 + "|" + sorted_input_uri_2 + ...
```

**Scope**: Step URIs and resolved input URIs only — no step configs. This is intentional: the
URI hash is a *naming* mechanism. It ensures that the same logical operation on the same inputs
always writes to the same URI, enabling idempotent writes without needing a database lookup.
Adding step configs would make the hash change when configs change but the output location
doesn't need to.

**These two hashes are always computed independently and are never equal.**

---

## Data Flow

```
BuildRun.__dispatch_target(target)
│
├─ __resolve_output_uris(target)
│    Substitutes {{ target_hash }} (URI hash, 8 chars) in output URI templates.
│    URIs with other templates (e.g. {{ checkpoint_id }}) are left as-is.
│
├─ __compute_target_def_hash(target)
│    Computes SHA-256 of env + steps + configs + sorted inputs.
│
└─ target_already_run_fn(def_hash)   [callback provided by BuildRunner]
     │
     ├─ None  ──►  create TargetRun(target_hash=def_hash)
     │              │
     │              └─ on each status event: EntityRunMetadata carries target_hash
     │                  BuildRunner stores target_hash in gb_targets on SUCCESS
     │
     └─ (uuid, {binding_id: [uri, ...]})  ──►  __handle_skipped_target(...)
                                                │
                                                ├─ emit STATUS_EVENT(type="Target", status=SUCCESS)
                                                │    EntityRunMetadata carries targetrun_id (new UUID)
                                                │    and skipped_for_prerun_target_id=uuid
                                                │    BuildRunner persists StoredTargetRun with
                                                │    skipped_for_prerun_target_id=uuid, target_hash=""
                                                │    then calls add_jobstats_for_build_target
                                                │
                                                └─ under input_status_update_lock:
                                                     for each (binding_id, uris):
                                                       mark binding available, append uris
                                                       if all inputs satisfied:
                                                         __dispatch_target(downstream)
```

---

## Storage

### `gb_targets` table — new columns

| Column | Type | Populated by | Empty on |
|--------|------|--------------|----------|
| `target_hash` | `TEXT` | Successful non-skipped runs | Skipped runs, pending/failed runs |
| `skipped_for_prerun_target_id` | `TEXT` | Skipped runs | All other runs |

`target_hash` has **no uniqueness constraint**. Multiple builds may store the same hash
(e.g. concurrent builds, or multiple retries of the same target). The skip lookup uses
`get_by_where` and takes the first matching successful row.

### Schema migration

Both columns are added by the auto-`ALTER TABLE` mechanism in `BaseSQLItemStorage`. On first
write after deployment, the columns are detected as missing and added automatically.

### `skipped_for_prerun_target_id` provenance

Skipped `StoredTargetRun` rows explicitly link back to the original run:

```
gb_targets row (original run):
  uuid:              "aaa-..."
  status:            SUCCESS
  target_hash:       "abc123...def456"   ← 64-char SHA-256
  skipped_for_prerun_target_id: ""

gb_targets row (skipped run):
  uuid:              "bbb-..."
  status:            SUCCESS
  target_hash:       ""                  ← intentionally empty (not the original's hash)
  skipped_for_prerun_target_id: "aaa-..."        ← points to original
```

This makes the skip relationship queryable and auditable.

---

## Downstream binding pre-resolution

When a target is skipped, its downstream targets must still receive their expected input URIs.
The mechanism differs from a normal run:

**Normal run**: The target's environment emits `ARTIFACT_PUSHED_EVENT` for each output.
`BuildRun._process_event()` handles these events, marks the binding available, and dispatches
any newly-satisfied downstream targets.

**Skipped run**: There is no execution, so no artifact events are emitted. Instead,
`__handle_skipped_target()` reads the resolved URIs returned by `target_already_run_fn` and
directly updates `inputs_status` for all downstream targets under `input_status_update_lock`.

```python
async with self.input_status_update_lock:
    for binding_id, uris in resolved_outputs.items():
        # find downstream targets waiting on this binding
        for bi in target_to_consider.inputs_status:
            if bi == binding_info:
                bi.available = True
                for uri_str in uris:
                    bi.uris.append(URI.get_uristr(URI.get_uri(uri_str)))
        # if all inputs now satisfied, dispatch immediately
        if all(b.available for b in target_to_consider.inputs_status):
            self.__dispatch_target(target_to_consider, tg)
```

**Checkpoint support**: `resolved_outputs` is `dict[str, list[str]]` (list of URIs per
binding), so targets that previously produced multiple checkpoint artifacts have all of their
URIs forwarded. Downstream targets receive all artifact URIs exactly as they would from a live
run.

**Dynamic URI support**: URIs that contain unresolved templates at dispatch time (e.g.
`{{ checkpoint_id }}`, filled in by the environment at runtime) are left as-is in the target
config — they were never substituted by `__resolve_output_uris()`. The resolved URIs come
entirely from `target_already_run_fn`'s artifact registry lookup, not from the config. This
means targets with fully dynamic output URIs are skippable as long as their prior run's
artifacts are in the registry.

---

## The callback contract

`BuildRunner` provides the callback; `Build`/`BuildRun` consume it. The callback signature:

```python
def target_already_run_fn(
    target_hash: str,
) -> Optional[tuple[str, dict[str, list[str]]]]:
    ...
```

**Input**: SHA-256 definition hash of the target to be dispatched.

**Output**:
- `None` — no prior successful run found; execute normally.
- `(uuid, resolved_outputs)` — prior run found.
  - `uuid`: the `StoredTargetRun.uuid` of the original run, stored as `skipped_for_prerun_target_id`.
  - `resolved_outputs`: `{binding_id: [uri_str, ...]}` built by looking up each artifact UUID
    from `StoredTargetRun.output_artifacts` in the `artifact_registry`.

**Implementation in `BuildRunner`**:

The callback is the private method `BuildRunner.__is_target_already_run`. It searches only
within the current build's **retry chain** — the set of build UUIDs obtained by following
`retry_of_build_id` links back to the root build:

```python
def __get_retry_chain_build_ids(self) -> list[str]:
    build_ids = [self.stored_build.uuid]
    current_id = self.stored_build.retry_of_build_id
    while current_id:
        build_ids.append(current_id)
        ancestor = self.storage.build_storage.get_by_uuid(current_id)
        if not isinstance(ancestor, StoredBuild):
            break
        current_id = ancestor.retry_of_build_id
    return build_ids

def __is_target_already_run(self, target_hash):
    chain_build_ids = self.__get_retry_chain_build_ids()
    results = self.storage.target_storage.get_by_where(
        {"target_hash": target_hash, "status": Status.SUCCESS.name, "build_id": chain_build_ids}
    )
    if not results:
        return None
    stored_target = results[0]
    resolved_outputs = {}
    for binding_id, artifact_uuids in stored_target.output_artifacts.items():
        uris = []
        for artifact_uuid in artifact_uuids:
            artifact = self.storage.artifact_registry.get_by_uuid(artifact_uuid)
            if artifact is None or artifact.status != ArtifactRegistrationStatus.SUCCESS:
                return None  # incomplete prior run — do not skip
            uris.append(artifact.uri)
        if uris:
            resolved_outputs[binding_id] = uris
    return (stored_target.uuid, resolved_outputs)
```

**When the callback is wired**: the callback is passed to `Build` only when both conditions
are true:

1. `self.stored_build.retry_of_build_id` is set — i.e. this is a retry build, not a fresh build.
2. `build_config.retries.target_reuse_enabled` is `True` (the default).

Build-resume runs and fresh (non-retry) builds always receive `None` for the callback.
Build-resume re-executes any incomplete target by definition. Fresh builds have no retry chain
to search, so the skip mechanism would never find anything useful.

---

## Hash correctness argument

For skipping to be safe, the outputs of two runs with the same definition hash must be
identical. The hash covers:

| Input | Why included |
|-------|-------------|
| `environment_uri` | Different compute environments may produce different outputs |
| `step_uri` (per step) | The container image URI identifies the exact code to run |
| `step_config` (JSON, per step) | Config parameters (LR, batch size, etc.) affect outputs |
| `sorted(input_uris)` | Different inputs produce different outputs |

**Target name is excluded** because it has no effect on execution. Two targets named differently
but configured identically produce the same outputs.

**Output URIs are excluded** because they are derived, not causal. The URI hash (for
`{{ target_hash }}` naming) is itself derived from a subset of the inputs.

**Build ID is included in the search scope** — the skip check is scoped to the retry chain,
not global. Only builds reachable by following `retry_of_build_id` links back to the root are
searched. This prevents targets from unrelated builds from accidentally satisfying a skip,
which could cause incorrect artifact reuse when inputs have changed between independent builds.

---

## Propagating `target_hash` and `skipped_for_prerun_target_id` through events

Both pieces of skip-related metadata travel from the engine layer (`BuildRun`) to the
orchestration layer (`BuildRunner`) via `EntityRunMetadata` fields on `BuildEvent`.

**`target_hash`** — for normal (non-skipped) runs:

```
TargetRun.__init__(target_hash=def_hash)
    │
    └─ get_runmetadata() → EntityRunMetadata(target_hash=def_hash)
         carried on every BuildEvent emitted by TargetRun/TargetStepRun
```

Written to `gb_targets.target_hash` on the SUCCESS update in
`__process_build_target_info_type_event`. If a target fails or is cancelled the row keeps
`target_hash = ""` and will never be found by `target_already_run_fn`.

**`skipped_for_prerun_target_id`** — for skipped runs:

```
BuildRun.__handle_skipped_target(skipped_for_prerun_target_id=uuid)
    │
    └─ STATUS_EVENT(type="Target", status=SUCCESS)
         EntityRunMetadata(targetrun_id=<new-uuid>, skipped_for_prerun_target_id=uuid)
```

`__process_build_target_info_type_event` creates the `StoredTargetRun` via the normal path,
then writes `skipped_for_prerun_target_id` from `run_info` before calling `add_jobstats_for_build_target`:

```python
if run_info.skipped_for_prerun_target_id:
    stored_target_run.skipped_for_prerun_target_id = run_info.skipped_for_prerun_target_id
    self.storage.target_storage.update(stored_target_run)
```

---

## Limitations and non-goals

- **No invalidation API**: there is no mechanism to manually invalidate a stored hash. If the
  environment image is updated but the `step_uri` stays the same (e.g. a `:latest` tag), the
  hash will not change and the target will be incorrectly skipped. Teams should use
  content-addressed URIs (digest-pinned images) to avoid this.

- **No TTL**: stored hashes persist indefinitely. A target that ran a year ago will still
  prevent re-runs today. Pruning old rows is an operational concern outside this feature.

- **Retry builds only**: the callback is not provided to fresh builds or build-resume runs.
  For fresh builds there is no retry chain to search. For build-resume, the semantics are
  already correct (re-run only incomplete targets); adding hash-skip on top would interfere
  with partial-completion state.

- **Opt-out via `target_reuse_enabled`**: setting `retries.target_reuse_enabled: false` in
  `build.yaml` disables the callback for that build's retry runs. Every target will re-execute
  from scratch even if it succeeded in a prior attempt.

- **No output-only hash**: if a target's step config changes in a way that doesn't affect the
  output (e.g. a verbosity flag), a cache miss occurs unnecessarily. This is a known trade-off
  in favour of correctness over efficiency.
