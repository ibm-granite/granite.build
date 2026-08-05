# In-place build retry: reuse the same StoredBuild / build id

## Context

Today, when a build fails and is eligible for retry, `BuildRunner` creates a
**brand-new `StoredBuild` with a fresh build id** and runs it as a new attempt
(`__prepare_retry` at [buildrunner.py:313](src/gbserver/buildrunner/buildrunner.py#L313)).
The attempts are stitched together with `retry_of_build_id` / `retry_build_id`
into a "retry chain", and successful targets from earlier attempts are re-recorded
in the new build as **skipped** target runs (`status=SUCCESS` +
`skipped_for_prerun_target_id`; there is no `SKIPPED` status enum — "SKIPPED" is
only a CLI display label).

We want retries to **reuse the same `StoredBuild` and build id**. Consequences
(all confirmed with the user):

- Target runs only ever have status **FAILED** or **SUCCESS** — the skip concept
  and `skipped_for_prerun_target_id` are removed entirely.
- When a FAILED target re-runs and succeeds in the same build, a **new** SUCCESS
  `StoredTargetRun` is created that **links back to the prior FAILED run** via a
  new field `retry_of_target_id`. Both records persist in the one build.
- Artifacts re-emitted by the re-run target are **re-associated** to the
  successful run (update `created_by_target_id` on the `ArtifactRegistration`).
- The build-level retry chain (`retry_of_build_id`, `retry_build_id`,
  `get_retry_chain_members`, chain-wide cancellation) is **removed fully**;
  `retry_count` is kept to enforce `max_retries`.
- **Clean forward-only** change — no migration for existing persisted data.
- **StoredStepRun is left as-is** — new step runs are simply created under the
  re-run target; no step-level linkage field.

Outcome: a retried build keeps one stable build id; its target runs read as an
honest FAILED→SUCCESS history within a single build, and artifacts point at the
run that actually produced the successful output.

## Approach

### 1. Models & storage

- [stored_build.py](src/gbserver/storage/stored_build.py): delete
  `retry_of_build_id` and `retry_build_id` fields (keep `retry_count`); delete
  the module-level `get_retry_chain_members()` function (lines ~236-264).
- [stored_target_run.py](src/gbserver/storage/stored_target_run.py): replace
  `skipped_for_prerun_target_id` with `retry_of_target_id: str = ""`
  ("UUID of the prior FAILED StoredTargetRun in the same build that this run
  retried; empty if this run is not a retry").
- [target_run_storage.py](src/gbserver/storage/target_run_storage.py) and
  [sql/target_run_storage.py](src/gbserver/storage/sql/target_run_storage.py):
  in `_get_column_values` (~41-51) drop the `skipped_for_prerun_target_id`
  promoted column. `retry_of_target_id` does **not** need to be indexed (we
  never query by it; the prior-failed-run lookup uses the already-indexed
  `build_id`/`name`/`status`).
- [buildevent.py](src/gbserver/types/buildevent.py): remove
  `skipped_for_prerun_target_id` from `EntityRunMetadata` (field decl ~134 and
  the `from_dict` read ~149). The linkage is set by the buildrunner directly, so
  it need not travel on the event.

### 2. BuildRunner retry loop (the core change)

[buildrunner.py](src/gbserver/buildrunner/buildrunner.py):

- **`__prepare_retry()`** (~313): rewrite to reuse the same build instead of
  creating a new one. Re-read `latest`; if `_should_retry(latest)` is False
  return None; otherwise `build_storage.update_fields(latest.uuid, {retry_count:
  latest.retry_count+1, status: Status.RETRY_PENDING, failure_reason: ""})`,
  re-read, and return the same build. No new `StoredBuild`, no
  `retry_of_build_id`/`retry_build_id`, no `__track_retry_chain_build`.
- **`start_and_wait()`** (~201): remove the chain-seed loop that calls
  `get_retry_chain_members` (~227-230). Keep the `while True` loop; the retry
  branch now just swaps to the same re-read build and re-runs. The
  `build_message_logger` re-creation (~287-289) is unneeded (same build) — drop
  or leave harmless.
- **`_should_retry()`** (~1127): unchanged (FAILED + `max_retries>0` +
  `retry_count<max_retries`).
- Delete `__get_retry_chain_build_ids` (~1306), `__track_retry_chain_build`
  (~303), the `_retry_chain_build_ids` / `_retry_chain_lock` state in
  `__init__`.
- **`__is_build_cancelled()`** (~667) and **`__cancel_build_run()`** (~802):
  drop the chain iteration — check/update only `self.stored_build.uuid`.
- **`__comment_on_original_pr()`** (~358): there is no separate original build;
  post the "retrying (attempt N of M)" comment on the build's **own**
  `source_uri` PR. Update the caller guard in `__setup` (~625) from
  `if self.stored_build.retry_of_build_id` to `if self.stored_build.retry_count`.

### 3. Which targets re-run, and skipping already-succeeded ones without a record

- **`__async_run_build`** (~440): the retry re-run stays on the normal
  (non-resume) path — run all `stored_build.targets`, and let
  `target_already_run_fn` skip the ones that already succeeded. Change the
  `target_already_run_fn` wiring (~512-517) gate from
  `self.stored_build.retry_of_build_id` to `self.stored_build.retry_count > 0`
  (still AND `target_reuse_enabled`).
- **`__is_target_already_run(target_hash)`** (~1320): change the query from
  `build_id in chain_build_ids` to `build_id == self.stored_build.uuid` — search
  the **same** build for a prior SUCCESS run with the matching `target_hash`
  (artifact-still-registered verification unchanged).
- **`__handle_skipped_target`** (buildrun.py ~295): keep the downstream
  binding propagation (`__propagate_binding` loop) but **remove the skip
  STATUS_EVENT dispatch** and the `skipped_for_prerun_target_id` parameter. The
  already-existing SUCCESS `StoredTargetRun` (from the earlier attempt in the
  same build) is what the API/CLI report; no new record is written. Update the
  caller `__dispatch_target` (~396) to match the new signature. (No effect on
  `run_and_wait` completion — the skip branch already creates no `TargetRun`.)

### 4. Failed→success target linkage

- **`__create_and_store_target_run`** ([buildrunner.py:1247](src/gbserver/buildrunner/buildrunner.py#L1247)):
  before `add`, look up a prior FAILED run for the same target in this build
  (`target_storage.get_by_where({build_id, name, status=Status.FAILED.name})`);
  if found, set `stored_target.retry_of_target_id = <that run's uuid>`. This is
  the single creation point for target runs, so it covers all paths.

### 5. Artifact re-association

- **`__process_artifact_event`** (~1043-1093, not-pushed branch): the
  chain-membership check becomes same-build only
  (`existing.created_by_build_id != self.stored_build.uuid` → raise). When
  reusing an `existing` artifact whose `created_by_target_id` differs from the
  current `target_id`, update it (`artifact_registry.update` /
  `update_fields(existing.uuid, {"created_by_target_id": target_id})`) so the
  artifact points at the successful re-run target. The buildrunner already has
  the artifact object in hand, so no extra (non-indexed) query is needed.

### 6. Consumers of removed fields

- **API** [api/builds.py](src/gbserver/api/builds.py): remove the
  `get_retry_chain_members` import and the `follow_retries`/`retry_chain`/
  `BuildChainMember` assembly (~470-482, ~569) and `_find_active_chain_member`
  (~605); a finished FAILED build is no longer cancellable via a chain, but an
  in-flight retry is the **same** build in `RETRY_PENDING`/`RUNNING`, so cancel
  by build id still works via the existing branch (~612). Keep `RETRY_PENDING`
  in the `Status` enum as the transient "about to retry in place" state.
- **CLI** [command_build.py](src/gbcli/commands/command_build.py): drop the
  SKIPPED branches in `target_status_label`/`target_status_emoji` (~99-108),
  the `retry_of_build_ids`/`retried_by_build_ids` lines (~115-124), the
  per-target foreign `Build ID` line (~182-189, now always same build), and the
  skipped-target section suppression (~176-193). Optionally surface
  `retry_of_target_id` as a small "retry of failed run …" note.
- **CLI service** [service_build.py](src/gbcli/services/service_build.py):
  remove the retry-chain merge and `retry_of_build_ids`/`retried_by_build_ids`
  (~1306-1382); `target_runs = build_status.get("target_runs")`. Replace the
  `skipped_for_prerun_target_id` passthroughs (~1953, ~2004) with
  `retry_of_target_id`.
- **Lineage** [wandb_jobstats.py](src/gbserver/lineage/wandb_jobstats.py):
  remove the skipped-target→original resolution block (~399-414); a reused
  target is now a real SUCCESS run with real outputs, so lineage uses it
  directly.

### 7. Docs

- Rewrite [docs/builds/build-retry.md](docs/builds/build-retry.md): in-place
  retry, one stable build id, `retry_count` only (drop the chain field table and
  chain-cancellation section), FAILED+SUCCESS target runs linked by
  `retry_of_target_id`, no skip.
- Update [docs/builds/target-reuse.md](docs/builds/target-reuse.md): reuse now
  means "don't re-run a target that already succeeded in this build" (no skip
  record, no `skipped_for_prerun_target_id`).

## Tests

- Update `test/integration/standalone/buildrunner/bash/test_buildrunner_retry.py`,
  `..._retry_exhaustion.py`, `..._retry_cancellation.py`: assert the build id is
  **stable** across retries; a re-run failed target yields a new SUCCESS run
  whose `retry_of_target_id` points at the FAILED run; **no** skip records; cancel
  by the (single) build id still cancels an in-flight retry.
- Rework `test/unit/buildrunner/test_artifact_reuse.py`: same-build reuse +
  `created_by_target_id` re-association (drop cross-build-chain cases).
- `test/unit/buildrunner/test_should_retry.py`: keep retry_count/max_retries
  logic; drop `retry_of_build_id` assertions.
- `test/integration/standalone/api/test_build_status_retry_chain.py`: rework to a
  single build with linked FAILED/SUCCESS runs (no chain response).
- `test/unit/gbcli/services/test_build_status_targets.py`: drop the SKIPPED-label
  test; assert `retry_of_target_id` passthrough.
- `test/libgbtest/buildrunner/buildtest.py` `_verify_skipped_target_and_steps`
  (~1086): replace with a helper verifying a reused target keeps its single
  SUCCESS run (no new/skip record) and a re-run target has a new linked run.
- `test/integration/ibm/lineage/test_wandb_jobstats.py`
  `test_create_jobstats_for_target_skipped` (~294): remove/replace.
- Add a focused unit test for `__create_and_store_target_run` setting
  `retry_of_target_id`, and for `__process_artifact_event` re-association.

## Risks / edge cases

- **Workspace/setup idempotency (highest risk):** the retry re-run reuses the
  same build id, so `workspace_dir` already exists from attempt 1. `__setup`
  returns early once `source_uri` is set (PR already created), and archive
  extraction uses a fresh `mkdtemp` per run, so this should be safe — but must be
  confirmed by the bash integration retry tests actually re-running a failed
  target in place.
- **Per-attempt artifact finalize:** `_finalize_artifact_status`
  ([build_utils.py](src/gbserver/buildrunner/build_utils.py)) may flip a failed
  attempt's PENDING artifacts to FAILED before the retry; the reuse path must
  accept a non-SUCCESS `existing` artifact (it does — status is re-set to SUCCESS
  on the next push). Verify timing.
- **`target_hash` reuse within one build:** `__is_target_already_run` matches on
  `target_hash` among SUCCESS runs of the same build; a FAILED run never has a
  hash, so it won't be matched and will correctly re-run.
- **RETRY_PENDING semantics:** retained as the transient in-place "about to
  retry" status so the BuildWatcher (polls PENDING only) never double-dispatches
  and cancel-by-id keeps working.

## Verification

1. `mypy`/typecheck the changed `src/gbserver` and `src/gbcli` modules.
2. Run the buildrunner bash retry integration tests (single tests, extended/live
   venv per repo conventions):
   `test_buildrunner_retry.py`, `..._retry_exhaustion.py`,
   `..._retry_cancellation.py` — confirm stable build id, FAILED→SUCCESS linkage,
   artifact re-association, no skip records.
3. Run `test/unit/buildrunner/test_artifact_reuse.py`,
   `test/unit/buildrunner/test_should_retry.py`, and the CLI
   `test_build_status_targets.py`.
4. Drive an end-to-end retry via the run-gbserver / gbmcp standalone build flow:
   submit a build with `max_retries: 1` and a target that fails on the first
   attempt then succeeds, and confirm `gb build status` shows one build id with
   the failed and successful target runs linked and outputs attributed to the
   successful run.
