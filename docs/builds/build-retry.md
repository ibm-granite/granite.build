# Build-level Retry

When a build fails, gbserver can automatically create a new build and run it as a retry
attempt. This is controlled by the `max_retries` field in `build.yaml` and is distinct from
the step-level retry described in [step-retry-configuration.md](step-retry-configuration.md), which re-launches a single step within the same build run.

## Configuration

Configure retries using the `retries` section of your `build.yaml`:

```yaml
llm.build:
  name: my-build
  retries:
    max_retries: 2              # retry up to 2 times on failure (default: 0)
    target_reuse_enabled: true  # reuse successful targets from earlier attempts (default: true)
  targets:
    my-target:
      environment_uri: space://environments/cpu
      steps:
        - step_uri: space://steps/my-step
```

`max_retries` defaults to `0`, meaning no automatic retries are attempted.

`target_reuse_enabled` defaults to `true`. Set it to `false` to force all targets to re-run
from scratch on every retry, even if they succeeded in an earlier attempt.

## Behaviour

When a build finishes with status `FAILED` and `retry_count < retries.max_retries`, gbserver:

1. Creates a new `StoredBuild` with the same configuration (`build_archive`, targets, tags,
   etc.) and status `RETRY_PENDING`.
2. Sets `retry_count` on the new build to `original.retry_count + 1`.
3. Sets `retry_of_build_id` on the new build to the UUID of the original (first) build — this
   field always points to the root of the retry chain, not just the previous attempt.
4. Updates `retry_build_id` on the failed build to point to the new retry build.
5. Runs the new build immediately in the same `BuildRunner` session.

The retry build is created with status `RETRY_PENDING` rather than `PENDING` on purpose: the
`BuildWatcher` only dispatches `PENDING` builds, so a distinct status keeps it from launching
a *second* runner for a retry that the in-process loop is already running. The `RETRY_PENDING` build
transitions to `RUNNING` as it executes, just like any other in-flight build.

Retries are only triggered for the `FAILED` status. Builds that end with `CANCELLED` or
`INVALID` are never retried.

## Cancellation

Cancelling a build with `max_retries > 0` cancels the **entire retry chain**, not just one
attempt. Because the whole chain is run by a single `BuildRunner`, cancelling any member of
the chain stops the work that is actually running and marks every build in the chain
`CANCELLED`.

How a cancellation request is handled (`POST /builds/{id}/cancel`):

- If the targeted build is **still in flight** (`PENDING`, `RUNNING`, or `RETRY_PENDING`), it is set to
  `CANCEL_REQUESTED` (or directly `CANCELLED` if it had not started yet).
- If the targeted build is **already finished** (for example the original, which is now
  `FAILED`) **but its retry chain still has an active member**, the request is accepted — the
  failed build is itself set to `CANCEL_REQUESTED`. This is a durable signal on a build that is
  not being re-run, so it cannot be clobbered by a concurrent status update. (Cancelling a
  finished build whose chain has **no** active member is still rejected with `412`.)

The `BuildRunner` checks the whole retry chain for a cancellation request after each attempt
(and while a step is running, where the environment supports interrupting it). As soon as any
member is `CANCEL_REQUESTED`/`CANCELLED`, it stops the active workload, **marks every build in
the chain `CANCELLED`**, and does not create any further retries. Earlier attempts that had
already failed are relabelled `CANCELLED` so the whole chain reflects the cancellation.

This means you can cancel a retrying build using the original build id you submitted, even
after that first attempt has failed and the chain has moved on to a later retry.

## Storage fields

| Field | Where set | Meaning |
|---|---|---|
| `retry_count` | retry build | Number of retry attempts so far (1 on first retry, 2 on second, etc.) |
| `retry_of_build_id` | retry build | UUID of the original failed build (root of the chain) |
| `retry_build_id` | original/previous build | UUID of the next retry build created for this build |

## Unified job status

Each attempt in a retry chain keeps its own status forever. The original build stays
`FAILED` even after a later attempt finishes every remaining target, so no single build
record answers the question a user actually asks — *did the build specification complete?*

The **job view** answers that. It treats the whole retry chain as one logical job and derives
an aggregate status on read from the same records the status endpoint already returns. Nothing
is retro-relabelled: there is no schema change, and every `StoredBuild` retains its per-attempt
status. The chain's root UUID is the job's stable identity — no new table or identifier.

### Requesting the job view

`GET /builds/{build_id}/status?follow_retries=true` adds a `job` object to the response
(alongside the `retry_chain` list). Without `follow_retries` the field is absent and the
response is unchanged. The `job` is a `JobSummary`:

| Field | Meaning |
|---|---|
| `job_id` | UUID of the chain root — the job's stable identity |
| `status` | The aggregate job status (see precedence below) |
| `attempts` | Number of builds in the chain (1-based) |
| `build_ids` | UUIDs of every chain member, root first |
| `targets` | Per spec-target outcome (`name`, `status`, `build_id`, `target_run_id`, `attempt`) |
| `counts` | Roll-up of the spec targets across the chain |

`counts` partitions the spec targets so that `total == succeeded + failed + running + not_run`:

| Count | Meaning |
|---|---|
| `total` | Number of spec targets |
| `succeeded` | Targets that succeeded on some attempt |
| `failed` | Targets that finished without succeeding |
| `running` | Targets whose latest run has not finished |
| `not_run` | Targets that were never dispatched (no run at all) |

### Status precedence

The job status is decided by the first matching rule, top to bottom:

| # | Condition | Job status |
|---|---|---|
| 1 | Any member is `CANCEL_REQUESTED` | `CANCEL_REQUESTED` |
| 2 | Any member is not finished | `RUNNING` |
| 3 | `counts.total == 0` (nothing to aggregate) | The latest member's own status |
| 4 | Every spec target succeeded (subject to the SUCCESS guard below) | `SUCCESS` |
| 5 | Any member is `CANCELLED` | `CANCELLED` |
| 6 | Any member is `INVALID` and nothing succeeded | `INVALID` |
| 7 | Otherwise | `FAILED` |

There is deliberately **no `PARTIAL` status**. A job that completed some but not all of its
targets is `FAILED` with `counts.succeeded > 0` — read the counts to distinguish a total
failure from a near miss. Adding a job-only status member to the shared status vocabulary would
leak the concept into per-build status, so it is expressed through the counts instead.

### The SUCCESS guard and the "never ran" limitation

Rule 4 (`SUCCESS`) carries a guard tied to whether the target list is **authoritative**. A
build submitted with an explicit `targets` list records exactly which targets were requested,
so a target that was never dispatched (for example, because an upstream dependency failed) is
known and counted as `not_run`. When the list is authoritative, "every spec target succeeded"
is trustworthy on its own.

When the build was submitted **without** an explicit `targets` list (the default), the job has
no record of targets that never ran — the only spec targets it can see are the ones that
actually produced a run. The `not_run` denominator is therefore incomplete: "every counted
target succeeded" would be trivially true for a build that died before dispatching the rest. To
avoid reporting `SUCCESS` for such a build, the guard additionally requires the **newest
attempt's own build status** to be `SUCCESS`. If it is not, the verdict falls through to the
member statuses and lands on `FAILED`.

This "never ran" limitation is intentional: counting never-dispatched targets from an implicit
target list would require parsing the build config on the read path, which the job view avoids.

### CLI

`gbcli build status` follows the retry chain by default. For a build that was actually retried
(more than one attempt), the headline **Status** now reports the *job* status. The queried
build's own status moves to a **This attempt** line, and a **Job result** line summarises the
roll-up:

```
- **Status**: SUCCESS
- **This attempt**: FAILED (attempt 1 of 2)
- **Job result**: 2 of 2 targets succeeded
```

The **Job result** line appends any non-zero `failed`, `running`, and `never ran` counts (the
CLI renders `not_run` as "never ran"). A build with a single attempt is unchanged: the headline
is its own status and the extra lines are omitted.

`gbcli build status --format json` includes a `job` object in the output (the same
`JobSummary`), or `null` when the retry chain is not followed.

## Examples

### Single retry on failure

```yaml
llm.build:
  name: fine-tune
  retries:
    max_retries: 1
  targets:
    train:
      environment_uri: space://environments/gpu
      steps:
        - step_uri: space://steps/my-training-step
```

If the build fails, gbserver creates one retry. If that retry also fails, the build is marked
`FAILED` with no further attempts (`retry_count == retries.max_retries`).

### No retry (default)

```yaml
llm.build:
  name: fine-tune
  targets:
    train:
      environment_uri: space://environments/gpu
      steps:
        - step_uri: space://steps/my-training-step
```

`max_retries` defaults to `0`. A failure ends the build immediately with no retry.

### Job status across a retry (worked example)

A build has two targets, `targetA` and `targetB`, and `max_retries: 1`:

1. The original build runs. `targetA` succeeds; `targetB` fails. The build ends `FAILED`.
2. gbserver creates one retry. `targetA` is reused (skipped) from the original attempt;
   `targetB` re-runs and succeeds. The retry build ends `SUCCESS`.

The two build records keep their own statuses — the original stays `FAILED`, the retry is
`SUCCESS`. Asking the retry chain for its job view returns `status = SUCCESS` with
`counts = {total: 2, succeeded: 2, failed: 0, running: 0, not_run: 0}`: both spec targets
succeeded somewhere in the chain, so the job specification completed. `gbcli build status`
reports **Status: SUCCESS**, **This attempt: FAILED** (when queried on the original), and
**Job result: 2 of 2 targets succeeded**.

## Target reuse across the retry chain

When a retry build runs, gbserver checks whether each target has already succeeded in any
earlier build in the same retry chain. If a matching successful run is found, the target is
**skipped** rather than re-executed, saving time and compute.

A target is considered a match when its `target_hash` — a SHA-256 digest of the target
definition (environment, steps, and input artifacts) — is identical to a previously successful
run within the retry chain.

When a target is skipped this way:

- Its `StoredTargetRun.status` is set to `SUCCESS`.
- Its `skipped_for_prerun_target_id` is set to the UUID of the original `StoredTargetRun`
  whose hash matched.
- No steps are dispatched and no new output artifacts are created for this build; the retry
  build resolves inputs from the original run's output artifacts.

This means a retry build only re-runs the targets that did not succeed in the original build,
making retries as cheap as possible.

See [target-reuse.md](target-reuse.md) for the full architecture, hash correctness argument,
and storage details.

## Relationship to step-level retry

These are two independent mechanisms:

| | Step-level retry | Build-level retry |
|---|---|---|
| Configured in | `build.yaml` step / `step.yaml` / env var | `build.yaml` `max_retries` |
| Scope | Re-launches a single failing step pod | Creates and runs a new build |
| Triggered by | Pod eviction, node failure, transient errors | Build status `FAILED` after all step retries exhausted |
| New build record created | No | Yes |

A build-level retry only fires after the build has fully failed — i.e. after all step-level
retries for that run have been exhausted.
