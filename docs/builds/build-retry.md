# Build-level Retry

When a build fails, gbserver can automatically re-run it as a retry attempt. This is controlled
by the `max_retries` field in `build.yaml` and is distinct from the step-level retry described in
[step-retry-configuration.md](step-retry-configuration.md), which re-launches a single step within
the same build run.

Retries happen **in place**: a retry reuses the *same* `StoredBuild` and build id rather than
creating a new build. The one build accumulates its target-run history across attempts.

## Configuration

Configure retries using the `retries` section of your `build.yaml`:

```yaml
llm.build:
  name: my-build
  retries:
    max_retries: 2              # retry up to 2 times on failure (default: 0)
    target_reuse_enabled: true  # don't re-run targets that already succeeded (default: true)
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

1. Bumps `retry_count` on the same build to `retry_count + 1`.
2. Sets the build's status back to `RUNNING` and clears its `failure_reason`.
3. Re-runs the same build immediately in the same `BuildRunner` session, keeping the same
   build id, `build_archive`, targets, tags, and PR.

The build is re-run in place as `RUNNING` rather than moved back to `PENDING` on purpose: the
retry loop re-runs it in the same thread, so it is genuinely in flight, and the `BuildWatcher`
only dispatches `SUBMITTED`/`PENDING` builds — a `RUNNING` status therefore keeps it from
launching a *second* runner for a retry the in-process loop is already running.

Retries are only triggered for the `FAILED` status. Builds that end with `CANCELLED` or
`INVALID` are never retried.

## Target runs across attempts

Because a retry reuses the same build, all of a target's runs live under the one build id and
read as an honest history:

- A target that **failed** on an attempt has a `StoredTargetRun` with status `FAILED`.
- When that target re-runs and **succeeds** on a later attempt, a **new** `StoredTargetRun`
  with status `SUCCESS` is created. Its `retry_of_target_id` points back to the prior FAILED
  run in the same build.
- Artifacts re-emitted by the successful re-run are **re-associated** to it: the
  `ArtifactRegistration.created_by_target_id` is updated to the successful run.

Target runs therefore only ever have status `FAILED` or `SUCCESS` — there is no "skipped"
status. A target that already succeeded in an earlier attempt simply is not re-run (see
[Target reuse](#target-reuse) below); its single existing SUCCESS run is what the API and CLI
report.

## Cancellation

Because a retrying build keeps its one stable build id, cancelling it is just cancelling that
build. There is no chain to walk.

How a cancellation request is handled (`POST /builds/{id}/cancel`):

- If the build is **in flight** (`RUNNING`, i.e. an attempt is running or is being re-run in
  place for a retry), it is set to `CANCEL_REQUESTED`.
- If the build has **not started yet** (`SUBMITTED` or `PENDING`), it is set directly to
  `CANCELLED`.
- If the build is **already finished** (`SUCCESS`, `FAILED` with retries exhausted, or
  `CANCELLED`), the request is rejected with `412`.

The `BuildRunner` checks for a cancellation request after each attempt (and while a step is
running, where the environment supports interrupting it). As soon as the build is
`CANCEL_REQUESTED`/`CANCELLED`, it stops the active workload, marks the build `CANCELLED`, and
does not create any further retries.

You cancel a retrying build using the same build id you submitted — it never changes.

## Storage fields

| Field | Where set | Meaning |
|---|---|---|
| `retry_count` | build | Number of retry attempts so far (1 on first retry, 2 on second, etc.) |
| `retry_of_target_id` | re-run target's SUCCESS run | UUID of the prior FAILED `StoredTargetRun` in the same build that this run retried; empty if not a retry |

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

If the build fails, gbserver retries it once. If that retry also fails, the build is marked
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

## Target reuse

When a build is retried, gbserver checks whether each target has already succeeded **in this
same build** on an earlier attempt. If a matching successful run is found, the target is not
re-executed, saving time and compute.

A target is considered a match when its `target_hash` — a SHA-256 digest of the target
definition (environment, steps, and input artifacts) — is identical to a previously successful
run within the same build.

When a target is reused this way, no new target run is written: its existing SUCCESS
`StoredTargetRun` remains, no steps are dispatched, and downstream targets resolve their inputs
from that run's output artifacts. Only the targets that did not yet succeed are re-run, making
retries as cheap as possible.

See [target-reuse.md](target-reuse.md) for the full architecture, hash correctness argument,
and storage details.

## Relationship to step-level retry

These are two independent mechanisms:

| | Step-level retry | Build-level retry |
|---|---|---|
| Configured in | `build.yaml` step / `step.yaml` / env var | `build.yaml` `max_retries` |
| Scope | Re-launches a single failing step pod | Re-runs the build in place |
| Triggered by | Pod eviction, node failure, transient errors | Build status `FAILED` after all step retries exhausted |
| New build record created | No | No (same build id is reused) |

A build-level retry only fires after the build has fully failed — i.e. after all step-level
retries for that run have been exhausted.

## Relationship to build restart

Build-level retry runs automatically, in the same runner, only for a `FAILED` build, within the
`max_retries` budget. To re-run a **finished build that did not fully succeed** (`FAILED`,
`INVALID`, or `CANCELLED`, for any reason) in a **fresh** runner — skipping targets that already
succeeded — use [build restart](build-restart.md) (`gb build restart <BUILD_ID>`), which reuses
the same target-reuse machinery but on a fresh `max_retries` budget.
