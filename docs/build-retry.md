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
   etc.) and status `PENDING`.
2. Sets `retry_count` on the new build to `original.retry_count + 1`.
3. Sets `retry_of_build_id` on the new build to the UUID of the original (first) build — this
   field always points to the root of the retry chain, not just the previous attempt.
4. Updates `retry_build_id` on the failed build to point to the new retry build.
5. Runs the new build immediately in the same `BuildRunner` session.

Retries are only triggered for the `FAILED` status. Builds that end with `CANCELLED` or
`INVALID` are never retried.

## Storage fields

| Field | Where set | Meaning |
|---|---|---|
| `retry_count` | retry build | Number of retry attempts so far (1 on first retry, 2 on second, etc.) |
| `retry_of_build_id` | retry build | UUID of the original failed build (root of the chain) |
| `retry_build_id` | original/previous build | UUID of the next retry build created for this build |

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
