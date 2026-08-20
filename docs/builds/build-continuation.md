# Build Continuation

Build continuation re-runs a previously-executed build in a **fresh** build runner,
**skipping targets that already succeeded** and re-running the rest. Use it to pick a build
back up from where it left off after a failure or interruption — without re-running work
that already completed.

```shell
gb build continue <BUILD_ID>
```

`<BUILD_ID>` may be a build id or a build URL. No local build folder is required — the build
definition, space, and targets are taken from the previous build.

The name is inspired by `curl -C` (resuming a partial download). It is distinct from
[build-level retry](build-retry.md) and from step-level retry.

## When to use it

A build can fail for many reasons — a build-definition error, a transient cluster problem, a
cancelled run. Continuation does not care *why* the previous build stopped: any **finished**
build can be continued. It just continues from where the previous run left off.

Continuation differs from re-initializing a fresh build with
`gb build init --from-build <ID>` followed by `gb build start`: that path creates a brand-new,
unrelated build and re-runs **every** target from scratch. Continuation reuses the targets that
already succeeded.

## Behaviour

`gb build continue <BUILD_ID>` creates a **new** build (a new build id) that extends the
previous build's chain, and submits it through the ordinary build path — so the BuildWatcher
dispatches a fresh runner for it, exactly like any other build. The new build:

1. Links to the previous build via `retry_of_build_id` (pointing at the root of the previous
   build's chain), so the runner's target-reuse machinery is engaged.
2. Starts with `retry_count = 0`, so the `max_retries` budget from `build.yaml` is counted
   **fresh** for the continuation, independent of how many retries the previous chain already
   consumed.
3. Re-runs the previous build's targets, **skipping** any target that already succeeded
   anywhere in the chain (see [target reuse](target-reuse.md)). A skipped target's
   `StoredTargetRun.skipped_for_prerun_target_id` points back to the original successful run.

Because the continuation is an ordinary build, everything that already works for a build works
for it: it retries its own remaining targets up to `max_retries`, cancellation spans the whole
chain, and `gb build status` shows the skipped and re-run targets together.

## The previous build must be finished

A continuation spins up a **fresh** runner, so the previous build must not still be active — a
build that is `PENDING`, `RUNNING`, `RETRY_PENDING`, or `CANCEL_REQUESTED` still has (or is
about to have) a runner working it. Continuing such a build is rejected (HTTP `409`). Only a
finished build (`SUCCESS`, `FAILED`, `INVALID`, or `CANCELLED`) can be continued.

Continuing a `SUCCESS` build is allowed and simply re-runs any targets that were not part of
that build's successful set (or completes immediately if there is nothing left to do).

## Continuing a build that was already retried

If the build you name is part of a retry chain (it was retried, or is itself a retry), the
continuation is linked to the **root** of that chain, so target reuse spans **all** prior
attempts in the chain — you may pass **any** member of the chain and the server resolves the
canonical root for you (it is reported back on the command line and in the `--format json`
output as `root_build_id`).

Continuations are **appended to the current tip** of the chain, so continuing repeatedly keeps
the chain linear (`A → B → C`) rather than branching into several chains that share a root. A
single `gb build status --follow-retries` therefore shows every attempt — original, auto-retries,
and continuations — in one view.

> **Note:** issuing two continuations of the *same finished tip* at the exact same moment can
> race on the tip's forward link (last write wins), leaving one continuation reachable only via
> its `retry_of_build_id` (the root) rather than the forward `--follow-retries` walk. This is a
> narrow, user-initiated window; continue a build once and wait for it to appear before
> continuing again.

## Relationship to retry

| | Build-level retry | Build continuation |
|---|---|---|
| Trigger | build ends `FAILED` and `retry_count < max_retries` | explicit `gb build continue` |
| Runner | same, in-process retry loop | a **fresh** runner |
| Applies to | only a `FAILED` build | **any** finished build |
| `max_retries` | consumed across the chain | **reset**: counted fresh |
| New build record | yes (linked via `retry_of_build_id`) | yes (linked via `retry_of_build_id`) |
| Target reuse | yes, across the chain | yes, across the chain |

Both mechanisms reuse the same [target-reuse](target-reuse.md) machinery; continuation simply
starts a fresh runner on a fresh `max_retries` budget for an arbitrary finished build.
