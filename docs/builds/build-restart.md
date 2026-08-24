# Build Restart

`gb build restart` re-runs a previously-executed build in a **fresh** build runner,
**reusing targets that already succeeded** and re-running the rest. Use it to pick a build
back up from where it left off after a failure or interruption — without re-running work
that already completed. (The build is *continued* in place rather than run from scratch.)

```shell
gb build restart <BUILD_ID>
```

`<BUILD_ID>` may be a build id or a build URL. No local build folder is required — the build
definition, space, and targets are taken from the build being continued.

Despite the `restart` verb, the build is **continued** in place, not run from scratch: targets
that already succeeded are reused. It is distinct from [build-level retry](build-retry.md) and
from step-level retry.

## When to use it

A build can fail for many reasons — a build-definition error, a transient cluster problem, a
cancelled run. Restart does not care *why* the previous build stopped: any **finished build that
did not fully succeed** can be restarted. It just continues from where the run left off. A build
that finished with status `SUCCESS` has nothing left to run and cannot be restarted (see
[The build must be finished and not `SUCCESS`](#the-build-must-be-finished-and-not-success)).

Continuation differs from re-initializing a fresh build with
`gb build init --from-build <ID>` followed by `gb build start`: that path creates a brand-new,
unrelated build and re-runs **every** target from scratch. Continuation reuses the targets that
already succeeded, in place.

## Behaviour

`gb build restart <BUILD_ID>` **re-opens the same build** — the build keeps its id. The
finished build is flipped back to `SUBMITTED` and submitted through the ordinary build path, so
the BuildWatcher dispatches a fresh runner for it, exactly like any other build. On re-open the
build:

1. Keeps its **same build id** — there is no new build record and no retry chain. The continued
   run's history (its failed and successful target runs) all live on the one build.
2. Resets `retry_count = 0`, so the `max_retries` budget from `build.yaml` is counted **fresh**
   for the continuation, independent of how many auto-retries the build already consumed.
3. Re-runs its targets, **reusing** any target that already succeeded in this build (see
   [target reuse](target-reuse.md)). A reused target is simply not re-run — its existing
   `SUCCESS` `StoredTargetRun` stands; no separate "skipped" record is written.

Because the continuation is an ordinary build, everything that already works for a build works
for it: it retries its own remaining targets up to `max_retries`, and cancellation by build id
cancels the in-flight continuation.

## The build must be finished and not `SUCCESS`

A restart spins up a **fresh** runner, so the build must not still be active — a build that
is `PENDING`, `RUNNING`, or `CANCEL_REQUESTED` still has (or is about to have) a
runner working it. Restarting such a build is rejected (HTTP `409`).

A build that finished with status `SUCCESS` is **also** rejected (HTTP `409`): every target
already succeeded, so target reuse would skip all of them and the fresh runner would do no work.
Only a finished build that did not fully succeed — `FAILED`, `INVALID`, or `CANCELLED` — can be
restarted. This is enforced atomically in `reopen_finished_build`, so a build that succeeds
concurrently with the restart request is rejected rather than needlessly re-opened.

## Relationship to retry

| | Build-level retry | Build continuation |
|---|---|---|
| Trigger | build ends `FAILED` and `retry_count < max_retries` | explicit `gb build restart` |
| Runner | same, in-process retry loop | a **fresh** runner |
| Applies to | only a `FAILED` build | any finished build **except `SUCCESS`** |
| Build id | same build, reused | same build, reused |
| `max_retries` | consumed as `retry_count` climbs | **reset**: counted fresh |
| Target reuse | yes, within the build | yes, within the build |

Both mechanisms run in place on the one build id and reuse the same
[target-reuse](target-reuse.md) machinery; continuation simply starts a fresh runner on a fresh
`max_retries` budget for an arbitrary finished build.
