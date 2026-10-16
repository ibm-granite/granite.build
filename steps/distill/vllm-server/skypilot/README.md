# vllm-server — authoring notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` publishes
as the asset's `README.md`. This file is about the step's construction.

## Why the server is backgrounded rather than `exec`ed

The `gold` step's in-allocation server branch `exec`s `run_vllm_serve.py`, which is right
there: the process *is* the node's job. Here it cannot be, because the step has to do
something **after** the server becomes ready — publish the URL — and an `exec`ed process
never returns to do it.

Publishing before the server is ready would defeat the entire design. The consumer
target dispatches the moment its bindings resolve, so an early publish means the trainer
starts racing a multi-minute model load, and the failure is a connection error minutes
into two allocations rather than a wait.

So: background the server, poll `/health`, publish, then `wait`. The `wait` matters as
much as the poll — a service step that returns ends its own step, and the trainer would
lose the server mid-run.

## The death check is simpler here than in the reference launcher

`gold-submit.sh` explains at length that a pid check alone is not enough, because its
`VLLM_PIDS` held the pid of a local `blaunch` rather than the remote python — on one job
the server died and `blaunch` did not exit, so the loop waited out the health timeout.
It therefore reads the server's own log for a traceback as the authoritative signal.

None of that applies here: SkyPilot's provisioner runs this script *on* the server node,
so `SERVER_PID` is a direct child and `kill -0` is authoritative. The log heuristic was
deliberately not ported — it exists to work around a layer this step does not have.

## Why the two markers are on separate lines

`get_events_from_log_line` reassigns `log_line = match[0]` after the first matching
event config, so two markers on one line leaves the second rule matching against a
truncated line. Both markers use the shipped monitor's generic
`GB_ARTIFACT_ID:… GB_ARTIFACT_STATE:…` rule, which is why the step needs no scrape
config of its own — and why they must not share a line.

## Why `extra_event_configs` and not `event_configs`

`resolve_monitor_config` **rejects** an overlay that sets `event_configs` on a `ref`.
Lists replace wholesale in the merge, so it would silently drop the referenced monitor's
artifact rules — the ones both markers depend on. `extra_event_configs` appends. The
step's only own rule is a `WORKLOAD_STATUS_EVENT` so the service reports RUNNING rather
than sitting in a pending state for its whole life.

## Adding a config key

Two places, one fewer than the `gold` step (there is no renderer here): the
`config.vllm_config` block in `step-template.yaml`, and its use in the `run:` script.
Add a test when the failure mode is worse than a wrong number — an unpublished binding
or a server that answers on an address the consumer cannot reach both present as a
consumer target that simply never starts.

## Tests

```shell
make -C steps/distill/vllm-server/skypilot test     # renders the Space, then runs test/
```

`test/test_step_template.py` reads the template directly rather than the rendered Space,
so it holds whether or not `make space` has run. It checks the shell parses, the
readiness gate really precedes the URL marker (by string position, not by inspection),
the markers' form and separation, the monitor overlay's shape, and the vLLM workarounds
carried from the reference launcher.

What it cannot check is the one thing that matters: whether a NCCL weight-sync group
spans two LSF allocations. That needs the cluster; build `9973e766` (the
`gold-onpolicy-smoke` recipe) showed that it does.
