# gold — authoring notes

User-facing documentation is in [USAGE.md](USAGE.md), which `make publish-step`
publishes as the asset's `README.md`. This file is for people changing the step.

## Why the config is rendered by a Python script

`src/render_gold_config.py`, not a heredoc in `run:`. Jinja works in `run:` (see
`openinstruct-rl`), so this is about which mistakes stay silent, and the trainer
has three requirements that corrupt a run without erroring:

1. `learning_rate` / `min_lr` must be YAML **floats**. PyYAML reads a bare
   `1e-05` as a string, which crashes min_lr handling partway into a run.
   `yaml.safe_dump` of a real float makes that structural.
2. Booleans must be lower-case. `safe_dump` does it by construction; a template
   emits `True` unless every site remembers `| lower`.
3. The six on-policy keys must appear **only** when `vllm_num_servers > 0` — one
   testable branch here, versus `{% if %}` inside a quoted heredoc inside a YAML
   literal block. This is also the seam the on-policy phase reopens.

Dumping with the library the trainer parses with means a config that renders is a
config the trainer can read. `test/test_render_gold_config.py` covers each rule.

## What is deliberately not ported

`launch_enroot.sh` and `bv_run_kd_distill.sh` from the ansible path. The
provisioner (LSF's `bsub`/`blaunch`/enroot, or SLURM's `sbatch`/`srun`/pyxis)
already does job submission and container import/creation on both subtypes.
The step reimplements only the inner logic: pick a role, render the config,
start `accelerate`. It reads SkyPilot's own generic topology vars
(`SKYPILOT_NODE_RANK` / `SKYPILOT_NUM_NODES` / `SKYPILOT_NODE_IPS` /
`SKYPILOT_NUM_GPUS_PER_NODE`) rather than LSF's provisioner-specific
`RANK`/`TOTAL_NODES`/`MASTER_ADDR`/`MASTER_PORT`, so one step serves both
subtypes with no `{% if %}` branching on which one it's running on — see the
step-template.yaml header comment for why (and why that's the house
convention here, not a shortcut).

Consequences worth keeping in mind:

- `run:` executes on **every** node with that node's own rank, so anything that
  must happen once is guarded by `[ "$NODE_RANK" = "0" ]`. The artifact marker in
  particular: the executor streams all nodes into one driver log, so an unguarded
  marker registers N artifacts.
- `MASTER_PORT` has no SkyPilot-generic equivalent on either subtype, so it's
  derived the same way LSF's own provisioner derives its port
  (`sky/provision/lsf/instance.py`): offset a base port by a job id mod 1000,
  using `SKYPILOT_INTERNAL_JOB_ID` instead of `LSB_JOBID` since that one is
  generic.
- The node count is read from the allocation, never from a build parameter, so the
  checkpoint path cannot claim a topology the run did not have.

## Adding a hyperparameter

Add it in three places, in this order: `config.gold_config` in
`step-template.yaml` (with a comment saying why a non-obvious default is what it
is), an argument in `render_gold_config.py`, and a flag in the `run:` block's
renderer invocation. A test in `test/test_render_gold_config.py` if the value has
a failure mode worse than "wrong number".

## Running the tests

```shell
make -C steps/distill/gold/skypilot test          # offline contracts only
GB_STEP_BLUEVELA_BUILD=1 make -C steps/distill/gold/skypilot test   # + real 2-node build
```

`make test` invokes pytest with no marker filter, so the cluster run is behind an
explicit opt-in rather than a marker: otherwise a routine `make test` would submit
a two-node LSF job to a shared cluster, and an interrupted run would leave it
pending, holding an allocation nobody is waiting for. The repository suites select
it by marker (`-m "not ibm"`) as usual, so CI behaviour is unchanged.
