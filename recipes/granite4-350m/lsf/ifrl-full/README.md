# IFRL full recipe

The complete ~1-day Instruction-Following RL run on BlueVela (LSF via SkyPilot).
This is the gbserver reproduction of the gbansible `run_rl_bluevela_if.sh`
script — same servers, same GRPO configuration, full training length. The
`ifrl-smoke` recipe is the minutes-long plumbing test for the same workflow;
this recipe runs it to convergence.

## What it runs

Four targets on `space://environments/skypilot/lsf`:

- **rm-server** — reward-model FastAPI service (`phi4-multilingual`); publishes
  its URL as the `rm_server_url` binding.
- **code-server** — code-execution FastAPI service; publishes `code_server_url`.
- **ifrl-training** — `openinstruct-rl` GRPO trainer. Resolves both server
  URLs, mounts the dataset, loads the SFT checkpoint, and runs the full GRPO
  loop (`TOTAL_EPISODES=500000`) on the 8-GPU topology (4 learners + 4 vLLM
  engines). Emits a converged `checkpoint` output.
- **teardown** — downs the rm/code server clusters once training emits its
  checkpoint.

## Difference from `ifrl-smoke`

Structurally identical (same `build.yaml`, same targets, same `H100:8`
topology). The only differences are the full-run training-length parameters,
matching the `run_rl_bluevela_if.sh` `-e` overrides:

| Parameter               | smoke | full   |
|-------------------------|-------|--------|
| `TOTAL_EPISODES`        | 1024  | 500000 |
| `SAVE_FREQ`             | 10    | 80     |
| `EVAL_FREQ`             | 10    | 80     |
| `CHECKPOINT_STATE_FREQ` | 10    | 80     |
| `REF_POLICY_UPDATE_FREQ`| 6     | 32     |

## What it does NOT include

- Downstream evals (the 27-eval fan-out is the separate full eval recipe, #39).
  Like the ansible script, this recipe stops after training.

## Run

    gb build start -f recipes/granite4-350m/lsf/ifrl-full/build.yaml \
      --parameters-path recipes/granite4-350m/lsf/ifrl-full/parameters.yaml \
      --space <your-space>

Outputs are namespaced under `…/granite4-350m-14-7180-ifrl-gb/` (note the `-gb`
suffix) so this gbserver run cannot clobber the gbansible run's output under
`…/granite4-350m-14-7180-ifrl/`. Override any value with `--param KEY=VALUE`
(see `parameters.yaml`).
