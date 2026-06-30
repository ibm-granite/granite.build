# Identity-RL full recipe

The complete Identity RL run on BlueVela (LSF via SkyPilot). This is the
gbserver reproduction of the gbansible `run_rl_bluevela_identity.sh` script —
RM server, same GRPO configuration, full training length. The
`identityrl-smoke` recipe is the minutes-long plumbing test for the same
workflow; this recipe runs it to convergence.

Identity RL trains the model's persona/identity responses ("Who are you?",
"What's your name?") on the `general_identity` dataset, scored by the reward
model. It typically takes an IFRL output checkpoint as its starting point.

## What it runs

Three targets on `space://environments/skypilot/lsf`:

- **rm-server** — reward-model FastAPI service (`phi4-multilingual`); publishes
  its URL as the `rm_server_url` binding.
- **identityrl-training** — `openinstruct-rl` GRPO trainer. Resolves the RM
  server URL, mounts the dataset, loads the starting checkpoint, and runs the
  full GRPO loop (`TOTAL_EPISODES=120000`) on the 8-GPU topology (4 learners +
  4 vLLM engines). Emits a converged `checkpoint` output.
- **teardown** — downs the rm-server cluster once training emits its checkpoint.

## No code-server

Unlike `ifrl-*`, identity RL has **no code-server** — the source script sets
`start_code_server=false` / `code_server_url=""`. The LSF `openinstruct-rl`
step gates the `CODE_SERVER_URL` export on a non-empty URL, so the recipe
simply omits the `code-server` target: the trainer takes only the
`rm_server_url` binding, and `teardown` downs only the rm-server cluster.

## Difference from `identityrl-smoke`

Structurally identical (same `build.yaml`, same targets, same `H100:8`
topology). The only differences are the full-run training-length parameters,
matching the `run_rl_bluevela_identity.sh` `-e` overrides:

| Parameter               | smoke | full   |
|-------------------------|-------|--------|
| `TOTAL_EPISODES`        | 1024  | 120000 |
| `SAVE_FREQ`             | 10    | 20     |
| `EVAL_FREQ`             | 10    | 20     |
| `CHECKPOINT_STATE_FREQ` | 10    | 20     |
| `REF_POLICY_UPDATE_FREQ`| 6     | 8      |

## Difference from `ifrl-full`

Same servers minus the code-server, plus the identity-specific deltas the
source scripts differ by:

| Parameter            | ifrl              | identityrl              |
|----------------------|-------------------|-------------------------|
| `RL_NAME`            | `if`              | `identity`              |
| `TEMPERATURE`        | 1.0               | 1.15                    |
| `TEMP_FINAL`         | 1.0               | 1.15                    |
| `ADD_GENERAL_REWARD` | true              | false                   |
| dataset              | `mix16_if/`       | `general_identity/`     |
| output namespace     | `exp_32`          | `exp_12`                |
| code-server          | yes               | none                    |

## What it does NOT include

- Downstream evals. Like the ansible script, this recipe stops after training.

## Run

    gb build start -f recipes/granite4-350m/lsf/identityrl-full/build.yaml \
      --parameters-path recipes/granite4-350m/lsf/identityrl-full/parameters.yaml \
      --space <your-space>

Outputs are namespaced under `…/granite4-350m-openinstruct-epoch-hf-2-identity/`
so this run cannot clobber the IFRL run's output. Override any value with
`--param KEY=VALUE` (see `parameters.yaml`).
