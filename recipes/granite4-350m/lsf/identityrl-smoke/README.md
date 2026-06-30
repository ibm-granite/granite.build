# Identity-RL smoke recipe

A minutes-long smoke test of the end-to-end Identity RL plumbing on BlueVela
(LSF via SkyPilot), before committing to the full run. Mirrors `ifrl-smoke`,
but for the identity workflow (gbansible `run_rl_bluevela_identity.sh`).

Identity RL trains the model's persona/identity responses ("Who are you?",
"What's your name?") on the `general_identity` dataset, scored by the reward
model.

## No code-server

Unlike `ifrl-smoke`, identity RL has **no code-server** — the source script
sets `start_code_server=false` / `code_server_url=""`. The recipe omits the
`code-server` target: the trainer takes only the `rm_server_url` binding, and
`teardown` downs only the rm-server cluster.

## What it validates

- Image pull + SkyPilot/LSF submission for all three targets.
- `rm-server` reaches a served state and publishes its URL as a binding
  (`rm_server_url`).
- `identityrl-training` resolves the RM URL, mounts the dataset, loads the
  model, and runs the GRPO loop for a couple of optimizer updates — exercising
  rollout, a reference-policy update, an eval, and a checkpoint write (all freqs
  forced low; `TOTAL_EPISODES=1024` → `1024 // (64×16) = 1` update).
- A `checkpoint` output is emitted under the smoke `OUTPUT_DIR`.
- `teardown` downs the rm-server cluster after training completes.

## What it does NOT validate

- Convergence / model quality (a handful of updates is far too few).
- Downstream evals.

## Run

    gb build start -f recipes/granite4-350m/lsf/identityrl-smoke/build.yaml \
      --parameters-path recipes/granite4-350m/lsf/identityrl-smoke/parameters.yaml \
      --space <your-space>

Outputs are namespaced under `…/granite4-350m-14-7180-identityrl-smoke/` so a
smoke run cannot clobber the real Identity RL run. Override any value with
`--param KEY=VALUE` (see `parameters.yaml`).
