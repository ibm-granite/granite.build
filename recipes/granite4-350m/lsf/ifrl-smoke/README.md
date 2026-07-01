# IFRL smoke recipe

A minutes-long smoke test of the end-to-end IFRL plumbing on BlueVela (LSF via
SkyPilot), before committing to the full ~1-day run. First integration of the
`openinstruct-rl` step (#36) with the `rm-server` / `code-server` auxiliary
targets (#37).

## What it validates

- Image pull + SkyPilot/LSF submission for all three targets.
- `rm-server` and `code-server` reach a served state and publish their URLs as
  bindings (`rm_server_url`, `code_server_url`).
- `ifrl-training` resolves both URLs, mounts the dataset, loads the model, and
  runs the full GRPO loop for **2 optimizer updates** — exercising rollout, a
  reference-policy update, an eval, and a checkpoint write (all freqs forced to
  1; `TOTAL_EPISODES=2048` → `2048 // (64×16) = 2` updates).
- A `checkpoint` output is emitted under the smoke `OUTPUT_DIR`.

## What it does NOT validate

- Convergence / model quality (2 updates is far too few).
- Full-scale topology (runs on `H100:4` with 2 learners + 2 vLLM engines, not
  the production 8-GPU layout).
- Downstream evals (the 27-eval fan-out is the full recipe, #39).

## Run

    gb build start -f recipes/granite4-350m/lsf/ifrl-smoke/build.yaml \
      --parameters-path recipes/granite4-350m/lsf/ifrl-smoke/parameters.yaml \
      --space <your-space>

Outputs are namespaced under `…/granite4-350m-14-7180-ifrl-smoke/` so a smoke
run cannot clobber the real IFRL run. Override any value with `--param KEY=VALUE`
(see `parameters.yaml`).
