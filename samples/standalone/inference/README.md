# Sample: `inference`

Run a single-prompt inference against any causal LM in the local **bash** environment — no
GPU, no container, no cloud credentials. Uses the
[`inference`](../../../configurations/assets/environments/bash/steps/inference/README.md) step.

## Run it

From the repo root, with the venv active (`make venv && source .venv/bin/activate`). Start the
standalone server in one terminal, then submit the build with `gb` (gbcli) in another:

```bash
# Terminal 1 — start the server
gbserver standalone --space-dir configurations/spaces/local
```

```bash
# Terminal 2 — submit the build
export GB_ENVIRONMENT=STANDALONE
gb build start -f samples/standalone/inference/build.yaml
```

`gb build start` returns a build ID; use it to check status and logs:

```bash
gb build status <build-id>
gb build log <build-id>
```

On first run the step `pip install`s `torch`/`transformers` into the venv (CPU-only), so it
takes a few minutes; subsequent runs are fast.

## What's configurable (all in `build.yaml`)

| Where | Field | Purpose |
|-------|-------|---------|
| `inputs.model.uri` | `hf:///ibm-granite/granite-4.0-h-350m` | The model. Swap to any HF causal LM — the step code doesn't change. |
| `config.bash.env.PROMPT` | prompt text | What to ask the model. |
| `config.bash.env.MAX_NEW_TOKENS` | `512` | Generation length. |

The model arrives in the step as `$LLMB_BASH_INPUT_MODEL` automatically (see
[bash-environment.md](../../../docs/operators/bash-environment.md#how-inputs-reach-your-script));
`PROMPT`/`MAX_NEW_TOKENS` arrive via `config.bash.env`.

## Output

The step writes `inference_result.json` (model type, prompt, response, timing) and
`response.txt`, and registers them as the `generation` artifact. Success is logged as
`INFERENCE_SUCCESS`.

A successful run creates an **`outputs/` folder** (relative to where you started
`gbserver standalone` — typically the repo root) and writes the result under it. The output
URI is defined in `build.yaml` as `file:outputs/inference_{{ binding.path | short_hash }}/`,
where `{{ binding.path | short_hash }}` makes the location **unique per run** (keyed on the
model), so repeated runs don't overwrite each other. For example:

```
outputs/inference_a1b2c3d4/
```

The registered artifact records the resolved absolute path. Inspect it with:

```bash
gb build status <build-id>              # shows each target's output artifacts (id + uri)
gb artifact list --build-id <build-id>  # lists the individual artifact entries
```
