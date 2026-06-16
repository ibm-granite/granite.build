# Sample: `inference`

Run a single-prompt inference against any causal LM in the local **bash** environment — no
GPU, no container, no cloud credentials. Uses the
[`inference`](../../../configurations/assets/environments/bash/steps/inference/README.md) step.

## Run it

From the repo root, with the venv active (`make venv && source .venv/bin/activate`):

```bash
GBSERVER_METADATA_STORAGE=sqlite \
GBSERVER_DEFAULT_BUILDRUNNER_TYPE=process \
GB_ENVIRONMENT=STANDALONE \
gbserver build run \
  --space-config-uri "file://$(pwd)/configurations/spaces/local" \
  samples/standalone/inference
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
`response.txt`, and registers them as the `generation` artifact (→
`file:outputs/inference/`). Success is logged as `INFERENCE_SUCCESS`.
