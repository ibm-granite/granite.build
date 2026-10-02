# distill-tokenizer-align — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`. This file is for people changing the step.

## What is ours and what is upstream

| | Where it lives | Change it where? |
|---|---|---|
| `step-template.yaml` | here | here |
| `src/run-align.sh` | here, a near-verbatim port | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_build_overlay.py` | here, a near-verbatim port | **upstream**, then re-sync |
| `gb_steps_post_training.distillation` (the actual work) | the delivered checkout — 15.6k lines, 33 modules | **upstream only** |

Ported files carry a header naming their upstream path and commit, so a re-sync is a three-way
merge rather than an eyeball diff. Keep them near-verbatim; behaviour changes belong upstream.

The shared package is deliberately **not vendored**: its `tokenizer_identity` module writes a
guard whose reader is the trainer in that same checkout, and a second copy is how the two halves
drift apart.

## Non-image step

There is no `Dockerfile`, so `steps/common.mk` auto-detects a non-image step
(`STEP_USES_IMAGE := $(if $(wildcard $(DOCKERFILE)),true,false)`), `IMAGE_REF` is empty, and
`image` / `publish-image` become no-ops. `make all` reduces to `make space`.

The image is named literally in the template rather than as `${IMAGE_REF}`. Its contents were
measured on BlueVela rather than assumed: python 3.12.14, torch 2.8.0+cu128, transformers 5.8.0,
tokenizers 0.22.1, safetensors 0.7.0, datasets 4.4.1, trl 0.26.2, deepspeed 0.17.5, accelerate
1.12.0, numpy 2.2.6, vllm 0.11.0, liger_kernel, pyyaml 6.0.3. **`clearml` and `wandb` are
absent** — they are imported lazily and only when a tracking project is configured, so tracking
must stay off.

## Two traps encoded in the template

- **`${#VAR}` opens a Jinja comment.** The renderer does not override `comment_start_string`, so
  it stays the default `{#`. A build.yaml carrying `${#TOKEN}` fails validation with
  "Missing end of comment tag" and nothing naming the real cause. Use `printf %s "$V" | wc -c`.
- **`[ -n "$X" ] && echo …` as the last statement of an if-body aborts the run.** The launcher
  prefixes `set -eu`, and that compound returns 1 when `X` is empty.
  `steps/distill/distill-gold/skypilot/step-template.yaml:219` still carries this latent bug; a test here
  asserts this step does not.

## Publishing

```bash
make space          # render the Space; cheap, offline
make test           # space + pytest
make publish-step   # -> configurations/assets/environments/skypilot/steps/distill/distill-tokenizer-align/
make check-published
```

`publish-step` writes to `configurations/assets/environments/skypilot/steps/` — **not** under
`lsf/ibm-bluevela/`. BlueVela still resolves it via the Tier-1 ancestor walk; the step gates
itself to LSF with `environment_configs.Skypilot.subtypes: [lsf]`.
