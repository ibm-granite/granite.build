# Templates

A template is a reusable `build.yaml` pattern that defines a complete pipeline
for a common workflow. Templates let you start quickly — instantiate one,
customize the parameters, and submit.

> **Audience:** users authoring builds. If you're looking for the build.yaml
> schema itself, see [`build-yaml-reference.md`](../users/build-yaml-reference.md).

## What's in a template

A template is just a directory containing:

```
assets/templates/<TemplateName>/
├── build.yaml    # the pipeline definition
└── README.md     # usage instructions (optional but recommended)
```

The `build.yaml` uses the same schema as any build — targets, steps, inputs,
outputs. The README describes what the template does, what inputs to customize,
and any prerequisites.

## Template categories

Templates generally fall into these categories:

| Category | Description |
|----------|-------------|
| **Data Generation** | Synthetic data generation using teacher models (e.g. DiGiT pipelines). |
| **Model Tuning** | Full fine-tuning, LoRA, EPT, reinforcement learning. |
| **Model Evaluation** | Benchmark evaluation, custom metrics, leaderboard generation. |
| **End-to-End** | Multi-stage pipelines combining data generation, tuning, and evaluation. |
| **Custom / BYOS** | Custom code steps for arbitrary workloads. |

## Listing available templates

```bash
gb template list
```

This shows templates registered in the active space. Use `--format json` for
machine-readable output.

## Using a template

```bash
gb build init --from-template <TemplateName> <your-build-dir>
cd <your-build-dir>
```

This copies the template's `build.yaml` into a new directory. Edit it to
customize parameters — model URIs, dataset paths, hyperparameters — then
submit:

```bash
gb build start
```

## Creating your own template

1. Create a directory under `assets/templates/` (or any location your space
   is configured to scan).
2. Add a `build.yaml` with your pipeline definition.
3. Add a `README.md` documenting the template's purpose, required inputs,
   expected outputs, and configuration knobs.

Follow the naming convention: `<Workflow>_<Variant>` (e.g.
`DiGiT_Skypilot`, `SFTFull_FMEval`).

### README structure for templates

A good template README covers:

- **What it does** — one-sentence summary.
- **Targets** — table of targets and what each produces.
- **Artifacts** — table of inputs and outputs with their types.
- **Prerequisites** — credentials, data, infrastructure.
- **Configuration** — what to edit in `build.yaml` before running.
- **Running** — the exact commands to submit and monitor.

See [`assets/templates/DiGiT_Skypilot/README.md`](../../assets/templates/DiGiT_Skypilot/README.md)
for a working example.

## Available templates

Templates shipped with this repo live in [`assets/templates/`](../../assets/templates/):

| Template | Description |
|----------|-------------|
| `DiGiT_Skypilot` | Synthetic data generation with DiGiT on SkyPilot (S3 file_mounts). |
| `DiGiT_Skypilot_PVC` | Same as above but outputs to a PVC. |
| `Sage_Skypilot` | LLM evaluation with Sage on SkyPilot. |
| `Sage_Skypilot_PVC` | Same as above but outputs to a PVC. |

## See also

- [`build.yaml` reference](../users/build-yaml-reference.md) — full schema
- [Steps](../steps/README.md) — built-in and custom steps
- [Bring Your Own Step](../users/bring-your-own-step.md) — custom code as a step
