# Granite Switch

Granite Switch lets you compose adapter functions into a single deployable
model. It brings the rigor and modularity of software engineering to LLMs.

## Available recipes

Recipes are grouped by the SkyPilot cluster they run on, as
`skypilot/<cluster>/<recipe>`; the cluster names match the step framework's
(`steps/*/skypilot/test/<cluster>/`): `aws`, `lsf-bluevela`, `slurm-bluevela`.

| Recipe | What it builds |
|---|---|
| [`skypilot/lsf-bluevela/answerability-sr`](skypilot/lsf-bluevela/answerability-sr/README.md) | A Shadow Residual answerability adapter on `granite-4.2-8b`, trained and then scored against the answerability eval set (IBM BlueVela, LSF). |
| [`skypilot/slurm-bluevela/answerability-lora`](skypilot/slurm-bluevela/answerability-lora/README.md) | An answerability LoRA or aLoRA adapter on `granite-4.2-8b`, trained with the `hf-peft-sft` step (IBM BlueVela, SLURM). |
| [`skypilot/aws/answerability-lora`](skypilot/aws/answerability-lora/README.md) | The same LoRA and/or aLoRA adapters, trained on AWS EC2 GPUs with a shared EFS filesystem. |
| [`skypilot/aws/answerability-sr`](skypilot/aws/answerability-sr/README.md) | The same Shadow Residual adapter, trained and scored on AWS EC2 GPUs with a shared EFS filesystem. |

More adapter-library recipes will be published here. Pick the `build.yaml` of
the recipe you want and run:

```bash
gb build start -f <path to the recipe>/build.yaml
```

Each recipe directory carries its own `README.md` and `parameters.yaml`; read
the README first, since a recipe may have a prerequisite to satisfy before the
first run — data to stage, or a credential to register.

Alternatively, access Granite.build through an agent session and simply ask
it to start a new build using that recipe.

For more details, see the Granite.build
[Getting started guide](../../docs/getting-started.md).

