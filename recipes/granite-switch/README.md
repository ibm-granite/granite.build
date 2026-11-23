# Granite Switch

Granite Switch lets you compose adapter functions into a single deployable
model. It brings the rigor and modularity of software engineering to LLMs.

## Available recipes

Recipes are grouped by the SkyPilot cloud they run on, as
`skypilot/<cloud>/<recipe>`: `aws`, or `slurm` (any SLURM cluster, described
by a site file).

| Recipe | What it builds |
|---|---|
| [`skypilot/slurm/answerability-lora`](skypilot/slurm/answerability-lora/README.md) | An answerability LoRA or aLoRA adapter on `granite-4.2-8b`, trained with the `hf-peft-sft` step (SkyPilot/SLURM). |
| [`skypilot/aws/answerability-lora`](skypilot/aws/answerability-lora/README.md) | A template for LoRA and/or aLoRA adapters on `granite-4.2-8b`, trained on AWS EC2 GPUs with a shared EFS filesystem. Bring your own data. |
| [`skypilot/aws/answerability-sr`](skypilot/aws/answerability-sr/README.md) | A template for a Shadow Residual adapter on `granite-4.2-8b`, trained and scored on AWS EC2 GPUs with a shared EFS filesystem. Bring your own data. |

The AWS recipes need a one-time setup in your own AWS account (credentials, an
EFS filesystem, and a private environment copy holding its id): see
[Running the granite-switch recipes on AWS](skypilot/aws/README.md).

The SLURM recipe needs a private site file describing your cluster (partitions,
account, QOS, shared paths, login nodes) and your data, and an environment
rendered from it: see the recipe's
[Setup](skypilot/slurm/answerability-lora/README.md#setup).

More adapter-library recipes will be published here. Pick the `build.yaml` of
the recipe you want and run:

```bash
gb build start -f <path to the recipe>/build.yaml
```

A recipe with a site file, like the SLURM one, also takes that file:

```bash
gb build start -f <path to the recipe>/build.yaml --param-file <your-site.yaml>
```

Each recipe directory carries its own `README.md` and `parameters.yaml`; read
the README first, since a recipe may have a prerequisite to satisfy before the
first run — data to stage, or a credential to register.

`<shared-fs>` in a recipe's README or comments stands for a directory on the
cluster's shared filesystem; see [Placeholders](../README.md#placeholders).

Alternatively, access Granite.build through an agent session and simply ask
it to start a new build using that recipe.

For more details, see the Granite.build
[Getting started guide](../../docs/getting-started.md).

