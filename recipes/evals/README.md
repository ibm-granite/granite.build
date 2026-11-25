# Eval recipes

One recipe per benchmark, at `<family>/<benchmark>/skypilot/`: a `build.yaml` and its
`parameters.yaml`. The `<family>` is the
[`steps/eval/<family>`](../../steps/eval/README.md) step the recipe runs, one granite-evals
image per family. Each recipe holds the GPU only while the model generates
(`<benchmark>-generate`), and most then grade that output on a CPU node (`<benchmark>`).

| Path | What it is |
|---|---|
| `<family>/<benchmark>/skypilot/` | One benchmark's recipe. The header of its `parameters.yaml` describes the recipe and gives a quick-check command. |
| [`site.example.yaml`](site.example.yaml) | Every site-specific key, with fake values. One file for all the recipes here. |
| [`../environments/skypilot-slurm.template.yaml`](../environments/skypilot-slurm.template.yaml) | The SkyPilot-on-SLURM environment, as a template filled from the site file. Shared with the other SLURM recipes. |

## Families

| Family | Benchmarks |
|---|---|
| `bfcl` | `bfcl-v4` |
| `bird` | `birdbench` |
| `ifbench` | `ifbench` |
| `judged` | `fortress-adversity`, `fortress-benign`, `gdpval`, `mcpatlas`, `multi-challenge`, `profbench`, `strongreject` |
| `lmeval` | `include-44`, `mmlu-prox-lite` |
| `mcpmark` | `mcpmark-filesystem`, `mcpmark-playwright`, `mcpmark-postgres` |
| `nemoskills` | `aa-lcr`, `aime24`, `aime25`, `aime26`, `arena-hard-v2`, `browsecomp`, `browsecomp-plus`, `critpt`, `critpt-tools`, `flores200`, `gpqa`, `hle`, `hle-tools`, `hmmt-feb25`, `human-eval`, `human-eval-tools`, `imo-answerbench`, `livecodebench-v6`, `math-500`, `mbpp`, `mbpp-tools`, `mmlu-pro`, `omniscience`, `omniscience-hallucination`, `ruler-64k`, `ruler-128k`, `ruler-256k`, `ruler-512k`, `ruler-1m`, `scicode`, `wmt24pp` |
| `officeqa` | `officeqa-full`, `officeqa-pro` |
| `swebench` | `swebench-multilingual`, `swebench-pro`, `swebench-verified` |
| `tau` | `tau3-airline`, `tau3-banking-knowledge`, `tau3-bench`, `tau3-retail`, `tau3-telecom` |
| `tbench` | `terminal-bench-2.1` |

The images are public tags on `quay.io/granitebuild`, pinned in each recipe's `IMAGE` /
`IMAGE_CPU`. A recipe that calls a judge, a user simulator or a search server leaves its URL
empty (`JUDGE_BASE_URL`, `USER_BASE_URL`, `SEARCH_MCP_URL`); set it for your site. The gb
secrets a recipe reads are listed under `secret_names_to_use_as_env_variable` in its
`build.yaml`.

## Setup

**Requires `gb` v0.3.10 or later**, for `--param-file` layered over the recipe's own
`parameters.yaml` ([#663](https://github.com/ibm-granite/granite.build/issues/663)).

**1. The site file.** Copy the example to a private location outside this repo and fill it
in. The real one names your hosts, users, accounts and paths, so never commit it. If you run
the [distill recipes](../distill/README.md) too, use your distill site file: it has every
key these recipes need.

```bash
mkdir -p ~/.granite.build/local-env
cp recipes/evals/site.example.yaml ~/.granite.build/local-env/evals-site.yaml
SITE=~/.granite.build/local-env/evals-site.yaml
```

**2. The environment.** Render the template with the site file:

```bash
scripts/setup-local-space.sh \
  --template recipes/environments/skypilot-slurm.template.yaml \
  --site "$SITE" --name <env-name>
```

It writes the environment into a standalone space (`~/.granite.build/local-space` by
default; `--out DIR` writes it for a private repo instead) and sets `ENVIRONMENT_URI` in the
site file to `space://environments/skypilot-local/<env-name>`. The distill README's
[Setup](../distill/README.md#setup) describes what it writes in full, and the SSH key secret
the environment needs.

**3. Run a recipe** with the site file, on a gbserver whose space holds the environment:

```bash
gb build start -f recipes/evals/nemoskills/aime24/skypilot/build.yaml \
  --param-file "$SITE" --param OUTPUT_ROOT=<root> --param LIMIT=5
```

`--param LIMIT=5` scores the first 5 examples, a quick check of the plumbing; leave it out
for the full benchmark. Add `--dry-run` to print the rendered `build.yaml` without
submitting it. Without `--param-file` the render fails on the first site key it meets
(`'ENVIRONMENT_URI' is undefined`), rather than sending the run to some other site.

## `OUTPUT_ROOT`

`OUTPUT_ROOT` is required and has no default. A run writes
`$OUTPUT_ROOT/$RUN_NAME/<benchmark>`, so point it at a directory on the cluster's shared
filesystem that the submitting user can write and the containers see (under one of the
environment's mounts). A relative value resolves against the build's `GB_BUILD_WORKDIR`.
The build still renders without it, but its job fails as soon as it starts. Pass it with
`--param OUTPUT_ROOT=<root>`, or put it in your site file.

Make `RUN_NAME` unique per run: with the same model and `RUN_NAME`, a re-run resolves the
same artifact URIs, and a reused target's downstream score target never dispatches. See
[`steps/eval/README.md`](../../steps/eval/README.md) for both.
