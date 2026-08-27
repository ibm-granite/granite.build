# dpk (SkyPilot)

Runs a [Data Prep Kit](https://github.com/data-prep-kit/data-prep-kit) (DPK) transform on a
SkyPilot cluster. You name the transform and its flags; the step derives the python module
and the pip dependencies, installs them on the node, wires the build's input/output paths
in, and registers the output artifact.

One step serves **every** DPK transform — tokenization, PII redaction, dedup, and the rest.
Switching transform is a change to the build.yaml, never to the step. It runs on the bare
launcher node by default (pure-python DPK, CPU-only), so it needs no container runtime and
works on clusters without the Pyxis SPANK plugin.

> **Developing or testing this step?** See `steps/dpk/skypilot/README.md` in the
> granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/dpk` URI:

```yaml
steps:
  - step_uri: space://steps/dpk
```

## Config contract (`dpk_config`)

All fields live under the step's `config.dpk_config`. There are two modes, selected by
which field you set — **`transform` XOR `command`**:

- **`transform`** — run a DPK transform. The step builds the invocation for you.
- **`command`** — run an arbitrary bash command, e.g. a bundled script. The step gets out
  of the way (this is the [`byoc`](../../byoc/skypilot/USAGE.md)-style escape hatch).

### Transform mode

| Field | Type | Required | Purpose |
|---|---|---|---|
| `transform` | string | **yes** | DPK transform short name, e.g. `tokenization2arrow`, `pii_redactor`, `ededup`. See [What `transform` derives](#what-transform-derives). |
| `input` | string | **yes** | Name of a declared target `inputs:` entry. Becomes the transform's `input_folder`. |
| `output` | string | **yes** | Name of a declared target `outputs:` entry. Used as the registered artifact's ID. |
| `output_path` | string | **yes** | Absolute path the transform writes to. **Must match the path in that output's `uri`** — see [Outputs](#outputs). |
| `args` | map | no | Transform flags, rendered in order as `--<key> '<value>'`. Keys are the **full flag name** as DPK spells it, without leading dashes. See [Transform flags](#transform-flags). |
| `dpk_version` | string | no | DPK release to install. Default `1.1.8`. |

### Command mode

| Field | Type | Required | Purpose |
|---|---|---|---|
| `command` | string | **yes** | Bash command, injected verbatim. Responsible for its own artifact marker — see [Outputs](#outputs). |

### Optional (both modes)

| Field | Type | Purpose |
|---|---|---|
| `packages` | list | Extra pip requirements installed alongside the transform, e.g. `["pyarrow"]`. In command mode this is the only install. |
| `pip_index_url` | string | Index for the pip install. Default `https://pypi.org/simple`. |
| `image` | string | Public container image to run in. Default `""` = the bare launcher node. When set, **no venv or pip install happens** — the image must already provide DPK. Rendered at runtime as SkyPilot `docker:<ref>`. |
| `module` | string | Override the derived module, e.g. `dpk_tokenization2arrow.ray.runtime`. |
| `extras` | list | Override the derived pip extra, e.g. `["ray"]`. |
| `no_extras` | bool | Install `data-prep-toolkit-transforms` with **no** extra. For the few transforms that declare none (`noop`, `c4_annotator`). |

## What `transform` derives

`transform: <name>` gives the step two things, by rule rather than by lookup table — which
is why a transform added to DPK later needs no change here:

```
transform: pii_redactor
  ├─ module → python -m dpk_pii_redactor.runtime          ("dpk_" + name + ".runtime")
  └─ pip    → data-prep-toolkit-transforms[pii-redactor]  (name, "_" → "-")
```

Every DPK transform exposes `dpk_<name>.runtime` (a `PythonTransformLauncher` accepting
`--data_local_config`), and `data-prep-toolkit-transforms` declares one pip extra per
transform. Deriving the extra means the dependency set is always the one DPK itself
declares for that transform — including awkward ones like `pii_redactor`'s pinned
`numpy<1.29` and its presidio/flair versions.

The step also auto-injects the launcher's data config, so you never write it:

```
--data_local_config "{'input_folder': '<input>', 'output_folder': '<output_path>'}"
```

## Transform flags

`args` keys are the **full flag name**, because DPK's flag prefix is *not* derivable from
the transform name. Roughly 40% of transforms use an arbitrary abbreviation:

| Transform | Flag prefix |
|---|---|
| `tokenization` | `tkn_` |
| `gopher_repetition_annotator` | `gra_` |
| `doc_quality` | `docq_` |
| `opensearch` | `os_` |
| `pii_redactor` | `pii_redactor_` |

So the step passes your keys through verbatim rather than guessing. Get the flag names from
the transform's DPK documentation or `python -m dpk_<name>.runtime --help`.

Value handling: `true` renders a bare `--flag`; `false` and null are omitted; everything
else renders as `--flag 'value'` (including `0`, which is meaningful for e.g.
`tkn_chunk_size`).

## Inputs and outputs

### Inputs

Declare each input on the **target** (a direct `uri:`, or a `binding:` to an upstream
target's output). The step exports every declared input as an environment variable holding
its staged local path:

```
inputs.<name>  →  $LLMB_INPUT_<name>
```

In transform mode, `input: <name>` selects which one feeds the transform. In command mode,
reference any of them directly from your `command`. Filesystem-backed schemes (`hf://`,
`env://`, `file://`, `s3://`, `lh://`) are staged by the assetstore before `run`; an
`hf://` input is downloaded during `setup` automatically.

### Outputs

Declare each output on the target, then make sure the bytes land at the path in its `uri`:

- **Transform mode** — set `output_path` to that path. The step creates it, points the
  transform's `output_folder` at it, and emits the artifact marker for you.
- **Command mode** — your command writes wherever it likes and prints the marker itself:

  ```
  LLMB_ARTIFACT_ID:<output-id> LLMB_ARTIFACT_PATH:<abs-path>
  ```

  `<output-id>` must match a declared `outputs.<id>`. Repeat the line to register several
  artifacts under one output. For `mem://` outputs use `LLMB_ARTIFACT_PATH` →
  `LLMB_ARTIFACT_STATE:<value>`.

> **`output_path` must agree with the output's `uri`.** The step cannot derive it — declared
> output URIs are not available in the runtime render context — so keeping the two in sync
> is the build author's job. A mismatch fails at run time, not at submit time.

### Choosing the output URI per endpoint

`validate`-style downstream targets read what an upstream target wrote, and the two may run
on **different nodes**. What works depends on the endpoint:

| Endpoint | Output URI | Why |
|---|---|---|
| `skypilot/slurm`, `skypilot/lsf` | `env:///shared/…` (+ matching `output_path`) | `/shared` is the environment's `shared_workdir`, mounted on every node — cross-node safe. |
| `skypilot/aws`, `skypilot/kubernetes` | `s3://bucket/…` with a **local** `output_path` | Each target gets its own instance with no shared filesystem. The S3 assetstore pushes the staged dir and the consumer pulls it. |

A node-local `env:///tmp/…` is **not** safe for a cross-target handoff: if the two targets
land on different nodes the consumer reads an absent directory.

## Working directory and paths

Both `setup` and `run` start in the same **working directory** (the step's per-run workdir),
so the step never needs its absolute location. The bundled `src/` is mounted at `./src`, and
in bare-node mode the virtualenv is created at `./venv` and activated for you. Use relative
paths from there; derive an absolute one at run time with `$(pwd)` when a marker needs it.

## Example build.yaml

Tokenize a HuggingFace dataset, then validate the result with the step's bundled script.
Both targets use the same step — the first in transform mode, the second in command mode.

```yaml
granite.build:
  name: dpk-tokenize
  targets:

    tokenize:
      environment_uri: space://environments/skypilot/slurm
      inputs:
        docs:
          uri: hf:///datasets/my-org/dpk-tokenization-sample
      outputs:
        tokens:
          uri: "env:///shared/dpk/tokens"
          type: dataset
      steps:
        - step_uri: space://steps/dpk
          config:
            poll_interval_seconds: 30
            dpk_config:
              transform: tokenization2arrow
              dpk_version: "1.1.8"
              input: docs
              output: tokens
              output_path: /shared/dpk/tokens      # matches the uri above
              args:
                tkn_tokenizer: hf-internal-testing/llama-tokenizer
                tkn_doc_id_column: document_id
                tkn_doc_content_column: contents
                tkn_text_lang: en
                tkn_chunk_size: 0
            compute_config:
              num_gpus_per_node: 0
              total_memory_per_node: "4Gi"

    validate:
      environment_uri: space://environments/skypilot/slurm
      inputs:
        tokens:
          binding: tokenize.tokens                 # dispatched once tokens registers
        docs:
          uri: hf:///datasets/my-org/dpk-tokenization-sample
      outputs:
        report:
          uri: "env:///tmp/dpk-validate-report"
          type: fileset
      steps:
        - step_uri: space://steps/dpk
          config:
            poll_interval_seconds: 30
            dpk_config:
              packages: ["pyarrow"]                # no transform => only this installs
              command: |
                REPORT=/tmp/dpk-validate-report
                mkdir -p "$REPORT"
                python src/validate_tokens.py "$LLMB_INPUT_tokens" "$REPORT" \
                  --input "$LLMB_INPUT_docs"
                echo "LLMB_ARTIFACT_ID:report LLMB_ARTIFACT_PATH:$REPORT"
```

Switching to a different transform touches only `dpk_config`:

```yaml
            dpk_config:
              transform: pii_redactor              # -> dpk_pii_redactor.runtime + [pii-redactor]
              input: docs
              output: clean
              output_path: /shared/dpk/clean
              args:
                pii_redactor_entities: "PERSON,EMAIL_ADDRESS,CREDIT_CARD"
                pii_redactor_operator: replace
                pii_redactor_score_threshold: 0.6
```

## Bundled scripts (`src/`)

The step ships `src/` to `./src` on the node:

- [`src/validate_tokens.py`](src/validate_tokens.py) — validates `tokenization2arrow`
  output. Checks that the Arrow token stream agrees with its `meta/*.docs` /
  `meta/*.docs.ids` sidecars (a truncated or duplicated write is caught, where an
  exists-and-non-empty check would pass), and with `--input <parquet_dir>` also checks
  **completeness**: that every non-empty source Parquet actually produced output. Exits
  non-zero with a per-file report, failing the build.

  ```
  python src/validate_tokens.py <tokenize_output_dir> <report_dir> [--input <source_parquet_dir>]
  ```

  It is read-only — it never deletes or rewrites anything.

## Notes and limitations

- **Pure-python DPK.** The step runs the CPU-only pure-python runtime. DPK's Ray runtime is
  faster on large corpora; it is reachable via `module: dpk_<name>.ray.runtime` +
  `extras: ["ray"]`, but provisioning a multi-node Ray cluster is out of scope here.
- **Runtime dependency install.** Dependencies are installed per cluster during `setup`, so
  the worker needs outbound access to `pip_index_url`. For an air-gapped cluster, or to
  avoid the install cost (`pii_redactor` pulls presidio + flair, which is heavy), pre-bake an
  image and set `image`.
- **Container images need Pyxis on SLURM/LSF.** Leave `image` empty on clusters without the
  Pyxis SPANK plugin — including the local Docker SLURM cluster used for testing.
- **`output_path` is unchecked.** See the note under [Outputs](#outputs).
- **Flag names are transform-specific.** The step derives the module and the dependencies,
  but not `args` keys — see [Transform flags](#transform-flags).
