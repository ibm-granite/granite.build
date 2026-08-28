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

## Two modes: pick one

The step has **two mutually exclusive modes** — set `transform` *or* `command`, never both.
Read this before the field tables; nearly everything else here applies to one mode only.

| | **Transform mode** (`transform:`) | **Command mode** (`command:`) |
|---|---|---|
| **Use it when** | the work *is* a DPK transform | the work is not a DPK transform — a script, a one-off, a validator |
| **Who builds the command line** | the step (from `transform` + `args`) | you, verbatim |
| **Dependencies** | derived from `transform` (+ `packages`) | `packages` only |
| **Artifact marker** | emitted for you, from `output`/`output_path` | your command prints it |

Command mode exists because a build often needs a non-transform step *beside* its
transforms — the shipped `src/validate_tokens.py` is exactly that, and it has no
`dpk_<name>.runtime` to invoke. Without it you would need a second step (or `byoc`) just to
run a script next to the transform, so the escape hatch is deliberate rather than
redundant. It behaves like [`byoc`](../../byoc/skypilot/USAGE.md)'s `command`.

## Config contract (`dpk_config`)

All fields live under the step's `config.dpk_config`.

### Transform mode

| Field | Type | Required | Purpose |
|---|---|---|---|
| `transform` | string | **yes** | DPK transform short name, e.g. `tokenization2arrow`, `pii_redactor`, `ededup`. See [What `transform` derives](#what-transform-derives). |
| `input` | string | **yes** | Name of a declared target `inputs:` entry. Becomes the transform's `input_folder`. |
| `output` | string | **yes** | Name of a declared target `outputs:` entry. Used as the registered artifact's ID. |
| `output_path` | string | no | Path the transform writes to. Defaults to `./output` in the step's working directory. **Set it explicitly when the output's `uri` names a path** — it must match. See [Outputs](#outputs). |
| `args` | map | no | Transform flags, rendered in order as `--<key> '<value>'`. Keys are the **full flag name** as DPK spells it, without leading dashes. See [Transform flags](#transform-flags). |
| `extra_args` | string | no | Flags appended **verbatim** to the transform's argv, after `args`. The escape hatch for anything `args` cannot express. See [Transform flags](#transform-flags). |
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
[the transform's DPK documentation](#per-transform-dpk-documentation) or from
`python -m dpk_<name>.runtime --help`, which is authoritative when the two disagree.

Value handling: `true` renders a bare `--flag`; `false` and null are omitted; everything
else renders as `--flag 'value'` (including `0`, which is meaningful for e.g.
`tkn_chunk_size`).

### `args` vs. `extra_args`

`args` quotes each value so it reaches the transform **byte-for-byte**. That is what you
want almost always, and it is what makes python-literal values work: `pii_redactor`'s
`--pii_redactor_entities` is `ast.literal_eval`'d, so `"['PERSON','EMAIL_ADDRESS']"` has to
arrive with its inner quotes intact.

`extra_args` is a single string appended verbatim after everything `args` rendered, and it
is **not** quoted — the remote shell word-splits and expands it. Use it when you *need* that:

```yaml
dpk_config:
  args:
    tkn_doc_id_column: document_id       # quoted for you
  extra_args: "--tkn_tokenizer $MY_TOKENIZER"   # expanded on the node
```

Because it is unquoted, correct quoting of any value containing spaces or quotes is yours to
get right — which is why `args` remains the default rather than a raw string being the only
option. Ignored in command mode, where you write the whole command anyway.

## Per-transform DPK documentation

The step derives the module and dependencies, but a transform's **flag names come from DPK**.
Each transform's docs live in the DPK repo, pinned below to the `v1.1.8` tag — the release
`dpk_version` installs by default, so the flags match the code you are running. Change the
tag in the URL if you set a different `dpk_version`.

The paths are not derivable from the transform name (the category is not encoded in it, and
`tokenization2arrow` shares a directory with `tokenization`), hence the table.

| Transform | Docs |
|---|---|
| `blocklist` | [universal/blocklist](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/blocklist/README.md) |
| `bloom` | [universal/bloom](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/bloom/README.md) |
| `c4_annotator` | [universal/c4_annotator](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/c4_annotator/README.md) |
| `code2parquet` | [code/code2parquet](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/code2parquet/README.md) |
| `code_profiler` | [code/code_profiler](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/code_profiler/README.md) |
| `code_quality` | [code/code_quality](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/code_quality/README.md) |
| `collapse` | [universal/collapse](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/collapse/README.md) |
| `doc_chunk` | [language/doc_chunk](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/doc_chunk/README.md) |
| `doc_id` | [universal/doc_id](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/doc_id/README.md) |
| `doc_quality` | [language/doc_quality](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/doc_quality/README.md) |
| `docling2parquet` | [language/docling2parquet](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/docling2parquet/README.md) |
| `ededup` | [universal/ededup](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/ededup/README.md) |
| `enrichment` | [language/enrichment](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/enrichment/README.md) |
| `extreme_tokenized` | [language/extreme_tokenized](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/extreme_tokenized/README.md) |
| `faces` | [images](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/images/README.md) (shared) |
| `fdedup` | [universal/fdedup](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/fdedup/README.md) |
| `filter` | [universal/filter](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/filter/README.md) |
| `fineweb_quality_annotator` | [universal/fineweb_quality_annotator](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/fineweb_quality_annotator/README.md) |
| `folder2parquet` | [universal/folder2parquet](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/folder2parquet/README.md) |
| `gneissweb_classification` | [language/gneissweb_classification](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/gneissweb_classification/README.md) |
| `gopher_repetition_annotator` | [universal/gopher_repetition_annotator](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/gopher_repetition_annotator/README.md) |
| `hap` | [universal/hap](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/hap/README.md) |
| `header_cleanser` | [code/header_cleanser](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/header_cleanser/README.md) |
| `html2parquet` | [language/html2parquet](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/html2parquet/README.md) |
| `lang_id` | [language/lang_id](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/lang_id/README.md) |
| `license_select` | [code/license_select](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/license_select/README.md) |
| `malware` | [code/malware](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/malware/README.md) |
| `ml_filter` | [language/ml_filter](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/ml_filter/README.md) |
| `nsfw` | [images](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/images/README.md) (shared) |
| `opensearch` | [universal/opensearch](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/opensearch/README.md) |
| `people` | [images](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/images/README.md) (shared) |
| `pii_redactor` | [language/pii_redactor](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/pii_redactor/README.md) |
| `profiler` | [universal/profiler](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/profiler/README.md) |
| `proglang_select` | [code/proglang_select](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/proglang_select/README.md) |
| `readability` | [language/readability](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/readability/README.md) |
| `rep_removal` | [universal/rep_removal](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/rep_removal/README.md) |
| `repo_level_order` | [code/repo_level_order](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/repo_level_order/README.md) |
| `resize` | [universal/resize](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/resize/README.md) |
| `similarity` | [language/similarity](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/similarity/README.md) |
| `text_encoder` | [language/text_encoder](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/language/text_encoder/README.md) |
| `tokenization` | [universal/tokenization](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/tokenization/README.md) |
| `tokenization2arrow` | [universal/tokenization (README-tkn2arrow.md)](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/tokenization/README-tkn2arrow.md) |
| `web2parquet` | [universal/web2parquet](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/universal/web2parquet/README.md) |
| `yara` | [code/yara](https://github.com/data-prep-kit/data-prep-kit/blob/v1.1.8/transforms/code/yara/README.md) |

Notes on the irregular entries, all verified against the tag:

- **`tokenization2arrow`** is documented in `README-tkn2arrow.md`, not the directory's
  `README.md` (which covers the older `tokenization`). The two transforms share a directory
  and a pip extra, but are different modules.
- **`faces` / `nsfw` / `people`** have no per-transform README; the shared
  `transforms/images/README.md` documents all three.
- **`c4_annotator`** declares no pip extra, so it needs `no_extras: true`.

The table lists the transforms a build would plausibly run. DPK also ships `noop` (a
no-op test fixture) and `dpk_transform_chain` (a chaining utility, not a data transform);
both work if you name them, but neither is documented here.

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

- **Transform mode** — the step creates the directory, points the transform's
  `output_folder` at it, and emits the artifact marker for you. `output_path` defaults to
  `./output` in the step's working directory; set it explicitly when the output's `uri` names
  a path, or when a **downstream target** reads the output — see
  [When the default is not enough](#when-the-default-is-not-enough).
- **Command mode** — your command writes wherever it likes and prints the marker itself:

  ```
  LLMB_ARTIFACT_ID:<output-id> LLMB_ARTIFACT_PATH:<abs-path>
  ```

  `<output-id>` must match a declared `outputs.<id>`. Repeat the line to register several
  artifacts under one output. For `mem://` outputs use `LLMB_ARTIFACT_PATH` →
  `LLMB_ARTIFACT_STATE:<value>`.

#### When the default is not enough

`output_path: ""` (the default) writes to `./output` in the step's working directory, and the
step absolutizes it before emitting the marker. That is right for two common cases:

- a **terminal** output nothing downstream consumes;
- an output an **assetstore pushes** (e.g. `s3://…`), where the local dir is only a staging
  area.

Set it explicitly in either of these cases:

1. **The output's `uri` names a path.** The step cannot derive it — declared output URIs are
   not in the runtime render context — so keeping `uri` and `output_path` in agreement is
   the build author's job. A mismatch fails at run time, not at submit time.
2. **A downstream target reads the output.** The working directory is the *per-run* workdir
   (`${shared_workdir}/builds/<build_id>/runs/<targetrun_id>`), which is keyed **per target**
   and removed when that target finishes. A consumer in another target would read a deleted
   directory, so give the output an explicit shared path instead — see
   [Choosing the output URI per endpoint](#choosing-the-output-uri-per-endpoint).

> **Why is there no `input_path`?** Because inputs and outputs reach the step differently, and
> this is the asymmetry the default only partly hides. An input is **staged before `run`** and
> its resolved path is handed to the step as `$LLMB_INPUT_<name>`, so there is nothing to
> specify. A declared **output does not exist yet** at render time: the runtime context is
> `bindings` + `run_metadata` + `setup_config` only, and declared output URIs reach *static
> validation* alone. Plumbing them into the runtime context would let the step derive
> `output_path` and drop the field — a gbserver change, tracked separately.

### Choosing the output URI per endpoint

`validate`-style downstream targets read what an upstream target wrote, and the two may run
on **different nodes**. What works depends on the endpoint:

| Endpoint | Output URI | Why |
|---|---|---|
| `skypilot/slurm`, `skypilot/lsf` | `env:///shared/…` + an **explicit** matching `output_path` | `/shared` is the environment's `shared_workdir`, mounted on every node — cross-node safe. The `./output` default will not do here: it lives in the per-target workdir, which is deleted at target teardown. |
| `skypilot/aws`, `skypilot/kubernetes` | `s3://bucket/…`, `output_path` may be **left default** | Each target gets its own instance with no shared filesystem. The S3 assetstore pushes the staged dir and the consumer pulls it, so a node-local `./output` is fine. |

A node-local `env:///tmp/…` is **not** safe for a cross-target handoff: if the two targets
land on different nodes the consumer reads an absent directory.

## Working directory and paths

Both `setup` and `run` start in the same **working directory** (the step's per-run workdir),
so the step never needs its absolute location. The bundled `src/` is mounted at `./src`, the
default `output_path` writes to `./output`, and in bare-node mode the virtualenv is created at
`./venv` and activated for you. Use relative paths from there; derive an absolute one at run
time with `$(pwd)` when a marker needs it.

That workdir is `${shared_workdir}/builds/<build_id>/runs/<targetrun_id>` where the
environment configures a `shared_workdir` (slurm, lsf), and SkyPilot's own `~/sky_workdir`
where it does not (aws, kubernetes). Either way it is **per target** and removed when the
target finishes — which is why anything a *downstream* target reads needs an explicit path
outside it.

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

Switching to a different transform touches only `dpk_config`. This one is a **single terminal
target**, so `output_path` is left out and the step writes to `./output`:

```yaml
      outputs:
        clean:
          uri: "env:///tmp/dpk-clean"              # no path to match => default is fine
          type: dataset
      steps:
        - step_uri: space://steps/dpk
          config:
            dpk_config:
              transform: pii_redactor              # -> dpk_pii_redactor.runtime + [pii-redactor]
              input: docs
              output: clean
              # output_path omitted -> ./output, absolutized in the marker
              args:
                # literal_eval'd by the transform, so a python list literal
                pii_redactor_entities: "['PERSON','EMAIL_ADDRESS']"
                pii_redactor_operator: replace
                pii_redactor_score_threshold: 0.6
```

Contrast the `tokenize` target above, which **must** set `output_path`: its `tokens` output is
read by the `validate` target, so it has to live on the shared filesystem rather than in the
per-target workdir.

## Bundled scripts (`src/`)

The step ships `src/` to `./src` on the node. Two of the three scripts are the step's own
machinery — you do not invoke them, but they are where the step's shell actually lives:

- `src/dpk_setup.sh` — the bare-node dependency install (`uv venv` + `uv pip install`),
  invoked from the step's `setup` phase. Skipped entirely when `image` is set.
- `src/dpk_run.sh` — the transform-mode invocation: creates and absolutizes the output
  directory, builds DPK's `--data_local_config`, runs `python -m <module>`, and emits the
  artifact marker. Not used in command mode, where your `command` runs instead.

The step.yaml computes the *values* (module, requirements, flags) and passes them to these
as arguments; the scripts do the shell. That keeps the shell in real files — checkable with
`shellcheck` and testable directly — rather than embedded in YAML behind Jinja.

The remaining script is a helper you *do* call, from a command-mode `command`:

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
- **Runtime dependency install.** Dependencies are installed per cluster during `setup` with
  [`uv`](https://github.com/astral-sh/uv) (bootstrapped with `pip` first, as DPK's own
  image does), so
  the worker needs outbound access to `pip_index_url`. For an air-gapped cluster, or to
  avoid the install cost (`pii_redactor` pulls presidio + flair, which is heavy), pre-bake an
  image and set `image`.
- **Container images need Pyxis on SLURM/LSF.** Leave `image` empty on clusters without the
  Pyxis SPANK plugin — including the local Docker SLURM cluster used for testing.
- **`output_path` is unchecked.** See the note under [Outputs](#outputs).
- **Flag names are transform-specific.** The step derives the module and the dependencies,
  but not `args` keys — see [Transform flags](#transform-flags).
