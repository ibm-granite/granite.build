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

All fields live under the step's `config.dpk_config`. The step runs **one DPK transform** —
you name it, and the step derives the python module and the pip dependencies.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `transform` | string | **yes** | DPK transform short name, e.g. `tokenization2arrow`, `pii_redactor`, `ededup`. See [What `transform` derives](#what-transform-derives). |
| `input` | string | **yes** | Name of a declared target `inputs:` entry. Becomes the transform's `input_folder`. |
| `output` | string | **yes** | Name of a declared target `outputs:` entry. Used as the registered artifact's ID. |
| `args` | map | no | Transform flags, rendered in order as `--<key> '<value>'`. Keys are the **full flag name** as DPK spells it, without leading dashes. See [Transform flags](#transform-flags). |
| `output_path` | string | no | Path the transform writes to. Defaults to `./output` in the step's working directory. **Set it explicitly when the output's `uri` names a path** — it must match. See [Inputs, outputs, and bundled scripts](#inputs-outputs-and-bundled-scripts). |
| `validate` | bool | no | Check the transform's output before registering it. Default `false`. See [Validating output](#validating-output). |
| `ray_enabled` | bool | no | Run on DPK's Ray runtime instead of pure python. Default `false`. See [Running on Ray](#running-on-ray). |
| `dpk_version` | string | no | DPK release to install. Default `1.1.8`. Ignored when `dpk_image` is set. |
| `dpk_image` | string | no | Container image to run in. Default `""` = the bare launcher node. See [Running in a container image](#running-in-a-container-image). |
| `packages` | list | no | Extra pip requirements installed alongside the transform, e.g. `["pyarrow"]`. |
| `pip_index_url` | string | no | Index for the pip install. Default `https://pypi.org/simple`. |
| `module` | string | no | Override the derived python module. An escape hatch — see [Running on Ray](#running-on-ray) for the case that needs it. |

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

## Running on Ray

DPK's Ray runtime is faster than the pure-python one on large corpora. `ray_enabled: true`
switches to it:

```yaml
dpk_config:
  transform: ededup
  input: docs
  output: deduped
  ray_enabled: true
```

That one flag does the **three** things Ray needs, which is why it is a single flag rather
than settings you have to keep in step — any subset of them fails on the node rather than at
submit time:

1. installs the `ray` pip extra *alongside* the transform's own;
2. points the module at `dpk_<transform>.ray.runtime`;
3. passes `--run_locally true`, so DPK starts a local Ray cluster. Its default is `false`,
   which means "connect to an existing cluster at `ray://localhost:10001`" — and this step
   provisions none, so without the flag the transform waits on a cluster nobody started.

This runs Ray's local runtime on the step's own node; provisioning a multi-node Ray cluster
is out of scope.

> **`ray_enabled` has no cluster test.** It is covered by render tests only — starting a Ray
> cluster inside the local Docker SLURM container (`RealMemory=1024`) is not something the
> test cluster can do reliably. Treat the first real Ray run as unproven.

> **Not every transform exposes `ray.runtime` in DPK 1.1.8.** 32 of the 43 data transforms
> do. The others either name their Ray entrypoint `ray.transform` (`code_profiler`,
> `doc_chunk`, `doc_quality`, `fdedup`, `hap`, `html2parquet`, `lang_id`) or ship no Ray
> package at all (`bloom`, `folder2parquet`, `web2parquet`, `similarity`). DPK is
> normalising these upstream, so the step derives by rule rather than carrying a list that
> would go stale. For one of those transforms, set `module` explicitly:
>
> ```yaml
>   transform: lang_id
>   ray_enabled: true
>   module: dpk_lang_id.ray.transform   # until DPK normalises this
> ```

## Running in a container image

By default the step runs on the bare launcher node and installs DPK at run time. Set
`dpk_image` to run inside a container instead:

```yaml
dpk_config:
  transform: tokenization2arrow
  dpk_image: "quay.io/my-org/dpk:1.1.8"
```

Two things to know:

* **Any registry works.** Give the plain reference (`quay.io/...`, `us.icr.io/...`, a private
  registry); the step adds SkyPilot's `docker:` scheme marker for you, which is what tells
  SkyPilot the value is a container rather than a cloud VM image.
* **The pip install is skipped entirely.** That is the point — a prebaked image starts
  faster. It also means **the image must already provide DPK**; `dpk_version` is ignored, so
  a base image without DPK will not work.

Container images need the Pyxis SPANK plugin on SLURM/LSF. Leave `dpk_image` empty on
clusters without it — including the local Docker SLURM cluster used for testing, which is
why image mode has no local cluster test.

## Validating output

`validate: true` runs a bundled validator against the transform's output, on the node,
**after** the transform and **before** the output is registered — so a failed check fails the
target and nothing downstream ever sees bad data.

```yaml
dpk_config:
  transform: tokenization2arrow
  input: docs
  output: tokens
  validate: true
```

**Which transforms have a validator.** The step looks for `src/validate_<transform>.py` — a
rule, not a lookup table, so a validator added for another transform works with no config
change. Today only **`tokenization2arrow`** ships one; it checks that the Arrow token stream
agrees with its `meta/` sidecars and, given the source, that every non-empty input file
produced output.

**For any other transform `validate: true` is a no-op** — the step prints

```
dpk: validate requested, but no validator for transform 'pii_redactor'
(expected ./src/validate_pii_redactor.py) — skipping
```

and carries on. It is deliberately not an error: the request is general even though the
coverage is not. It is equally deliberately not *silent* — a skipped check that said nothing
would look exactly like a passed one on a green build, so check the step log if you are
relying on validation.

**Where the report goes.** The validator writes `validation.json` into the output directory,
so the record travels with the data it describes rather than in a separate artifact that can
drift from it.

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

Value handling: every value renders as `--flag 'value'`, booleans included —
`true`/`false` become `--flag 'true'`/`--flag 'false'`. DPK declares its boolean arguments
as value-taking (`str2bool`) rather than presence flags, so a bare `--flag` would make
argparse consume the next token. Only **null** omits a flag: `false` is a real setting and
is sent. `0` is likewise sent, which matters for e.g. `tkn_chunk_size`.

### Values that vary per run

`args` is the only way to pass transform flags, so there is one quoting model to learn:
every value is quoted for you and reaches the transform byte-for-byte. That is what makes
python-literal values work — `pii_redactor`'s `--pii_redactor_entities` is
`ast.literal_eval`'d, so `"['PERSON','EMAIL_ADDRESS']"` has to arrive with its inner quotes
intact — and it means a value containing spaces or quotes can never be word-split by
accident.

The consequence: `args` values are **not** shell-expanded, so `"$MY_VAR"` reaches the
transform as the literal characters `$MY_VAR`. Parameterise the build instead — two
mechanisms, both resolved before the step runs:

**`$${PARAM}` — a build parameter** (the usual choice). Values come from a
`parameters.yaml` beside the build, and any of them can be overridden per run on the command
line:

```yaml
# parameters.yaml
TOKENIZER: "hf-internal-testing/llama-tokenizer"
DOC_COLUMN: "contents"
```

```yaml
# build.yaml
dpk_config:
  transform: tokenization2arrow
  input: docs
  output: tokens
  args:
    tkn_tokenizer: "$${TOKENIZER}"
    tkn_doc_content_column: "$${DOC_COLUMN}"
```

```
gb build start -f build.yaml --parameters-path parameters.yaml
gb build start -f build.yaml --parameters-path parameters.yaml --param TOKENIZER=bigcode/starcoder
```

**`{{ run_metadata.* }}` — a per-run value the server knows**, for things no one can supply
by hand, such as keying an output path to the run so concurrent builds do not collide:

```yaml
  output_path: "/shared/tokens/{{ run_metadata.targetsteprun_id | short_hash }}"
```

Both are better than shell expansion, not merely equivalent: the value is resolved before
the step renders, so it appears in the persisted step config and in build lineage instead of
being decided invisibly on a node. `gb build start --dry-run` prints the fully resolved
build.yaml if you want to check what a run will actually use.

If you need genuine shell logic — a computed path, a conditional, a pipeline — that is not
this step's job. Use the [`byoc`](../../byoc/skypilot/USAGE.md) step or the built-in
`command` step, either of which runs an arbitrary command alongside your DPK targets.

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
The table lists the transforms a build can actually run. Three DPK entries are deliberately
absent because they are **not installable from PyPI** — verified against the published
`data-prep-toolkit-transforms==1.1.8` wheel, which ships 44 `dpk_*` modules and declares 50
extras:

- **`c4_annotator`** and **`noop`** — neither module is in the wheel and neither is a
  declared extra, so `transform: noop` / `transform: c4_annotator` cannot resolve at all
  (`noop` is a test fixture in any case). They are not a dependency problem; there is
  nothing to install.
- **`dpk_transform_chain`** — a chaining utility, not a data transform.

## Inputs, outputs, and bundled scripts

How to declare a target's `inputs:`/`outputs:`, which URI schemes stage where, how
`output_path` must line up with an output's `uri`, and what the step ships in `src/` are all
documented in `steps/dpk/skypilot/README.md` in the granite.build repository, alongside the
step's other implementation detail.

The short version for writing a build:

* Every declared input is exported to the step as `$GB_INPUT_<name>`; `input:` names which
  one feeds the transform.
* The step registers the output for you, from `output` and the path it wrote to.
* If an output's `uri` names a path, set `output_path` to match it. On `skypilot/slurm` and
  `skypilot/lsf` a path another target reads must be on the shared filesystem
  (`env:///shared/…`); the `./output` default lives in the per-target workdir and is removed
  when that target finishes.

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

Tokenize a HuggingFace dataset and validate the result — one target, because `validate: true`
folds the check into the same step.

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
          uri: "env:///tokens"          # no path to match => default output_path
          type: dataset
      steps:
        - step_uri: space://steps/dpk
          config:
            poll_interval_seconds: 30
            dpk_config:
              transform: tokenization2arrow
              input: docs
              output: tokens
              validate: true            # runs the bundled tokenization validator
              args:
                tkn_tokenizer: hf-internal-testing/llama-tokenizer
                tkn_doc_id_column: document_id
                tkn_doc_content_column: contents
                tkn_text_lang: en
                tkn_chunk_size: 0
            compute_config:
              num_gpus_per_node: 0
              total_memory_per_node: "4Gi"
```

Switching transform touches only `dpk_config`. This one is also a single terminal target, so
`output_path` is left out and the step writes to `./output`:

```yaml
      outputs:
        clean:
          uri: "env:///tmp/dpk-clean"
          type: dataset
      steps:
        - step_uri: space://steps/dpk
          config:
            dpk_config:
              transform: pii_redactor   # -> dpk_pii_redactor.runtime + [pii-redactor]
              input: docs
              output: clean
              args:
                # literal_eval'd by the transform, so a python list literal
                pii_redactor_entities: "['PERSON','EMAIL_ADDRESS']"
                pii_redactor_operator: replace
                pii_redactor_score_threshold: 0.6
```

### When a downstream target reads the output

The examples above are terminal — nothing else consumes them — so the default `output_path`
is fine. If another target binds this output, it must live on the shared filesystem, because
the default lands in the per-target workdir that is removed when the target finishes:

```yaml
      outputs:
        tokens:
          uri: "env:///shared/dpk/tokens"
          type: dataset
      steps:
        - step_uri: space://steps/dpk
          config:
            dpk_config:
              transform: tokenization2arrow
              input: docs
              output: tokens
              output_path: /shared/dpk/tokens    # must match the uri above
```

## Notes and limitations

- **Runtime dependency install.** On the bare launcher node, dependencies are installed per
  cluster during `setup` with [`uv`](https://github.com/astral-sh/uv), so the worker needs
  outbound access to `pip_index_url`. For an air-gapped cluster, or to avoid the install cost
  (`pii_redactor` pulls presidio + flair, which is heavy), pre-bake an image and set
  `dpk_image` — see [Running in a container image](#running-in-a-container-image).
- **`output_path` is unchecked.** Nothing validates that it agrees with the output's declared
  `uri`; a mismatch fails at run time, not at submit time.
- **Flag names are transform-specific.** The step derives the module and the dependencies,
  but not `args` keys — see [Transform flags](#transform-flags).
- **One transform per step.** To chain transforms, declare one target per transform and bind
  each output to the next target's input. For work that is not a DPK transform, use
  [`byoc`](../../byoc/skypilot/USAGE.md) or the built-in `command` step.
