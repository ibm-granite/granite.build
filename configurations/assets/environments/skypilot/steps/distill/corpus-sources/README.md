# corpus-sources (SkyPilot / LSF)

Combines several raw SFT splits into one `train.jsonl` that corpus prep can read. The
`space://steps/distill/corpus-prep` step takes a single file, reads the `messages` key,
and stops after its first N kept rows, so this step renames `conversations` to
`messages`, samples each split in proportion to its size, and shuffles the result. It is
CPU-only, uses only the Python standard library, and needs no trainer source.

> **Developing or testing this step?** See `steps/distill/corpus-sources/skypilot/README.md`
> in the granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/corpus-sources` URI:

```yaml
steps:
  - step_uri: space://steps/distill/corpus-sources
```

## Config contract (`sources_config`)

All fields live under the step's `config.sources_config`.

### Required

| Field | Type | Purpose |
|---|---|---|
| `sources` | list | The raw splits, as `.jsonl` paths visible from the compute node. The default is `[]`, which fails the step with `FATAL: no sources given`. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `target_rows` | integer | Rows to keep across all splits. `0`, or anything at least the corpus size, keeps every row. Default: `0`. |
| `shuffle_seed` | integer | Seeds the per-split sampling and the final shuffle. Default: `0`. |
| `output_dir` | string | Directory `train.jsonl` is written to; created if missing. A relative path lands under `GB_BUILD_WORKDIR`. Default: `sources`. |
| `python` | string | Interpreter in the image that runs `src/build_sources.py`. Default: `/stage/.venv/bin/python`. |

> **The same `sources` list, `target_rows` and `shuffle_seed` always give the same file.**
> See [Reproducibility](#reproducibility-and-source-order) for what changes the selection.

The build also supplies `compute_config` (e.g. `{num_nodes: 1, num_cpus_per_node: 4}`)
and `launcher_config` (`image_id`, and `resources` such as `cluster`, `zone`, `memory`);
the step sets no resources of its own.

## Inputs and outputs

### Inputs

The step declares no named inputs (`inputs: {allow_unknown: true}`): its sources are a
variable-length list, so there is no fixed set of names to declare. It reads the files
named in `sources_config.sources` directly, so they must already be on a filesystem the
compute node can reach. Recipes also declare each split as an `env://` input of type
`dataset` on the target, so the corpus's provenance is recorded as artifacts; the pull is
a no-op, and the step still takes the paths from `sources`:

```yaml
inputs:
  source_general: {uri: "env:///data/general.jsonl", type: dataset}
```

### Outputs

| Output | Type | What it is |
|---|---|---|
| `corpus_source` (required) | dataset | The written `<output_dir>/train.jsonl`, one `{"messages": [...]}` row per line. |

`src/build_sources.py` registers it as its last log line:

```
GB_ARTIFACT_ID:corpus_source GB_ARTIFACT_PATH:<output_dir>/train.jsonl
```

Declare it on the target as `env://{{ binding.path }}`. The
`space://steps/distill/corpus-prep` step consumes it as its source dataset.

## Working directory and paths

The run script starts in the step's per-run workdir (`GB_BUILD_WORKDIR`, or the current
directory if that is unset). The step's `src/` is mounted at `./src` and run as
`./src/build_sources.py`. A relative `output_dir` is resolved against the workdir; an
absolute one is used as is. The step writes only `<output_dir>/train.jsonl`.

## Example build.yaml

A `sources` target feeding corpus prep:

```yaml
granite.build:
  name: corpus-sources-example
  targets:
    sources:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      outputs:
        corpus_source:
          uri: "env://{{ binding.path }}"
          type: dataset
      steps:
        - step_uri: space://steps/distill/corpus-sources
          config:
            compute_config: {num_nodes: 1, num_cpus_per_node: 4}
            launcher_config:
              resources: {cluster: "bluevela", zone: "normal", memory: 64}
            sources_config:
              sources: [/data/general.jsonl, /data/tools.jsonl, /data/rag.jsonl]
              target_rows: 100000
              shuffle_seed: 42
              output_dir: /proj/run/sources
    prep:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        source_dataset: {binding: sources.corpus_source}
      steps:
        - step_uri: space://steps/distill/corpus-prep
          # ...
```

## Notes and limitations

### Why the step exists

Three things are true at once:

1. Corpus prep takes **one** path and refuses a directory, so there is no glob or list
   form to hand it several files with.
2. The raw rows are spelled `conversations`, not `messages`. Prep reads `messages` and
   drops everything else as `no_messages`; handed the files unchanged it would drop all
   5,081,504 rows. The rename is the step's most load-bearing line.
3. The splits are very unequal — general 84.86%, tools 13.98%, rag 1.17% — and prep's
   row limit stops after N kept rows in input order. A plain concatenation plus a row
   limit would silently give a subset of pure `general`. Sampling proportionally here,
   then shuffling, is what makes a subset representative rather than just short.

### Sampling

Two passes: one counts each split's lines, one reservoir-samples each split, because the
per-split quota cannot be computed until the totals are known. Memory is bounded by the
sample, not the corpus.

- Each split's quota is its share of `target_rows`, rounded, and at least 1 row, so a
  tiny split is never dropped. The quotas need not sum to exactly `target_rows`.
- Quotas are computed from raw line counts; unusable rows are skipped during sampling,
  so a split with many of them can yield fewer rows than its quota.
- One reservoir per split keeps each split's share fixed, which a single reservoir over
  the concatenation would only achieve in expectation.

The log shows the line count and quota per split (`SOURCES count`, `SOURCES quota`), how
many rows were sampled from each (`SOURCES sampled`), and how many were renamed, already
used `messages`, or were unusable and skipped (`SOURCES renamed=… already_messages=…
bad=…`).

### Reproducibility and source order

Each split's random stream is seeded from `shuffle_seed` and the source's **file name**
(not its directory or its position in the list), and the final shuffle from
`shuffle_seed` alone. So:

- the same list, `target_rows` and `shuffle_seed` always give the same file;
- reordering `sources` keeps the same rows but changes their order in `train.jsonl`,
  because the final shuffle permutes the splits' concatenation in list order;
- two sources with the same file name in different directories share a random stream.

### Failures

The step fails on:

- no sources given, a source that is not a file, or every source empty;
- no row anywhere carrying `conversations` or `messages`. That means the source schema
  has changed, and prep would otherwise drop every row.

Unusable individual rows (invalid JSON, or neither key) are skipped and counted in a
`SOURCES WARNING unusable rows skipped: N` line, not fatal.

### No `code_config`

The step needs no trainer source, so it carries none of the other distillation steps'
`code_config` / source-delivery contract. The sampling method, and the measurements
behind it, are documented in `src/build_sources.py`.
