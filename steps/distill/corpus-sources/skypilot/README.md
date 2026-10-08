# corpus-sources

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

Builds one prep-ready `train.jsonl` from several raw SFT splits, for
the `corpus-prep` step to consume. It does three things, and the rationale for each is
in [`src/build_sources.py`](src/build_sources.py):

1. renames `conversations` to `messages`, the key prep reads;
2. reservoir-samples each split in proportion to its share of the corpus, so a subset
   keeps every split rather than only the largest;
3. shuffles the result, so an early truncation downstream still sees every split.

## Config (`sources_config`)

| key | meaning |
|---|---|
| `sources` | list of `.jsonl` paths visible from the compute node; each split's RNG stream is seeded from `shuffle_seed` and the file name, and list order sets the order the final shuffle starts from |
| `target_rows` | rows to keep in total; `0` keeps every row |
| `shuffle_seed` | seed for the per-split reservoirs and the final shuffle |
| `output_dir` | where `train.jsonl` lands; relative paths go under `GB_BUILD_WORKDIR` |
| `python` | interpreter in the image (`/stage/.venv/bin/python`) |

Output: `corpus_source`, the written `train.jsonl`.

## Example

```yaml
sources:
  outputs:
    corpus_source:
      uri: "env://{{ binding.path }}"
      type: dataset
  steps:
    - step_uri: space://steps/distill/corpus-sources
      config:
        sources_config:
          sources: [/data/general.jsonl, /data/tools.jsonl, /data/rag.jsonl]
          target_rows: 100000
          shuffle_seed: 42
          output_dir: /proj/run/sources
```

Stdlib-only, CPU-only, and no trainer source, so it does not carry the other distillation steps'
`code_config` contract.
