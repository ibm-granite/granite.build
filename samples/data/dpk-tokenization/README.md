# DPK tokenization sample input

Five small Parquet files used as the input artifact for the
[`DPK_Tokenize_Skypilot`](../../../configurations/assets/templates/DPK_Tokenize_Skypilot/)
template. Total size ~44 KB, so it is committed directly rather than fetched at
build time — the template runs offline apart from the tokenizer download.

## Provenance

Copied verbatim from the [Data Prep Kit](https://github.com/data-prep-kit/data-prep-kit)
project (Apache-2.0), release 1.1.8:

```
transforms/universal/tokenization/test-data/tkn2arrow-ds01/input/
```

These are DPK's own test fixtures for the `tokenization2arrow` transform, which
is why they exercise the interesting edge cases rather than just the happy path.
Attribution is recorded in the repo [`NOTICE`](../../../NOTICE).

## Layout and schema

```
input/
└── lang=en/
    ├── pq01.parquet                                    3 rows
    ├── pq02.parquet                                    3 rows
    ├── dataset=cybersecurity_v2.0/version=2.3.2/
    │   └── pq03.snappy.parquet                         non-empty, snappy
    └── dataset=empty/
        ├── dpv08_cc01.snappy.parquet                   0 rows
        └── dpv08_cc02.snappy.parquet                   0 rows
```

Every file has two columns:

| Column | Type | Description |
|---|---|---|
| `document_id` | string | Unique document identifier (e.g. `d01`). Must be unique across the dataset. |
| `contents` | string | The document text to tokenize. |

The column names are passed to the transform as `--tkn_doc_id_column` and
`--tkn_doc_content_column`; both are parameterized in the template.

The two `dataset=empty/` files are deliberate: the transform skips empty tables
and reports them as `skipped empty tables` in `metadata.json`, so this input
covers that path. The nested Hive-style partition directories also confirm the
transform preserves input directory structure in its output.

## Expected output

Tokenizing with `hf-internal-testing/llama-tokenizer` (the template default)
yields 3 `.arrow` files plus a `meta/` tree, and these `metadata.json` stats:

| Stat | Value |
|---|---|
| `source_files` | 5 |
| `result_files` | 3 |
| `skipped empty tables` | 2 |
| `num_rows` | 6 |
| `num_tokens` | 85 |

These are the values DPK's own `expected/` fixtures assert, so they are a useful
correctness check when changing the template — see the template README's
"Verifying the output" section.
