# corpus-pin-check (SkyPilot / LSF)

Checks that a pinned corpus, one built by an earlier run and reused by this one, was
built the way this build would have built it. It reads the corpus's
`corpus_manifest.json`, compares it against this build's teacher, prep policies and
retagged tokenizer, and fails on any mismatch. It runs before any GPU allocation is held,
so a stale or mismatched corpus fails in seconds rather than after a training run.
CPU-only, standard library only, and it needs no trainer source.

> **Developing or testing this step?** See `steps/distill/corpus-pin-check/skypilot/README.md`
> in the granite.build repository for how the step is generated, tested, and published.

## Referencing the step

Point your build's Space at one that provides the step, then reference it by the stable
`space://steps/distill/corpus-pin-check` URI:

```yaml
steps:
  - step_uri: space://steps/distill/corpus-pin-check
```

## Config contract (`pin_check_config`)

All fields live under the step's `config.pin_check_config`. Use the same values the
recipe would pass to corpus prep: a pin check fed different values from the prep it
stands in for checks the wrong thing.

### Required

| Field | Type | Purpose |
|---|---|---|
| `corpus_dir` | string | The pinned corpus directory, holding `corpus_manifest.json`, `train.jsonl` and (when the corpus has an eval split) `eval.jsonl`. A directory with no `corpus_manifest.json` fails immediately. |
| `teacher_model` | string | The teacher this build distils from. Its basename (trailing `/` ignored) must equal the manifest's `tokenizer_identity`. |
| `max_length` | integer | Must equal the manifest's `policies.max_length`. The default `0` never matches a real corpus. |
| `think_policy` | string | Must equal the manifest's `policies.think_policy`. |
| `documents_policy` | string | Must equal the manifest's `policies.documents_policy`. |
| `eval_fraction` | number | Must equal the manifest's `eval_fraction`, and decides whether `eval.jsonl` must exist. The default `0.0` matches only a corpus prepped without an eval split. |
| `tokenizer_dir` | string | This build's retagged tokenizer, byte-compared against the manifest's `tokenizer_path` when that directory still exists. |

### Optional

| Field | Type | Purpose |
|---|---|---|
| `output_dir` | string | Where `corpus_pin.json` is written. A relative path lands under `$GB_BUILD_WORKDIR`. Default: `corpus-pin`. |
| `python` | string | The interpreter in the image. Default: `/stage/.venv/bin/python`. |

## Inputs and outputs

### Inputs

The step declares no `inputs:` of its own; it reads `corpus_dir` and `tokenizer_dir`
from config. The shipped recipes declare two target inputs anyway:

- `pinned_corpus` — a direct `uri:` to the corpus's `train.jsonl` (`type: dataset`), so the
  pinned corpus appears in the build's lineage. When a recipe skips corpus prep, this is the
  only place it is named as an input.
- `tokenizer` — `binding: align.retagged_student`, the output of the
  `space://steps/distill/tokenizer-align` step. It is passed in as
  `tokenizer_dir: "{{ bindings.tokenizer.binding.path }}"`, and the binding is the ordering
  edge that guarantees align has run before the comparison is made.

### Outputs

| Output | Type | What it is |
|---|---|---|
| `pin_check` | `fileset` | `<output_dir>/corpus_pin.json`, written only on acceptance. It records the corpus directory, `accepted: true`, which tokenizer check ran, and the manifest's tokenizer identity, policies, eval fraction, seed, counts and splits. |

`src/check_corpus_pin.py` registers it with:

```
GB_ARTIFACT_ID:pin_check GB_ARTIFACT_PATH:<output_dir>/corpus_pin.json
```

Bind `pin_check` into every target that reads the pinned corpus (in the shipped recipes:
the SFT and GOLD training targets and the baseline eval). That makes the check an
ordering edge rather than a suggestion: nothing can train on the corpus before it is
accepted.

## Working directory and paths

The run block sets `WORK` to `$GB_BUILD_WORKDIR` (falling back to the current directory).
A relative `output_dir` becomes `$WORK/<output_dir>`; it is created before the check runs.
`src/` is file-mounted beside the step and the workload runs
`<python> ./src/check_corpus_pin.py` with the config values as positional arguments.
`corpus_dir`, `teacher_model` and `tokenizer_dir` are used as given, so pass absolute
paths.

## Example build.yaml

```yaml
granite.build:
  name: corpus-pin-check-example
  version: 0.0.1
  targets:
    corpus-pin-check:
      environment_uri: space://environments/skypilot/lsf/ibm-bluevela
      inputs:
        pinned_corpus:
          uri: "env:///proj/run/corpus/train.jsonl"
          type: dataset
        tokenizer:
          binding: align.retagged_student
      outputs:
        pin_check:
          uri: "env://{{ binding.path }}"
          type: fileset
      steps:
        - step_uri: space://steps/distill/corpus-pin-check
          config:
            compute_config: {num_nodes: 1, num_cpus_per_node: 4}
            launcher_config:
              resources: {cluster: "bluevela", zone: "normal", memory: 32}
            pin_check_config:
              corpus_dir: /proj/run/corpus
              teacher_model: /proj/models/granite-4.1-3b-pinned
              max_length: 4096
              think_policy: keep
              documents_policy: keep
              eval_fraction: 0.005
              tokenizer_dir: "{{ bindings.tokenizer.binding.path }}"
              output_dir: /proj/run/corpus-pin
    train:
      inputs:
        pin_check: {binding: corpus-pin-check.pin_check}
      # ...
```

The `align` target is a `space://steps/distill/tokenizer-align` target defined elsewhere
in the same build.

## Notes and limitations

### When to use it

When a recipe skips corpus prep and reads an existing corpus directory instead. The
corpus is defined by the retagged tokenizer and by the prep policies: the rows were
rendered through that chat template, the assistant masks were built with that tokenizer,
and rows (5,908 in the reference corpus) were dropped for exceeding that `max_length`. A pin taken from a run that
differed in any of those trains on a corpus this build does not describe, and every
metric downstream would still look normal.

### What it checks

- `train.jsonl` is present, and so is `eval.jsonl` when `eval_fraction > 0`. Corpus prep
  writes `eval.jsonl` only when `eval_fraction > 0`, so without an eval split its absence
  is the consistent state.
- The manifest's `tokenizer_identity` is the basename of `teacher_model`.
- `max_length`, `think_policy` and `documents_policy` match the manifest's `policies`, and
  `policies.completion_boundary` is `last_message`. That one is not a parameter: the
  recipes hard-code `last_message` because the eval measures divergence over the final
  assistant turn.
- `eval_fraction` matches the manifest's (compared as a float).
- When the manifest's `tokenizer_path` still exists, its `tokenizer.json` and
  `chat_template.jinja` are byte-identical (SHA-256) to the ones in `tokenizer_dir`; a file
  missing on either side is a mismatch. When `tokenizer_path` no longer exists, that
  comparison is skipped — a pin should outlive the build that produced it — and
  `corpus_pin.json` records `identity-only (pinned tokenizer_path no longer on disk)`
  instead of `byte-compared against the pinned tokenizer_path`.

The reasoning behind each comparison is in `src/check_corpus_pin.py`.

### Failures

Every mismatch is reported in one run, under a `CORPUS-PIN REJECTED <corpus_dir>` line,
and the target fails with `FATAL [corpus-pin-check]: N mismatch(es) between this build
and the pinned corpus manifest`. A directory with no `corpus_manifest.json` fails
immediately with `FATAL: no corpus_manifest.json under <corpus_dir>`. On acceptance the
log shows `CORPUS-PIN ACCEPTED <corpus_dir> (<mode>)`.

### No source delivery

The step is stdlib-only and CPU-only, and needs no trainer source, so it does not carry
the `code_config` source-delivery contract the other ported distillation steps share.
