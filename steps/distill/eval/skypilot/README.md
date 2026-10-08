# eval — development notes

User-facing documentation is [USAGE.md](USAGE.md), which `make publish-step` copies to the
released step as its `README.md`.

## What is ours and what is upstream

| | Where | Change it where? |
|---|---|---|
| `step-template.yaml` | here — but see the source-delivery note | here |
| `src/run-eval.sh` | here, a near-verbatim port | **upstream**, then re-sync |
| `test/test_step_template.py` | here | here |
| `test/test_divergence.py` | here, a near-verbatim port | **upstream**, then re-sync |
| `gb_steps_post_training.distillation` (`run_divergence`, `eval_state`) | the delivered checkout | **upstream only** |

## One upstream bug the port fixed

Upstream printed the `eval_metrics` marker **twice**: once from `run-eval.sh`'s
`publish_artifacts()` (called on both the success and SKIP paths) and again from the
step-template after the script returned. That would register two `NEWARTIFACT` events for one
id. The port keeps the script's marker — it is the one that also fires on the resume path — and
drops the template's. Tests assert both halves: the script prints it, the template does not.

## The monitor is overridden

`log_retrieval.mode` is `periodic`, not the library default `on_completion`, for the same reason
`distill-gold` overrides it: a teacher forward pass over 256 samples is not a seconds-long job,
and `on_completion` surfaces nothing until the end — so a stalled run looks identical to a slow
one. Both intervals stay recipe-overridable.

## The source-delivery region is spliced, not written

`code_config` and the delimited region at the top of `run:` are **byte-identical copies** from
`steps/distill/tokenizer-align`, the reference step. Do not edit them here — change the
reference and re-splice, or that step's `test_source_contract.py` fails and names this one.

## Tests

```bash
make test                                                   # 31 contract tests, no checkout needed
GB_DISTILL_CODE_DIR=/path/to/checkout make test              # + 21 ported (10 skip without torch)
GB_STEP_BLUEVELA_BUILD=1 make test                           # + the end-to-end run on BlueVela
```

Ten of the ported tests need torch and skip without it. That is upstream's design, not an
oversight: the metric arithmetic is worth testing against real tensors, and the image that has
torch is the cluster's, not CI's.

The end-to-end test (`test/lsf/test_skypilot_lsf_distill_eval.py`, fixtures in
`test-data/lsf/`) is a three-target chain (align -> corpus -> eval), because the student has to
be a retagged model whose tokenizer agrees with the teacher's and the corpus has to be an eval
split. It asks for one H100 and eight samples, so it costs minutes. Without
`GB_STEP_BLUEVELA_BUILD=1`, or when the BlueVela login node is unreachable, it skips.

## Publishing

```bash
make space && make test && make publish-step && make check-published
```
