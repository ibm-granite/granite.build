# corpus-pin-check

Refuses a pinned corpus whose `corpus_manifest.json` does not describe this build,
before any GPU allocation is held. The comparisons, and why each one exists, are in
[`src/check_corpus_pin.py`](src/check_corpus_pin.py):

- `train.jsonl` and `eval.jsonl` are both present;
- the manifest's `tokenizer_identity` is the basename of this build's teacher;
- `max_length`, `think_policy`, `documents_policy` and `completion_boundary`
  (always `last_message`) match;
- `eval_fraction` matches;
- when the manifest's `tokenizer_path` still exists, `tokenizer.json` and
  `chat_template.jinja` are byte-identical to this build's retagged tokenizer.

Every mismatch is reported in one run. On success it writes `corpus_pin.json` and
registers it as `pin_check`. Every consumer of the pinned corpus should bind that
output, which makes the check an ordering edge rather than a suggestion.

## Config (`pin_check_config`)

`corpus_dir`, `teacher_model`, `max_length`, `think_policy`, `documents_policy`,
`eval_fraction`, `tokenizer_dir`, `output_dir` (relative paths land under
`GB_BUILD_WORKDIR`) and `python`.

Stdlib-only, CPU-only, and no trainer source, so it does not carry the distill-* steps'
`code_config` contract.
