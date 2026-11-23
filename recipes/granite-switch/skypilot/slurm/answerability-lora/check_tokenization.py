"""Check that the prepared prompt/completion data tokenizes to what was trained.

The source parquet (training_data/answerability/parquet_granite4.2_8b in the HF
dataset repo named by SOURCE_DATA_URI) carries pre-tokenized `input_ids` / `labels`
next to the `prompt` / `completion` they were tokenized from. The prepare
target keeps only `prompt` / `completion`, and hf-peft-sft re-tokenizes them
with TRL's prompt-completion path. This replays that path (the same steps as
trl 1.17.1 SFTTrainer's tokenize_fn: chat template on the prompt with
add_generation_prompt, on prompt + completion without, the completion starting
where the two diverge) on a sample of rows and compares them with the stored
columns.

    python -I check_tokenization.py <tokenizer dir or HF id> <source parquet> [N]

Needs transformers + jinja2 + pyarrow (CPU only; no model weights). Get the
source parquet with:

    hf download <org>/<dataset> --repo-type dataset \
      --include 'training_data/answerability/parquet_granite4.2_8b/*' --local-dir .
"""

import sys

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _ids(x):
    return list(x["input_ids"] if hasattr(x, "keys") else x)


def replay(tok, prompt, completion):
    """(input_ids, labels) as TRL's prompt-completion tokenization builds them."""
    prompt_ids = _ids(tok.apply_chat_template(prompt, add_generation_prompt=True))
    full_ids = _ids(tok.apply_chat_template(prompt + completion))
    n = 0
    while n < min(len(prompt_ids), len(full_ids)) and prompt_ids[n] == full_ids[n]:
        n += 1
    return full_ids, [-100] * n + full_ids[n:]


def main(tokenizer, parquet, n=200):
    tok = AutoTokenizer.from_pretrained(tokenizer)
    table = pq.read_table(
        parquet, columns=["input_ids", "labels", "prompt", "completion"]
    )
    n = min(n, table.num_rows)
    rows = table.slice(0, n).to_pylist()
    ids_ok = labels_ok = 0
    first_bad = None
    for i, row in enumerate(rows):
        ids, labels = replay(tok, row["prompt"], row["completion"])
        stored_ids, stored_labels = row["input_ids"], row["labels"]
        # The stored rows are right-padded per transform batch; compare the
        # unpadded prefix.
        n_ids = len(ids)
        a = stored_ids[:n_ids] == ids
        b = stored_labels[:n_ids] == labels and all(
            t == -100 for t in stored_labels[n_ids:]
        )
        ids_ok += a
        labels_ok += b
        if not (a and b) and first_bad is None:
            first_bad = (i, ids, labels, stored_ids, stored_labels)
    print(f"rows checked: {n}")
    print(f"input_ids match: {ids_ok}/{n}")
    print(f"labels match:    {labels_ok}/{n}")
    _, labels0 = replay(tok, rows[0]["prompt"], rows[0]["completion"])
    print(f"row 0 trained span: {tok.decode([t for t in labels0 if t != -100])!r}")
    if first_bad:
        i, ids, labels, sids, slabels = first_bad
        print(f"first mismatch at row {i}:")
        print("  replay trained:", repr(tok.decode([t for t in labels if t != -100])))
        print("  stored trained:", repr(tok.decode([t for t in slabels if t != -100])))
        print("  replay len", len(ids), "stored len", len(sids))
    return 0 if ids_ok == labels_ok == n else 1


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:])))
