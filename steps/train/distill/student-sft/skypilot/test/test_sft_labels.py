# SPDX-License-Identifier: Apache-2.0
"""student-sft's labels under trl 1.x are the labels trl 0.26.2 trained on.

trl 0.26.2 built the labels in its collator, from the `assistant_masks` (always) and
`completion_mask` (with completion_only_loss) columns our map chain writes
(trl/trainer/sft_trainer.py:213-222 at 0.26.2). trl 1.x builds them in
SFTTrainer._prepare_dataset, which CustomSFTTrainer replaces, and its collator ignores the
masks (sft_trainer.py:173 at 1.18.2): without sft.py's `build_labels` map, the loss would
cover the prompt. These tests run the real map chain on a tiny offline tokenizer and compare
the trl 1.x collator's labels with the 0.26.2 rule, re-implemented below from those lines.

Needs torch, transformers, tokenizers, datasets and trl, so it runs in the step image and
skips elsewhere.
"""

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("tokenizers")

RESPONSE_TEMPLATE = "<|im_start|>assistant\n"
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="module")
def sft():
    # sft.py loads the Mamba2 hub kernels at import; the label path needs none of them.
    from gb_steps_post_training.distillation import hub_kernel_pins

    real = hub_kernel_pins.prepare_hub_kernels
    hub_kernel_pins.prepare_hub_kernels = lambda: None
    try:
        from gb_steps_post_training.distillation import sft as module
    finally:
        hub_kernel_pins.prepare_hub_kernels = real
    return module


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<pad>", "<|im_start|>", "<|im_end|>", "\n", " ", "user", "assistant"]
    words += ["hi", "there", "hello", "world", "again", "[UNK]"]
    tok = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\s"), behavior="isolated")
    tok.add_special_tokens(
        [AddedToken(t, special=True) for t in ("<pad>", "<|im_start|>", "<|im_end|>")]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="<pad>",
        eos_token="<|im_end|>",
    )
    fast.chat_template = CHAT_TEMPLATE
    return fast


def _args():
    return types.SimpleNamespace(
        assistant_only_loss=False,
        dataset_num_proc=None,
        dataset_text_field="text",
        max_length=64,
        packing_strategy=None,
        seed=0,
        shuffle_dataset=False,
        use_liger_kernel=False,
    )


def _prepare(sft, tokenizer, rows, completion_only_loss=True):
    from datasets import Dataset

    trainer = types.SimpleNamespace(
        response_template=RESPONSE_TEMPLATE,
        precomputed_logits_dir="",
        _is_vlm=False,
        completion_only_loss=completion_only_loss,
    )
    return sft.CustomSFTTrainer._run_map_chain(
        trainer, Dataset.from_list(rows), tokenizer, _args(), False, None, "train"
    )


def _labels_trl_0_26_2(example, completion_only_loss):
    """trl 0.26.2's collator rule (sft_trainer.py:213-222), one example, no padding."""
    labels = list(example["input_ids"])
    if completion_only_loss and "completion_mask" in example:
        labels = [l if m else -100 for l, m in zip(labels, example["completion_mask"])]
    if "assistant_masks" in example:
        labels = [l if m else -100 for l, m in zip(labels, example["assistant_masks"])]
    return labels


def _collate_trl_1(tokenizer, examples):
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    # The collator sees only the columns that survive signature pruning (input_ids, labels).
    return collator([{k: e[k] for k in ("input_ids", "labels")} for e in examples])[
        "labels"
    ]


MESSAGES = [
    {
        "messages": [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello world"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "hello"},
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "world"},
        ]
    },
]
PROMPT_COMPLETION = [
    {
        "prompt": [{"role": "user", "content": "hi there"}],
        "completion": [{"role": "assistant", "content": "hello world"}],
    }
]


@pytest.mark.parametrize(
    "rows", [MESSAGES, PROMPT_COMPLETION], ids=["messages", "prompt-completion"]
)
def test_labels_equal_the_trl_0_26_2_labels(sft, tokenizer, rows):
    prepared = _prepare(sft, tokenizer, rows)
    examples = [prepared[i] for i in range(len(prepared))]
    got = _collate_trl_1(tokenizer, examples)
    for i, example in enumerate(examples):
        want = _labels_trl_0_26_2(example, completion_only_loss=True)
        assert got[i, : len(want)].tolist() == want
        assert all(v == -100 for v in got[i, len(want) :].tolist())  # padding
        # Not vacuous: the prompt is masked and the assistant tokens are labels.
        assert -100 in want and any(v != -100 for v in want)


def test_only_assistant_tokens_are_labels(sft, tokenizer):
    """What the labels ARE, independent of the reference: the assistant spans, through the
    <|im_end|> that closes each (the response-template scan's end), and nothing else."""
    prepared = _prepare(sft, tokenizer, MESSAGES[1:])
    labels = prepared[0]["labels"]
    ids = prepared[0]["input_ids"]
    kept = tokenizer.convert_ids_to_tokens(
        [t for t, l in zip(ids, labels) if l != -100]
    )
    assert kept == ["world", "<|im_end|>"]


def test_prompt_completion_mask_covers_the_whole_prompt(sft, tokenizer):
    """The transformers 5.x BatchEncoding fix: completion_mask must be 0 over every prompt
    token, not over len(BatchEncoding) == 2 of them."""
    prepared = _prepare(sft, tokenizer, PROMPT_COMPLETION)
    prompt_len = len(
        tokenizer.apply_chat_template(
            PROMPT_COMPLETION[0]["prompt"], tokenize=True, add_generation_prompt=True
        )["input_ids"]
    )
    mask = prepared[0]["completion_mask"]
    assert mask[:prompt_len] == [0] * prompt_len
    assert mask[prompt_len:] == [1] * (len(mask) - prompt_len)


def test_without_the_labels_map_the_trl_1_collator_trains_on_the_prompt(tokenizer):
    """Pins why build_labels exists: trl 1.x's collator, given rows with masks but no
    labels, labels every token."""
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    row = {"input_ids": [1, 5, 3, 7, 2], "assistant_masks": [0, 0, 0, 1, 1]}
    assert collator([row])["labels"].tolist() == [[1, 5, 3, 7, 2]]


def test_build_labels_is_trl_1s_rule(sft):
    row = {
        "input_ids": [1, 2, 3, 4],
        "completion_mask": [0, 1, 1, 1],
        "assistant_masks": [0, 0, 1, 1],
    }
    assert sft.build_labels(row, ["completion_mask", "assistant_masks"]) == {
        "labels": [-100, -100, 3, 4]
    }
    assert sft.build_labels(row, []) == {"labels": [1, 2, 3, 4]}
