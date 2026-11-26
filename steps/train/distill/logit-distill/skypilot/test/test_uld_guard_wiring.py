# SPDX-License-Identifier: Apache-2.0

"""ULDAlignmentGuard is wired into the real CustomGOLDTrainer, not only correct in isolation.

test_loss_numerics.py checks the guard's arithmetic. These check the wiring, which nothing
else exercises: __init__ registers the callback for a CustomULDLoss, the callback's methods
are hooks transformers actually calls, a run whose rows never align raises at its first
optimizer step, an aligned run's log lines carry the guard's counts, and evaluation does
not leak its rows into the next training step.

The trainer is real and runs on CPU: CustomGOLDTrainer, CustomDataCollatorForChatML and
CustomGOLDConfig as gold.py builds them. It uses a 1-layer Llama and a character-level
tokenizer built here, so nothing is downloaded. Student and teacher share the vocabulary.
The teacher's chat template either renders the assistant content as the student's does
(the rows align) or prefixes it with "x" (no row aligns).
"""

import inspect
import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("datasets")

from datasets import Dataset  # noqa: E402
from gb_steps_post_training.distillation import custom_gold_trainer as cgt  # noqa: E402
from gb_steps_post_training.distillation.custom_gold_config import (  # noqa: E402
    CustomGOLDConfig,
)
from gb_steps_post_training.distillation.utils import (  # noqa: E402
    CustomDataCollatorForChatML,
)
from transformers import (  # noqa: E402
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    TrainerCallback,
)

_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 ?.!,:\n"
_SPECIALS = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}{{ PREFIX }}{% endif %}{{ m.content }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
_ROWS = [
    [
        {"role": "user", "content": "what is two plus two?"},
        {"role": "assistant", "content": "four."},
    ],
    [
        {"role": "user", "content": "name a colour."},
        {"role": "assistant", "content": "red is one."},
    ],
    [
        {"role": "user", "content": "say hi."},
        {"role": "assistant", "content": "hi there!"},
    ],
    [
        {"role": "user", "content": "count to 3."},
        {"role": "assistant", "content": "1, 2, 3."},
    ],
]
# Two assistant turns per row: what last_message_only decides about.
_MULTI_TURN_ROWS = [
    [
        {"role": "user", "content": "say hi."},
        {"role": "assistant", "content": "hi there!"},
        {"role": "user", "content": "what is two plus two?"},
        {"role": "assistant", "content": "four."},
    ],
    [
        {"role": "user", "content": "name a colour."},
        {"role": "assistant", "content": "red is one."},
        {"role": "user", "content": "count to 3."},
        {"role": "assistant", "content": "1, 2, 3."},
    ],
]


def _model_dir(path, assistant_prefix):
    """A tokenizer + tiny Llama saved at `path`. `assistant_prefix` is rendered before every
    assistant turn's content, which is how the teacher is made to disagree."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(_SPECIALS + list(_CHARS))}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split(Regex("[\\s\\S]"), behavior="isolated")
    raw.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = _TEMPLATE.replace(
        "{{ PREFIX }}", json.dumps(assistant_prefix).join(["{{ ", " }}"])
    )
    tok.save_pretrained(path)
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(tok),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    LlamaForCausalLM(cfg).save_pretrained(path)
    return str(path), tok


def _trainer(
    tmp_path,
    *,
    teacher_prefix="",
    max_steps=2,
    logging_steps=1,
    rows=_ROWS,
    collator_lmo=None,
    **config,
):
    """`config` overrides the CustomGOLDConfig fields below. `collator_lmo` gives the
    collator a last_message_only other than the config's, i.e. a config that bypassed
    CustomGOLDConfig's check."""
    student, tok = _model_dir(tmp_path / "student", "")
    teacher, _ = _model_dir(tmp_path / "teacher", teacher_prefix)
    kw = dict(
        output_dir=str(tmp_path / "out"),
        use_uld_loss=True,
        teacher_model_name_or_path=teacher,
        teacher_tokenizer_name_or_path=teacher,
        response_template="<|im_start|>assistant\n",
        last_message_only=True,
        lmbda=0.0,
        max_length=256,
        max_completion_length=64,
        max_steps=max_steps,
        per_device_train_batch_size=2,
        logging_steps=logging_steps,
        learning_rate=1e-3,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        bf16=False,
        dataset_num_proc=None,
        use_vllm=False,
        seed=0,
    )
    kw.update(config)
    args = CustomGOLDConfig(**kw)
    collator = CustomDataCollatorForChatML(
        tokenizer=tok,
        max_length=args.max_length,
        response_template=args.response_template,
        last_message_only=(
            args.last_message_only if collator_lmo is None else collator_lmo
        ),
    )
    return cgt.CustomGOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=args,
        train_dataset=Dataset.from_list([{"messages": m} for m in rows]),
        eval_dataset=(
            Dataset.from_list([{"messages": m} for m in rows])
            if kw.get("eval_strategy", "no") != "no"
            else None
        ),
        processing_class=tok,
        data_collator=collator,
    )


def _guard_callbacks(trainer):
    return [
        cb
        for cb in trainer.callback_handler.callbacks
        if isinstance(cb, cgt._ULDAlignmentGuardCallback)
    ]


def test_every_callback_method_is_a_transformers_hook():
    """A misspelt or renamed hook is never called, and nothing else would notice."""
    mine = {
        name
        for name, value in vars(cgt._ULDAlignmentGuardCallback).items()
        if inspect.isfunction(value) and not name.startswith("_")
    }
    assert mine == {"on_pre_optimizer_step", "on_train_end", "on_evaluate"}
    for name in mine:
        assert name in vars(TrainerCallback), name


def test_evaluation_rows_do_not_reach_the_next_step():
    """Except a completion at position 0, which is refused wherever it came from: it is
    left for the next step's guard."""
    loss_fn = types.SimpleNamespace(
        _alignment_unaligned=3, _alignment_rows=5, _alignment_unpredictable=1
    )
    callback = cgt._ULDAlignmentGuardCallback(
        types.SimpleNamespace(uld_loss_fn=loss_fn)
    )
    callback.on_evaluate(None, None, None, metrics={})
    assert (loss_fn._alignment_unaligned, loss_fn._alignment_rows) == (0, 0)
    assert loss_fn._alignment_unpredictable == 1


def test_init_registers_the_guard_for_the_uld_loss(tmp_path):
    trainer = _trainer(tmp_path)
    assert isinstance(trainer.uld_loss_fn, cgt.CustomULDLoss)
    assert isinstance(trainer._uld_alignment_guard, cgt.ULDAlignmentGuard)
    assert len(_guard_callbacks(trainer)) == 1


def test_an_aligned_run_trains_and_logs_the_guards_counts(tmp_path):
    """The rows align, so the run trains; log() drains the guard into every log line."""
    trainer = _trainer(tmp_path)
    trainer.train()
    lines = [h for h in trainer.state.log_history if "loss" in h]
    assert len(lines) == 2
    for h in lines:
        assert h["grad_norm"] > 0
        assert h["train/alignment_rows"] == 2
        assert h["train/alignment_unaligned_rows"] == 0
        assert h["train/alignment_unaligned_rate"] == 0.0


def test_a_run_whose_rows_never_align_raises_at_the_first_step(tmp_path):
    """The teacher renders every assistant turn with a leading "x" the student never
    has, so no row aligns. Without the guard this trained at loss 0.0."""
    trainer = _trainer(tmp_path, teacher_prefix="x")
    with pytest.raises(RuntimeError, match="ULD step 1: all 2 rows"):
        trainer.train()
    assert trainer.state.global_step == 0


def test_a_run_shorter_than_one_window_is_judged_at_train_end(tmp_path):
    """Two steps, a ten-step logging window: only on_train_end can judge it. A guard
    with a 0.0 threshold and one misaligned row per step trips there, not before."""
    trainer = _trainer(tmp_path, max_steps=2, logging_steps=10)
    guard = trainer._uld_alignment_guard
    guard.max_unaligned_fraction = 0.0
    real = trainer.uld_loss_fn._build_alignment_groups_from_ids
    calls = {"n": 0}

    def every_other_row_unaligned(s_ids, t_ids):
        calls["n"] += 1
        return ([], []) if calls["n"] % 2 else real(s_ids, t_ids)

    trainer.uld_loss_fn._build_alignment_groups_from_ids = every_other_row_unaligned
    with pytest.raises(RuntimeError, match=r"ULD steps 1-2: 2 of 4 rows"):
        trainer.train()
    assert trainer.state.global_step == 2


def test_a_position_zero_row_in_the_final_evaluation_is_refused_at_train_end(
    tmp_path,
):
    """An evaluation at the last step runs after that step's guard, so no later optimizer
    step would see its rows; on_train_end must. The rows align, and every evaluated row is
    made to report a completion at position 0."""
    trainer = _trainer(tmp_path, max_steps=2, eval_strategy="steps", eval_steps=2)
    loss_fn = trainer.uld_loss_fn
    real = loss_fn._unpredictable_start
    evaluated = {"rows": 0}

    def at_zero_when_evaluating(student_start, teacher_start):
        if trainer.model.training:
            return real(student_start, teacher_start)
        evaluated["rows"] += 1
        loss_fn._alignment_unpredictable += 1
        return True

    loss_fn._unpredictable_start = at_zero_when_evaluating
    with pytest.raises(RuntimeError, match="ULD, at train end: .* position 0"):
        trainer.train()
    assert trainer.state.global_step == 2
    assert evaluated["rows"] > 0


# ── Before the guard: the config refuses the known systematic cause ──────────────────


def _config(tmp_path, **overrides):
    kw = dict(output_dir=str(tmp_path), report_to=[], use_cpu=True, use_uld_loss=True)
    kw.update(overrides)
    return CustomGOLDConfig(**kw)


@pytest.mark.parametrize("lmo", [None, False], ids=["default", "false"])
def test_the_config_refuses_uld_without_last_message_only(tmp_path, lmo):
    """The same requirement as render_gold_config.py's uld arm, for a config that never
    went through the renderer, as CustomGOLDConfig already does for sampled_opd."""
    overrides = {} if lmo is None else {"last_message_only": lmo}
    with pytest.raises(
        ValueError, match="use_uld_loss requires last_message_only=true"
    ):
        _config(tmp_path, **overrides)


def test_the_config_accepts_uld_with_last_message_only(tmp_path):
    assert _config(tmp_path, last_message_only=True).last_message_only is True


@pytest.mark.parametrize(
    "mode",
    [
        {"uld_per_turn_regions": True, "uld_use_hybrid_loss": True},
        {"uld_reuse_student_input": True},
        {"uld_rebuild_student_input_with_teacher_template": True},
    ],
    ids=["per-turn-regions", "reuse-student-input", "rebuild-with-teacher-template"],
)
def test_the_config_exempts_the_modes_that_render_the_teacher_otherwise(tmp_path, mode):
    """Per-turn regions give the teacher one region per assistant turn and REQUIRE false;
    reusing the student's input gives the teacher the student's own spans; rebuilding the
    student from the teacher's texts gives the student the teacher's last-turn span."""
    assert _config(tmp_path, last_message_only=False, **mode).use_uld_loss is True


def test_last_message_only_false_leaves_multi_turn_rows_unaligned(tmp_path):
    """The control for the test below: the collator labels both assistant turns, the
    teacher renders the last one alone, and no row aligns (the config is bypassed to get
    here, as a config that never went through CustomGOLDConfig would)."""
    trainer = _trainer(tmp_path, rows=_MULTI_TURN_ROWS, max_steps=1, collator_lmo=False)
    with pytest.raises(RuntimeError, match="ULD step 1: all 2 rows"):
        trainer.train()


def test_rebuilding_the_student_aligns_multi_turn_rows_without_last_message_only(
    tmp_path,
):
    """uld_rebuild_student_input_with_teacher_template replaces the student's input_ids
    and labels with ones built from the teacher's (prompt, completion) texts, so the
    student's span is the last turn alone whatever the collator labelled: every row
    aligns with last_message_only false, which is why the config exempts the mode."""
    trainer = _trainer(
        tmp_path,
        rows=_MULTI_TURN_ROWS,
        max_steps=1,
        last_message_only=False,
        uld_rebuild_student_input_with_teacher_template=True,
    )
    assert trainer.data_collator.last_message_only is False
    trainer.train()
    (line,) = [h for h in trainer.state.log_history if "loss" in h]
    assert line["train/alignment_rows"] == 2
    assert line["train/alignment_unaligned_rows"] == 0


def test_reusing_the_student_input_keeps_every_rows_completion(tmp_path, monkeypatch):
    """uld_reuse_student_input at batch 2. The collator left-pads, so the rows end
    together; row 1's prompt is shorter and its completion longer, so its completion
    starts before the batch's longest prompt ends. Cropped at that prompt, the teacher
    lost row 1's first tokens and its labels started at position 0. Cropped at the
    earliest completion, the teacher scores exactly the student's completion on every
    row, from position 1 or later, and both rows align. (With the hybrid loss: the
    mode's tokenizer check reads `jaccard`, which only the hybrid loss's vocabulary
    mapping sets.)"""
    trainer = _trainer(tmp_path, uld_reuse_student_input=True, uld_use_hybrid_loss=True)
    inputs = trainer.data_collator([trainer.train_dataset[i] for i in (0, 1)])
    first = [int((row != -100).nonzero()[0]) for row in inputs["labels"]]
    assert min(first) < inputs["prompts"].shape[1]  # the layout this is about

    loss_fn = trainer.uld_loss_fn
    seen = {}
    real_call = cgt.CustomULDLoss.__call__

    def spy(self, **kw):  # trl's ULDLoss defines __call__, not forward
        seen.update(kw)
        return real_call(self, **kw)

    monkeypatch.setattr(cgt.CustomULDLoss, "__call__", spy)
    trainer.compute_loss(trainer.model, inputs)

    for row in (0, 1):
        s_mask = seen["student_labels"][row] != -100
        t_mask = seen["teacher_labels"][row] != -100
        assert torch.equal(
            seen["teacher_input_ids"][row][t_mask],
            seen["student_input_ids"][row][s_mask],
        )
        assert int(t_mask.nonzero()[0]) >= 1
    assert loss_fn._alignment_unpredictable == 0
    assert (loss_fn._alignment_rows, loss_fn._alignment_unaligned) == (2, 0)


def test_the_config_leaves_other_arms_alone(tmp_path):
    assert _config(tmp_path, use_uld_loss=False).last_message_only is False
