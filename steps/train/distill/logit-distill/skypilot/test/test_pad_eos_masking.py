# SPDX-License-Identifier: Apache-2.0
"""pad == eos must not mask the EOS out of the distillation labels.

gold.py sets ``pad_token = eos_token`` for tokenizers without a pad token, so
pad == eos is the common case. Masking labels by token value then also masks
every real EOS, and the student never gets a KD signal for stopping. The
teacher builder and both ULD label sets go through ``mask_pad_labels_by_value``,
which skips the value mask when pad == eos; these tests pin that.

TRL's in-process ``generate_on_policy_outputs`` masks by value too, and is not
ours to patch: render_gold_config.py refuses the only configuration that
reaches it (lmbda > 0 with vllm_num_servers 0), see test_render_gold_config.py.

Torch-gated like test_loss_numerics.py: skips where torch and trl are absent
(the repo venv, CI), and runs in the GPU image.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")  # custom_gold_trainer imports trl at module load

from gb_steps_post_training.distillation.custom_gold_trainer import (  # noqa: E402
    build_teacher_inputs_from_texts_with_eos_control,
    mask_pad_labels_by_value,
)

EOS = 2
PAD = 0


class _Tokenizer:
    """The three things the teacher builder reads: ids, pad and eos."""

    def __init__(self, pad_token_id, eos_token_id=EOS):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._ids = {"short": [5], "a longer prompt": [5, 6, 7], "out": [8, 9]}

    def __call__(self, texts, add_special_tokens=True):
        return {"input_ids": [list(self._ids[t]) for t in texts]}


class TestMaskPadLabelsByValue:
    def test_pad_equal_to_eos_keeps_the_eos(self):
        labels = torch.tensor([[-100, 8, 9, EOS]])
        out = mask_pad_labels_by_value(labels.clone(), EOS, EOS)
        assert out.tolist() == [[-100, 8, 9, EOS]]

    def test_a_distinct_pad_is_masked(self):
        labels = torch.tensor([[-100, 8, PAD, EOS]])
        out = mask_pad_labels_by_value(labels.clone(), PAD, EOS)
        assert out.tolist() == [[-100, 8, -100, EOS]]

    def test_no_pad_token_is_a_no_op(self):
        labels = torch.tensor([[-100, 8, 9, EOS]])
        out = mask_pad_labels_by_value(labels.clone(), None, EOS)
        assert out.tolist() == [[-100, 8, 9, EOS]]

    def test_it_edits_in_place(self):
        """The ULD call sites rely on this: they do not reassign the result."""
        labels = torch.tensor([[8, PAD]])
        mask_pad_labels_by_value(labels, PAD, EOS)
        assert labels.tolist() == [[8, -100]]


class TestTeacherBuilder:
    @staticmethod
    def _labels(tokenizer):
        _, labels, _, _ = build_teacher_inputs_from_texts_with_eos_control(
            tokenizer, ["short", "a longer prompt"], ["out", "out"]
        )
        return labels

    def test_pad_equal_to_eos_keeps_the_appended_eos(self):
        """The regression: with pad == eos, value masking used to turn the appended
        EOS into -100 in every row."""
        labels = self._labels(_Tokenizer(pad_token_id=EOS))
        for row in labels.tolist():
            assert row[-1] == EOS, row

    def test_prompt_and_left_padding_are_still_masked_positionally(self):
        labels = self._labels(_Tokenizer(pad_token_id=EOS))
        # Prompts are left-padded to 3 tokens; only the completion and EOS train.
        assert labels.tolist() == [
            [-100, -100, -100, 8, 9, EOS],
            [-100, -100, -100, 8, 9, EOS],
        ]

    def test_a_distinct_pad_gives_the_same_labels(self):
        """With a real pad token nothing in the completion matches it, so the
        labels must not depend on which pad the tokenizer has."""
        assert (
            self._labels(_Tokenizer(pad_token_id=PAD)).tolist()
            == self._labels(_Tokenizer(pad_token_id=EOS)).tolist()
        )
