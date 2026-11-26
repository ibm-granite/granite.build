# SPDX-License-Identifier: Apache-2.0
"""Numeric property tests for the vendored distillation losses.

These need torch (and, for the import, trl), so they skip wherever those are
absent: the repo venv does not ship them, while the GPU image and a dev
checkout with the image's pins do. They exist because config-and-plumbing
tests cannot catch a wrong loss: the ``adaptive_kld`` arm shipped
mathematically wrong for its whole life (per-entry forward/reverse KL
switching whose teacher is not a stationary point — see
``REJECTED_LOSS_ARMS`` in render_gold_config.py) and every suite was green
until these properties were checked.

Properties, from the #540 review:

- non-negative, and zero when the student equals the teacher;
- STATIONARY at the teacher: the gradient of the loss with respect to the
  student logits is ~0 at p == q — the property ``adaptive_kld`` violates;
- reference agreement: beta 0 = forward KL, beta 1 = reverse KL, the mixture
  JSD, the convex KL interpolation, DistiLLM-2's on/off-policy arms, and
  sft.py's top-K forward-KD ``_compute_loss_kd``.

The ULD arm (``CustomULDLoss``, the cross-tokenizer loss) is checked
separately in ``TestULD``: it is a sorted-distribution distance, not a
divergence over one shared vocabulary, so only some of these properties apply
to it, and its point is the EOS-only completion its parent scores at zero.
"""

import dataclasses
import math
import types

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
pytest.importorskip("trl")  # custom_gold_trainer imports trl at module load

from gb_steps_post_training.distillation import custom_gold_trainer as cgt  # noqa: E402
from gb_steps_post_training.distillation.custom_gold_config import (  # noqa: E402
    CustomGOLDConfig,
)
from gb_steps_post_training.distillation.custom_gold_trainer import (  # noqa: E402
    CustomGOLDTrainer,
    CustomULDLoss,
)
from gb_steps_post_training.distillation.sft import CustomSFTTrainer  # noqa: E402
from trl.experimental.gold.gold_trainer import ULDLoss  # noqa: E402

# float64 throughout: the properties assert against tight tolerances, and the
# stationarity check in particular is meaningless in float32.
DTYPE = torch.float64

JSD = CustomGOLDTrainer.generalized_jsd_loss
D2 = CustomGOLDTrainer.distillm2_loss


def _logits(batch=4, seq=6, vocab=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, vocab, generator=g, dtype=DTYPE)


def _stationarity_gap(loss_fn, **kwargs):
    """|d loss / d student logits| at p == q, max over all logits."""
    teacher = _logits(seed=1)
    student = teacher.clone().requires_grad_(True)
    loss = loss_fn(student, teacher, **kwargs)
    loss.backward()
    return student.grad.abs().max().item()


class TestNonnegativityAndMatch:
    @pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize("temperature", [1.0, 2.0])
    def test_jsd_is_nonnegative(self, beta, temperature):
        loss = JSD(_logits(seed=2), _logits(seed=3), beta=beta, temperature=temperature)
        assert loss.item() >= -1e-9

    @pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_jsd_is_zero_when_the_student_equals_the_teacher(self, beta):
        t = _logits(seed=4)
        assert JSD(t.clone(), t, beta=beta).item() == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("on_policy", [True, False])
    def test_distillm2_is_nonnegative_and_zero_at_match(self, on_policy):
        s, t = _logits(seed=5), _logits(seed=6)
        assert D2(s, t, on_policy=[on_policy] * 4).item() >= -1e-9
        assert D2(t.clone(), t, on_policy=[on_policy] * 4).item() == pytest.approx(
            0.0, abs=1e-9
        )


class TestStationarity:
    """At p == q every divergence here must have zero gradient in the student:
    the teacher is the optimum, so a non-zero gradient keeps training past it
    (the student sharpens its own head indefinitely). That is exactly the
    defect that got adaptive_kld refused."""

    @pytest.mark.parametrize("beta", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_jsd_is_stationary_at_the_teacher(self, beta):
        assert _stationarity_gap(lambda s, t: JSD(s, t, beta=beta)) < 1e-8

    def test_kl_interpolation_is_stationary_at_the_teacher(self):
        assert (
            _stationarity_gap(
                lambda s, t: JSD(s, t, beta=0.4, use_kl_interpolation=True)
            )
            < 1e-8
        )

    @pytest.mark.parametrize("on_policy", [True, False])
    def test_distillm2_is_stationary_at_the_teacher(self, on_policy):
        assert (
            _stationarity_gap(lambda s, t: D2(s, t, on_policy=[on_policy] * 4)) < 1e-8
        )

    def test_adaptive_kld_is_NOT_stationary_at_the_teacher(self):
        """The property the arm violates, kept as the reason it is refused.
        Random teacher logits keep both head and tail entries in the top-0.5
        mask for most vocabularies, so the per-entry switch is exercised."""
        gap = _stationarity_gap(lambda s, t: JSD(s, t, beta=0.5, use_adaptive_kld=True))
        assert gap > 1e-6, (
            f"adaptive_kld looked stationary (gap {gap:.3e}); if the arm was "
            "fixed upstream to weight whole distributions, re-enable it in "
            "render_gold_config.py and delete this test with the refusal"
        )


class TestReferences:
    """Each arm against the same math written out directly."""

    def test_beta_zero_is_forward_kl(self):
        s, t = _logits(seed=7), _logits(seed=8)
        expected = F.kl_div(
            F.log_softmax(s, dim=-1),
            F.log_softmax(t, dim=-1),
            reduction="batchmean",
            log_target=True,
        )
        assert JSD(s, t, beta=0.0).item() == pytest.approx(expected.item(), abs=1e-10)

    def test_beta_one_is_reverse_kl(self):
        s, t = _logits(seed=7), _logits(seed=8)
        sl, tl = F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1)
        expected = (torch.exp(sl) * (sl - tl)).sum() / s.size(0)
        assert JSD(s, t, beta=1.0).item() == pytest.approx(expected.item(), abs=1e-10)

    def test_mixture_jsd_matches_reference(self):
        beta = 0.3
        s, t = _logits(seed=9), _logits(seed=10)
        sl, tl = F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1)
        mixture = torch.logaddexp(sl + math.log(1 - beta), tl + math.log(beta))
        kl_teacher = (torch.exp(tl) * (tl - mixture)).sum() / s.size(0)
        kl_student = (torch.exp(sl) * (sl - mixture)).sum() / s.size(0)
        expected = beta * kl_teacher + (1 - beta) * kl_student
        assert JSD(s, t, beta=beta).item() == pytest.approx(expected.item(), abs=1e-10)

    def test_kl_interpolation_matches_reference(self):
        beta = 0.4
        s, t = _logits(seed=11), _logits(seed=12)
        sl, tl = F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1)
        fkl = (torch.exp(tl) * (tl - sl)).sum() / s.size(0)
        rkl = (torch.exp(sl) * (sl - tl)).sum() / s.size(0)
        expected = (1 - beta) * fkl + beta * rkl
        assert JSD(s, t, beta=beta, use_kl_interpolation=True).item() == (
            pytest.approx(expected.item(), abs=1e-10)
        )

    def test_distillm2_splits_kl_by_policy_class(self):
        """On-policy rows get reverse KL, off-policy rows forward KL, summed
        over tokens and divided by the token count (the batchmean the
        docstring promises)."""
        s, t = _logits(seed=13), _logits(seed=14)
        on_policy = [True, False, True, False]
        sl, tl = F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1)
        per_row_rkl = (torch.exp(sl) * (sl - tl)).sum(dim=-1).sum(dim=-1)
        per_row_fkl = (torch.exp(tl) * (tl - sl)).sum(dim=-1).sum(dim=-1)
        n_tokens = s.size(0) * s.size(1)
        expected = (
            per_row_rkl[0] + per_row_rkl[2] + per_row_fkl[1] + per_row_fkl[3]
        ) / n_tokens
        assert D2(s, t, on_policy=on_policy).item() == pytest.approx(
            expected.item(), abs=1e-10
        )

    def test_jsd_labels_mask_positions(self):
        s, t = _logits(batch=2, seed=15), _logits(batch=2, seed=16)
        labels = torch.tensor([[0, 0, -100, 0, -100, -100], [0, -100, -100, 0, 0, 0]])
        masked = JSD(s, t, labels=labels, beta=0.5)
        keep = labels != -100
        s_kept = s.reshape(-1, s.size(-1))[keep.view(-1)]
        t_kept = t.reshape(-1, t.size(-1))[keep.view(-1)]
        sl, tl = F.log_softmax(s_kept, dim=-1), F.log_softmax(t_kept, dim=-1)
        mixture = torch.logaddexp(sl + math.log(0.5), tl + math.log(0.5))
        kl_t = (torch.exp(tl) * (tl - mixture)).sum()
        kl_s = (torch.exp(sl) * (sl - mixture)).sum()
        # with labels the reduction is sum / mask.sum(): a token mean.
        expected = (0.5 * kl_t + 0.5 * kl_s) / keep.sum()
        assert masked.item() == pytest.approx(expected.item(), abs=1e-10)


class TestComputeLossKd:
    """sft.py's top-K forward-KD loss against the same math written directly,
    through a stub trainer and a model whose logits are a known parameter."""

    def test_matches_reference(self):
        B, S, V, K = 2, 5, 13, 4
        g = torch.Generator().manual_seed(0)
        # The model returns [B, S+1, V]; _compute_loss_kd drops the last
        # position (shifted labels), so the reference uses [:, :-1].
        logits = torch.randn(B, S + 1, V, generator=g, dtype=DTYPE)

        class _Model:
            training = True

            def __call__(self, **_kwargs):
                return types.SimpleNamespace(logits=logits)

        model = _Model()
        stub = types.SimpleNamespace(
            model=model,
            _total_train_tokens=0,
            kd_temperature=2.0,
            kd_weight=0.7,
            ce_weight=0.0,
            accelerator=types.SimpleNamespace(
                device=torch.device("cpu"),
                gather_for_metrics=lambda x: x,
            ),
            _metrics={"train": {"mean_token_accuracy": []}, "eval": {}},
            args=types.SimpleNamespace(logging_steps=1000),
        )

        # A top-K teacher that puts all mass on the student's argmax: the
        # renormalized top-K teacher is one-hot there, so the KL reduces to
        # -log softmax(student)[argmax] per token. The teacher tensors arrive
        # already flattened to [N, K] with N the masked token count.
        used = logits[:, :-1, :].reshape(-1, V)  # [N=B*S, V]
        top_idx = used.argmax(dim=-1)  # [N]
        k_idx = torch.zeros(used.size(0), K, dtype=torch.long)
        k_idx[:, 0] = top_idx
        k_lg = torch.full((used.size(0), K), -1e30, dtype=DTYPE)
        k_lg[:, 0] = 0.0  # softmax -> one-hot on the argmax

        inputs = {
            "input_ids": torch.zeros(B, S + 1, dtype=torch.long),
            "attention_mask": torch.ones(B, S + 1, dtype=torch.long),
            "labels": torch.zeros(B, S + 1, dtype=torch.long),
            "teacher_topk_indices": k_idx,
            "teacher_topk_logits": k_lg,
        }
        loss = CustomSFTTrainer._compute_loss_kd(stub, model, inputs)

        T = 2.0
        s_log_probs = F.log_softmax(used / T, dim=-1)
        s_at_argmax = s_log_probs.gather(-1, top_idx.unsqueeze(-1)).squeeze(-1)
        expected_kl = (-s_at_argmax).mean()  # t log t - t log s at one-hot t
        expected = 0.7 * (T * T) * expected_kl
        assert loss.item() == pytest.approx(expected.item(), abs=1e-8)


# --- ULD -----------------------------------------------------------------------------------
#
# What the reference follows. `loss_arm: uld` renders `use_uld_loss: true` and, required by
# the arm, `last_message_only: true` (render_gold_config.py LOSS_ARMS["uld"]); the latter
# shapes the collator's labels, not the loss. So every other ULD knob is a dataclass default:
# use_extended_uld True, uld_use_hybrid_loss False, cross-entropy weight 0, distillation
# weight 1, both temperatures 1. With the hybrid loss off, CustomULDLoss runs
# _compute_base_distillation_loss: TRL's non-hybrid ULD (trl 0.26.2,
# trl/experimental/gold/gold_trainer.py:325-423) with skip_*_eos forced False, its own
# _build_alignment_groups_from_ids (bodies aligned greedily, the two EOS positions appended
# as one final group), and two fixes to TRL (TestULDLogitAlignment, TestULDUnalignedRow).
# Per row:
#
#   1. takes the completion as [first label, first label + label count), and reads the
#      logits one position EARLIER, so probs[k] predicts token k. (TRL reads the logits AT
#      those positions, :367-368, which predict the token after each one.)
#   2. softmaxes each side over its own vocabulary (:371-372);
#   3. merges each alignment group into one distribution: a single position as is, several
#      as the renormalised product of their probabilities (:521-557, via log-sum-softmax);
#   4. sorts each merged distribution descending, zero-pads the narrower vocabulary, and
#      takes the L1 distance summed over groups and vocabulary, divided by the group count
#      (:405-419);
#   5. averages the rows and scales by the distillation weight (:422-423).
#
# The reference below writes the same steps out with the alignment groups spelled out by
# hand, so a regression in the class's grouping is caught rather than reproduced. TRL builds
# the merged tensor with torch.zeros and no dtype (:538), so the loss is float32 whatever the
# logits are; hence the 1e-6 tolerances here rather than the 1e-10 elsewhere in this file.

ULD_TOL = 1e-6

# Student vocabulary (7) and teacher vocabulary (9): different sizes, and different
# segmentations of the same text ("ab" is one student token, "abc" one teacher token).
S_PIECES = ["</s>", "a", "b", "c", "ab", "ca", "<pad>"]
T_PIECES = ["<pad>", "<|eos|>", "a", "b", "c", "abc", "bc", "x", "y"]
S_EOS, T_EOS = 0, 1
S_PAD, T_PAD = 6, 0
IGNORE = -100


class _PieceTokenizer:
    """What the ULD losses read from a tokenizer on the path the step uses: eos_token_id
    (CustomULDLoss.__init__) and decode over each prefix of a completion, which the
    alignment walks to recover every id's text piece. Concatenating pieces is exactly a
    decode for these vocabularies, and it needs no download."""

    def __init__(self, pieces, eos_token_id):
        self._pieces = pieces
        self.eos_token_id = eos_token_id

    def decode(
        self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    ):
        return "".join(self._pieces[i] for i in ids)

    def get_vocab(self):
        return {p: i for i, p in enumerate(self._pieces)}


def _uld_config(**overrides):
    """The step's ULD configuration: every CustomGOLDConfig default, plus the one flag the
    uld arm sets. Read from the dataclass fields rather than by constructing the
    TrainingArguments, which would validate output dirs and devices this test has no use for.
    """
    cfg = {}
    for f in dataclasses.fields(CustomGOLDConfig):
        if f.default is not dataclasses.MISSING:
            cfg[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            cfg[f.name] = f.default_factory()
    cfg["use_uld_loss"] = True
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def _custom_uld(**overrides):
    return CustomULDLoss(
        _uld_config(**overrides),
        student_tokenizer=_PieceTokenizer(S_PIECES, S_EOS),
        teacher_tokenizer=_PieceTokenizer(T_PIECES, T_EOS),
    )


def _parent_uld(**overrides):
    return ULDLoss(
        _uld_config(**overrides),
        student_tokenizer=_PieceTokenizer(S_PIECES, S_EOS),
        teacher_tokenizer=_PieceTokenizer(T_PIECES, T_EOS),
    )


def _rows(rows, pad):
    """[(prompt ids, completion ids)] -> right-padded (input_ids, labels), prompt and pad
    unlabelled, the completion (EOS included) labelled with its own ids."""
    width = max(len(p) + len(c) for p, c in rows)
    ids, labels = [], []
    for prompt, completion in rows:
        n_pad = width - len(prompt) - len(completion)
        ids.append(prompt + completion + [pad] * n_pad)
        labels.append([IGNORE] * len(prompt) + completion + [IGNORE] * n_pad)
    return torch.tensor(ids), torch.tensor(labels)


# Two rows. Row 0 is "abcab": student ab|c|ab, teacher abc|a|b, so the greedy alignment
# groups student [0,1] with teacher [0] ("abc") and student [2] with teacher [1,2] ("ab").
# Row 1 is "cab": student ca|b, teacher c|a|b, grouping student [0] with teacher [0,1]
# ("ca") and student [1] with teacher [2] ("b"). Each row then ends with its EOS group.
# Prompt lengths differ by side and by row, so the completions start at different offsets.
S_ROWS = [([1, 2], [4, 3, 4, S_EOS]), ([3, 1, 2], [5, 2, S_EOS])]
T_ROWS = [([7], [5, 2, 3, T_EOS]), ([8, 8], [4, 2, 3, T_EOS])]
S_GROUPS = [[[0, 1], [2], [3]], [[0], [1], [2]]]
T_GROUPS = [[[0], [1, 2], [3]], [[0, 1], [2], [3]]]


def _uld_batch(seed=0):
    s_ids, s_labels = _rows(S_ROWS, S_PAD)
    t_ids, t_labels = _rows(T_ROWS, T_PAD)
    g = torch.Generator().manual_seed(seed)
    s_logits = torch.randn(*s_ids.shape, len(S_PIECES), generator=g, dtype=DTYPE)
    t_logits = torch.randn(*t_ids.shape, len(T_PIECES), generator=g, dtype=DTYPE)
    return dict(
        student_logits=s_logits,
        teacher_logits=t_logits,
        student_labels=s_labels,
        teacher_labels=t_labels,
        student_input_ids=s_ids,
        teacher_input_ids=t_ids,
    )


def _uld_reference(batch, s_groups, t_groups, at_labels=False):
    """Sorted-distribution ULD, written out (see the block comment above for each step).
    `at_labels` reads the logits at the label positions instead, as TRL does."""

    def merged(logits, labels, groups):
        # step 1-2: the logits BEFORE each label (or AT it, for TRL's indexing)
        where = (labels != IGNORE).nonzero().squeeze(-1) - (0 if at_labels else 1)
        probs = F.softmax(logits[where], dim=-1)
        out = []
        for g in groups:  # step 3: renormalised product over the group
            q = probs[g].prod(dim=0)
            out.append(q / q.sum())
        return torch.stack(out)

    row_losses = []
    for i in range(batch["student_logits"].size(0)):
        s = merged(batch["student_logits"][i], batch["student_labels"][i], s_groups[i])
        t = merged(batch["teacher_logits"][i], batch["teacher_labels"][i], t_groups[i])
        width = max(s.size(-1), t.size(-1))  # step 4: sort, zero-pad, L1 per group
        s = F.pad(s.sort(dim=-1, descending=True).values, (0, width - s.size(-1)))
        t = F.pad(t.sort(dim=-1, descending=True).values, (0, width - t.size(-1)))
        row_losses.append((s - t).abs().sum() / len(s_groups[i]))
    return torch.stack(row_losses).mean()  # step 5, at distillation weight 1


class TestULD:
    """CustomULDLoss on the configuration the uld arm renders."""

    def test_the_step_config_takes_the_sorted_path(self):
        """The reference below is the non-hybrid, extended-alignment path at unit weights.
        If a re-vendor changes any of these defaults, the arm computes something else.
        """
        loss = _custom_uld()
        assert loss.use_extended_uld is True
        assert loss.use_hybrid_loss is False and loss._vocab_mapping is None
        assert loss.crossentropy_weight == 0.0
        assert loss.distillation_weight == 1.0
        assert loss.student_temperature == loss.teacher_temperature == 1.0
        assert loss.skip_student_eos is False and loss.skip_teacher_eos is False

    def test_alignment_groups_are_the_hand_derived_ones(self):
        """The class aligns the bodies greedily and appends EOS as its own group."""
        loss = _custom_uld()
        for (_, s_c), (_, t_c), s_g, t_g in zip(S_ROWS, T_ROWS, S_GROUPS, T_GROUPS):
            assert loss._build_alignment_groups_from_ids(s_c, t_c) == (s_g, t_g)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_reference(self, seed):
        batch = _uld_batch(seed)
        expected = _uld_reference(batch, S_GROUPS, T_GROUPS)
        assert _custom_uld()(**batch).item() == pytest.approx(
            expected.item(), abs=ULD_TOL
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_is_nonnegative(self, seed):
        assert _custom_uld()(**_uld_batch(seed)).item() >= 0.0

    def test_is_zero_when_the_student_matches_the_teacher(self):
        """Across two vocabularies "matches" means: the same distribution up to a relabelling
        of tokens, with the teacher's surplus tokens at probability zero. That is the zero
        set of a sorted, zero-padded distance. The text "abc" is a|b|c on both sides, so
        every group is one position and no merge perturbs the probabilities."""
        s_ids, s_labels = _rows([([5], [1, 2, 3, S_EOS])], S_PAD)
        t_ids, t_labels = _rows([([7], [2, 3, 4, T_EOS])], T_PAD)
        g = torch.Generator().manual_seed(3)
        t_logits = torch.randn(1, 5, len(T_PIECES), generator=g, dtype=DTYPE)
        t_logits[..., 7:] = float("-inf")  # the two surplus teacher tokens
        relabel = torch.randperm(
            7, generator=g
        )  # student token k <- teacher token relabel[k]
        s_logits = t_logits[..., relabel].clone()
        batch = dict(
            student_logits=s_logits,
            teacher_logits=t_logits,
            student_labels=s_labels,
            teacher_labels=t_labels,
            student_input_ids=s_ids,
            teacher_input_ids=t_ids,
        )
        assert _custom_uld()(**batch).item() == pytest.approx(0.0, abs=ULD_TOL)
        # and the same pair at an unrelated student is not zero, so the check has teeth.
        batch["student_logits"] = torch.randn(
            1, 5, len(S_PIECES), generator=g, dtype=DTYPE
        )
        assert _custom_uld()(**batch).item() > 1e-3

    @pytest.mark.parametrize("side", ["student", "teacher", "both"])
    def test_is_unchanged_by_a_vocabulary_permutation(self, side):
        """Sorting discards token identity, so relabelling either vocabulary (the logit
        columns; the ids, and so the alignment, unchanged) cannot change the loss."""
        batch = _uld_batch(seed=4)
        before = _custom_uld()(**batch).item()
        g = torch.Generator().manual_seed(5)
        if side in ("student", "both"):
            perm = torch.randperm(len(S_PIECES), generator=g)
            batch["student_logits"] = batch["student_logits"][..., perm]
        if side in ("teacher", "both"):
            perm = torch.randperm(len(T_PIECES), generator=g)
            batch["teacher_logits"] = batch["teacher_logits"][..., perm]
        assert _custom_uld()(**batch).item() == pytest.approx(before, abs=ULD_TOL)


class TestULDEosOnlyCompletion:
    """The bug CustomULDLoss exists for (its docstring): when the completion is only EOS,
    TRL's ULDLoss subtracts the EOS from each answer size (uld_skip_*_eos default True,
    gold_trainer.py:338-341), every size is then 0, and it returns a constant zero
    (:343-349) -- no loss and no gradient for the student's decision to stop."""

    @staticmethod
    def _eos_only_batch():
        s_ids, s_labels = _rows([([1, 2], [S_EOS])], S_PAD)
        t_ids, t_labels = _rows([([7], [T_EOS])], T_PAD)
        g = torch.Generator().manual_seed(6)
        return dict(
            student_logits=torch.randn(1, 3, len(S_PIECES), generator=g, dtype=DTYPE),
            teacher_logits=torch.randn(1, 2, len(T_PIECES), generator=g, dtype=DTYPE),
            student_labels=s_labels,
            teacher_labels=t_labels,
            student_input_ids=s_ids,
            teacher_input_ids=t_ids,
        )

    def test_parent_scores_it_zero(self):
        """The defect, pinned on the parent, so the subclass's fix has something to fix:
        if trl stops dropping the EOS, this fails and the subclass's override can go."""
        batch = self._eos_only_batch()
        batch["student_logits"].requires_grad_(True)
        loss = _parent_uld()(**batch)
        assert loss.item() == 0.0
        loss.backward()
        assert batch["student_logits"].grad.abs().max().item() == 0.0

    def test_custom_scores_it_and_matches_reference(self):
        batch = self._eos_only_batch()
        batch["student_logits"].requires_grad_(True)
        loss = _custom_uld()(**batch)
        # One row, one group: the two EOS positions, aligned by the subclass. Pinned
        # separately because without that group the parent falls back to the unmerged
        # positions (gold_trainer.py:532-533), which for a one-token completion scores the
        # same number for the wrong reason.
        assert _custom_uld()._build_alignment_groups_from_ids([S_EOS], [T_EOS]) == (
            [[0]],
            [[0]],
        )
        expected = _uld_reference(batch, [[[0]]], [[[0]]])
        assert loss.item() > 1e-3
        assert loss.item() == pytest.approx(expected.item(), abs=ULD_TOL)
        loss.backward()
        assert batch["student_logits"].grad.abs().max().item() > 1e-6


def _single_row(s_completion, t_completion, seed, s_prompt=(5,), t_prompt=(7,)):
    s_ids, s_labels = _rows([(list(s_prompt), s_completion)], S_PAD)
    t_ids, t_labels = _rows([(list(t_prompt), t_completion)], T_PAD)
    g = torch.Generator().manual_seed(seed)
    return dict(
        student_logits=torch.randn(
            *s_ids.shape, len(S_PIECES), generator=g, dtype=DTYPE
        ),
        teacher_logits=torch.randn(
            *t_ids.shape, len(T_PIECES), generator=g, dtype=DTYPE
        ),
        student_labels=s_labels,
        teacher_labels=t_labels,
        student_input_ids=s_ids,
        teacher_input_ids=t_ids,
    )


class TestULDLogitAlignment:
    """Which logits score a completion token. The trainer passes both models' logits and
    labels in input coordinates, unshifted (compute_loss; the teacher crop keeps
    `teacher_prompt_length - 1` so the first completion token keeps its predictor), and in
    a causal LM logits[t] predict token t + 1. So the distribution for completion token p
    is logits[p - 1]: what the JSD path (logits[:, :-1] against labels[:, 1:]) and the
    hybrid ULD path (start-1) read. TRL's base path read logits[p] (gold_trainer.py:367-368),
    one position late: every token was scored by the prediction of the token after it, and
    the EOS by the prediction past the end of the completion.

    The construction makes the two readings disagree. Text "abc" is a|b|c on both sides, so
    every group is one position. At the positions that PREDICT the completion (the last
    prompt position through the one before EOS) the student is the teacher relabelled, so
    the correct reading gives exactly zero; at the EOS position, whose logits predict past
    the end, the student is unrelated, so the late reading does not."""

    @staticmethod
    def _batch():
        batch = _single_row([1, 2, 3, S_EOS], [2, 3, 4, T_EOS], seed=7)
        t_logits = batch["teacher_logits"]
        t_logits[..., 7:] = float("-inf")  # the two surplus teacher tokens
        relabel = torch.randperm(7, generator=torch.Generator().manual_seed(8))
        predictors = slice(0, 4)  # positions 0..3 predict completion tokens 1..4
        batch["student_logits"][:, predictors] = t_logits[:, predictors][..., relabel]
        return batch

    def test_each_token_is_scored_by_the_logits_before_it(self):
        batch = self._batch()
        groups = [[[0], [1], [2], [3]]]
        assert _uld_reference(batch, groups, groups).item() == pytest.approx(
            0.0, abs=ULD_TOL
        )
        assert _uld_reference(batch, groups, groups, at_labels=True).item() > 1e-3
        assert _custom_uld()(**batch).item() == pytest.approx(0.0, abs=ULD_TOL)

    def test_trl_defaults_to_the_observed_strategy_and_needs_byte_offsets(self):
        """The upstream defect, pinned on the parent with the subclass's EOS settings, so
        the override has something to fix. trl 1.x's base path aligns by byte offsets,
        which this trainer does not pass, so the parent cannot score these inputs at all;
        what is pinned is that its default strategy, "observed", is the one that reads the
        logits AT the labels (only the opt-in "bayesian" reads start-1). If trl's default
        moves to the shifted reading, this fails and fix 1 is to be re-judged."""
        batch = self._batch()
        parent = _parent_uld()
        parent.skip_student_eos = parent.skip_teacher_eos = False
        assert parent.token_merge_strategy == "observed"
        with pytest.raises(ValueError, match="Byte offsets are required"):
            parent(**batch)

    def test_an_eos_only_completion_is_scored_where_the_stop_is_predicted(self):
        """The EOS fix is only worth having at the right position: the stop decision is
        the last PROMPT position's distribution. Agree there and disagree past the end, and
        the loss must be zero; the reverse, and it must be the predictor's distance."""
        batch = _single_row([S_EOS], [T_EOS], seed=9)
        t_logits = batch["teacher_logits"]
        t_logits[..., 7:] = float("-inf")
        relabel = torch.randperm(7, generator=torch.Generator().manual_seed(10))
        agree_at_stop = {k: v.clone() for k, v in batch.items()}
        agree_at_stop["student_logits"][:, 0] = t_logits[:, 0][..., relabel]
        assert _custom_uld()(**agree_at_stop).item() == pytest.approx(0.0, abs=ULD_TOL)

        agree_past_end = {k: v.clone() for k, v in batch.items()}
        agree_past_end["student_logits"][:, 1] = t_logits[:, 1][..., relabel]
        loss = _custom_uld()(**agree_past_end).item()
        assert loss > 1e-3
        assert loss == pytest.approx(
            _uld_reference(agree_past_end, [[[0]]], [[[0]]]).item(), abs=ULD_TOL
        )


class TestULDUnalignedRow:
    """A row whose alignment yields NO groups. TRL's merge then returns the probabilities
    unmerged (gold_trainer.py:532-533), so the base path scored positions that were never
    paired: a shape RuntimeError when the completions differ in length, unless one side has
    a single position, which F.l1_loss broadcasts against the other's with only a warning;
    and unaligned positions against each other when they have the same length. The
    class's alignment returns no groups when the decoded completions differ before the
    first common boundary (a tokenizer or render mismatch: special tokens, whitespace,
    normalisation, truncation) or when matching texts reach max_alignment_group_size
    tokens on either side before one. Such a row now contributes no loss, as on the hybrid
    path, and is counted for `train/alignment_unaligned_rows`."""

    @staticmethod
    def _with_good_row(s_bad, t_bad):
        """Row 0 is the aligned "abcab" row of S_ROWS/T_ROWS; row 1 the unaligned one."""
        s_ids, s_labels = _rows([S_ROWS[0], ([1], s_bad)], S_PAD)
        t_ids, t_labels = _rows([T_ROWS[0], ([8], t_bad)], T_PAD)
        g = torch.Generator().manual_seed(11)
        return dict(
            student_logits=torch.randn(
                *s_ids.shape, len(S_PIECES), generator=g, dtype=DTYPE
            ),
            teacher_logits=torch.randn(
                *t_ids.shape, len(T_PIECES), generator=g, dtype=DTYPE
            ),
            student_labels=s_labels,
            teacher_labels=t_labels,
            student_input_ids=s_ids,
            teacher_input_ids=t_ids,
        )

    @staticmethod
    def _row0_reference(batch):
        """The aligned row on its own, i.e. the batch's row 0 with row 1 dropped."""
        return _uld_reference(
            {k: v[:1] for k, v in batch.items()}, S_GROUPS[:1], T_GROUPS[:1]
        )

    @pytest.mark.parametrize(
        "s_bad, t_bad",
        [
            # "ab" vs "abc", 3 positions vs 2: TRL's L1 raised a shape RuntimeError.
            ([1, 2, S_EOS], [5, T_EOS]),
            # "ab" vs "abc", 2 positions each: TRL compared position k with position k.
            ([4, S_EOS], [5, T_EOS]),
            # "ab" (no EOS) vs "abc", 1 position vs 2: TRL's L1 broadcast the one student
            # position against both teacher positions, with only a UserWarning.
            ([4], [5, T_EOS]),
        ],
        ids=["lengths-differ", "lengths-equal", "one-position-broadcasts"],
    )
    def test_texts_that_differ_contribute_nothing_and_are_counted(self, s_bad, t_bad):
        loss_fn = _custom_uld()
        assert loss_fn._build_alignment_groups_from_ids(s_bad, t_bad) == ([], [])
        batch = self._with_good_row(s_bad, t_bad)
        batch["student_logits"].requires_grad_(True)
        loss = loss_fn(**batch)
        # TRL's base-path mean is over rows, an empty row counting as zero (as it does
        # for a row with no completion), so the aligned row's loss is halved.
        expected = self._row0_reference(batch) / 2
        assert loss.item() == pytest.approx(expected.item(), abs=ULD_TOL)
        assert loss_fn._alignment_unaligned == 1
        assert loss_fn._alignment_rows == 2
        loss.backward()
        assert batch["student_logits"].grad[1].abs().max().item() == 0.0

    def test_the_hybrid_path_counts_an_unaligned_region(self):
        """The hybrid path skips an unaligned region the same way and must count it the
        same way, or the guard below cannot see a hybrid arm training on nothing."""
        loss_fn = _custom_uld(uld_use_hybrid_loss=True)
        assert loss_fn.use_hybrid_loss and loss_fn._vocab_mapping is not None
        batch = self._with_good_row([1, 2, S_EOS], [5, T_EOS])
        batch["student_logits"].requires_grad_(True)
        loss_fn(**batch).backward()
        assert loss_fn._alignment_unaligned == 1
        assert loss_fn._alignment_rows == 2
        assert batch["student_logits"].grad[1].abs().max().item() == 0.0
        assert batch["student_logits"].grad[0].abs().max().item() > 0.0

    def test_matching_texts_past_the_group_size_limit_contribute_nothing(self):
        """Not only a mismatch: "abcab" aligns at the default limit (S_GROUPS), but its
        first group needs two student tokens, so at max_alignment_group_size 1 no group
        closes and the row is unaligned too."""
        loss_fn = _custom_uld(max_alignment_group_size=1)
        s_c, t_c = S_ROWS[0][1], T_ROWS[0][1]
        assert loss_fn._build_alignment_groups_from_ids(s_c, t_c) == ([], [])
        batch = _single_row(s_c, t_c, seed=12)
        assert loss_fn(**batch).item() == 0.0
        assert loss_fn._alignment_unaligned == 1


class TestULDCompletionAtPositionZero:
    """Both paths read the logits at start-1. A completion at position 0 has no logit
    predicting its first token, and start-1 = -1 would slice from the end of the row, so
    each path refuses it rather than scoring the wrong positions."""

    @pytest.mark.parametrize("path", ["base", "hybrid"])
    @pytest.mark.parametrize("side", ["student", "teacher"])
    def test_raises(self, path, side):
        s_c, t_c = S_ROWS[0][1], T_ROWS[0][1]
        batch = _single_row(
            s_c,
            t_c,
            seed=14,
            s_prompt=() if side == "student" else (5,),
            t_prompt=() if side == "teacher" else (7,),
        )
        first = "student 0" if side == "student" else "teacher 0"
        loss_fn = _custom_uld(uld_use_hybrid_loss=(path == "hybrid"))
        assert loss_fn.use_hybrid_loss is (path == "hybrid")
        kind = "region" if path == "hybrid" else "row"
        with pytest.raises(ValueError, match=f"ULD {kind} 0: .*{first}"):
            loss_fn(**batch)


def _counts(unaligned, rows):
    """A stand-in for the loss object's two rank-local counters, which is all the guard
    reads."""
    return types.SimpleNamespace(_alignment_unaligned=unaligned, _alignment_rows=rows)


def _one_rank(values):
    return list(values)


class TestULDAlignmentGuard:
    """Skipping an unaligned row is allowed; skipping every row is not. Without the guard a
    step whose rows were all unaligned trained at loss 0.0 and grad_norm 0.0, and the run
    checkpointed and reported success: what the step's defaults (last_message_only=false,
    response_template <|im_start|>assistant) do to granite-4.2 data, where the student's
    span opens with '\\n<think></think>' and the teacher's does not."""

    def test_a_step_with_every_row_unaligned_raises(self):
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=1.0)
        with pytest.raises(RuntimeError, match="all 2 rows .* no alignment groups"):
            guard.end_step(_counts(2, 2), step=7, reduce=_one_rank, window_steps=10)

    def test_an_all_unaligned_batch_through_the_loss_raises(self):
        """The same, from a real loss call rather than hand-set counters."""
        loss_fn = _custom_uld()
        loss_fn(**_single_row([4, S_EOS], [5, T_EOS], seed=13))
        assert (loss_fn._alignment_unaligned, loss_fn._alignment_rows) == (1, 1)
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=1.0)
        with pytest.raises(RuntimeError, match="all 1 rows"):
            guard.end_step(loss_fn, step=1, reduce=_one_rank, window_steps=1)

    def test_a_rare_unaligned_row_is_skipped_and_the_counters_drained(self):
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.5)
        counts = _counts(1, 4)
        guard.end_step(counts, step=1, reduce=_one_rank, window_steps=1)
        assert (counts._alignment_unaligned, counts._alignment_rows) == (0, 0)
        assert guard.pop_log() == {
            "train/alignment_unaligned_rows": 1,
            "train/alignment_rows": 4,
            "train/alignment_unaligned_rate": 0.25,
        }
        assert guard.pop_log() == {}

    def test_no_aligned_rows_at_all_is_not_judged(self):
        """A step with no row that reached alignment (no completions, or an arm without
        extended alignment) has no rate; it is neither all-unaligned nor logged."""
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.0)
        guard.end_step(_counts(0, 0), step=1, reduce=_one_rank, window_steps=1)
        assert guard.pop_log() == {}

    def test_the_window_threshold(self):
        """Two steps of 1 in 4 unaligned: 0.25 over the window. Strictly above the
        threshold raises, at the step that closes the window; at it, it does not."""
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.2)
        guard.end_step(_counts(1, 4), step=1, reduce=_one_rank, window_steps=2)
        with pytest.raises(RuntimeError, match=r"steps 1-2: 2 of 8 rows \(25.0%\)"):
            guard.end_step(_counts(1, 4), step=2, reduce=_one_rank, window_steps=2)

        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.25)
        for step in (1, 2, 3, 4):
            guard.end_step(_counts(1, 4), step=step, reduce=_one_rank, window_steps=2)

    def test_the_final_partial_window_is_judged_at_train_end(self):
        """A run shorter than one window (max_steps < logging_steps), or the steps after
        the last full window, would otherwise never be judged."""
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.2)
        guard.end_step(_counts(1, 4), step=1, reduce=_one_rank, window_steps=10)
        guard.end_step(_counts(1, 4), step=2, reduce=_one_rank, window_steps=10)
        with pytest.raises(RuntimeError, match=r"steps 1-2: 2 of 8 rows"):
            guard.end_train()

    def test_train_end_judges_only_what_no_window_has(self):
        """A window that closed is not judged twice, and a run with no step since is
        not judged at all."""
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.2)
        guard.end_train()
        guard.end_step(_counts(0, 4), step=1, reduce=_one_rank, window_steps=1)
        guard.end_train()
        guard.end_step(_counts(0, 4), step=2, reduce=_one_rank, window_steps=1)
        guard.end_step(_counts(1, 4), step=3, reduce=_one_rank, window_steps=10)
        with pytest.raises(RuntimeError, match=r"steps 3-3: 1 of 4 rows"):
            guard.end_train()

    def test_the_window_resets_and_its_own_setting_wins(self):
        """A bad window followed by a clean one: the clean one is judged on its own rows.
        And `window_steps` set on the guard (uld_unaligned_window_steps) overrides the
        logging window the trainer passes."""
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.5, window_steps=1)
        guard.end_step(_counts(2, 4), step=1, reduce=_one_rank, window_steps=100)
        guard.end_step(_counts(0, 4), step=2, reduce=_one_rank, window_steps=100)
        with pytest.raises(RuntimeError, match="3 of 4 rows"):
            guard.end_step(_counts(3, 4), step=3, reduce=_one_rank, window_steps=100)

    def test_the_default_threshold_passes_one_bad_row_in_a_small_step(self):
        """The default is loose on purpose: 2 rows a step, logged every step, and one of
        them unaligned is exactly 0.5, which does not stop the run."""
        assert (
            CustomGOLDConfig.__dataclass_fields__["uld_max_unaligned_fraction"].default
            == 0.5
        )
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=0.5)
        guard.end_step(_counts(1, 2), step=1, reduce=_one_rank, window_steps=1)

    def test_the_trainer_wires_the_guard_to_its_loss_and_logging_window(self):
        """CustomGOLDTrainer._uld_alignment_step_end, on a single process: it drains the
        trainer's own loss object, numbers the step being taken, and defaults the window to
        the logging window."""
        loss_fn = _counts(1, 3)
        trainer = types.SimpleNamespace(
            accelerator=types.SimpleNamespace(
                distributed_type=cgt.DistributedType.NO, device=torch.device("cpu")
            ),
            args=types.SimpleNamespace(logging_steps=2),
            uld_loss_fn=loss_fn,
            _uld_alignment_guard=cgt.ULDAlignmentGuard(max_unaligned_fraction=0.3),
        )
        state = types.SimpleNamespace(global_step=4, logging_steps=2)
        CustomGOLDTrainer._uld_alignment_step_end(trainer, state)  # window 1 of 2: 1/3
        assert (loss_fn._alignment_unaligned, loss_fn._alignment_rows) == (0, 0)
        loss_fn._alignment_unaligned, loss_fn._alignment_rows = 1, 3
        with pytest.raises(RuntimeError, match="ULD steps 5-6: 2 of 6 rows"):
            CustomGOLDTrainer._uld_alignment_step_end(
                trainer, types.SimpleNamespace(global_step=5, logging_steps=2)
            )

    def test_the_rate_is_reduced_across_ranks(self, tmp_path):
        """Two real gloo ranks, through CustomGOLDTrainer._uld_alignment_step_end with a
        distributed type other than NO. Rank 0 aligned both its rows, rank 1 neither of
        its: the step is NOT all-unaligned (2 of 4), and both ranks log the global rate
        0.5, not their own 0.0 and 1.0. Then both ranks all-unaligned: both raise,
        together, so neither is left waiting in a collective. Rank-local judging (a forced
        or inverted `distributed`) makes rank 1 raise alone at the first step: on N GPUs,
        one rank dies and the others hang."""
        mp = pytest.importorskip("torch.multiprocessing")
        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available")
        mp.spawn(
            _guard_rank,
            args=(2, str(tmp_path / "rdzv"), str(tmp_path)),
            nprocs=2,
            join=True,
        )
        for rank in (0, 1):
            log, raised = (tmp_path / f"rank{rank}.txt").read_text().split("|")
            assert log == "2/4/0.5"
            assert raised == "raised"


def _guard_rank(rank, world, rendezvous, out_dir):
    """One rank of test_the_rate_is_reduced_across_ranks (module level: spawn pickles it)."""
    import os
    from pathlib import Path

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world
    )
    try:
        # Through the trainer's own step-end path, with a distributed type that is not NO,
        # as accelerate reports it on N GPUs (MULTI_GPU) or N CPU processes (MULTI_CPU): what
        # _uld_alignment_step_end derives `distributed` from. Forced or inverted, the reduction
        # is skipped and each rank judges its own rows.
        guard = cgt.ULDAlignmentGuard(max_unaligned_fraction=1.0)
        trainer = types.SimpleNamespace(
            accelerator=types.SimpleNamespace(
                distributed_type=cgt.DistributedType.MULTI_CPU,
                device=torch.device("cpu"),
            ),
            args=types.SimpleNamespace(logging_steps=1),
            uld_loss_fn=_counts(0, 2) if rank == 0 else _counts(2, 2),
            _uld_alignment_guard=guard,
        )
        state = types.SimpleNamespace(global_step=0, logging_steps=1)
        CustomGOLDTrainer._uld_alignment_step_end(trainer, state)
        log = guard.pop_log()
        result = "%d/%d/%s" % (
            log["train/alignment_unaligned_rows"],
            log["train/alignment_rows"],
            log["train/alignment_unaligned_rate"],
        )
        try:
            trainer.uld_loss_fn = _counts(2, 2)
            state.global_step = 1
            CustomGOLDTrainer._uld_alignment_step_end(trainer, state)
            outcome = "did not raise"
        except RuntimeError as e:
            outcome = "raised" if "all 4 rows" in str(e) else f"wrong: {e}"
        Path(os.path.join(out_dir, f"rank{rank}.txt")).write_text(f"{result}|{outcome}")
    finally:
        torch.distributed.destroy_process_group()
