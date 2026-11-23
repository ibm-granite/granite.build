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
# What the reference follows. `loss_arm: uld` renders `use_uld_loss: true` and nothing else
# (render_gold_config.py LOSS_ARMS["uld"]), so every other ULD knob is a dataclass default:
# use_extended_uld True, uld_use_hybrid_loss False, cross-entropy weight 0, distillation
# weight 1, both temperatures 1. With the hybrid loss off, CustomULDLoss hands the whole
# computation to TRL's ULDLoss._compute_distillation_loss (trl 0.26.2,
# trl/experimental/gold/gold_trainer.py:325-423), and overrides only two things it uses:
# skip_student_eos / skip_teacher_eos forced False, and _build_alignment_groups_from_ids,
# which aligns the bodies and appends the two EOS positions as one final group. Per row,
# TRL's base path:
#
#   1. takes the completion as [first label, first label + label count) (:367-368) -- the
#      logits AT those positions, i.e. each one's prediction of the NEXT token. The hybrid
#      path (custom_gold_trainer.py:2023) reads start-1, the prediction of the token itself;
#      the reference follows the base path, which is the one the arm runs;
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


def _uld_reference(batch, s_groups, t_groups):
    """Sorted-distribution ULD, written out (see the block comment above for each step)."""

    def merged(logits, labels, groups):
        probs = F.softmax(logits[labels != IGNORE], dim=-1)  # step 1-2: AT the labels
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
