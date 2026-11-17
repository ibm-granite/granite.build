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
"""

import math
import types

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional
pytest.importorskip("trl")  # custom_gold_trainer imports trl at module load

from gb_steps_post_training.distillation.custom_gold_trainer import (  # noqa: E402
    CustomGOLDTrainer,
)
from gb_steps_post_training.distillation.sft import CustomSFTTrainer  # noqa: E402

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
