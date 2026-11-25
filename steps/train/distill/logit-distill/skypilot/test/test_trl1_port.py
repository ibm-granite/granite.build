# SPDX-License-Identifier: Apache-2.0
"""The behaviour decisions of the trl 0.26.2 -> 1.x port of the GOLD trainer (#676).

Each test pins one decision, against the real trl when it is installed (the step images):

- D-1: the data path stays the Trainer's (no GOLD buffer, sampler, dataloader or batch fetch).
- weight sync: the 0.26.2 callback is back, delegating to trl's VLLMGeneration.sync_weights.
- D-5: prompts go to vLLM as the trainer's token IDs.
- D-2/D-4: the sampled token's logprob is index 0 of trl's [seq][tok][k] structure.
- D-3: a NaN logprob (None from trl's client) masks the token out of importance sampling.
- D-7: the ULD probability merge is trl 0.26.2's, bit for bit.
- MultiVLLMClient fans generation and weight updates out over the servers.
"""

import math
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
transformers = pytest.importorskip("transformers")

from gb_steps_post_training.distillation import custom_gold_trainer as cgt  # noqa: E402
from gb_steps_post_training.distillation.custom_gold_trainer import (  # noqa: E402
    CustomGOLDTrainer,
    CustomULDLoss,
    MultiVLLMClient,
)
from trl.experimental.gold.gold_trainer import GOLDTrainer  # noqa: E402


def _bare_trainer():
    """A CustomGOLDTrainer with no __init__ run: enough to call the methods under test."""
    return object.__new__(CustomGOLDTrainer)


# --- D-1 -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, args",
    [
        ("_prepare_inputs", ({"x": 1},)),
        ("get_train_dataloader", ()),
        ("get_batch_samples", ("it", 2, "cpu")),
    ],
)
def test_the_data_path_skips_gold_and_reaches_the_trainer(monkeypatch, name, args):
    """GOLDTrainer overrides these to build its generation buffer; this trainer must land
    on the Trainer's (or SFTTrainer's) instead."""
    assert name in vars(GOLDTrainer), f"trl's GOLDTrainer no longer overrides {name}"
    calls = []
    monkeypatch.setattr(
        transformers.Trainer,
        name,
        lambda self, *a: calls.append(a) or "trainer",
        raising=True,
    )
    monkeypatch.setattr(
        GOLDTrainer, name, lambda self, *a: pytest.fail("GOLD's buffered path ran")
    )
    assert getattr(_bare_trainer(), name)(*args) == "trainer"
    assert calls == [args]


def test_the_sampler_is_the_trainers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        transformers.Trainer,
        "_get_train_sampler",
        lambda self, *a, **k: calls.append("trainer") or "sampler",
    )
    monkeypatch.setattr(
        GOLDTrainer,
        "_get_train_sampler",
        lambda self, *a, **k: pytest.fail("GOLD's RepeatSampler was chosen"),
    )
    t = _bare_trainer()
    t.args = types.SimpleNamespace(train_sampling_strategy="random")
    t.train_dataset = None
    assert t._get_train_sampler() == "sampler"
    assert calls == ["trainer"]


# --- weight sync ----------------------------------------------------------------------


def test_weights_sync_every_vllm_sync_frequency_steps():
    synced = []
    trainer = types.SimpleNamespace(
        _last_vllm_sync_step=-2,
        vllm_sync_frequency=2,
        accelerator=types.SimpleNamespace(sync_gradients=True),
    )
    trainer._move_model_to_vllm = lambda: synced.append(True)
    cb = cgt._VLLMWeightSyncCallback(trainer)
    for step in range(1, 7):
        cb.on_step_end(None, types.SimpleNamespace(global_step=step), None)
    assert len(synced) == 3  # steps 2, 4, 6
    assert trainer._last_vllm_sync_step == 6
    cb.on_step_end(None, types.SimpleNamespace(global_step=6), None)
    assert len(synced) == 3  # never twice for one step


def test_a_resumed_run_syncs_before_its_first_rollout():
    synced = []
    trainer = types.SimpleNamespace(_last_vllm_sync_step=-1, vllm_sync_frequency=10)
    trainer._move_model_to_vllm = lambda: synced.append(True)
    cb = cgt._VLLMWeightSyncCallback(trainer)
    cb.on_train_begin(None, types.SimpleNamespace(global_step=0), None)
    assert synced == []  # a fresh run: the server already holds these weights
    cb.on_train_begin(None, types.SimpleNamespace(global_step=1200), None)
    assert synced == [True] and trainer._last_vllm_sync_step == 1200


def test_move_model_to_vllm_is_trls_sync_weights():
    t = _bare_trainer()
    calls = []
    t.vllm_generation = types.SimpleNamespace(
        sync_weights=lambda: calls.append(True),
        _weight_metadata=[("a", "bf16", [1])] * 3,
    )
    t._timing_weight_sync = 0.0
    t._move_model_to_vllm()
    assert calls == [True]
    assert t._timing_weight_sync_params == 3


# --- D-5: token-ID prompts --------------------------------------------------------------


def test_prompts_are_the_trainers_token_ids_without_padding():
    inputs = {
        "prompts": torch.tensor([[0, 0, 5, 6, 7], [8, 9, 10, 11, 12]]),
        "prompt_attention_mask": torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
    }
    assert CustomGOLDTrainer._prompt_token_ids(inputs, 100) == [
        [5, 6, 7],
        [8, 9, 10, 11, 12],
    ]
    # Over the prompt budget: the LAST tokens are kept, as the text path truncated.
    assert CustomGOLDTrainer._prompt_token_ids(inputs, 2) == [[6, 7], [11, 12]]


def test_the_request_is_built_from_vllm_generation():
    t = _bare_trainer()
    t.vllm_generation = types.SimpleNamespace(
        repetition_penalty=1.0,
        temperature=0.7,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        max_completion_length=64,
        logprobs=0,
        structured_outputs_regex=None,
        generation_kwargs={"min_tokens": 4},
    )
    kw = t._vllm_request_kwargs()
    assert kw["logprobs"] == 0 and kw["temperature"] == 0.7 and kw["max_tokens"] == 64
    assert kw["generation_kwargs"] == {"min_tokens": 4}
    # trl's client keyword, not 0.26.2's guided_decoding_regex.
    assert "structured_outputs_regex" in kw and "guided_decoding_regex" not in kw


# --- D-2/D-4/D-3: logprobs ---------------------------------------------------------------


def test_the_sampled_logprob_is_index_0_and_none_becomes_nan():
    # trl's client: [seq][tok][k], sorted, NaN as None; logprobs=0 -> k == 1.
    lps = [[[-0.1], [None], [-2.5]], [[-0.3]]]
    out = CustomGOLDTrainer._sampled_token_logprobs(lps, [[1, 2, 3], [4]])
    assert out[0][0] == pytest.approx(-0.1) and math.isnan(out[0][1])
    assert out[0][2] == pytest.approx(-2.5) and out[1] == [pytest.approx(-0.3)]


def test_no_logprobs_at_all_is_nan_not_zero():
    out = CustomGOLDTrainer._sampled_token_logprobs(None, [[1, 2], [3]])
    assert all(math.isnan(x) for row in out for x in row) and [len(r) for r in out] == [
        2,
        1,
    ]


def test_a_nan_logprob_is_masked_out_of_importance_sampling():
    """Compare against the same batch with the NaN token removed: identical loss, and the
    stats describe only the tokens that have a weight."""
    torch.manual_seed(0)
    s = torch.randn(1, 4, 7)
    t = torch.randn(1, 4, 7)
    labels = torch.tensor([[1, 2, 3, 4]])
    s_lp = torch.log_softmax(s, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    vllm = s_lp.detach().clone()  # IS weight 1 everywhere: inside any truncation band
    vllm_nan = vllm.clone()
    vllm_nan[0, 2] = float("nan")

    loss_nan, stats_nan = CustomGOLDTrainer.sampled_opd_loss(
        s, t, labels, vllm_logprobs=vllm_nan
    )
    keep = [0, 1, 3]
    loss_ref, stats_ref = CustomGOLDTrainer.sampled_opd_loss(
        s[:, keep], t[:, keep], labels[:, keep], vllm_logprobs=vllm[:, keep]
    )
    assert torch.isfinite(loss_nan)
    assert loss_nan.item() == pytest.approx(loss_ref.item(), abs=1e-6)
    assert all(math.isfinite(v) for v in stats_nan.values())
    assert stats_nan["mean"] == pytest.approx(stats_ref["mean"])
    assert stats_nan["masked_fraction"] == pytest.approx(0.25)


# --- D-7: the ULD merge ---------------------------------------------------------------


def _merge_0262(probs, groups):
    """trl 0.26.2's _merge_probabilities_with_alignment_groups, restated for reference."""
    if not groups:
        return probs
    out = torch.zeros(len(groups), probs.size(-1))
    for i, g in enumerate(groups):
        if len(g) > 1:
            logp = torch.log(probs[g[0]].clamp_min(1e-8))
            for idx in g[1:]:
                if idx < probs.size(0):
                    logp = logp + torch.log(probs[idx].clamp_min(1e-8))
            out[i] = torch.softmax(logp, dim=-1)
        elif len(g) == 1:
            out[i] = probs[g[0]]
    return out


def test_the_uld_merge_is_trl_0262s_bit_for_bit():
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(6, 11), -1)
    groups = [[0], [1, 2], [3, 4, 5]]
    loss = object.__new__(CustomULDLoss)
    got = loss._merge_probabilities_with_alignment_groups(probs, groups)
    assert torch.equal(got, _merge_0262(probs, groups))
    # Multi-token groups without token_ids: trl 1.x raises, the vendored merge does not.
    assert got.shape == (3, 11)
    # No groups: unmerged, as in 0.26.2 (the caller skips such rows).
    assert loss._merge_probabilities_with_alignment_groups(probs, []) is probs


# --- MultiVLLMClient ---------------------------------------------------------------------


class _FakeClient:
    def __init__(self, tag):
        self.tag, self.sent, self.updates = tag, [], 0
        self.base_url = f"http://{tag}:1"

    def generate(self, prompts, **kw):
        return {
            "prompt_ids": prompts,
            "completion_ids": [[self.tag]] * len(prompts),
            "logprobs": [[[-1.0]]] * len(prompts),
            "logprob_token_ids": [[[self.tag]]] * len(prompts),
        }

    def weight_update(self):
        client = self

        class _Ctx:
            def __enter__(self):
                client.updates += 1

            def __exit__(self, *a):
                return False

        return _Ctx()

    def update_named_params(self, metadata, named_params):
        self.sent.extend(name for name, _ in named_params)


def _multi(n):
    m = object.__new__(MultiVLLMClient)
    m.clients = [_FakeClient(i) for i in range(n)]
    m._streams = [types.SimpleNamespace(synchronize=lambda: None) for _ in range(n)]
    return m


def test_multi_client_splits_and_merges_in_prompt_order():
    m = _multi(2)
    out = m.generate(prompts=[[1], [2], [3]], n=1)
    assert out["prompt_ids"] == [[1], [2], [3]]
    assert out["completion_ids"] == [[0], [0], [1]]  # 2 to server 0, 1 to server 1
    assert len(out["logprobs"]) == 3 and len(out["logprob_token_ids"]) == 3


def test_multi_client_streams_every_tensor_to_every_server_in_one_update(monkeypatch):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda s: None)
    m = _multi(3)
    meta = [("a", "bfloat16", [1]), ("b", "bfloat16", [1])]
    m.update_named_params(meta, iter([("a", torch.zeros(1)), ("b", torch.zeros(1))]))
    for c in m.clients:
        assert c.sent == ["a", "b"]
        assert c.updates == 1  # one weight update per server per sync
