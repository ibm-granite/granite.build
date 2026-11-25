# SPDX-License-Identifier: Apache-2.0
"""How precompute_logits loads the teacher: which device, which dtype, under transformers 5.16+.

The teacher-logits build test on dev-676-5e931046 died in the lm_head probe with
"Expected all tensors to be on the same device, but got mat2 is on cpu": transformers 5.16+
turns a bare tp_plan="auto" into DistributedConfig(tp_size=None), discards the caller's
device_mesh, and at tp_size 1 places nothing, so the teacher stayed on the CPU. No test
looked at the from_pretrained arguments. These do, and one loads a tiny model through the
real transformers with them.
"""

import warnings

import pytest
from gb_steps_post_training.distillation import precompute_logits as P


def test_one_gpu_per_rank_is_placed_by_device_map():
    kw = P.teacher_load_kwargs("bf16", tp_size=1, mesh_dp=1, device="cuda:0")
    assert kw["device_map"] == {"": "cuda:0"}
    assert kw["dtype"] == "bf16"
    # The keys that silently left the model on the CPU, or warn as deprecated.
    assert not {"tp_plan", "device_mesh", "torch_dtype", "distributed_config"} & set(kw)


def test_several_gpus_use_transformers_tensor_parallel_config():
    pytest.importorskip("transformers")
    kw = P.teacher_load_kwargs("bf16", tp_size=8, mesh_dp=1, device="cuda:3")
    cfg = kw["distributed_config"]
    assert cfg.tp_size == 8 and cfg.tp_plan == "auto"
    # TP and device_map are mutually exclusive in transformers (prepare_distribute_model).
    assert "device_map" not in kw and "torch_dtype" not in kw


def test_a_multi_node_rendezvous_is_refused():
    with pytest.raises(RuntimeError, match="rendezvous"):
        P.teacher_load_kwargs("bf16", tp_size=8, mesh_dp=2, device="cuda:0")


def test_the_real_transformers_places_and_casts_the_model(tmp_path):
    """The TP-1 keywords through the real from_pretrained: the weights land on the requested
    device in the requested dtype, with no torch_dtype deprecation warning."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    transformers.LlamaForCausalLM(cfg).save_pretrained(tmp_path)
    kw = P.teacher_load_kwargs(torch.bfloat16, tp_size=1, mesh_dp=1, device="cpu")
    kw.pop("trust_remote_code")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path, **kw)
    lm_head = model.get_output_embeddings()
    assert lm_head.weight.device.type == "cpu"
    assert lm_head.weight.dtype == torch.bfloat16
    assert not [w for w in caught if "torch_dtype" in str(w.message)]
