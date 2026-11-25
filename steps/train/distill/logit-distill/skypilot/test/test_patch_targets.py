# SPDX-License-Identifier: Apache-2.0
"""patch_targets.py's source probes, against fake upstream trees.

The probes read installed upstream source (located without importing it), so each is
tested here with a minimal fake tree in both directions: the shape the port relies on
passes, and a moved or reverted upstream is reported. One test runs the whole report
against the real stack when it is installed (the GPU image).
"""

import textwrap
from pathlib import Path

import pytest
from gb_steps_post_training.distillation import patch_targets as pt


@pytest.fixture()
def fake_tree(tmp_path, monkeypatch):
    """Write files under tmp_path/<package>/... and point _source_path at them."""

    def write(package, relative, text):
        path = tmp_path / package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")

    def source_path(package, relative):
        path = tmp_path / package / relative
        return path if path.is_file() else None

    monkeypatch.setattr(pt, "_source_path", source_path)
    return write


_GOLD = "experimental/gold/gold_trainer.py"


def _uld_source(drop=()):
    methods = "".join(
        f"    def {m}(self):\n        pass\n" for m in pt._ULD_METHODS if m not in drop
    )
    attrs = "".join(
        f"        self.{a} = None\n" for a in pt._ULD_ATTRIBUTES if a not in drop
    )
    return f"class ULDLoss:\n    def _setup(self):\n{attrs}\n{methods}"


def test_gold_overrides_are_needed_while_trl_defines_them(fake_tree):
    fake_tree(
        "trl",
        _GOLD,
        """
        class GOLDTrainer:
            def _prepare_inputs(self, inputs): pass
            def get_train_dataloader(self): pass
    """,
    )
    p = pt.probe_gold_trainer_overrides()
    assert p.exists and p.needed
    assert "_prepare_inputs" in p.detail


def test_gold_overrides_look_moot_when_trl_drops_them(fake_tree):
    fake_tree("trl", _GOLD, "class GOLDTrainer:\n    pass\n")
    p = pt.probe_gold_trainer_overrides()
    assert p.exists and p.needed is False


def test_a_missing_gold_module_is_missing(fake_tree):
    p = pt.probe_gold_trainer_overrides()
    assert p.exists is False


def test_uld_surface_passes_when_complete(fake_tree):
    fake_tree("trl", _GOLD, _uld_source())
    assert pt.probe_uld_loss_surface().exists


@pytest.mark.parametrize("dropped", ["_get_start_and_size_answers", "_vocab_mapping"])
def test_uld_surface_names_what_is_gone(fake_tree, dropped):
    fake_tree("trl", _GOLD, _uld_source(drop=(dropped,)))
    p = pt.probe_uld_loss_surface()
    assert p.exists is False and dropped in p.detail


def _client_source(
    generate_args="self, prompts, logprobs=0, structured_outputs_regex=None, generation_kwargs=None",
):
    return f"""
        class VLLMClient:
            def generate({generate_args}): pass
            def init_communicator(self, device=0): pass
            def weight_update(self): pass
            def update_named_params(self, metadata, named_params): pass
            def reset_prefix_cache(self): pass
    """


def test_vllm_client_api_passes(fake_tree):
    fake_tree("trl", "generation/vllm_client.py", _client_source())
    fake_tree(
        "trl",
        "generation/vllm_generation.py",
        "class VLLMGeneration:\n    def sync_weights(self): pass\n",
    )
    assert pt.probe_vllm_client_api().exists


def test_vllm_client_api_reports_a_lost_parameter(fake_tree):
    fake_tree(
        "trl",
        "generation/vllm_client.py",
        _client_source("self, prompts, generation_kwargs=None"),
    )
    fake_tree(
        "trl",
        "generation/vllm_generation.py",
        "class VLLMGeneration:\n    def sync_weights(self): pass\n",
    )
    p = pt.probe_vllm_client_api()
    assert p.exists is False and "logprobs" in p.detail


_MODELING = "models/granitemoehybrid/modeling_granitemoehybrid.py"


def test_hub_kernel_targets_need_registration(fake_tree):
    fake_tree(
        "transformers", _MODELING, "\n".join(f"@{d}" for d, _ in pt._KERNEL_DECORATIONS)
    )
    p = pt.probe_hub_kernel_registration()
    assert p.exists and p.needed


def test_hub_kernel_targets_report_a_return_to_lazy_loading(fake_tree):
    fake_tree("transformers", _MODELING, 'mamba_ssm = lazy_load_kernel("mamba-ssm")\n')
    p = pt.probe_hub_kernel_registration()
    assert p.exists is False and "lazy_load_kernel" in p.detail


def test_deleted_rope_patch_stays_deleted(fake_tree):
    fake_tree(
        "vllm",
        "model_executor/models/granite.py",
        "class GraniteAttention:\n    def f(self, config):\n        x = config.rope_parameters\n",
    )
    fake_tree(
        "vllm",
        "model_executor/models/granitemoehybrid.py",
        "class GraniteMoeHybridAttention:\n    def f(self, config):\n        get_rope(rope_parameters=config.rope_parameters)\n",
    )
    probes = pt.probe_vllm_rope_parameters()
    assert all(p.exists and p.deleted and p.needed is False for p in probes)


def test_deleted_rope_patch_is_reported_when_the_bug_returns(
    fake_tree, capsys, monkeypatch
):
    fake_tree(
        "vllm",
        "model_executor/models/granite.py",
        "class GraniteDecoderLayer:\n    def f(self, config):\n        t = config.rope_theta\n",
    )
    fake_tree(
        "vllm",
        "model_executor/models/granitemoehybrid.py",
        "class GraniteMoeHybridAttention:\n    pass\n",
    )
    probes = pt.probe_vllm_rope_parameters()
    assert all(p.needed for p in probes)
    # And main() treats a returned bug as a failure, not as a MOOT cleanup.
    for name in (
        "probe_deepspeed_defaults",
        "probe_generation_aliases",
        "probe_trl_bool_guards",
        "probe_gold_trainer_overrides",
        "probe_uld_loss_surface",
        "probe_vllm_client_api",
        "probe_hub_kernel_registration",
    ):
        monkeypatch.setattr(
            pt, name, lambda: pt.Probe("ok", "here", True, True, "fine")
        )
    assert pt.main([]) == 1
    assert "bug is BACK" in capsys.readouterr().out


def test_the_real_stack_reports_no_broken_target():
    """In the GPU image: every target the port relies on exists, and no deleted patch's
    bug is back."""
    pytest.importorskip("deepspeed")
    pytest.importorskip("trl")
    pytest.importorskip("vllm")  # the rope guards read vLLM's source
    assert pt.main([]) == 0
