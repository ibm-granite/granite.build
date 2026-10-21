"""Wrapper for trl's vllm_serve that patches compatibility bugs.

Patches applied:
1. TRL 0.26.2 _vllm_ascend_available tuple bug
2. vLLM 0.11.0 missing rope_theta attribute on GraniteMoeHybridConfig
2b. Same rope_theta fix for regular GraniteForCausalLM (GraniteDecoderLayer)
3. NaN logprobs → 0.0 to prevent Pydantic ResponseValidationError (HTTP 500)
"""

import importlib

# Patch 1: TRL stores _is_package_available results as raw tuples.
import_utils = importlib.import_module("trl.import_utils")
import_utils._vllm_ascend_available = False

# Patch 2: vLLM accesses config.rope_theta directly, but transformers 5.x
# stores it inside config.rope_parameters dict. Monkey-patch the init.
_vllm_gmh = importlib.import_module("vllm.model_executor.models.granitemoehybrid")
_orig_gmh_attn_init = _vllm_gmh.GraniteMoeHybridAttention.__init__

def _patched_gmh_attn_init(self, config, *args, _orig=_orig_gmh_attn_init, **kwargs):
    if not hasattr(config, "rope_theta"):
        rp = getattr(config, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            config.rope_theta = rp["rope_theta"]
        else:
            config.rope_theta = 10000
    _orig(self, config, *args, **kwargs)

_vllm_gmh.GraniteMoeHybridAttention.__init__ = _patched_gmh_attn_init
del _vllm_gmh, _orig_gmh_attn_init

# Patch 2b: Same issue for regular GraniteForCausalLM (not MoeHybrid).
# GraniteDecoderLayer.__init__ reads config.rope_theta with a fallback to 10000,
# which is wrong for models like granite-4.1-20b that have rope_theta=50000000.
_vllm_granite = importlib.import_module("vllm.model_executor.models.granite")
_orig_granite_layer_init = _vllm_granite.GraniteDecoderLayer.__init__

def _patched_granite_layer_init(self, config, *args, _orig=_orig_granite_layer_init, **kwargs):
    if not hasattr(config, "rope_theta"):
        rp = getattr(config, "rope_parameters", None)
        if isinstance(rp, dict) and "rope_theta" in rp:
            config.rope_theta = rp["rope_theta"]
        else:
            config.rope_theta = 10000
    _orig(self, config, *args, **kwargs)

_vllm_granite.GraniteDecoderLayer.__init__ = _patched_granite_layer_init
del _vllm_granite, _orig_granite_layer_init

# Patch 3: sanitize_logprob returns None for NaN logprobs, but the Pydantic
# response schema (GenerateResponse.logprobs: list[list[float]]) rejects None,
# causing a FastAPI ResponseValidationError → HTTP 500. Replace NaN with 0.0.
import math as _math
_vllm_serve = importlib.import_module("trl.scripts.vllm_serve")

def _patched_sanitize_logprob(logprob):
    value = logprob.logprob
    if _math.isnan(value):
        return 0.0
    return value

_vllm_serve.sanitize_logprob = _patched_sanitize_logprob
del _vllm_serve

# Patch 4: register `granite_swa` with the HF Auto* APIs (transformers 5.8.0 has
# no upstream registration) and with vLLM's ModelRegistry, using the vendored
# modeling files. Unlike patches 1-3 above -- which fix genuine upstream bugs that
# affect every model -- this one only matters when the served model IS a GraniteSWA,
# and the vendored package is not part of this collection. Gated on the SWA arm so
# that serving a granite 4.0/4.1 student does not require it. See _swa_arm.py.
from gb_steps_post_training.distillation._swa_arm import (
    activate_swa_arm,
    activate_swa_vllm_model,
)
activate_swa_arm()
activate_swa_vllm_model()

from trl.scripts.vllm_serve import main, make_parser

if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()

    # Pre-flight tokenizer check: vLLM's `tokenizer_mode="auto"` path uses
    # `AutoTokenizer.from_pretrained(...)` under the hood, which silently
    # overrides the trained pre_tokenizer with GPT2Tokenizer's plain ByteLevel
    # when tokenizer_config.json declares `tokenizer_class: "GPT2Tokenizer"`.
    # Detect this here — before vLLM binds a port — so LSF jobs die loudly
    # instead of streaming wrong-tokenized prompts through on-policy rollouts.
    #
    # NB the trigger is that config key, NOT the presence of legacy
    # vocab.json + merges.txt sidecars, which are inert under this tree's
    # transformers 5.8.0, confirmed by direct measurement. An earlier revision of this
    # comment said sidecars; see docs/tokenizer_mismatch.md, which now carries
    # the variant table. The check below is unaffected and correct either way:
    # verify_fast_tokenizer compares the resolved backend against
    # tokenizer.json rather than keying on which files are present.
    from transformers import AutoTokenizer

    from gb_steps_post_training.distillation.utils import verify_fast_tokenizer

    _model_path = getattr(script_args, "model", None) or getattr(script_args, "model_name_or_path", None)
    if _model_path is not None:
        _tok = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)
        verify_fast_tokenizer(_tok, _model_path, source_label="vllm-server")

    main(script_args)
