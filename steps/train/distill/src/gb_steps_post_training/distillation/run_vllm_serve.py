# SPDX-License-Identifier: Apache-2.0

"""Start the student's rollout server through trl's `vllm-serve`, after a tokenizer check.

WHAT trl's `vllm-serve` IS NOW (trl 1.x). `trl.scripts.vllm_serve.main` no longer runs a
server of its own: it builds a `vllm serve` command line (NCCL weight-transfer engine,
processed logprobs, `--max-logprobs -1`) and `os.execve`s it with
`VLLM_SERVER_DEV_MODE=1`. The process that answers requests is vLLM's OpenAI-compatible
server, with the same PID as this one.

So nothing patched in THIS interpreter reaches the server, and this wrapper no longer
patches anything. What it used to patch, and where each went (#676):

1. trl 0.26.2's tuple-valued `_vllm_ascend_available` guard: gone in trl 1.x, whose
   `import_utils` returns plain bools.
2. / 2b. vLLM 0.11.0 reading `config.rope_theta` in `GraniteMoeHybridAttention` and
   `GraniteDecoderLayer`: vLLM 0.34.0 reads `config.rope_parameters` itself.
3. trl 0.26.2's `sanitize_logprob` turning a NaN logprob into None, which its response
   schema rejected (HTTP 500): vLLM's server serialises NaN, trl's client turns it into
   None, and the trainer masks that token out of its importance-sampling terms.
4. `granite_swa` registration with vLLM's ModelRegistry: a registration made here is lost
   at the exec, so the SWA arm cannot be served this way at all. It is refused below
   rather than served without the model class.

What is left is the tokenizer pre-flight, which runs before the exec and so still guards
the server, and the hand-off to trl.
"""

from gb_steps_post_training.distillation import _swa_arm

_SWA_IN_SERVER_MODE = (
    "GOLD_SWA_ARM=1, but the rollout server is vLLM's own server, which trl's vllm-serve "
    "execs: a granite_swa registration made in this process cannot reach it, so the SWA "
    "student would not be servable. (The arm is also not ported; see _swa_arm.py.)"
)


def main(argv=None) -> None:
    if _swa_arm.SWA_ARM:
        raise RuntimeError(_SWA_IN_SERVER_MODE)

    from trl.scripts.vllm_serve import main as trl_main
    from trl.scripts.vllm_serve import make_parser

    parser = make_parser()
    # Flags trl's ScriptArguments does not know are passed to `vllm serve` as-is, the way
    # `trl vllm-serve` itself does.
    script_args, extra_args = parser.parse_args_and_config(
        args=argv, return_remaining_strings=True
    )

    # Pre-flight tokenizer check: vLLM's `tokenizer_mode="auto"` path uses
    # `AutoTokenizer.from_pretrained(...)` under the hood, which silently
    # overrides the trained pre_tokenizer with GPT2Tokenizer's plain ByteLevel
    # when tokenizer_config.json declares `tokenizer_class: "GPT2Tokenizer"`.
    # Detect this here — before vLLM binds a port — so jobs die loudly
    # instead of streaming wrong-tokenized prompts through on-policy rollouts.
    #
    # NB the trigger is that config key, NOT the presence of legacy
    # vocab.json + merges.txt sidecars, which are inert under transformers 5.x,
    # confirmed by direct measurement on 5.8.0. See docs/tokenizer_mismatch.md, which
    # carries the variant table. The check is unaffected either way:
    # verify_fast_tokenizer compares the resolved backend against
    # tokenizer.json rather than keying on which files are present.
    from gb_steps_post_training.distillation.utils import verify_fast_tokenizer
    from transformers import AutoTokenizer

    model_path = getattr(script_args, "model", None)
    if model_path is not None:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        verify_fast_tokenizer(tok, model_path, source_label="vllm-server")

    # Does not return: execs `vllm serve`.
    trl_main(script_args, extra_args)


if __name__ == "__main__":
    main()
