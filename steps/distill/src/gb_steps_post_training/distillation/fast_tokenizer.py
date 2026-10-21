"""Load a fast tokenizer the way every step in this collection must: never AutoTokenizer.

WHY THIS MODULE EXISTS AT ALL. A Granite directory's `tokenizer_config.json` declares
`tokenizer_class: "GPT2Tokenizer"`. `AutoTokenizer.from_pretrained` honours that, constructs
GPT2TokenizerFast, and that class imposes its OWN plain ByteLevel `pre_tokenizer` over the
one stored in `tokenizer.json`. It does not error. It silently mis-segments: 26.1 versus
3.29 PPL/token on the granite-4.1 base student, whose trained pre_tokenizer is
`Sequence[Split(regex), ByteLevel]` (measured directly across three separate runs, and see
docs/tokenizer_mismatch.md). Deleting the key is not enough either: with a `config.json`
present, the `model_type: granite` fallback through TOKENIZER_MAPPING_NAMES revives the
override.

`PreTrainedTokenizerFast(tokenizer_file=...)` loads the serialized tokenizer and nothing
else, so it is immune BY CONSTRUCTION rather than by remembering to pass the right
directory. `retag_student.py` and `prep_corpus.py` each arrived at this independently; the
third consumer (`distill-eval`'s divergence metrics) is where it became worth sharing, and
the SECOND time it was worth sharing -- an earlier exploratory JSD script called
`AutoTokenizer.from_pretrained` and would have carried the bug into the ported step.

That call is worth spelling out, because "the tokenizer is slightly wrong" sounds cosmetic
for a divergence metric and is not. Both models are fed the SAME mis-segmented ids, so the
computation does not crash and the numbers look plausible -- they are simply measured on text
in a segmentation neither model was ever trained on. A JSD that says the student moved
toward the teacher, computed on out-of-distribution inputs, is worse than no JSD.
"""
from __future__ import annotations

import json
from pathlib import Path


class TokenizerLoadError(Exception):
    """Message is the operator-facing explanation, including what to pass instead."""


def load(tokenizer_dir, *, require_chat_template: bool = True):
    """Return (tokenizer, template_source). `template_source` is where the template came from.

    require_chat_template=False is for consumers that only need ids (entropy over a corpus
    already rendered, say). Anything that renders conversations wants the default: a
    tokenizer with no template cannot apply one, and the failure would otherwise surface as
    an empty string rather than as an error naming the directory.
    """
    from transformers import PreTrainedTokenizerFast

    tokenizer_dir = Path(tokenizer_dir)
    tok_file = tokenizer_dir / "tokenizer.json"
    if not tok_file.is_file():
        raise TokenizerLoadError(
            f"{tok_file} is missing, so there is no fast tokenizer to load. Pass a model "
            "directory or one of distill-tokenizer-align's outputs (retagged_student, "
            "teacher_overlay, student_overlay).")
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))

    # pad_token_id is set from eos when absent, because batching needs one and a base model
    # frequently has no pad token. Done here rather than in each caller so the two metric
    # scripts cannot disagree about it -- padding positions are excluded from every metric
    # by the attention mask, so the choice only has to be consistent.
    cfg_path = tokenizer_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        val = cfg.get(key)
        if isinstance(val, dict):
            val = val.get("content")
        if isinstance(val, str) and getattr(tok, key, None) is None:
            setattr(tok, key, val)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    jinja = tokenizer_dir / "chat_template.jinja"
    if jinja.is_file():
        tok.chat_template = jinja.read_text()
        return tok, str(jinja)
    inline = cfg.get("chat_template")
    if inline:
        tok.chat_template = inline
        return tok, f"{cfg_path} (inline chat_template)"
    if require_chat_template:
        raise TokenizerLoadError(
            f"no chat template in {tokenizer_dir} (looked for chat_template.jinja and an "
            "inline chat_template in tokenizer_config.json). Rendering conversations is not "
            "possible without one; distill-tokenizer-align installs it via --chat-template.")
    return tok, ""
