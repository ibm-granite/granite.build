# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared chat-template helpers and prompt construction for the dataset builders.

The module has two halves, separated by banner comments:

1. **Chat-template plumbing** — document-style detection, rendering, token-length
   stats, tokenizer loading, output-path helpers.
2. **Prompt construction** — the guardian (factuality), atomizer, and NLI prompt
   texts plus their ``get_*_messages`` / ``get_*_prompt`` builders.

Granite generations consume the RAG ``documents`` argument incompatibly, and
``apply_chat_template`` never raises when a template ignores the kwarg — it just
renders a prompt with the context missing. The capability detection here is what
every builder relies on to adapt:

- **Granite 3.x** reads ``document['doc_id']`` and ``document['text']``
  explicitly, emitting one ``<|start_of_role|>document {...}<|end_of_role|>``
  block per document. ``doc_id`` is required.
- **Granite 4.0 / 4.1** render ``document | tojson`` into a ``<documents>`` XML
  block in the system message. Keys appear *verbatim*, so a stray ``doc_id``
  would leak into the prompt; only ``text`` is emitted.
- **Granite 4.2** dropped ``documents`` support entirely (and moved to ChatML).
  There the passages are inlined into a leading system message instead. It also
  defaults to ``enable_thinking=True``, which would leave the prompt inside an
  open ``<think>`` block — wrong for a structured SFT target.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

# =============================================================================
# Chat-template plumbing
# =============================================================================

# Document-passing styles, keyed off what the chat template actually supports.
DOC_STYLE_GRANITE3 = "granite3"  # native documents=, requires doc_id + text
DOC_STYLE_GRANITE4 = "granite4"  # native documents=, verbatim keys -> text only
DOC_STYLE_INLINE = "inline"  # template ignores documents= -> inline into messages

DOC_STYLES = (DOC_STYLE_GRANITE3, DOC_STYLE_GRANITE4, DOC_STYLE_INLINE)

# Mirrors the Granite 4.x documents system message so inlined prompts look
# like the native rendering the model was trained on.
_INLINE_DOCS_PREFIX = (
    "You are a helpful assistant with access to the following documents. "
    "You may use one or more documents to assist with the user query.\n\n"
    "You are given a list of documents within <documents></documents> XML tags:\n"
    "<documents>"
)
_INLINE_DOCS_SUFFIX = (
    "\n</documents>\n\nWrite the response to the user's input by strictly aligning "
    "with the facts in the provided documents. If the information needed to answer "
    "the question is not available in the documents, inform the user that the "
    "question cannot be answered based on the available data."
)


def detect_document_style(chat_template: Optional[str]) -> str:
    """Classify how a chat template consumes the ``documents=`` kwarg.

    Pure string introspection so it stays unit-testable and needs no model
    download. Returns one of ``DOC_STYLES``.

    - ``doc_id`` referenced          -> Granite 3.x (indexes doc_id directly)
    - ``documents`` referenced       -> Granite 4.0/4.1 (tojson, verbatim keys)
    - ``document`` absent entirely   -> template ignores documents (4.2, Llama)
    """
    if not chat_template:
        return DOC_STYLE_INLINE
    if "doc_id" in chat_template:
        return DOC_STYLE_GRANITE3
    if "document" in chat_template:
        return DOC_STYLE_GRANITE4
    return DOC_STYLE_INLINE


def template_supports_kwarg(chat_template: Optional[str], name: str) -> bool:
    """True when ``name`` appears in the template, i.e. passing it does something."""
    return bool(chat_template) and name in chat_template


def tokenizer_document_style(tokenizer) -> str:
    """Detect the document style from a tokenizer's chat template."""
    return detect_document_style(getattr(tokenizer, "chat_template", None))


def build_documents(contexts: Sequence[str], doc_style: str) -> List[Dict[str, str]]:
    """Build the documents list for ``doc_style``, one document per passage.

    Granite 4.x renders each document as its own JSON line with verbatim keys,
    so ``doc_id`` is omitted there to keep it out of the prompt. Granite 3.x
    indexes ``doc_id`` directly and would fail without it.
    """
    if doc_style == DOC_STYLE_GRANITE3:
        return [{"doc_id": str(i), "text": text} for i, text in enumerate(contexts)]
    return [{"text": text} for text in contexts]


def inline_documents_message(documents: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """Render documents into a leading system message.

    Used when the chat template ignores ``documents=`` (Granite 4.2, non-Granite
    models), where the kwarg would otherwise be silently dropped.
    """
    body = "\n".join(json.dumps(d, ensure_ascii=False) for d in documents)
    return {"role": "system", "content": _INLINE_DOCS_PREFIX + "\n" + body + _INLINE_DOCS_SUFFIX}


def split_documents(
    contexts: Sequence[str],
    doc_style: str,
) -> Tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    """Return ``(inline_documents, template_documents)`` for ``doc_style``.

    Exactly one is populated: inline styles carry the context in the messages,
    native styles pass it via the ``documents=`` kwarg / column.
    """
    documents = build_documents(contexts, doc_style)
    if doc_style == DOC_STYLE_INLINE:
        return documents, None
    return [], documents


def render_chat_input(
    messages: List[Dict[str, str]],
    tokenizer,
    documents: Optional[Sequence[Dict[str, str]]] = None,
    *,
    thinking: bool = False,
) -> str:
    """Render messages through the chat template.

    ``documents`` is forwarded only when non-empty (callers pass ``None`` for
    templates that ignore it, having inlined the passages instead).
    ``enable_thinking`` is passed only when the template supports it (Granite
    4.2), where it otherwise defaults to True and would leave the prompt inside
    an open ``<think>`` block — wrong for a structured SFT target.
    """
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if documents:
        kwargs["documents"] = list(documents)
    if template_supports_kwarg(getattr(tokenizer, "chat_template", None), "enable_thinking"):
        kwargs["enable_thinking"] = thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def summarize(name: str, lengths: Sequence[int]) -> Dict[str, Any]:
    """Summary stats for a series of token lengths (sorted-percentile based)."""
    if not lengths:
        return {"name": name, "count": 0}
    s = sorted(lengths)
    n = len(s)

    def _pct(p: float) -> int:
        # Nearest-rank percentile; index clamped to the last element.
        idx = min(n - 1, int(p / 100.0 * n))
        return s[idx]

    return {
        "name": name,
        "count": n,
        "min": s[0],
        "max": s[-1],
        "mean": round(sum(s) / n, 1),
        "p50": _pct(50),
        "p90": _pct(90),
        "p95": _pct(95),
        "p99": _pct(99),
    }


def print_token_stats(stats: List[Dict[str, Any]], tag: str) -> None:
    """Print a small table of token-length stats (one row per series)."""
    if not stats or stats[0].get("count", 0) == 0:
        print(f"[{tag}] no records written; skipping token stats")
        return
    cols = ["count", "min", "max", "mean", "p50", "p90", "p95", "p99"]
    header = "  ".join(["series".ljust(14)] + [c.rjust(8) for c in cols])
    print(f"[{tag}] token-length stats:")
    print(f"  {header}")
    for row in stats:
        cells = [row["name"].ljust(14)] + [str(row.get(c, "")).rjust(8) for c in cols]
        print(f"  {'  '.join(cells)}")


def num_tokens(tokenizer, text: str) -> int:
    """Token count for a string, without adding special tokens."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


class IdentityTokenizer:
    """Stand-in tokenizer for chat-only conversion, which never renders a template."""

    chat_template = None

    def __call__(self, *args, **kwargs):
        raise RuntimeError("chat-format conversion should not tokenize")

    def apply_chat_template(self, *args, **kwargs):
        raise RuntimeError("chat-format conversion should not render a template")


def resolve_formats(fmt: str) -> List[str]:
    """Expand ``--format`` into the concrete formats to write."""
    return ["formatted", "chat"] if fmt == "both" else [fmt]


def output_name_for(input_path: str, fmt: str, *parts: str) -> str:
    """Derive an output filename from the input file, optional parts, and the format.

    The stem of the input file leads, then any extra ``parts`` (e.g. a task name),
    then the format::

        ("eli5_raw_train.json", "chat")               -> eli5_raw_train_chat.jsonl
        ("eli5_raw_train.json", "chat", "detection")  -> eli5_raw_train_detection_chat.jsonl
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    middle = "".join(f"_{p}" for p in parts)
    return f"{stem}{middle}_{fmt}.jsonl"


def load_tokenizer(model: Optional[str], needs_tokenizer: bool, reason: str):
    """Load a real tokenizer when needed, else a null-object stand-in."""
    if not needs_tokenizer:
        return IdentityTokenizer()
    if not model:
        raise SystemExit(f"--model is required {reason}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


# =============================================================================
# Prompt construction
# =============================================================================
# Each task exposes a ``get_*_messages`` builder (a list of chat-role dicts) and
# a ``get_*_prompt`` builder (the chat-templated string), so the messages can be
# sent to a chat API (e.g. RITS) while the templated string drives local
# generation. Keeping both here means the training-side builders and the
# eval-side callers can never drift apart.


# ------------------------------- guardian -------------------------------------
# Every judging task is expressed as one guardian block:
#
#     <guardian>{system prompt}
#
#     ### Criteria: {CRITERIA[criteria_id]}
#
#     ### Scoring Schema: {schema}[ {json hint}]
#
# A single builder, :func:`guardian_text`, renders all of them — it dispatches on
# ``criteria_id`` against CRITERIA (the criteria text) and _GUARDIAN_SPECS (the
# preamble, schema, and JSON hint). Adding a judging task means adding one entry
# to each dict, with no new assembly code. Tasks: factuality (detection and
# correction variants), atomizer decomposition, and NLI classification.

CRITERIA = {
    "factuality": (
        "A factually incorrect response occurs when the assistant's message contains "
        "one or more factual claims that are unsupported by, inconsistent with, or "
        "directly contradicted by the information provided in the documents or context. "
        "This includes situations where the assistant: introduces details not grounded "
        "in the context, misstates or distorts facts contained within the context, "
        "misinterprets the meaning or implications of the context, supplies erroneous "
        "or conflicting information relative to the context. Even if only a small "
        "portion of the response contains such inaccuracies, the overall message is "
        "considered factually incorrect."
    ),
    "atomicity": (
        "An atomic unit is a single, self-contained statement that expresses exactly "
        "one piece of information from the text. It should be minimal (it cannot be "
        "split further without losing meaning), independently verifiable, and "
        "understandable on its own without reference to the other units. Resolve "
        "pronouns and references so that each unit stands alone."
    ),
    # Deliberately detailed: NLI accuracy hinges on the three labels being
    # separated crisply, so each one gets its test, its decision rules, and the
    # confusions that most often cause mislabeling.
    "entailment": (
        "Natural language inference judges the logical relationship from a premise "
        "(text A) to a hypothesis (text B), assuming both describe the same situation. "
        "The relationship is exactly one of three labels.\n\n"
        "'entailment': a reader accepting the premise as true must also accept the "
        "hypothesis as true. The hypothesis adds no unsupported information; it may "
        "restate the premise, paraphrase it, generalize it, or state part of it.\n\n"
        "'contradiction': a reader accepting the premise as true must reject the "
        "hypothesis as false. The premise and hypothesis cannot both hold of the same "
        "situation, because they conflict over an entity, action, attribute, quantity, "
        "time, place, or negation.\n\n"
        "'neutral': the premise neither compels nor rules out the hypothesis. The "
        "hypothesis could be true or false while the premise stays true — typically "
        "because it adds detail the premise never mentions, states a plausible but "
        "unstated motive, cause, or outcome, or makes a broader claim the premise only "
        "partly supports.\n\n"
        "Apply these rules:\n"
        "- Judge only what the premise states or necessarily implies. Do not rely on "
        "outside world knowledge, and do not assume unstated details.\n"
        "- Unmentioned is not the same as false. Information absent from the premise "
        "makes a hypothesis 'neutral', not 'contradiction'.\n"
        "- Plausible is not the same as entailed. A likely inference the premise does "
        "not guarantee is 'neutral', not 'entailment'.\n"
        "- A hypothesis more specific than the premise is usually 'neutral'; a "
        "hypothesis more general than the premise is usually 'entailment'.\n"
        "- Weigh negation, quantifiers ('all', 'some', 'none'), and numbers carefully: "
        "they often flip the label on their own.\n"
        "- Choose 'contradiction' only for genuine incompatibility, not mere difference "
        "in wording, emphasis, or level of detail."
    ),
}

# Default criteria id used by the factuality builders.
CRITERIA_ID = "factuality"

# Criteria id used by the atomizer builder.
ATOMIZER_CRITERIA_ID = "atomicity"

# Criteria id used by the NLI builder.
NLI_CRITERIA_ID = "entailment"

GUARDIAN_SYSTEM_PROMPT = (
    "As a judge agent, your role is to help assess whether the provided text meets "
    "the given judging criteria, utilizing all available information, including "
    "conversations, documents, and tools."
)

# The atomizer decomposes rather than scores, so it gets its own preamble.
ATOMIZER_SYSTEM_PROMPT = (
    "As a judge agent, your role is to decompose the provided text into units that "
    "meet the given criteria, utilizing all available information, including "
    "conversations, documents, and tools."
)

# NLI classifies the relation between two texts rather than scoring one.
NLI_SYSTEM_PROMPT = (
    "As a judge agent, your role is to classify the logical relationship between "
    "the two provided texts according to the given criteria, utilizing all "
    "available information, including conversations, documents, and tools."
)

GUARDIAN_DETECTION_SCHEMA = "If the last assistant's text meets the criteria, return 'yes'; otherwise, return 'no'."

GUARDIAN_CORRECTION_SCHEMA = (
    "If the last assistant's text meets the criteria, return a corrected version of "
    "the assistant's message based on the given context; otherwise, return 'none'."
)

GUARDIAN_ATOMIZER_SCHEMA = (
    "Decompose the last assistant's text into the units that meet the criteria. "
    "Return one unit per line, with no numbering, bullets, or extra commentary."
)

GUARDIAN_NLI_SCHEMA = (
    "Judge the relationship from text A to text B and return exactly one of "
    "'entailment', 'contradiction', or 'neutral'."
)

# JSON-dict hints appended by the prompt-style variants (``--for-prompt``).
GUARDIAN_DETECTION_JSON_HINT = (
    'Provide the final answer as a JSON dict with the following format: {"score": "yes" or "no"}.'
)

GUARDIAN_CORRECTION_JSON_HINT = (
    'Provide the final answer as a JSON dict with the following format: {"correction": "corrected message" or "none"}.'
)

GUARDIAN_NLI_JSON_HINT = (
    "Provide the final answer as a JSON dict with the following "
    'format: {"label": "entailment" | "contradiction" | "neutral"}.'
)

# The atomizer target is newline-separated plain text, so there is no JSON hint.


# Per-task guardian specs, keyed by criteria id. ``detection`` and ``correction``
# share the "factuality" criteria but score differently, so the factuality entry
# maps a variant name to its (schema, json hint) pair; single-variant tasks use
# the lone ``None`` key. ``always_hint`` marks tasks whose target *is* the JSON
# dict (NLI), so the hint is not gated on ``for_prompt``.
_GUARDIAN_SPECS = {
    "factuality": {
        "system_prompt": GUARDIAN_SYSTEM_PROMPT,
        "variants": {
            "detection": (GUARDIAN_DETECTION_SCHEMA, GUARDIAN_DETECTION_JSON_HINT),
            "correction": (GUARDIAN_CORRECTION_SCHEMA, GUARDIAN_CORRECTION_JSON_HINT),
        },
        "default_variant": "detection",
        "always_hint": False,
    },
    "atomicity": {
        "system_prompt": ATOMIZER_SYSTEM_PROMPT,
        # Target is newline-separated text, so no JSON hint exists.
        "variants": {None: (GUARDIAN_ATOMIZER_SCHEMA, None)},
        "default_variant": None,
        "always_hint": False,
    },
    "entailment": {
        "system_prompt": NLI_SYSTEM_PROMPT,
        "variants": {None: (GUARDIAN_NLI_SCHEMA, GUARDIAN_NLI_JSON_HINT)},
        "default_variant": None,
        "always_hint": True,
    },
}


def _guardian_spec(criteria_id: str, variant: Optional[str]):
    """Resolve ``(system_prompt, schema, json_hint, always_hint)`` for a task."""
    if criteria_id not in CRITERIA:
        raise KeyError(f"unknown criteria id {criteria_id!r}; known: {sorted(CRITERIA)}")
    spec = _GUARDIAN_SPECS.get(criteria_id)
    if spec is None:
        raise KeyError(f"criteria id {criteria_id!r} has no guardian spec; known: {sorted(_GUARDIAN_SPECS)}")
    variants = spec["variants"]
    if variant is None:
        variant = spec["default_variant"]
    if variant not in variants:
        known = sorted(v for v in variants if v is not None) or ["<none>"]
        raise KeyError(f"unknown variant {variant!r} for {criteria_id!r}; known: {known}")
    schema, json_hint = variants[variant]
    return spec["system_prompt"], schema, json_hint, spec["always_hint"]


def guardian_text(
    criteria_id: str = CRITERIA_ID,
    *,
    variant: Optional[str] = None,
    for_prompt: bool = False,
    tag: str = "guardian",
) -> str:
    """Assemble a ``<guardian>`` user-turn block for any judging task.

    Every task renders the same three-part block::

        <{tag}>{system prompt}

        ### Criteria: {CRITERIA[criteria_id]}

        ### Scoring Schema: {schema}[ {json hint}]

    ``criteria_id`` selects both the criteria text (:data:`CRITERIA`) and the
    task's spec (:data:`_GUARDIAN_SPECS`) — ``"factuality"``, ``"atomicity"``, or
    ``"entailment"``. ``variant`` picks between a task's schemas where it has more
    than one (factuality: ``"detection"`` / ``"correction"``; default
    ``"detection"``). ``for_prompt`` appends the JSON dict hint; tasks whose
    target *is* JSON (NLI) always include it.

    Adding a judging task means adding a :data:`CRITERIA` entry and a
    :data:`_GUARDIAN_SPECS` entry — no new assembly code.
    """
    system_prompt, schema, json_hint, always_hint = _guardian_spec(criteria_id, variant)
    if json_hint and (for_prompt or always_hint):
        schema = f"{schema} {json_hint}"
    return f"<{tag}>{system_prompt}\n\n### Criteria: {CRITERIA[criteria_id]}\n\n### Scoring Schema: {schema}"


# Convenience constants for the two factuality variants (no JSON hint).
FACT_DETECTION_GUARDIAN_TEXT = guardian_text(CRITERIA_ID, variant="detection")
FACT_CORRECTION_GUARDIAN_TEXT = guardian_text(CRITERIA_ID, variant="correction")

# Atomizer and NLI guardian blocks (kept under their historical names).
ATOMIZER_TEXT = guardian_text(ATOMIZER_CRITERIA_ID)
NLI_TEXT = guardian_text(NLI_CRITERIA_ID)


def _guardian_messages(query, response, unique_contexts, text):
    """Chat messages for detection/correction (contexts folded into the user turn).

    For local generation the contexts ride in the chat template's ``documents``
    slot; for a plain chat API (RITS) they are inlined into the guardian user turn
    so no template-specific document handling is needed.
    """
    context_text = "\n\n".join(unique_contexts)
    with_docs = f"{text}\n\n### Documents:\n{context_text}" if context_text else text
    return [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
        {"role": "user", "content": with_docs},
    ]


def _guardian_prompt(query, response, unique_contexts, tokenizer, text, *, thinking=False):
    """Chat-templated guardian prompt, adapted to what the template supports.

    The document shape is not portable across Granite generations: 3.x indexes
    ``doc_id``, 4.0/4.1 render keys verbatim (so ``doc_id`` would leak), and 4.2
    dropped ``documents`` entirely — passing it there silently drops the context.
    :func:`split_documents` picks the right shape and, for templates that ignore
    the kwarg, hands back documents to inline into a leading system message.
    """
    inline_docs, template_docs = split_documents(unique_contexts, tokenizer_document_style(tokenizer))
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
        {"role": "user", "content": text},
    ]
    if inline_docs:
        messages.insert(0, inline_documents_message(inline_docs))
    return render_chat_input(messages, tokenizer, template_docs, thinking=thinking)


def get_detection_messages(query, response, unique_contexts):
    """Chat messages for factuality detection."""
    return _guardian_messages(query, response, unique_contexts, FACT_DETECTION_GUARDIAN_TEXT)


def get_correction_messages(query, response, unique_contexts):
    """Chat messages for factuality correction."""
    return _guardian_messages(query, response, unique_contexts, FACT_CORRECTION_GUARDIAN_TEXT)


def get_detection_prompt(query, response, unique_contexts, tokenizer, *, thinking=False):
    """Create the factuality detection prompt (chat-templated string)."""
    return _guardian_prompt(
        query, response, unique_contexts, tokenizer, FACT_DETECTION_GUARDIAN_TEXT, thinking=thinking
    )


def get_correction_prompt(query, response, unique_contexts, tokenizer, *, thinking=False):
    """Create the factuality correction prompt (chat-templated string)."""
    return _guardian_prompt(
        query, response, unique_contexts, tokenizer, FACT_CORRECTION_GUARDIAN_TEXT, thinking=thinking
    )


# ------------------------------- atomizer ------------------------------------
# Decomposes an assistant response into atomic units. No RAG documents involved.
# The prompt block itself is a guardian block built above from
# ``CRITERIA["atomicity"]`` — see :func:`atomizer_text` / :data:`ATOMIZER_TEXT`.


def extract_assistant_text(input_value):
    """The response to atomize, from an atomizer dataset's ``input``.

    The atomizer ``input`` is canonically a chat-message list whose **assistant**
    turn holds the response to decompose (the user turn is the atomizer
    instruction). Returns that assistant content. If ``input`` is already a plain
    string, it is treated as the response itself.
    """
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        for msg in input_value:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # No explicit assistant turn — fall back to the first message's content.
        if input_value and isinstance(input_value[0], dict):
            return input_value[0].get("content", "")
    raise ValueError("Atomizer input must be a chat-message list or a string.")


def get_atomizer_messages(response, query=None):
    """Training-format atomizer messages: the text to decompose is the assistant turn.

    Used for the *local* chat-templated path (mirrors the intrinsic's training
    data). The RITS backend instead sends the dataset's own ``input`` message list
    as-is.

    ``query`` is optional: when given, it becomes a leading user turn so the
    conversation the response came from is preserved (the shape
    ``build_atomizer_dataset.py`` trains on). Omit it for a bare decomposition.
    """
    messages = [
        {"role": "assistant", "content": response},
        {"role": "user", "content": ATOMIZER_TEXT},
    ]
    if query is not None:
        messages.insert(0, {"role": "user", "content": query})
    return messages


def get_atomizer_prompt(response, tokenizer, query=None, *, thinking=False):
    """Create the atomizer prompt (chat-templated string) for the local backend."""
    return render_chat_input(get_atomizer_messages(response, query), tokenizer, thinking=thinking)


# ------------------------------- NLI -----------------------------------------
# Judges the relation between two texts. No RAG documents involved.
# The prompt block itself is a guardian block built above from
# ``CRITERIA["entailment"]`` — see :func:`nli_text` / :data:`NLI_TEXT`.


def get_nli_messages(premise, hypothesis):
    """Chat messages for NLI: premise (text A) vs hypothesis (text B) -> label."""
    user_turn = f"Text A (premise): {premise}\n\nText B (hypothesis): {hypothesis}\n\n{NLI_TEXT}"
    return [{"role": "user", "content": user_turn}]


def get_nli_prompt(premise, hypothesis, tokenizer, *, thinking=False):
    """Create the NLI prompt (chat-templated string)."""
    return render_chat_input(get_nli_messages(premise, hypothesis), tokenizer, thinking=thinking)
