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
  There the passages go in as a ``tool`` message, which 4.2 renders as a
  ``<tool_response>[...]</tool_response>`` block — how it natively represents RAG
  context. Because that block is one JSON *array*, all passages are merged into a
  **single** document (blank-line separated) so the rendered context matches 4.1's
  single ``<documents>`` block rather than fragmenting into N array entries. For
  judge/guardian data the message is placed *before the first assistant turn*,
  since the documents ground the answer being judged. It also defaults to
  ``enable_thinking=True``, which would leave the prompt inside an open ``<think>``
  block — wrong for a structured SFT target.
- **Other templates that ignore ``documents=``** (plain Llama, etc.) fall back to
  inlining the passages into a leading system message.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

# =============================================================================
# Chat-template plumbing
# =============================================================================

# Document-passing styles, keyed off what the chat template actually supports.
DOC_STYLE_GRANITE3 = "granite3"  # native documents=, requires doc_id + text
DOC_STYLE_GRANITE4 = "granite4"  # native documents=, verbatim keys -> text only
DOC_STYLE_TOOL = "tool"  # ChatML/4.2: documents ride in a `tool` message
DOC_STYLE_INLINE = "inline"  # template ignores documents= -> inline into messages

DOC_STYLES = (DOC_STYLE_GRANITE3, DOC_STYLE_GRANITE4, DOC_STYLE_TOOL, DOC_STYLE_INLINE)

# The ChatML turn marker. Its presence is what distinguishes Granite 4.2's ChatML
# template from 4.1's `<|start_of_role|>` role tags.
CHATML_MARKER = "<|im_start|>"

# Default `source` for a normalized tool-response document.
_TOOL_DOC_SOURCE = "knowledge_base"

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


def is_chatml_template(chat_template: Optional[str]) -> bool:
    """True when the template is ChatML (Granite 4.2), detected by the turn marker.

    Granite 4.2 moved to ChatML and dropped ``documents=``; there the RAG context
    goes in as a ``tool`` message instead (see :func:`tool_documents_message`).
    """
    return bool(chat_template and CHATML_MARKER in chat_template)


def tokenizer_is_chatml(tokenizer: Any) -> bool:
    """True for a ChatML tokenizer: template probe first, then vocab membership.

    The template probe keeps this unit-testable with a template stand-in and no
    model download. The vocab fallback catches a real 4.2 tokenizer whose template
    text stops matching. ``convert_tokens_to_ids`` is deliberately *not* used as a
    fallback: for an unknown token it returns the unk id (often 0), which would
    false-positive on every non-ChatML tokenizer.
    """
    if is_chatml_template(getattr(tokenizer, "chat_template", None)):
        return True
    vocab_get = getattr(tokenizer, "get_vocab", None)
    if callable(vocab_get):
        try:
            if CHATML_MARKER in vocab_get():
                return True
        except Exception:  # pragma: no cover - defensive
            pass
    added = getattr(tokenizer, "added_tokens_encoder", None)
    if isinstance(added, dict) and CHATML_MARKER in added:
        return True
    return False


def detect_document_style(chat_template: Optional[str]) -> str:
    """Classify how a chat template consumes the ``documents=`` kwarg.

    Pure string introspection so it stays unit-testable and needs no model
    download. Returns one of ``DOC_STYLES``.

    - ChatML marker present          -> Granite 4.2 (documents= dropped; use a tool
      message). Checked **first**: 4.2's template may still mention ``document``
      elsewhere, and falling through to ``granite4`` would silently drop the context.
    - ``doc_id`` referenced          -> Granite 3.x (indexes doc_id directly)
    - ``documents`` referenced       -> Granite 4.0/4.1 (tojson, verbatim keys)
    - ``document`` absent entirely   -> template ignores documents (Llama, etc.)
    """
    if not chat_template:
        return DOC_STYLE_INLINE
    if is_chatml_template(chat_template):
        return DOC_STYLE_TOOL
    if "doc_id" in chat_template:
        return DOC_STYLE_GRANITE3
    if "document" in chat_template:
        return DOC_STYLE_GRANITE4
    return DOC_STYLE_INLINE


def template_supports_kwarg(chat_template: Optional[str], name: str) -> bool:
    """True when ``name`` appears in the template, i.e. passing it does something."""
    return bool(chat_template and name in chat_template)


def tokenizer_document_style(tokenizer: Any) -> str:
    """Detect the document style from a tokenizer.

    Goes through :func:`tokenizer_is_chatml` first so a real Granite 4.2 tokenizer
    resolves to ``tool`` via its vocab even if the template text does not match.
    """
    if tokenizer_is_chatml(tokenizer):
        return DOC_STYLE_TOOL
    return detect_document_style(getattr(tokenizer, "chat_template", None))


def build_documents(contexts: Sequence[str], doc_style: str) -> List[Dict[str, str]]:
    """Build the documents list for ``doc_style``.

    One document per passage, except for ``tool`` (Granite 4.2) where all passages
    are merged into a single document.

    Granite 4.x renders each document as its own JSON line with verbatim keys,
    so ``doc_id`` is omitted there to keep it out of the prompt. Granite 3.x
    indexes ``doc_id`` directly and would fail without it.

    Granite 4.2 renders a ``tool`` message as one ``<tool_response>[...]``
    JSON array, so N documents become N array entries — a different prompt shape
    from 4.1's single ``<documents>`` block. Merging the passages into one document
    (separated by a blank line) keeps the 4.2 block equivalent to the 4.1 one, so
    data built for either generation trains the same context layout.
    """
    if doc_style == DOC_STYLE_GRANITE3:
        return [{"doc_id": str(i), "text": text} for i, text in enumerate(contexts)]
    if doc_style == DOC_STYLE_TOOL:
        return [{"text": "\n\n".join(contexts)}] if contexts else []
    return [{"text": text} for text in contexts]


def inline_documents_message(documents: Sequence[Dict[str, str]]) -> Dict[str, str]:
    """Render documents into a leading system message.

    Used when the chat template ignores ``documents=`` (Granite 4.2, non-Granite
    models), where the kwarg would otherwise be silently dropped.
    """
    body = "\n".join(json.dumps(d, ensure_ascii=False) for d in documents)
    return {"role": "system", "content": _INLINE_DOCS_PREFIX + "\n" + body + _INLINE_DOCS_SUFFIX}


def normalize_document(doc: Any, idx: int) -> Dict[str, str]:
    """Normalize a passage into the agentic ``{source, document_id, content}`` record.

    Accepts a plain string (the builders' shape), a ``{"text": ...}`` dict, or a
    ``{"content": ...}`` dict. ``document_id`` falls back to the positional index
    when absent *or blank* — an empty ``doc_id`` would otherwise render as an empty
    field in the prompt.
    """
    if isinstance(doc, dict):
        content = doc.get("content", doc.get("text", ""))
        doc_id = doc.get("document_id") or doc.get("doc_id")
        return {
            "source": str(doc.get("source") or _TOOL_DOC_SOURCE),
            "document_id": str(doc_id) if doc_id else str(idx),
            "content": content if isinstance(content, str) else str(content),
        }
    return {
        "source": _TOOL_DOC_SOURCE,
        "document_id": str(idx),
        "content": doc if isinstance(doc, str) else str(doc),
    }


def tool_documents_message(documents: Sequence[Any]) -> Dict[str, str]:
    """Build the ``tool`` message carrying the retrieved documents.

    Granite 4.2 renders a ``tool`` message as a ``<tool_response>[...]</tool_response>``
    block — how 4.2 natively represents RAG context. No assistant ``tool_calls`` turn
    is synthesized and no tool schema is declared; 4.2 renders a bare ``tool`` message.

    One array entry per element of ``documents``. The builders hand this a single
    merged document (see :func:`build_documents`) so the block mirrors 4.1's single
    ``<documents>`` block; callers passing pre-built records keep one entry each,
    since their ``document_id``/``source`` values are theirs to preserve.
    """
    payload = [normalize_document(d, i) for i, d in enumerate(documents)]
    return {"role": "tool", "content": json.dumps(payload, indent=2, ensure_ascii=False)}


def insert_documents_as_tool(
    messages: Sequence[Dict[str, str]],
    documents: Sequence[Any],
    *,
    before_first_assistant: bool = True,
) -> List[Dict[str, str]]:
    """Return a copy of ``messages`` with the documents inserted as a ``tool`` message.

    ``before_first_assistant=True`` (judge/guardian data) places it immediately before
    the first assistant turn: the documents are the grounding for the answer being
    judged, while a later user turn carries the judging *instruction*. Placing them
    after the last user turn would bury the context after the instruction and just
    before the label.

    With no assistant turn (e.g. an inference prompt) there is no answer to precede,
    so the message lands after the last user turn instead.
    """
    result = list(messages)
    tool_msg = tool_documents_message(documents)

    if before_first_assistant:
        idx = next((i for i, m in enumerate(result) if m.get("role") == "assistant"), None)
        if idx is not None:
            return result[:idx] + [tool_msg] + result[idx:]

    last_user = None
    for i, m in enumerate(result):
        if m.get("role") == "user":
            last_user = i
    if last_user is None:
        idx = next((i for i, m in enumerate(result) if m.get("role") == "assistant"), len(result))
        return result[:idx] + [tool_msg] + result[idx:]
    return result[: last_user + 1] + [tool_msg] + result[last_user + 1 :]


def documents_ride_in_messages(doc_style: str) -> bool:
    """True when ``doc_style`` carries documents in the messages, not the kwarg."""
    return doc_style in (DOC_STYLE_TOOL, DOC_STYLE_INLINE)


def attach_documents(
    messages: Sequence[Dict[str, str]],
    documents: Sequence[Dict[str, str]],
    doc_style: str,
    *,
    before_first_assistant: bool = True,
) -> List[Dict[str, str]]:
    """Attach message-borne documents to ``messages`` according to ``doc_style``.

    The single seam for the two message-borne styles, so the dataset builders and the
    eval-side prompt builders cannot drift:

    - ``tool``   -> a ``tool`` message (Granite 4.2's native RAG representation)
    - ``inline`` -> a leading system message mimicking Granite 4.x's ``<documents>``

    Native styles (granite3/granite4) pass documents via the kwarg, so this is a
    no-op for them.
    """
    if not documents:
        return list(messages)
    if doc_style == DOC_STYLE_TOOL:
        return insert_documents_as_tool(messages, documents, before_first_assistant=before_first_assistant)
    if doc_style == DOC_STYLE_INLINE:
        return [inline_documents_message(documents), *messages]
    return list(messages)


def split_documents(
    contexts: Sequence[str],
    doc_style: str,
) -> Tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    """Return ``(message_documents, template_documents)`` for ``doc_style``.

    Exactly one is populated: message-borne styles (``tool``, ``inline``) carry the
    context inside the messages, native styles pass it via the ``documents=`` kwarg /
    column. Hand the first to :func:`attach_documents` and the second to
    :func:`render_chat_input`.
    """
    documents = build_documents(contexts, doc_style)
    if documents_ride_in_messages(doc_style):
        return documents, None
    return [], documents


def thinking_kwargs(tokenizer: Any, thinking: bool = False) -> Dict[str, Any]:
    """``{"enable_thinking": thinking}`` when the chat template supports it, else ``{}``.

    Granite 4.2 defaults to ``enable_thinking=True``, which ends the rendered
    prompt inside an *open* ``<think>`` block — so an SFT target would be trained
    as reasoning content rather than as the answer. Older templates ignore the
    kwarg, and a non-Granite template might reject an unknown one, so the flag is
    only forwarded when the template actually mentions it.

    Args:
        tokenizer: A tokenizer whose ``chat_template`` is inspected.
        thinking: The ``enable_thinking`` value to forward.

    Returns:
        Keyword arguments to spread into ``apply_chat_template``.
    """
    if template_supports_kwarg(getattr(tokenizer, "chat_template", None), "enable_thinking"):
        return {"enable_thinking": thinking}
    return {}


def apply_chat_template_to_df(df: Any, tokenizer: Any, input_col: str, thinking: bool = False) -> Any:
    """Convert message-list rows in ``df[input_col]`` to plain strings via the chat template.

    Handles optional ``documents`` and ``tools`` columns for RAG / tool-use
    prompts. Returns ``df`` unchanged if the column doesn't hold message lists
    (detected by inspecting the first row) or if the DataFrame is empty.

    ``thinking`` is off by default; see :func:`thinking_kwargs`.

    Shared by the multi-GPU drivers and between the Arrow and Ray Data backends
    to keep chat-template handling in a single place.

    Args:
        df: A pandas DataFrame of training rows.
        tokenizer: The tokenizer whose chat template renders each row.
        input_col: The column holding the message lists.
        thinking: Forwarded as ``enable_thinking`` when the template supports it.

    Returns:
        ``df`` with ``input_col`` rendered to strings, or ``df`` unchanged.
    """
    if input_col not in df.columns or len(df) == 0:
        return df

    first_val = df[input_col].iloc[0]
    if not isinstance(first_val, list):
        return df

    has_documents = "documents" in df.columns and isinstance(df["documents"].iloc[0], list)
    has_tools = "tools" in df.columns and isinstance(df["tools"].iloc[0], list)

    base_kwargs = {"tokenize": False, "add_generation_prompt": True, **thinking_kwargs(tokenizer, thinking)}

    def _render(row: Any) -> Any:
        kwargs = dict(base_kwargs)
        if has_documents:
            kwargs["documents"] = row["documents"]
        if has_tools:
            kwargs["tools"] = row["tools"]
        return tokenizer.apply_chat_template(row[input_col], **kwargs)

    df[input_col] = df.apply(_render, axis=1)
    return df


def render_chat_input(
    messages: List[Dict[str, str]],
    tokenizer: Any,
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
    kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, **thinking_kwargs(tokenizer, thinking)}
    if documents:
        kwargs["documents"] = list(documents)
    return cast(str, tokenizer.apply_chat_template(messages, **kwargs))


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


def num_tokens(tokenizer: Any, text: str) -> int:
    """Token count for a string, without adding special tokens."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


class IdentityTokenizer:
    """Stand-in tokenizer for chat-only conversion, which never renders a template."""

    chat_template = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("chat-format conversion should not tokenize")

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> Any:
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


def load_tokenizer(model: Optional[str], needs_tokenizer: bool, reason: str) -> Any:
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

GUARDIAN_DETECTION_SCHEMA_JSON = (
    "If the last assistant's text meets the criteria, provide the final answer as a JSON dict with the following "
    'format: {"score": "yes" or "no"}.'
)

GUARDIAN_CORRECTION_SCHEMA = (
    "If the last assistant's text meets the criteria, return a corrected version of "
    "the assistant's message based on the given context; otherwise, return 'none'."
)

GUARDIAN_CORRECTION_SCHEMA_JSON = (
    "If the last assistant's text meets the criteria, provide the final answer as a JSON dict with the following "
    'format: {"correction": "corrected message" or "none"}.'
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
_GUARDIAN_SPECS: Dict[str, Dict[str, Any]] = {
    "factuality": {
        "system_prompt": GUARDIAN_SYSTEM_PROMPT,
        "variants": {
            "detection": (GUARDIAN_DETECTION_SCHEMA, GUARDIAN_DETECTION_JSON_HINT),
            "correction": (GUARDIAN_CORRECTION_SCHEMA, GUARDIAN_CORRECTION_JSON_HINT),
        },
        # Granite 4.2 asks for the verdict as a JSON dict directly in the schema
        # instead of the "return 'yes'/'no'" phrasing plus an appended hint.
        "json_schemas": {
            "detection": GUARDIAN_DETECTION_SCHEMA_JSON,
            "correction": GUARDIAN_CORRECTION_SCHEMA_JSON,
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


def _guardian_spec(criteria_id: str, variant: Optional[str], *, json_schema: bool = False) -> Tuple[Any, Any, Any, Any]:
    """Resolve ``(system_prompt, schema, json_hint, always_hint)`` for a task.

    ``json_schema`` selects the task's JSON-dict schema where it has one (factuality
    on Granite 4.2). That schema *replaces* the plain one and already spells out the
    JSON format, so the appended hint is suppressed by returning ``None`` for it —
    otherwise the format spec would appear twice.
    """
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
    if json_schema:
        json_only = spec.get("json_schemas", {}).get(variant)
        if json_only is not None:
            return spec["system_prompt"], json_only, None, spec["always_hint"]
    return spec["system_prompt"], schema, json_hint, spec["always_hint"]


def guardian_text(
    criteria_id: str = CRITERIA_ID,
    *,
    variant: Optional[str] = None,
    for_prompt: bool = False,
    json_schema: bool = False,
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

    ``json_schema`` swaps in the task's JSON-dict schema instead (factuality on
    Granite 4.2), which states the JSON format itself — so the appended hint is
    suppressed and ``for_prompt`` becomes a no-op for that task.

    Adding a judging task means adding a :data:`CRITERIA` entry and a
    :data:`_GUARDIAN_SPECS` entry — no new assembly code.
    """
    system_prompt, schema, json_hint, always_hint = _guardian_spec(criteria_id, variant, json_schema=json_schema)
    if json_hint and (for_prompt or always_hint):
        schema = f"{schema} {json_hint}"
    return f"<{tag}>{system_prompt}\n\n### Criteria: {CRITERIA[criteria_id]}\n\n### Scoring Schema: {schema}"


# Convenience constants for the two factuality variants (no JSON hint).
FACT_DETECTION_GUARDIAN_TEXT = guardian_text(CRITERIA_ID, variant="detection")
FACT_CORRECTION_GUARDIAN_TEXT = guardian_text(CRITERIA_ID, variant="correction")

# Granite 4.2 variants: the schema itself asks for a JSON dict (see the
# ``json_schemas`` entry in :data:`_GUARDIAN_SPECS`).
FACT_DETECTION_GUARDIAN_TEXT_JSON = guardian_text(CRITERIA_ID, variant="detection", json_schema=True)
FACT_CORRECTION_GUARDIAN_TEXT_JSON = guardian_text(CRITERIA_ID, variant="correction", json_schema=True)


def fact_guardian_text(variant: str, *, json_schema: bool = False) -> str:
    """The factuality guardian block for ``variant``, plain or JSON-schema flavored.

    The single lookup shared by the dataset builder and the eval-side prompt
    builders, so a 4.2 run cannot end up training on one schema and evaluating
    against the other.
    """
    if json_schema:
        return FACT_DETECTION_GUARDIAN_TEXT_JSON if variant == "detection" else FACT_CORRECTION_GUARDIAN_TEXT_JSON
    return FACT_DETECTION_GUARDIAN_TEXT if variant == "detection" else FACT_CORRECTION_GUARDIAN_TEXT


def uses_json_schema(doc_style: str) -> bool:
    """True when ``doc_style`` implies the JSON-dict guardian schema (Granite 4.2).

    Tied to the ChatML/``tool`` style rather than a model name: ``model_type`` is
    ``granite`` for both 4.1 and 4.2, so the template is the only reliable signal,
    and it is already resolved wherever documents are shaped.
    """
    return doc_style == DOC_STYLE_TOOL


# Atomizer and NLI guardian blocks (kept under their historical names).
ATOMIZER_TEXT = guardian_text(ATOMIZER_CRITERIA_ID, tag="atomizer")
NLI_TEXT = guardian_text(NLI_CRITERIA_ID, tag="nli")


def _guardian_messages(query: str, response: str, unique_contexts: Sequence[str], text: str) -> List[Dict[str, str]]:
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


def _guardian_prompt(
    query: str,
    response: str,
    unique_contexts: Sequence[str],
    tokenizer: Any,
    text: str,
    *,
    thinking: bool = False,
) -> str:
    """Chat-templated guardian prompt, adapted to what the template supports.

    The document shape is not portable across Granite generations: 3.x indexes
    ``doc_id``, 4.0/4.1 render keys verbatim (so ``doc_id`` would leak), and 4.2
    dropped ``documents`` entirely — passing it there silently drops the context.
    :func:`split_documents` picks the right shape and :func:`attach_documents` puts
    message-borne documents in the right place (a ``tool`` message on 4.2, a leading
    system message on templates that ignore the kwarg) — the same seam the dataset
    builders use, so training and eval prompts cannot drift.
    """
    doc_style = tokenizer_document_style(tokenizer)
    message_docs, template_docs = split_documents(unique_contexts, doc_style)
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
        {"role": "user", "content": text},
    ]
    # before_first_assistant: the documents ground the response being judged, and the
    # trailing user turn is the judging instruction.
    messages = attach_documents(messages, message_docs, doc_style, before_first_assistant=True)
    return render_chat_input(messages, tokenizer, template_docs, thinking=thinking)


def get_detection_messages(query: str, response: str, unique_contexts: Sequence[str]) -> List[Dict[str, str]]:
    """Chat messages for factuality detection."""
    return _guardian_messages(query, response, unique_contexts, FACT_DETECTION_GUARDIAN_TEXT)


def get_correction_messages(query: str, response: str, unique_contexts: Sequence[str]) -> List[Dict[str, str]]:
    """Chat messages for factuality correction."""
    return _guardian_messages(query, response, unique_contexts, FACT_CORRECTION_GUARDIAN_TEXT)


def get_detection_prompt(
    query: str, response: str, unique_contexts: Sequence[str], tokenizer: Any, *, thinking: bool = False
) -> str:
    """Create the factuality detection prompt (chat-templated string).

    The schema flavor follows the tokenizer's own template (JSON-dict on Granite
    4.2), matching what the dataset builder wrote for that model.
    """
    text = fact_guardian_text("detection", json_schema=uses_json_schema(tokenizer_document_style(tokenizer)))
    return _guardian_prompt(query, response, unique_contexts, tokenizer, text, thinking=thinking)


def get_correction_prompt(
    query: str, response: str, unique_contexts: Sequence[str], tokenizer: Any, *, thinking: bool = False
) -> str:
    """Create the factuality correction prompt (chat-templated string).

    The schema flavor follows the tokenizer's own template (JSON-dict on Granite
    4.2), matching what the dataset builder wrote for that model.
    """
    text = fact_guardian_text("correction", json_schema=uses_json_schema(tokenizer_document_style(tokenizer)))
    return _guardian_prompt(query, response, unique_contexts, tokenizer, text, thinking=thinking)


# ------------------------------- atomizer ------------------------------------
# Decomposes an assistant response into atomic units. No RAG documents involved.
# The prompt block itself is a guardian block built above from
# ``CRITERIA["atomicity"]`` — see :func:`atomizer_text` / :data:`ATOMIZER_TEXT`.


def extract_assistant_text(input_value: Union[str, Sequence[Any]]) -> str:
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
                return cast(str, msg.get("content", ""))
        # No explicit assistant turn — fall back to the first message's content.
        if input_value and isinstance(input_value[0], dict):
            return cast(str, input_value[0].get("content", ""))
    raise ValueError("Atomizer input must be a chat-message list or a string.")


def get_atomizer_messages(response: str, query: Optional[str] = None) -> List[Dict[str, str]]:
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


def get_atomizer_prompt(response: str, tokenizer: Any, query: Optional[str] = None, *, thinking: bool = False) -> str:
    """Create the atomizer prompt (chat-templated string) for the local backend."""
    return render_chat_input(get_atomizer_messages(response, query), tokenizer, thinking=thinking)


# ------------------------------- NLI -----------------------------------------
# Judges the relation between two texts. No RAG documents involved.
# The prompt block itself is a guardian block built above from
# ``CRITERIA["entailment"]`` — see :func:`nli_text` / :data:`NLI_TEXT`.


def get_nli_messages(premise: str, hypothesis: str) -> List[Dict[str, str]]:
    """Chat messages for NLI: premise (text A) vs hypothesis (text B) -> label."""
    user_turn = f"Text A (premise): {premise}\n\nText B (hypothesis): {hypothesis}\n\n{NLI_TEXT}"
    return [{"role": "user", "content": user_turn}]


def get_nli_prompt(premise: str, hypothesis: str, tokenizer: Any, *, thinking: bool = False) -> str:
    """Create the NLI prompt (chat-templated string)."""
    return render_chat_input(get_nli_messages(premise, hypothesis), tokenizer, thinking=thinking)
