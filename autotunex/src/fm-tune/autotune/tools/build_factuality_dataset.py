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

"""Build factuality intrinsics datasets (detection / correction) from raw ELI5 splits.

Reads one raw JSON file holding a top-level list of records and emits JSONL in
one or both of two shapes:

- ``formatted``: ``input`` is a string rendered through the model's chat
  template; ``output`` is the target JSON string.
- ``chat``: ``input`` is a list of chat messages, ``documents`` is the separate
  documents column, and ``output`` is the target JSON string.

Usage:

    python -m autotune.tools.build_factuality_dataset \
        --input /path/to/eli5_raw_train.json \
        --output-dir /path/to/out \
        --task detection \
        --model ibm-granite/granite-4.1-3b

The output filename is derived from the input file, the task, and the format:
``eli5_raw_train.json`` with ``--task detection`` yields
``eli5_raw_train_detection_formatted.jsonl`` and
``eli5_raw_train_detection_chat.jsonl`` in ``--output-dir`` (``--format both`` is
the default; pass ``--format formatted`` or ``--format chat`` for just one).

Granite document handling differs by generation
-----------------------------------------------
The RAG context ("documents") is passed to ``apply_chat_template`` differently
by each Granite generation, so this tool inspects the tokenizer's chat template
and adapts (see ``detect_document_style``):

- **Granite 3.x** reads ``document['doc_id']`` and ``document['text']``
  explicitly, emitting one ``<|start_of_role|>document {...}<|end_of_role|>``
  block per document. ``doc_id`` is required.
- **Granite 4.0 / 4.1** render ``document | tojson`` into a ``<documents>``
  XML block in the system message. Keys appear *verbatim* in the prompt, so a
  stray ``doc_id`` would leak as a meaningless field; only ``text`` is emitted.
- **Granite 4.2** dropped ``documents`` support entirely (and moved to ChatML).
  Passing ``documents=`` there is silently ignored.

``apply_chat_template`` never raises on an unsupported ``documents=`` kwarg, so
a template that ignores it would otherwise yield training data whose context is
missing from the prompt — teaching the judge to guess. For those templates this
tool inlines the documents into a leading system message instead (portable: the
inlined text also survives 3.x/4.x templates), and refuses to emit anything if
inlining is disabled.

Note: Granite 3.3 injects a live ``Today's Date:`` line into rendered prompts,
so ``formatted`` files built with it are not byte-reproducible across days.
Granite 4.x templates carry no date.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Re-exported under the historical private names so existing callers and tests
# keep working now that the capability helpers live in _chat_utils.
from ._chat_utils import (  # noqa: F401
    CRITERIA,
    CRITERIA_ID,
    DOC_STYLE_GRANITE3,
    DOC_STYLE_GRANITE4,
    DOC_STYLE_INLINE,
    DOC_STYLES,
    IdentityTokenizer,
    detect_document_style,
    guardian_text,
    render_chat_input,
    template_supports_kwarg,
)
from ._chat_utils import build_documents as _build_documents  # noqa: F401
from ._chat_utils import inline_documents_message as _inline_documents_message  # noqa: F401
from ._chat_utils import output_name_for as _output_name_for
from ._chat_utils import resolve_formats as _resolve_formats  # noqa: F401
from ._chat_utils import split_documents as _split_documents  # noqa: F401


def _guardian_block(is_detection: bool, for_prompt: bool) -> str:
    """Build the ``<guardian>`` user-turn content.

    The preamble, criteria (from ``CRITERIA[CRITERIA_ID]``), and scoring schema
    are assembled by ``_chat_utils.guardian_text`` so the training and eval
    prompts can never drift. ``for_prompt`` appends the JSON dict schema hint
    that the prompt-style variants use.
    """
    return guardian_text(
        CRITERIA_ID,
        variant="detection" if is_detection else "correction",
        for_prompt=for_prompt,
    )


def _unique_strings(strings: List[str]) -> List[str]:
    seen = set()
    result = []
    for s in strings:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _context_sort_key(key: str) -> Tuple[int, str]:
    """Sort ``c_a*`` keys by trailing integer so c_a2 precedes c_a10.

    Raw JSON preserves file order, which is not necessarily numeric; keys with
    no trailing digits sort last, alphabetically.
    """
    m = re.search(r"(\d+)$", key)
    return (int(m.group(1)), key) if m else (sys.maxsize, key)


# Raw factuality files come in two shapes. ELI5-style records carry one context
# per ``c_a*`` key (``{"text": ...}`` dicts); Biographies-style records carry a
# single list-of-strings field instead. ``contexts_short`` is preferred over the
# full ``contexts`` because the latter averages ~8.7k tokens per record (peaking
# near 26k), which would blow past a normal ``max_seq_length``.
CONTEXT_LIST_FIELDS = ("contexts_short", "contexts")


def _extract_contexts(
    record: Dict[str, Any],
    context_field: Optional[str] = None,
) -> List[str]:
    """Pull the context passages out of a raw record and deduplicate.

    Handles both raw schemas: ``c_a*`` keys (ELI5) and a list-of-strings field
    (Biographies). ``context_field`` pins which list field to prefer; by default
    :data:`CONTEXT_LIST_FIELDS` is tried in order.
    """
    keys = [k for k, v in record.items() if k.startswith("c_a") and isinstance(v, dict) and v.get("text") is not None]
    if keys:
        return _unique_strings([record[k]["text"] for k in sorted(keys, key=_context_sort_key)])

    # Biographies-style: a flat list of context strings.
    fields = (context_field,) if context_field else CONTEXT_LIST_FIELDS
    for field in fields:
        value = record.get(field)
        if isinstance(value, list):
            texts = [v.strip() for v in value if isinstance(v, str) and v.strip()]
            if texts:
                return _unique_strings(texts)
    return []


def _build_messages(
    query: str,
    response: str,
    is_detection: bool,
    for_prompt: bool,
    documents: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the 3-turn judge conversation, optionally prefixed with inlined docs."""
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
        {"role": "user", "content": _guardian_block(is_detection, for_prompt)},
    ]
    if documents:
        messages.insert(0, _inline_documents_message(documents))
    return messages


def _record_detection(
    dp: Dict[str, Any],
    tokenizer,
    fmt: str,
    for_prompt: bool,
    doc_style: str = DOC_STYLE_GRANITE4,
    thinking: bool = False,
    context_field: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    query = dp["query"]
    response = dp["response"]["text"]
    label = dp["response"]["label"]

    contexts = _extract_contexts(dp, context_field)
    if not contexts:
        # No grounding context -> the judge would have to guess. Drop the record.
        return None
    inline_docs, template_docs = _split_documents(contexts, doc_style)
    messages = _build_messages(query, response, is_detection=True, for_prompt=for_prompt, documents=inline_docs)
    output_seq = json.dumps({"score": label.lower()})

    if fmt == "formatted":
        return {
            "input": render_chat_input(messages, tokenizer, template_docs, thinking=thinking),
            "output": output_seq,
        }
    record: Dict[str, Any] = {"input": messages, "output": output_seq}
    if template_docs:
        record["documents"] = template_docs
    return record


def _record_correction(
    dp: Dict[str, Any],
    tokenizer,
    fmt: str,
    for_prompt: bool,
    max_length: int,
    include_meta: bool,
    doc_style: str = DOC_STYLE_GRANITE4,
    thinking: bool = False,
    context_field: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    query = dp["query"]
    response = dp["response"]["text"]
    label = dp["response"]["label"]
    correction = dp["correction"]["text"] if "correction" in dp else "none"

    contexts = _extract_contexts(dp, context_field)
    if not contexts:
        # No grounding context -> the judge would have to guess. Drop the record.
        return None
    inline_docs, template_docs = _split_documents(contexts, doc_style)
    messages = _build_messages(query, response, is_detection=False, for_prompt=for_prompt, documents=inline_docs)
    output_seq = json.dumps({"correction": correction})

    # Length gate the target — mirrors the scratchpad behavior.
    num_tokens = len(tokenizer(output_seq)["input_ids"])
    if num_tokens >= max_length:
        return None

    if fmt == "formatted":
        record: Dict[str, Any] = {
            "input": render_chat_input(messages, tokenizer, template_docs, thinking=thinking),
            "output": output_seq,
        }
        if include_meta:
            record.update({"query": query, "response": response, "label": label})
        return record

    record = {"input": messages, "output": output_seq}
    if template_docs:
        record["documents"] = template_docs
    if include_meta:
        # Same meta as the formatted path, ``label`` included: an eval file needs
        # the gold label to score the generated correction.
        record.update({"query": query, "response": response, "label": label})
    return record


def convert_split(
    input_path: str,
    output_path: str,
    task: str,
    fmt: str,
    tokenizer,
    *,
    for_prompt: bool = False,
    max_correction_length: int = 1024,
    doc_style: str = DOC_STYLE_GRANITE4,
    thinking: bool = False,
    include_meta: bool = False,
    context_field: Optional[str] = None,
) -> None:
    """Convert one raw JSON file to the target JSONL.

    Records with no grounding context are dropped: a factuality judge trained on
    a context-free prompt can only guess. ``include_meta`` echoes the original
    ``query``/``response``/``label`` onto each correction record so eval can
    recover the originals — useful for a held-out file, off by default.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{input_path}: expected a top-level JSON list")

    written = 0
    skipped_length = 0
    skipped_no_context = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for dp in records:
            if not isinstance(dp, dict):
                continue
            # Checked up front so the no-context and over-length drops stay
            # distinguishable in the summary line.
            if not _extract_contexts(dp, context_field):
                skipped_no_context += 1
                continue
            if task == "detection":
                record = _record_detection(dp, tokenizer, fmt, for_prompt, doc_style, thinking, context_field)
            else:
                record = _record_correction(
                    dp,
                    tokenizer,
                    fmt,
                    for_prompt,
                    max_correction_length,
                    include_meta,
                    doc_style,
                    thinking,
                    context_field,
                )
                if record is None:
                    skipped_length += 1
                    continue
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    skips = [f"{skipped_no_context} without context"]
    if task == "correction":
        skips.append(f"{skipped_length} over-length")
    print(f"[{task}/{fmt}] wrote {written} records to {output_path} (skipped {', '.join(skips)})")


class _IdentityTokenizer:
    """Stand-in tokenizer for chat-only conversion, which never renders a template."""

    chat_template = None

    def __call__(self, text: str):
        raise RuntimeError("chat-format conversion should not tokenize")

    def apply_chat_template(self, *args, **kwargs):
        raise RuntimeError("chat-format conversion should not render a template")


def _load_tokenizer(model: Optional[str], fmt: str, task: str):
    # formatted needs a real template; correction length-gates via tokenization.
    formats = _resolve_formats(fmt)
    needs_tokenizer = "formatted" in formats or task == "correction"
    if not needs_tokenizer:
        return _IdentityTokenizer()
    if not model:
        raise SystemExit(
            "--model is required for --format formatted/both or --task correction "
            "(used for chat template rendering and/or output length gating)"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def resolve_document_style(
    requested: str,
    tokenizer,
    *,
    inline_documents: bool,
    renders_template: bool,
) -> str:
    """Resolve ``--doc-style`` against the tokenizer's actual template.

    Raises ``SystemExit`` when the template would silently drop the documents
    and inlining is disabled — emitting such records would train the judge
    against context absent from its prompt.
    """
    template = getattr(tokenizer, "chat_template", None)
    if requested == "auto":
        if template is None and renders_template:
            raise SystemExit(
                "--model has no chat template; cannot render --format formatted/both. "
                "Pick a model with a chat template, or use --format chat with --doc-style."
            )
        doc_style = detect_document_style(template)
        if template is None:
            print(
                f"[doc-style] no chat template available; assuming '{doc_style}' (override with --doc-style)",
                file=sys.stderr,
            )
        else:
            print(f"[doc-style] detected '{doc_style}' from chat template")
    else:
        doc_style = requested
        print(f"[doc-style] using '{doc_style}' (explicit)")

    if doc_style == DOC_STYLE_INLINE:
        if not inline_documents:
            raise SystemExit(
                "The chat template does not consume `documents=` (e.g. Granite 4.2), so the "
                "context would be silently dropped from every prompt, and --no-inline-documents "
                "disables the fallback. Re-run without --no-inline-documents, or choose a model "
                "whose template supports documents."
            )
        print("[doc-style] documents will be inlined into a leading system message")
    return doc_style


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_factuality_dataset",
        description="Convert a raw factuality JSON file into training-ready JSONL.",
    )
    p.add_argument("--input", required=True, help="Raw input JSON file (top-level list of records).")
    p.add_argument("--output-dir", required=True, help="Directory to write the JSONL output into.")
    p.add_argument(
        "--task",
        required=True,
        choices=["detection", "correction"],
        help="Factuality task variant.",
    )
    p.add_argument(
        "--format",
        dest="fmt",
        default="both",
        choices=["formatted", "chat", "both"],
        help="'formatted' renders input via the chat template; 'chat' keeps messages + "
        "documents; 'both' writes one file of each (default: both).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name or path for tokenizer / chat template. "
        "Required for --format formatted/both and for --task correction.",
    )
    p.add_argument(
        "--doc-style",
        default="auto",
        choices=["auto", *DOC_STYLES],
        help="How to pass RAG documents. 'auto' (default) inspects the chat template: "
        "granite3 (doc_id+text), granite4 (text only, verbatim keys), "
        "inline (template ignores documents= — inlined into a system message).",
    )
    p.add_argument(
        "--no-inline-documents",
        dest="inline_documents",
        action="store_false",
        help="Fail instead of inlining documents when the chat template ignores `documents=`.",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable the template's thinking mode (Granite 4.2). Off by default so the "
        "target JSON is not trained inside a <think> block.",
    )
    p.add_argument(
        "--max-correction-length",
        type=int,
        default=1024,
        help="Skip correction records whose tokenized target is >= this length.",
    )
    p.add_argument(
        "--for-prompt",
        action="store_true",
        help="Append the JSON-dict schema hint to the guardian block.",
    )
    p.add_argument(
        "--context-field",
        default=None,
        choices=list(CONTEXT_LIST_FIELDS),
        help="For raw records that carry contexts as a flat list of strings "
        "(Biographies-style), which field to read. Default: try "
        f"{' then '.join(CONTEXT_LIST_FIELDS)}. Ignored for records with c_a* keys.",
    )
    p.add_argument(
        "--include-meta",
        action="store_true",
        help="Echo the original query/response/label onto each correction record so eval "
        "can recover the originals (use for a held-out file).",
    )
    return p


def output_name_for(input_path: str, task: str, fmt: str) -> str:
    """Derive the output filename from the input file, the task, and the format.

    ``eli5_raw_train.json`` -> ``eli5_raw_train_detection_chat.jsonl``.
    """
    return _output_name_for(input_path, fmt, task)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"[error] {args.input} not found", file=sys.stderr)
        return 1

    formats = _resolve_formats(args.fmt)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = _load_tokenizer(args.model, args.fmt, args.task)
    doc_style = resolve_document_style(
        args.doc_style,
        tokenizer,
        inline_documents=args.inline_documents,
        renders_template="formatted" in formats,
    )

    for fmt in formats:
        output_path = os.path.join(args.output_dir, output_name_for(args.input, args.task, fmt))
        convert_split(
            input_path=args.input,
            output_path=output_path,
            task=args.task,
            fmt=fmt,
            tokenizer=tokenizer,
            for_prompt=args.for_prompt,
            max_correction_length=args.max_correction_length,
            doc_style=doc_style,
            thinking=args.thinking,
            include_meta=args.include_meta,
            context_field=args.context_field,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
