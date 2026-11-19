# SPDX-License-Identifier: Apache-2.0

"""Post-export verification for the lm-engine -> Hugging Face conversion.

The export is two stages of upstream code: ``lm_engine.training.unshard`` writes an
unsharded ``gpt_base`` checkpoint, then ``tools.model_conversion.export_to_huggingface``
rewrites it as a ``GraniteConfig`` / ``LlamaConfig`` / ``GraniteMoeHybridConfig`` model.
Both stages can succeed and still hand vLLM a model that is quietly not the one that
was trained, so this module checks the result and fails the target instead. Every
check here is one incident this project has already paid for:

**``rope_theta`` dropped on export.** transformers 5.x nests the RoPE settings under
``rope_parameters``; 4.x keeps ``rope_theta`` at the top level. An exporter that writes
only one shape, read by a server that looks at the other, loses the value. A granite-4
model trained at theta=1e7 was served at the 1e4 default: it produced fluent text and
scored 0.000 on a tool-calling benchmark whose baseline was 0.8075, and the failure was
chased as a tool-format bug for days. So ``rope_theta`` is accepted at the top level,
else under ``rope_parameters.rope_theta``, else under ``rope_scaling`` -- and *absent*
is an error, never "the default is fine".

**Tokenizer re-saving corrupts the tokenizer.** ``save_pretrained`` on a loaded
tokenizer leaked the trainer's runtime ``truncation.max_length`` into the exported
``tokenizer.json`` and dropped a trained ``Split`` pre-tokenizer; the model mis-segmented
text and every eval recorded against it was understated. Upstream's exporter does
exactly that (``tokenizer.save_pretrained(save_path, legacy_format=False)``), and when it
cannot load a tokenizer at all it swallows the exception and writes *none*. So the step
copies the tokenizer files byte for byte with ``shutil.copy2`` and :func:`check_tokenizer`
asserts byte-identity -- a real byte comparison, not a size or mtime check.

**muP multipliers.** ``_export_granite_config`` maps lm-engine's ``m_emb`` / ``m_residual``
/ ``m_width`` / ``attention_multiplier`` onto ``embedding_multiplier`` /
``residual_multiplier`` / ``logits_scaling`` / ``attention_multiplier``, substituting ``1``
for any that is ``None``. Losing one changes the model's effective parametrization while
the config still looks plausible.

Both entry points are pure and stdlib-only (``json``, ``pathlib``, ``filecmp``): they run
in the export pod after the heavy work, and must not need torch or transformers. Each
returns the *complete* list of human-readable problems, the way
``render_config.validate`` does, so one failed export tells the user everything that is
wrong rather than one thing at a time.
"""

from __future__ import annotations

import filecmp
import json
from pathlib import Path
from typing import Any

__all__ = ["check_config", "check_tokenizer", "TOKENIZER_FILES"]

#: Training knob name -> the field the Hugging Face config writes it to. The names on
#: the left are the pretrain step's knobs (``render_config.DEFAULT_KNOBS``); ``expected``
#: is keyed by them so a recipe can pass the knobs it trained with unchanged.
#: ``sequence_length`` is the one rename: lm-engine's ``GPTBaseConfig`` carries it as
#: ``max_position_embeddings``, which is what the HF config and vLLM read.
_PLAIN_FIELDS = {
    "vocab_size": "vocab_size",
    "tie_word_embeddings": "tie_word_embeddings",
    "bos_token_id": "bos_token_id",
    "eos_token_id": "eos_token_id",
    "pad_token_id": "pad_token_id",
    "sequence_length": "max_position_embeddings",
}

#: The muP mapping ``_export_granite_config`` applies. Checked only for a ``granite``
#: export, as the recipe's model_type resolution guarantees: a standard-parametrization
#: run goes out as ``llama``, whose config has no multiplier fields at all. (Upstream's
#: ``granitemoehybrid`` exporter writes the same four fields, so widening the gate is a
#: one-line change once a MoE recipe exists.)
_MUP_FIELDS = {
    "m_emb": "embedding_multiplier",
    "m_residual": "residual_multiplier",
    "m_width": "logits_scaling",
    "attention_multiplier": "attention_multiplier",
}

#: Nested blocks a transformers version may hide ``rope_theta`` in, in the order they
#: are consulted.
_ROPE_PARENTS = ("rope_parameters", "rope_scaling")

#: Every file that is part of a tokenizer. The export copies the ones that exist and
#: nothing else; a name here that appears only in the export was synthesised.
TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
)

_SPECIAL_TOKENS = ("bos_token", "eos_token", "pad_token")


def check_config(cfg: dict, expected: dict) -> list[str]:
    """Compare a written ``config.json`` against the values it was trained with.

    ``cfg`` is the parsed exported ``config.json``; ``expected`` is the training-side
    dict keyed by the pretrain step's knob names. Only keys present in ``expected`` are
    checked, so a recipe decides what to pin -- but a key that *is* in ``expected`` and
    missing from ``cfg`` is an error: an absent field means the file does not state the
    behaviour, and the HF defaults differ between architectures and versions.

    Returns every problem found, as human-readable strings; empty means the export is
    faithful.
    """
    errors: list[str] = []
    if not expected:
        return errors

    model_type = cfg.get("model_type")
    if "model_type" in expected and model_type != expected["model_type"]:
        errors.append(
            f"model_type is {model_type!r}, but the recipe asked to export as "
            f"{expected['model_type']!r}"
        )

    if "rope_theta" in expected:
        errors += _check_rope_theta(cfg, expected["rope_theta"])

    for knob, field in _PLAIN_FIELDS.items():
        if knob in expected:
            errors += _compare(cfg, field, expected[knob], knob)

    # muP: only a granite export carries the multipliers (see _MUP_FIELDS).
    if model_type == "granite":
        for knob, field in _MUP_FIELDS.items():
            if knob in expected:
                errors += _compare(cfg, field, expected[knob], knob)

    return errors


def check_tokenizer(src: Path, dst: Path, cfg: dict) -> list[str]:
    """Check that the exported tokenizer is the trained one, byte for byte.

    ``src`` is the directory the tokenizer files were copied *from*, ``dst`` the exported
    model directory, and ``cfg`` its parsed ``config.json``. Every tokenizer file in
    ``src`` must exist in ``dst`` and be byte-identical; a tokenizer file that exists only
    in ``dst`` is an error, because the export copies and never generates, so such a file
    came out of ``save_pretrained``. Files that are not part of a tokenizer (the weights,
    ``config.json``, ``generation_config.json``) are ignored.

    Finally, each special token the exported tokenizer *declares* must resolve to the id
    ``cfg`` uses for it. A token the tokenizer does not declare is not compared: training
    may set ``bos_token_id`` without the tokenizer emitting a BOS, which is a style
    choice rather than a mis-export.

    Returns every problem found.
    """
    errors: list[str] = []
    for label, path in (("source", src), ("exported", dst)):
        if not path.is_dir():
            errors.append(f"the {label} tokenizer directory {path} does not exist")
    if errors:
        return errors

    src_files = _tokenizer_files(src)
    dst_files = _tokenizer_files(dst)

    if not src_files:
        errors.append(
            f"no tokenizer files in {src}: the exported model would have no tokenizer "
            "and could not be served"
        )

    for name in sorted(src_files):
        target = dst / name
        if not target.is_file():
            errors.append(
                f"{name} is missing from the exported model (present in {src}); "
                "export_to_huggingface writes no tokenizer at all when it cannot load one"
            )
        elif not filecmp.cmp(src / name, target, shallow=False):
            errors.append(
                f"{name} differs byte for byte from {src / name}: the tokenizer was "
                "re-saved instead of copied, which leaks the trainer's truncation "
                "settings and can drop trained pre-tokenizer rules"
            )

    for name in sorted(dst_files - src_files):
        errors.append(
            f"{name} exists only in the exported model: nothing trained it, so it was "
            "synthesised by save_pretrained"
        )

    # Only compare ids once the copy itself is clean: resolving a special token through
    # a file that is missing or already known to differ reports the same single fault
    # three more times, which buries the fault that matters.
    if not errors:
        errors += _check_token_ids(dst, cfg)
    return errors


# --------------------------------------------------------------------- internals


def _compare(cfg: dict, field: str, want: Any, knob: str) -> list[str]:
    where = f"{field} (trained as {knob})" if field != knob else field
    if field not in cfg:
        return [
            f"{where} is absent from the exported config, but training set it to "
            f"{want!r}; the Hugging Face defaults differ by architecture, so the "
            "exported config must state it"
        ]
    got = cfg[field]
    if _same(got, want):
        return []
    return [f"{where} is {got!r} in the exported config, but training used {want!r}"]


def _same(got: Any, want: Any) -> bool:
    """Exact, type-aware equality.

    ``None`` matches only ``None``: a null id on one side and a number on the other is a
    real divergence. Booleans match only booleans, so ``1`` is not ``True`` -- an
    exported ``tie_word_embeddings`` of ``1`` means something wrote an int where HF
    expects a flag. Numbers compare exactly, with no tolerance: these are config scalars
    that reach the file through a JSON round-trip of the very same Python float, never
    through arithmetic, so any difference at all means the exporter transformed the
    value (collapsed a ``None`` to ``1``, or narrowed a float) -- which is exactly what
    is being looked for.
    """
    if got is None or want is None:
        return got is None and want is None
    if isinstance(got, bool) or isinstance(want, bool):
        return isinstance(got, bool) and isinstance(want, bool) and got == want
    return got == want


def _rope_theta_sites(cfg: dict) -> list[tuple[str, Any]]:
    """Every place the config states a non-null ``rope_theta``, as (path, value)."""
    sites: list[tuple[str, Any]] = []
    if cfg.get("rope_theta") is not None:
        sites.append(("rope_theta", cfg["rope_theta"]))
    for parent in _ROPE_PARENTS:
        block = cfg.get(parent)
        if isinstance(block, dict) and block.get("rope_theta") is not None:
            sites.append((f"{parent}.rope_theta", block["rope_theta"]))
    return sites


def _check_rope_theta(cfg: dict, want: Any) -> list[str]:
    sites = _rope_theta_sites(cfg)
    if not sites:
        return [
            f"rope_theta is absent from the exported config (training used {want!r}); "
            "a server that finds no value serves the architecture default, which has "
            "already cost this project a 0.000 benchmark score -- it is never "
            "'the default is fine'"
        ]
    values = {value for _, value in sites}
    if len(values) > 1:
        shown = ", ".join(f"{path}={value!r}" for path, value in sites)
        return [
            f"the exported config states rope_theta more than once and the values "
            f"disagree ({shown}); which copy a server honours depends on its "
            "transformers version, so the served theta is unpredictable"
        ]
    path, value = sites[0]
    if _same(value, want):
        return []
    return [f"{path} is {value!r} in the exported config, but training used {want!r}"]


def _tokenizer_files(directory: Path) -> set[str]:
    return {
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.name in TOKENIZER_FILES
    }


def _load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f"{path.name} in the exported model could not be read: {exc}")
        return None


def _token_vocabulary(directory: Path, errors: list[str]) -> dict[str, int]:
    """Map token text -> id from the exported tokenizer's added tokens and vocab."""
    vocab: dict[str, int] = {}
    tokenizer_json = directory / "tokenizer.json"
    if tokenizer_json.is_file():
        data = _load_json(tokenizer_json, errors)
        if isinstance(data, dict):
            model = data.get("model")
            if isinstance(model, dict) and isinstance(model.get("vocab"), dict):
                vocab.update(model["vocab"])
            for entry in data.get("added_tokens") or []:
                if isinstance(entry, dict) and "content" in entry and "id" in entry:
                    vocab[entry["content"]] = entry["id"]
    added = directory / "added_tokens.json"
    if added.is_file():
        data = _load_json(added, errors)
        if isinstance(data, dict):
            vocab.update(data)
    return vocab


def _declared_tokens(directory: Path, errors: list[str]) -> dict[str, str]:
    """The token text the exported tokenizer declares for each special-token role.

    ``tokenizer_config.json`` holds each one either as a plain string or as an
    ``AddedToken`` mapping with a ``content`` key; ``special_tokens_map.json`` is the
    fallback for a tokenizer that only writes it there. Each file is read once, so an
    unreadable one is reported once rather than per role.
    """
    declared: dict[str, str] = {}
    for name in ("special_tokens_map.json", "tokenizer_config.json"):
        path = directory / name
        if not path.is_file():
            continue
        data = _load_json(path, errors)
        if not isinstance(data, dict):
            continue
        for role in _SPECIAL_TOKENS:
            value = data.get(role)
            if isinstance(value, str):
                declared[role] = value
            elif isinstance(value, dict) and isinstance(value.get("content"), str):
                declared[role] = value["content"]
    return declared


def _check_token_ids(directory: Path, cfg: dict) -> list[str]:
    errors: list[str] = []
    vocab = _token_vocabulary(directory, errors)
    declared = _declared_tokens(directory, errors)
    for role in _SPECIAL_TOKENS:
        content = declared.get(role)
        if content is None:
            continue
        field = f"{role}_id"
        if field not in cfg:
            continue
        want = cfg[field]
        if content not in vocab:
            errors.append(
                f"the exported tokenizer declares {role} {content!r}, which is not in "
                f"its own vocabulary, so {field} ({want!r}) cannot be honoured"
            )
        elif not _same(vocab[content], want):
            errors.append(
                f"the exported tokenizer resolves {role} {content!r} to id "
                f"{vocab[content]!r}, but the exported config's {field} is {want!r}"
            )
    return errors
