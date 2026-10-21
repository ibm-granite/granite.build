"""Config validation shared by every training step's config renderer.

WHY THIS MODULE EXISTS. `distill-gold-train` and `distill-sft-baseline` are separate steps
with separate images, and `distill-sft-baseline`'s README states the rule they follow:
SHARE THE CODE, NOT THE STEP. The two steps must stay separate -- a control that shares a
step with the treatment is a control you can silently mis-configure into the treatment -- but
their *validation* must not diverge, because a control validated by a second, slightly
different copy of these rules is not a control either. Every guard here was written for a
failure that actually happened; a copy that drifts by one line reintroduces one of them.

The step renderers keep their own arm tables, flag surfaces and emission logic. What lives
here is only what would otherwise be duplicated verbatim: the corpus/student tokenizer
identity comparison, the resume preflight, the checkpoint predicate, and the topology bounds.

Each function raises ConfigError with an operator-facing message and returns None. They do
not print, and they never repair a config: a guard that fixes things up silently changes the
experiment the operator asked for, which is the failure mode all of this exists to prevent.
"""
from __future__ import annotations

import json
from pathlib import Path

from gb_steps_post_training.distillation import tokenizer_identity

# auto | never | require. Shared because it is the SAME preflight in both steps for the same
# reason: both entrypoints resume on the mere PRESENCE of a checkpoint directory and offer no
# way to ask for anything else, so the mode can only be enforced before the trainer starts.
RESUME_MODES = ("auto", "never", "require")


class ConfigError(Exception):
    """A config the step refuses to run. Message is the operator-facing explanation."""


def corpus_tokenizer_identity(corpus_path: Path) -> str | None:
    """Return the tokenizer identity recorded in the corpus manifest, or None.

    THE CORPUS IS NOT TOKENIZED, and an earlier version of this docstring said it was.
    distill-corpus-prep emits a TEXT-level JSONL of `messages` conversations, and the trainer
    tokenizes at train time. So the file holds no token ids at all.

    It is still tokenizer-SPECIFIC, which is why this comparison exists: corpus-prep drops
    conversations that exceed max_length, and that length is measured with a specific
    tokenizer. Swap the tokenizer and the filtering decisions become wrong -- silently, since
    nothing downstream re-checks lengths. The pre_tokenizer difference between the two
    families (Split-regex on 4.1 vs plain ByteLevel on 4.2) is exactly large enough to move
    examples across that boundary.

    Returning None means "no manifest" (an older corpus), which callers must treat as
    unverifiable rather than as agreement.
    """
    manifest = _corpus_manifest(corpus_path)
    if manifest is None:
        return None
    identity = manifest.get("tokenizer_identity") or manifest.get("tokenizer")
    return str(identity) if identity is not None else None


def _corpus_manifest(corpus_path: Path) -> dict | None:
    """The corpus's manifest as a dict, or None when it has none. Both readers use this."""
    for name in ("manifest.json", "corpus_manifest.json"):
        candidate = corpus_path.parent / name if corpus_path.is_file() else corpus_path / name
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except ValueError as exc:
                raise ConfigError(f"corpus manifest {candidate} is not valid JSON: {exc}") from exc
    return None


def corpus_tokenizer_hash(corpus_path: Path) -> str | None:
    """The corpus's tokenizer in `sha256:<16 hex>` form, or None if it cannot be established.

    Three sources, in descending order of trust, because corpora built at different times
    record different things:
      1. `tokenizer_sha256`, which distill-corpus-prep writes from 2026-09-21 on.
      2. `tokenizer_identity`, WHEN it is already hash-form -- which is every corpus prepped
         from a student that carried no tokenizer_identity.json, i.e. every corpus on disk
         before that date.
      3. Hashing `tokenizer_path`'s tokenizer.json. Last because the path is a record of
         where the tokenizer WAS: the directory can be gone, or worse, still there and since
         rewritten, and then this is a statement about today's bytes rather than the ones the
         corpus was filtered with. Used anyway because the alternative is no verdict at all,
         and a rewritten retag output is a much rarer event than a missing sha field.

    None means unverifiable -- callers must not read it as agreement. See
    tokenizer_identity.comparable_hash for why the hash and not the name is what the guard
    compares.
    """
    manifest = _corpus_manifest(corpus_path)
    if manifest is None:
        return None
    recorded = manifest.get("tokenizer_sha256")
    if recorded:
        return str(recorded)
    identity = manifest.get("tokenizer_identity") or manifest.get("tokenizer")
    if identity and str(identity).startswith("sha256:"):
        return str(identity)
    tok_path = manifest.get("tokenizer_path")
    if tok_path and Path(tok_path).is_dir():
        return tokenizer_identity.hash_tokenizer(Path(tok_path))
    return None


def student_tokenizer_identity(student_path: Path) -> str | None:
    """The student's tokenizer identity: recorded name if present, else a content hash.

    Delegated to the shared helper so that the side WRITING an identity
    (distill-tokenizer-align) and the side READING it here cannot drift; a guard whose two
    halves compute identity differently is worse than no guard, because it either fires on
    matching tokenizers or passes on mismatched ones.

    An earlier version of this returned None whenever tokenizer_identity.json was absent --
    which, since nothing in the repo wrote that file, was ALWAYS. The comparison it fed was
    therefore skipped on every run it was meant to protect. The named form is still preferred
    (a mismatch should read "granite-4.2-3b != granite-4.1-3b-base", not "abc123 != def456");
    the hash is the fallback that keeps the check real for tokenizers no step produced, e.g.
    any base model in a smoke run.
    """
    try:
        return tokenizer_identity.read(student_path)
    except tokenizer_identity.IdentityError as exc:
        raise ConfigError(str(exc)) from exc


def existing_checkpoints(output_dir: Path) -> list[Path]:
    """Checkpoint dirs the trainer would find. Mirrors the entrypoints' glob deliberately:
    if this predicate and theirs disagree, the validation is worse than none."""
    if not output_dir.is_dir():
        return []
    return sorted(p for p in output_dir.glob("checkpoint-*") if p.is_dir())


def validate_resume(mode: str, output_dir: str | Path, *, entrypoint: str) -> None:
    """Enforce resume=auto|never|require against what is on disk.

    `entrypoint` names the script whose behaviour is being constrained (e.g. "gold.py"), so
    the message points at the code that would do the resuming rather than at this module.

    The mode itself is checked by the caller's argparse/validate, since a bad mode should be
    reported whether or not paths are being inspected. This half needs the filesystem.
    """
    found = existing_checkpoints(Path(output_dir))
    if mode == "never" and found:
        raise ConfigError(
            f"resume='never' but {len(found)} checkpoint(s) already exist under "
            f"{output_dir} (e.g. {found[-1].name}). {entrypoint} resumes on their mere "
            "presence, so this run would restore the previous optimizer and scheduler "
            "state while applying the current config -- a silent hybrid that reports as a "
            "clean run. Point output_dir somewhere new, or move the existing checkpoints "
            "aside yourself. This step will not delete training output for you."
        )
    if mode == "require" and not found:
        raise ConfigError(
            f"resume='require' but no checkpoint-* exists under {output_dir}. "
            f"{entrypoint} would silently train from scratch instead of resuming; failing "
            "here rather than after hours of training that was supposed to continue."
        )


def validate_student_against_corpus(student_path: str | Path, corpus_path: str | Path) -> None:
    """The student must carry a chat template, and must be the tokenizer the corpus was built for.

    Both halves are assertions rather than notes in a README because both failures are
    SILENT. A student without chat_template.jinja does not error -- it segments prompt from
    completion wrongly. A corpus built with the other tokenizer does not error either: after
    retagging the ids are interchangeable, but the pre_tokenizers still differ (a Split regex
    on 4.1 vs plain ByteLevel on 4.2), so the same text segments differently and the run
    trains on mis-segmented text at full speed with a falling loss.
    """
    student = Path(student_path)
    if not (student / "chat_template.jinja").is_file():
        raise ConfigError(
            f"{student}/chat_template.jinja is missing. distill-tokenizer-align installs "
            "it; a student without it will not segment prompt from completion correctly."
        )

    corpus_tok = corpus_tokenizer_identity(Path(corpus_path))
    student_tok = student_tokenizer_identity(student)

    # COMPARE THE HASHES, REPORT THE NAMES. The identities above are the legible form and
    # they are not always the same KIND of value on both sides: a corpus prepped before
    # distill-tokenizer-align began writing tokenizer_identity.json carries a content hash,
    # while a student retagged after that carries a name, and comparing those two refuses a
    # byte-identical pair. Measured on the bullet-4 pair, 2026-09-21 -- see
    # tokenizer_identity.comparable_hash for the numbers. The hash is symmetric by
    # construction, so it is the comparison; the names only have to make the message readable.
    corpus_sha = corpus_tokenizer_hash(Path(corpus_path))
    student_sha = tokenizer_identity.comparable_hash(student)
    if corpus_sha is not None and student_sha is not None:
        if corpus_sha != student_sha:
            raise ConfigError(
                f"corpus was tokenized with {corpus_tok!r} ({corpus_sha}) but the student's "
                f"tokenizer is {student_tok!r} ({student_sha}). The corpus is "
                "tokenizer-specific and cannot be reused across tokenizers; rebuild it with "
                "distill-corpus-prep against this student."
            )
        return

    # Neither side could produce a hash, so fall back to the identities as written. This is
    # weaker on purpose rather than by omission: it is the only comparison available for a
    # corpus whose manifest records no sha and whose tokenizer_path is gone, and refusing
    # outright there would block runs the guard has no evidence against.
    if corpus_tok is not None and student_tok is not None and corpus_tok != student_tok:
        raise ConfigError(
            f"corpus was tokenized with {corpus_tok!r} but the student's tokenizer is "
            f"{student_tok!r}, and neither side records a tokenizer sha to compare instead. "
            "The corpus is tokenizer-specific and cannot be reused across tokenizers; "
            "rebuild it with distill-corpus-prep against this student."
        )


def validate_topology(gpus_per_node: int, nodes: int) -> None:
    """Bounds that are a property of the cluster, not of any one objective."""
    if gpus_per_node < 1 or gpus_per_node > 8:
        raise ConfigError(
            f"gpus_per_node must be 1..8 (a validated host has 8 GPUs); got {gpus_per_node}."
        )
    if nodes < 1:
        raise ConfigError(f"nodes must be >= 1; got {nodes}.")
