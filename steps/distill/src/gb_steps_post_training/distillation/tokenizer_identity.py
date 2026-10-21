"""One definition of "which tokenizer is this", shared by every step that has an opinion.

WHY IT IS SHARED CODE AND NOT A CONVENTION. Three steps need to agree about this:
distill-tokenizer-align WRITES an identity onto the retagged student, distill-corpus-prep
COPIES it into the corpus manifest, and distill-gold-train COMPARES the two and refuses a
mismatch. If any of the three computed it differently the guard would either fire on
matching tokenizers or -- much worse -- pass on mismatched ones. So it is one function.

WHY THE GUARD MATTERS. After retagging, teacher and student ids are interchangeable, but
the two pre_tokenizers still differ (Sequence[Split(regex), ByteLevel] on granite-4.1
versus plain ByteLevel on 4.2), so the SAME TEXT can segment differently. A corpus built
with one tokenizer and trained with the other is not a crash; it is a quietly worse model.

HOW IT WAS FOUND BROKEN. render_gold_config.py already implemented the comparison and read
the identity from `<model>/tokenizer_identity.json` -- a file NOTHING in the repo wrote. So
it returned None on every real retagged student, and the check was skipped every time it
was supposed to fire. An assertion that cannot fail is worse than no assertion, because the
plan doc counted it as done.

TWO FORMS, deliberately, and the preference order is not arbitrary:
  1. A NAME, when `tokenizer_identity.json` is present. render_gold_config's original
     comment made the right argument for this: a mismatch should read "retagged-from-4.2-30b
     != granite-4.1-3b-base", which tells an operator what happened, not
     "abc123 != def456", which tells them nothing.
  2. A CONTENT HASH of tokenizer.json (`sha256:<16 hex>`) when the file is absent.
     The original returned None there, which made the check unverifiable for any tokenizer
     not produced by the retag step -- including every base model, i.e. every smoke run.
     A hash is a poor error message but a real check, and the fallback is symmetric on both
     sides, so it can only compare equal for byte-identical tokenizers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

IDENTITY_FILE = "tokenizer_identity.json"
IDENTITY_KEY = "tokenizer_identity"


class IdentityError(Exception):
    """A tokenizer identity that cannot be read. Message is operator-facing."""


def derive_name(model_dir: Path) -> str:
    """A LEGIBLE name for the tokenizer in `model_dir`, decoding the HF cache layout.

    `Path.name` is not good enough and the difference is not cosmetic. A model resolved from
    the hub cache lives at
        <HF_HOME>/hub/models--ibm-granite--granite-4.2-3b/snapshots/8b6ac672.../
    so the basename is a 40-char commit SHA. Recording that as the identity produces exactly
    the mismatch message the guard was designed to avoid: "8b6ac672... != a50b46ce..." says
    nothing about which tokenizer was wrong. Confirmed by direct measurement, where the first
    version of this defaulting recorded precisely those SHAs.

    So walk up to the `models--org--name` directory and decode it back to `org/name`. Any
    other layout (a plain checkout, a step's output directory) keeps its basename, which is
    already meaningful there.
    """
    p = Path(model_dir).resolve()
    for parent in (p, *p.parents):
        if parent.name.startswith("models--"):
            return parent.name[len("models--"):].replace("--", "/")
    return p.name


def hash_tokenizer(model_dir: Path) -> str | None:
    """`sha256:<16 hex>` over tokenizer.json, or None if there is no tokenizer.json.

    Truncated to 16 hex chars: this is a change-detector between two directories that are
    meant to hold the same file, not a security boundary, and a full 64-char digest makes
    the log line unreadable for no gain.
    """
    fp = Path(model_dir) / "tokenizer.json"
    if not fp.is_file():
        return None
    h = hashlib.sha256(fp.read_bytes()).hexdigest()[:16]
    return f"sha256:{h}"


def comparable_hash(model_dir: Path) -> str | None:
    """The `sha256:<16 hex>` form for `model_dir`, from the recorded value or recomputed.

    WHY THIS EXISTS SEPARATELY FROM read(). read() prefers the NAME form, for the good
    legibility reason above -- but a name and a hash are two different KINDS of answer, and
    the two sides of the guard do not always produce the same kind. Measured on 2026-09-21,
    building the bullet-4 pair:

        granite-4.0-1b-base_retagged_v2   'ibm-granite/granite-4.2-30b'  (retagged today,
                                           so distill-tokenizer-align wrote an identity file)
        en-sft-4.1-0.2-16K-v2 manifest    'sha256:883975314d587437'      (prepped 2026-08-26
                                           from a student that had no identity file to copy)

    Byte-identical tokenizers -- same 100,352-entry vocab, same 96 added tokens, same
    tokenizer.json sha -- and the guard REFUSED the pair. That is the exact failure this
    module's header warns about ("fire on matching tokenizers"), arriving through the
    producer boundary rather than through a disagreement in the arithmetic: the align step
    started writing identity files after the corpora were built, so every corpus that
    predates that change is name-incomparable with every student that postdates it.

    The hash is the form that is symmetric by construction, so it is what a COMPARISON
    should use; the name stays what an error MESSAGE should use. Recorded in preference to
    recomputed only because a retagged directory records the hash of the tokenizer.json it
    wrote, and re-reading 7 MB per validate() call buys nothing.
    """
    model_dir = Path(model_dir)
    meta = model_dir / IDENTITY_FILE
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text())
        except ValueError as exc:
            raise IdentityError(f"{meta} is not valid JSON: {exc}") from exc
        recorded = payload.get("tokenizer_sha256")
        if recorded:
            return str(recorded)
    return hash_tokenizer(model_dir)


def read(model_dir: Path) -> str | None:
    """The identity of the tokenizer in `model_dir`: recorded name, else content hash.

    None only when the directory holds no tokenizer.json at all, which means "not a
    tokenizer directory" rather than "unverifiable".
    """
    model_dir = Path(model_dir)
    meta = model_dir / IDENTITY_FILE
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text())
        except ValueError as exc:
            raise IdentityError(f"{meta} is not valid JSON: {exc}") from exc
        value = payload.get(IDENTITY_KEY)
        if value:
            return str(value)
        # A present-but-empty identity is a bug in whatever wrote it, not a licence to
        # fall through to the hash: silently substituting a different identity scheme is
        # how a guard starts comparing two things that were never the same kind of thing.
        raise IdentityError(
            f"{meta} exists but has no non-empty {IDENTITY_KEY!r} key. Delete the file to "
            f"fall back to a content hash, or write a real identity into it."
        )
    return hash_tokenizer(model_dir)


def write(model_dir: Path, identity: str, **extra: object) -> Path:
    """Record `identity` for the tokenizer in `model_dir`. Returns the file written.

    `extra` is recorded alongside for provenance (who produced this tokenizer, from what)
    and is never read back by the comparison -- only IDENTITY_KEY is. Keeping the compared
    field to a single scalar is deliberate: a structured identity invites partial matches.
    """
    model_dir = Path(model_dir)
    if not identity:
        raise IdentityError("identity must be a non-empty string")
    payload = {IDENTITY_KEY: identity, "tokenizer_sha256": hash_tokenizer(model_dir), **extra}
    dest = model_dir / IDENTITY_FILE
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return dest
