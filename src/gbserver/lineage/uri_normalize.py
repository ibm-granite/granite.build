#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Normalize an artifact URI into a stable lineage identity.

The lineage index keys nodes by URI: two rows describe the same artifact exactly
when their ``input``/``output`` strings are equal. So every raw spelling of one
artifact has to converge here, and -- far more importantly -- two spellings of
*different* artifacts must never converge. A missed merge shows up as two
disconnected subgraphs, which is visible and fixable; a wrong merge invents
provenance, and someone concludes a model was trained on data it never saw.

That asymmetry is why this module is conservative: it applies only per-scheme
rules that are grounded in the URI handlers under :mod:`gbcommon.uri`, and it
returns ``""`` -- never a guess -- for anything it does not recognize.

Why not just ``str(URI.get_uri(raw))``:

- :meth:`URI.get_uri` runs the string through ``fill_template`` with the
  thread-local space config (``uri.py``), so it is not a pure function of its
  argument. Lineage identity must not depend on ambient state.
- Only ``HfURI``, ``EnvURI`` and ``MemURI`` define ``custom_str``; for every
  other scheme ``str()`` is the raw ``geturl()``, which normalizes nothing.
- ``LhURI.__init__`` *rewrites* the URI it is given (injecting a default
  revision) and raises on inputs this function must simply decline.

So the dispatch is explicit, and it borrows the handlers' logic only where that
logic is a pure string transform.
"""

from typing import Dict, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from gbcommon.uri.cos import COS_SCHEME, S3_SCHEME
from gbcommon.uri.git import split_repo_path
from gbcommon.uri.hf import (
    DEFAULT_REVISION,
    HF_HOST,
    HF_TYPE_TO_SEGMENT,
    HF_URI_SCHEME,
    HF_WEB_RESERVED_OWNERS,
    HF_WEB_REVISION_MARKERS,
    HF_WEB_TREE,
    URLSEGMENT_BUCKETS,
)
from gbcommon.uri.lh import (
    DEFAULT_FILESET_VERSION,
    DEFAULT_MODEL_REVISION,
    FILESET_REVISION_INDEX,
    LH_URI_SCHEME,
    MODEL_REVISION_INDEX,
    TYPE_INDEX,
    URLSEGMENT_FILES,
)
from gbcommon.uri.lh import URLSEGMENT_MODELS as LH_URLSEGMENT_MODELS
from gbserver.storage.stored_lineage_row import MAX_LINEAGE_URI_LENGTH
from gbserver.types.constants import FILE_SCHEME, MEM_URI_SCHEME
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# The HF path segments that name a repo type. A URI whose *host* is one of these
# is the missing-slash typo `hf://models/o/r`, which HfURI only warns about.
_HF_TYPE_SEGMENTS = frozenset(HF_TYPE_TO_SEGMENT.values())


# LH types that carry a revision/version segment: the default LhURI injects in
# __init__ when the URI omits one, and the segment index it sits at. The same
# artifact has two spellings depending on whether it passed through that class.
_LH_INJECTED_DEFAULT: Dict[str, Tuple[str, int]] = {
    LH_URLSEGMENT_MODELS: (DEFAULT_MODEL_REVISION, MODEL_REVISION_INDEX),
    URLSEGMENT_FILES: (DEFAULT_FILESET_VERSION, FILESET_REVISION_INDEX),
}

_GIT_SCHEMES = frozenset({"git+https", "git+git", "git+ssh"})
_GIT_CANONICAL_SCHEME = "git+https"


def normalize_uri(raw: str) -> str:
    """Return a stable identity for an artifact URI, or ``""`` when unknown.

    Args:
        raw: the URI as some producer recorded it, in any spelling.

    Returns:
        The normalized URI, or ``""`` when ``raw`` is empty, unparseable, or of a
        scheme with no identity rule.

    Returning ``""`` rather than falling back to ``raw`` is deliberate, and
    matches the contract the identity layer had before it: an endpoint that
    cannot be identified is *dropped*, and a dropped endpoint is countable at the
    call site. Passing the raw string through would instead mint a node that
    silently fails to merge with the same artifact spelled any other way -- the
    same damage as a wrong merge, only quieter. ``""`` is also the row's terminal
    marker, so the walk already stops on it.

    Total by construction: every handler call is guarded, because several raise
    on input this function must merely decline (``LhURI`` on an empty table name,
    ``HfURI`` on a short path).
    """
    if not raw:
        return ""
    candidate = raw.strip()
    if not candidate:
        return ""

    try:
        parsed = urlparse(candidate)
    except Exception:
        logger.debug("Lineage URI is unparseable, dropping endpoint: %r", raw)
        return ""

    # urlparse lowercases the scheme already; be explicit so the dispatch cannot
    # depend on that behavior.
    scheme = (parsed.scheme or "").lower()
    if not scheme:
        # A bare path is not an artifact identity. URI.get_uri would read it as
        # scheme "git" via its default_scheme and then fail to find a handler;
        # decline directly rather than inherit that surprise.
        return ""

    normalizer = _NORMALIZERS.get(scheme)
    if normalizer is None:
        logger.debug("No lineage identity rule for scheme %r in %r", scheme, raw)
        return ""

    try:
        normalized = normalizer(candidate, parsed) or ""
    except Exception:
        # Never propagate: a malformed URI on one row must not fail a whole walk
        # or a whole scan.
        logger.debug("Lineage URI failed to normalize, dropping endpoint: %r", raw)
        return ""

    if len(normalized) > MAX_LINEAGE_URI_LENGTH:
        # Drop rather than truncate. The column would truncate silently, and two
        # distinct artifacts sharing a long prefix would then collapse into one
        # node -- inventing provenance, which is the one failure worth losing a
        # node to avoid.
        logger.warning(
            "Lineage URI exceeds %d characters, dropping endpoint: %r",
            MAX_LINEAGE_URI_LENGTH,
            raw,
        )
        return ""
    return normalized


def _normalize_hf(candidate: str, parsed) -> str:
    """Canonicalize an ``hf://`` URI to its ``https://huggingface.co`` web URL.

    The web URL is the identity rather than ``hf://`` because it is the spelling
    every other system understands: a browser, a model card, another lineage
    source. ``hf://`` is granite.build's own scheme and means nothing outside it.

    The parsing is still ``HfURI``'s -- its ``custom_str`` is already an idempotent
    canonicalizer (always the host and the type segment, the revision dropped when
    it is the default and no path follows) -- and only the final spelling is
    translated, so this cannot drift from what push/pull actually resolve.

    One fixup on top: ``hf://models/owner/repo`` (two slashes) parses the type
    segment as the *host*, which ``HfURI`` only logs a warning about. Left alone
    it would canonicalize to a bogus ``hf://models/...`` endpoint that never
    merges with the correctly spelled artifact.

    A repo on a host other than ``huggingface.co`` keeps its ``hf://`` form: there
    is no rule that recognizes an arbitrary ``https`` host as a hub, so its web URL
    would not normalize back to itself.
    """
    # Import locally: gbcommon.uri.hf imports huggingface_hub at module scope,
    # and normalize_uri is called on paths that have no business requiring it.
    from gbcommon.uri.hf import HfType, HfURI

    netloc = parsed.netloc or ""
    # ``/o/m//x`` is HfURI's empty revision -- the default -- with ``x`` in the
    # repo. Read that before collapsing, which would shift ``x`` into the revision
    # slot and merge it with ``/o/m/x`` (revision ``x``).
    raw = HfURI.parse(candidate)
    default_revision_path = (
        _collapse_slashes(raw.get_path_in_repo() or "").strip("/")
        if raw.get_revision() == "" and raw.get_path_in_repo()
        else None
    )
    path = _collapse_slashes(parsed.path or "")
    if netloc.lower() in _HF_TYPE_SEGMENTS:
        # Re-spell as the three-slash form the author meant, then canonicalize.
        candidate = f"{HF_URI_SCHEME}://{HF_HOST}/{netloc}{path}"
    else:
        # Rebuilt so a doubled slash inside the path is dropped, as the web
        # spelling's segment split already does.
        candidate = f"{HF_URI_SCHEME}://{netloc}{path}"

    hf = HfURI.parse(candidate)
    if (hf.get_host() or "").lower() != HF_HOST:
        return str(hf)
    owner = (hf.get_owner() or "").lower()
    if owner in _HF_TYPE_SEGMENTS:
        # The Hub reserves these as owner names; read as one, the web URL would be
        # another type's repo (``models/spaces/m`` -> the Space ``spaces/m``).
        return ""
    if owner in HF_WEB_RESERVED_OWNERS:
        # The web spelling refuses these owners as site pages, so keeping one here
        # would give an id that does not normalize to itself.
        return ""

    # The web URL omits the type segment for a model (the one difference from the
    # ``hf://`` layout), and spells a revision or a path inside the repo as
    # /tree/<revision>/<path>. A bucket has no revision.
    hf_type = hf.get_hf_type()
    revision = hf.get_revision()
    path_in_repo = hf.get_path_in_repo()
    if default_revision_path and hf_type != HfType.BUCKET:
        revision, path_in_repo = DEFAULT_REVISION, default_revision_path
    segments = []
    if hf_type != HfType.MODEL:
        segments.append(HF_TYPE_TO_SEGMENT[hf_type])
    segments.extend([hf.get_owner(), hf.get_repo()])
    if hf_type == HfType.BUCKET:
        if path_in_repo:
            segments.append(path_in_repo)
    elif revision and (revision != DEFAULT_REVISION or path_in_repo):
        segments.extend([HF_WEB_TREE, revision])
        if path_in_repo:
            segments.append(path_in_repo)
    return f"https://{HF_HOST}/" + "/".join(segments)


def _normalize_lh(candidate: str, parsed) -> str:
    """Canonicalize an ``lh://`` URI by segment, without constructing ``LhURI``.

    Two transforms, both mirroring behavior that already exists in ``lh.py`` but
    only on paths this function cannot use:

    1. **Drop an injected default revision.** ``LhURI.__init__`` appends
       ``granite-dot-build`` to a model/fileset URI that omits its
       revision/version, so one artifact has two spellings. Stripping that
       segment converges them -- only at the revision position, so a *label*
       named ``granite-dot-build`` is kept.

       The injection is also *not deterministic*: its guard is
       ``if not revision in uristr`` -- a substring test over the whole URI -- so
       a URI whose namespace or table merely contains ``granite-dot-build``
       never gets the append at all. Normalizing by removal rather than by
       addition is what makes both spellings converge regardless.

    2. **Lowercase the host.** ``prod``/``staging`` are case-insensitive. Any
       other host is kept as written rather than folded to ``prod`` the way
       ``_get_uri_from_name`` does when *building* a URI: a stored URI on an
       unknown host may be a different lakehouse, and merging it with ``prod``
       would invent provenance.

    Doubled and trailing slashes are dropped, so ``.../tbl/`` and ``.../tbl``
    are one artifact, as they are for ``s3://``.

    ``LhURI`` is deliberately not instantiated: it would re-inject the very
    default this strips, and it raises on URIs that should simply be declined.
    """
    # Index 0 is the empty string before the leading slash, as in lh.py's indexes.
    segments = _collapse_slashes(parsed.path or "").split("/")
    if len(segments) > TYPE_INDEX:
        injected = _LH_INJECTED_DEFAULT.get(segments[TYPE_INDEX])
        # Only the segment at the revision position is the injected default. The
        # same string elsewhere (a namespace, a table, a label) is part of the
        # artifact's real name and must be preserved.
        if injected is not None:
            default, index = injected
            if len(segments) == index + 1 and segments[index] == default:
                segments = segments[:-1]

    host = (parsed.netloc or "").lower()
    path = "/".join(segments) if len(segments) > 1 else ""
    return urlunparse((LH_URI_SCHEME, host, path, "", "", ""))


def _normalize_cos(candidate: str, parsed) -> str:
    """Canonicalize ``cos://`` and ``s3://`` to one scheme.

    ``CosURI`` advertises both and treats them as interchangeable, stripping
    whichever prefix it finds -- but nothing ever rewrites one to the other, so
    the same object has two identities today. Pick ``s3://`` (the form in the
    class's own docstring example).

    The bucket is lowercased because a bucket name is case-insensitive; the key
    is left exactly as given because object keys are case-sensitive, and folding
    them would merge genuinely different objects.
    """
    bucket = (parsed.netloc or "").lower()
    if not bucket:
        return ""
    path = (parsed.path or "").rstrip("/")
    return urlunparse((S3_SCHEME, bucket, path, "", "", ""))


def _normalize_file(candidate: str, parsed) -> str:
    """Canonicalize a ``file://`` URI, preserving its trailing slash.

    A ``file://`` URI is host-local: the same string on two machines can name
    different bytes, so this is the one scheme whose identity is inherently
    weak. It is still normalized rather than dropped, because a single-host
    standalone deployment is a real and common case.

    The trailing slash is **load-bearing** and must survive: ``FileURI.pull``
    treats it as "the directory's contents" versus the directory itself, and
    ``absolutize_file_uri`` goes out of its way to restore one that
    ``normpath`` stripped. Only a relative URI is left alone -- resolving it
    would require a cwd, which would make this function impure and machine
    dependent.
    """
    path = parsed.path or ""
    if not path:
        return ""
    if not path.startswith("/"):
        # Relative: cannot be absolutized purely. Keep the spelling verbatim
        # rather than resolving against an ambient cwd. Not rebuilt through
        # urlunparse: newer patch releases (e.g. 3.11.17) turn ``file:x`` into
        # ``file:///x``, which would make it absolute.
        return candidate

    trailing = path.endswith("/") and path != "/"
    collapsed = _collapse_slashes(path)
    if trailing and not collapsed.endswith("/"):
        collapsed = f"{collapsed}/"
    return urlunparse((FILE_SCHEME, parsed.netloc or "", collapsed, "", "", ""))


def _collapse_slashes(path: str) -> str:
    """Collapse runs of ``/`` in a path, keeping a single leading slash."""
    parts = [segment for segment in path.split("/") if segment]
    return "/" + "/".join(parts)


def _normalize_git(candidate: str, parsed) -> str:
    """Canonicalize a ``git+*`` URI to one scheme, owner and repo.

    ``git+https``, ``git+git`` and ``git+ssh`` are three spellings of one
    repository, so they fold to one. ``split_repo_path`` supplies the
    owner/repo parsing, including the case-insensitive ``.git`` strip -- reused
    rather than restated so this cannot disagree with build validation about
    what repo a URI names.

    Credentials in the authority are dropped. A URI carrying a token both fails
    to merge with the same repo fetched another way *and* writes a secret into
    the lineage table, where it would survive in every graph response.

    The pip-style ``@<ref>`` and the ``subdirectory`` fragment key are kept: two
    subdirectories of a monorepo are different artifacts, and so is a checkout at
    a different ref. Folding them would merge things that are not the same. Any
    other fragment key is dropped, since it does not change what is checked out.
    """
    host = (parsed.netloc or "").rsplit("@", 1)[-1].lower()
    if not host:
        return ""
    raw_path = parsed.path or ""
    owner, repo = split_repo_path(raw_path)
    if not owner or not repo:
        return ""
    if "@" in owner:
        # Not an owner name any host allows; ``@`` belongs to the ref, after the repo.
        return ""
    path = f"/{owner}/{repo}"
    # The ref follows the repo (``repo.git@ref``): read it after the owner segment.
    after_owner = raw_path.lstrip("/").split("/", 1)[1]
    _, at, ref = after_owner.partition("@")
    ref = ref.strip("/")
    if at and ref:
        path = f"{path}@{ref}"
    subdirectory = parse_qs(parsed.fragment or "").get("subdirectory", [""])[0]
    subdirectory = _collapse_slashes(subdirectory).strip("/")
    fragment = (
        urlencode({"subdirectory": subdirectory}, safe="/") if subdirectory else ""
    )
    return urlunparse((_GIT_CANONICAL_SCHEME, host, path, "", "", fragment))


def _normalize_mem(candidate: str, parsed) -> str:
    """Return a ``mem://`` URI byte for byte.

    ``MemURI`` is an opaque key into the build's in-memory state, and its own
    ``custom_str`` exists precisely to avoid normalization: the value it carries
    may itself be a URL (``http://host:8000``), which path normalization
    corrupts. Passing it through unchanged is the rule, not an omission.
    """
    return candidate


def _normalize_https(candidate: str, parsed) -> str:
    """Canonicalize a HuggingFace web URL, the identity every HF spelling reaches.

    The same model is written both ways in practice -- a browser URL pasted into
    a build, and the ``hf://`` URI the runtime resolves -- and they must be one
    node. The web URL is translated into the ``hf://`` path layout and sent
    through :func:`_normalize_hf`, so both spellings share one canonicalizer.

    The web layout differs from ``hf://`` in one place: a revision is introduced
    by a ``tree``/``blob``/``resolve`` marker rather than following the repo
    directly. Any other page under a repo (``discussions``, ``commit/<sha>``,
    ``settings``) is declined rather than guessed: it names something about the
    repo, not an artifact, and folding it onto the repo -- or reading ``commit``
    as a revision -- would merge things that are not the same.

    Only ``huggingface.co`` is translated. Any other ``https`` host is declined:
    there is no rule that would make a generic web URL an artifact identity, and
    inventing one would merge unrelated things.
    """
    host = (parsed.netloc or "").lower()
    if host != HF_HOST:
        return ""
    segments = [segment for segment in (parsed.path or "").split("/") if segment]
    if not segments:
        return ""

    type_segments = []
    if segments[0] in _HF_TYPE_SEGMENTS:
        type_segments, segments = [segments[0]], segments[1:]
    if len(segments) < 2 or segments[0] in HF_WEB_RESERVED_OWNERS:
        return ""
    repo_segments, rest = segments[:2], segments[2:]

    if rest and type_segments != [URLSEGMENT_BUCKETS]:
        # A bucket has no revision, so what follows it is a path taken as is
        # (``tree`` included); for every other type an unmarked segment is a
        # page, not a revision.
        if rest[0] not in HF_WEB_REVISION_MARKERS:
            return ""
        rest = rest[1:]
        if not rest:
            return ""

    path = "/".join(type_segments + repo_segments + rest)
    translated = f"{HF_URI_SCHEME}://{HF_HOST}/{path}"
    return _normalize_hf(translated, urlparse(translated))


# Scheme -> normalizer. A scheme absent from this table has no identity rule and
# is dropped; that is the conservative default, and adding a scheme should be a
# deliberate act with a test rather than a fallback that guesses.
_NORMALIZERS = {
    HF_URI_SCHEME: _normalize_hf,
    LH_URI_SCHEME: _normalize_lh,
    COS_SCHEME: _normalize_cos,
    S3_SCHEME: _normalize_cos,
    FILE_SCHEME: _normalize_file,
    MEM_URI_SCHEME: _normalize_mem,
    "https": _normalize_https,
    **{scheme: _normalize_git for scheme in _GIT_SCHEMES},
}
