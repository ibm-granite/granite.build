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

from typing import Dict
from urllib.parse import urlparse, urlunparse

from gbcommon.uri.cos import COS_SCHEME, S3_SCHEME
from gbcommon.uri.git import split_repo_path
from gbcommon.uri.hf import (
    DEFAULT_REVISION,
    HF_HOST,
    HF_URI_SCHEME,
    URLSEGMENT_BUCKETS,
    URLSEGMENT_DATASETS,
    URLSEGMENT_MODELS,
    URLSEGMENT_SPACES,
)
from gbcommon.uri.lh import (
    DEFAULT_FILESET_VERSION,
    DEFAULT_MODEL_REVISION,
    LH_URI_SCHEME,
    PRODUCTION_HOST,
    STAGING_HOST,
    URLSEGMENT_FILES,
)
from gbcommon.uri.lh import URLSEGMENT_MODELS as LH_URLSEGMENT_MODELS
from gbserver.storage.stored_lineage_row import MAX_LINEAGE_URI_LENGTH
from gbserver.types.constants import FILE_SCHEME, MEM_URI_SCHEME
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# The HF path segments that name a repo type. A URI whose *host* is one of these
# is the missing-slash typo `hf://models/o/r`, which HfURI only warns about.
_HF_TYPE_SEGMENTS = frozenset(
    {
        URLSEGMENT_MODELS,
        URLSEGMENT_DATASETS,
        URLSEGMENT_SPACES,
        URLSEGMENT_BUCKETS,
    }
)

# How the web URL spells each non-model repo type. A model has no segment, which
# is the one difference from the ``hf://`` layout. Keyed by ``HfType`` value.
_HF_WEB_TYPE_SEGMENT = {
    "dataset": URLSEGMENT_DATASETS,
    "space": URLSEGMENT_SPACES,
    "bucket": URLSEGMENT_BUCKETS,
}

# The marker the canonical web URL puts before a revision, and every marker a web
# URL may use there: ``tree`` for a directory, ``blob`` for a file page, ``resolve``
# for a raw download. All three name the same revision and path.
_HF_WEB_TREE = "tree"
_HF_WEB_REVISION_MARKERS = frozenset({"tree", "blob", "resolve"})

# LH types that carry a revision/version segment, and the default injected for
# each. LhURI appends these in __init__ when the URI omits one, so the same
# artifact has two spellings depending on whether it passed through that class.
_LH_INJECTED_DEFAULT: Dict[str, str] = {
    LH_URLSEGMENT_MODELS: DEFAULT_MODEL_REVISION,
    URLSEGMENT_FILES: DEFAULT_FILESET_VERSION,
}

# Index of the type segment in an lh:// path split on "/". Index 0 is the empty
# string before the leading slash.
_LH_TYPE_INDEX = 2

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

    if (parsed.netloc or "").lower() in _HF_TYPE_SEGMENTS:
        # Re-spell as the three-slash form the author meant, then canonicalize.
        path = parsed.path or ""
        candidate = f"{HF_URI_SCHEME}://{HF_HOST}/{parsed.netloc}{path}"

    hf = HfURI.parse(candidate)
    canonical = str(hf)
    parts = hf._parts()
    if parts.host != HF_HOST:
        return canonical

    # The web URL omits the type segment for a model, and spells a revision or a
    # path inside the repo as /tree/<revision>/<path>. A bucket has no revision.
    segments = []
    if parts.hf_type != HfType.MODEL:
        segments.append(_HF_WEB_TYPE_SEGMENT[str(parts.hf_type)])
    segments.extend([parts.owner, parts.repo])
    if parts.hf_type == HfType.BUCKET:
        if parts.path_in_repo:
            segments.append(parts.path_in_repo)
    elif parts.revision and (parts.revision != DEFAULT_REVISION or parts.path_in_repo):
        segments.extend([_HF_WEB_TREE, parts.revision])
        if parts.path_in_repo:
            segments.append(parts.path_in_repo)
    return f"https://{HF_HOST}/" + "/".join(segments)


def _normalize_lh(candidate: str, parsed) -> str:
    """Canonicalize an ``lh://`` URI by segment, without constructing ``LhURI``.

    Two transforms, both mirroring behavior that already exists in ``lh.py`` but
    only on paths this function cannot use:

    1. **Drop an injected default revision.** ``LhURI.__init__`` appends
       ``granite-dot-build`` to a model/fileset URI that omits its
       revision/version, so one artifact has two spellings. Stripping that
       trailing segment converges them.

       The injection is also *not deterministic*: its guard is
       ``if not revision in uristr`` -- a substring test over the whole URI -- so
       a URI whose namespace or table merely contains ``granite-dot-build``
       never gets the append at all. Normalizing by removal rather than by
       addition is what makes both spellings converge regardless.

    2. **Fold the host.** ``_get_uri_from_name`` lowercases the environment and
       collapses any legacy hostname to ``staging``/``prod``, but only when
       *building* a URI -- a hand-written or stored one is never folded. Apply
       the same fold here so both reach one identity.

    ``LhURI`` is deliberately not instantiated: it would re-inject the very
    default this strips, and it raises on URIs that should simply be declined.
    """
    segments = (parsed.path or "").split("/")
    if len(segments) > _LH_TYPE_INDEX:
        lh_type = segments[_LH_TYPE_INDEX]
        injected = _LH_INJECTED_DEFAULT.get(lh_type)
        # Only a *trailing* segment is the injected default. The same string
        # elsewhere in the path (a namespace, a table) is part of the artifact's
        # real name and must be preserved.
        if injected is not None and segments[-1] == injected:
            segments = segments[:-1]

    host = _fold_lh_host(parsed.netloc or "")
    path = "/".join(segments)
    return urlunparse((LH_URI_SCHEME, host, path, "", "", ""))


def _fold_lh_host(netloc: str) -> str:
    """Fold a lakehouse host to ``staging``/``prod``, as the builders do.

    Mirrors ``LhURI._get_uri_from_name``: lowercase, then map any host that is
    not already one of the two identifiers by looking for ``staging`` in it.
    """
    host = netloc.lower()
    if not host or host in (STAGING_HOST, PRODUCTION_HOST):
        return host
    return STAGING_HOST if "staging" in host else PRODUCTION_HOST


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

    The ``@<ref>`` and ``#subdirectory=`` parts are intentionally not carried:
    an artifact's identity is the repository it came from, and keeping a ref
    would split one repo into a node per branch.
    """
    host = (parsed.netloc or "").rsplit("@", 1)[-1].lower()
    if not host:
        return ""
    owner, repo = split_repo_path(parsed.path or "")
    if not owner or not repo:
        return ""
    return urlunparse((_GIT_CANONICAL_SCHEME, host, f"/{owner}/{repo}", "", "", ""))


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
    if len(segments) < 2:
        return ""
    repo_segments, rest = segments[:2], segments[2:]

    if rest:
        if rest[0] in _HF_WEB_REVISION_MARKERS:
            rest = rest[1:]
            if not rest:
                return ""
        elif type_segments != [URLSEGMENT_BUCKETS]:
            # A bucket has no revision, so what follows it is a path; for every
            # other type an unmarked segment is a page, not a revision.
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
