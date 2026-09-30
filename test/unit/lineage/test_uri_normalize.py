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

"""Tests for URI normalization, the lineage index's node identity.

Two properties matter, and they are not symmetric. Spellings of one artifact must
converge, or the graph splits into disconnected pieces -- visible and fixable.
Spellings of *different* artifacts must never converge, because that invents
provenance: someone reads that a model was trained on data it never saw. The
negative tests here are therefore as load-bearing as the positive ones.
"""

import pytest

from gbserver.lineage.uri_normalize import normalize_uri, normalized_or_none
from gbserver.storage.stored_lineage_row import MAX_LINEAGE_URI_LENGTH

# Groups of raw spellings that must each collapse to one identity. Every member
# of a group has to normalize to the same string as every other member.
CONVERGING_GROUPS = [
    pytest.param(
        [
            "hf://huggingface.co/models/ibm-research/modelX",
            "hf:///ibm-research/modelX",
            "hf:///models/ibm-research/modelX",
            "hf://huggingface.co/ibm-research/modelX",
            "hf://huggingface.co/models/ibm-research/modelX/main",
            "hf://huggingface.co/models/ibm-research/modelX/",
            "https://huggingface.co/ibm-research/modelX",
            "https://huggingface.co/models/ibm-research/modelX",
            "https://huggingface.co/ibm-research/modelX/tree/main",
            "https://huggingface.co/ibm-research/modelX/",
            # The missing-slash typo: HfURI parses "models" as the host and only
            # warns, which would otherwise mint a bogus separate node.
            "hf://models/ibm-research/modelX",
            "  hf://huggingface.co/models/ibm-research/modelX  ",
        ],
        id="hf-model",
    ),
    pytest.param(
        [
            "hf://huggingface.co/datasets/org/ds",
            "hf:///datasets/org/ds",
            "hf://datasets/org/ds",
            "https://huggingface.co/datasets/org/ds",
        ],
        id="hf-dataset",
    ),
    pytest.param(
        [
            # tree / blob / resolve all name the same revision and path.
            "hf://huggingface.co/models/org/repo/main/config.json",
            "https://huggingface.co/org/repo/tree/main/config.json",
            "https://huggingface.co/org/repo/blob/main/config.json",
            "https://huggingface.co/org/repo/resolve/main/config.json",
        ],
        id="hf-path-in-repo",
    ),
    pytest.param(
        [
            # LhURI injects this default when a model URI omits its revision, so
            # one artifact has two spellings depending on whether it was built
            # through that class.
            "lh://prod/ns/models/tbl/mylabel",
            "lh://prod/ns/models/tbl/mylabel/granite-dot-build",
            "lh://PROD/ns/models/tbl/mylabel",
            # Legacy hostnames fold to prod, as the URI builders already do.
            "lh://some-lakehouse.ibm.com/ns/models/tbl/mylabel",
        ],
        id="lh-model-default-revision",
    ),
    pytest.param(
        [
            "lh://prod/ns/filesets/tbl/lbl",
            "lh://prod/ns/filesets/tbl/lbl/granite-dot-build",
        ],
        id="lh-fileset-default-version",
    ),
    pytest.param(
        [
            "lh://staging/ns/tables/mytable",
            "lh://STAGING/ns/tables/mytable",
            "lh://my-lakehouse-staging.example.com/ns/tables/mytable",
        ],
        id="lh-staging-host",
    ),
    pytest.param(
        # One class advertises both schemes and treats them as interchangeable,
        # but nothing rewrote one to the other before this.
        ["s3://bucket/some/key", "cos://bucket/some/key", "s3://BUCKET/some/key"],
        id="cos-s3-alias",
    ),
    pytest.param(
        [
            "git+https://host.com/org/repo",
            "git+ssh://host.com/org/repo.git",
            "git+git://host.com/org/repo.GIT",
            "git+ssh://Host.com/org/repo.git",
            # A ref and a subdirectory scope a checkout, not the artifact's
            # origin; keeping them would split one repo into a node per branch.
            "git+ssh://host.com/org/repo.git@some-branch",
            "git+https://host.com/org/repo.git#subdirectory=steps/x",
            # A credential must never reach the identity, let alone the table.
            "git+https://user:token@host.com/org/repo.git",
        ],
        id="git-schemes",
    ),
    pytest.param(
        ["file:///tmp/out/model", "file:///tmp//out///model"],
        id="file-slash-collapse",
    ),
]


@pytest.mark.parametrize("spellings", CONVERGING_GROUPS)
def test_spellings_of_one_artifact_converge(spellings):
    """Every spelling in a group yields one identity."""
    normalized = {normalize_uri(raw) for raw in spellings}
    assert len(normalized) == 1, f"expected one identity, got {sorted(normalized)}"
    assert normalized.pop(), "an identity must not be empty"


@pytest.mark.parametrize("spellings", CONVERGING_GROUPS)
def test_normalization_is_idempotent(spellings):
    """Normalizing an already-normalized URI changes nothing.

    Required because the walk matches stored values against freshly normalized
    request input: if the function were not idempotent, a stored row would stop
    matching the artifact it describes.
    """
    for raw in spellings:
        once = normalize_uri(raw)
        assert normalize_uri(once) == once


@pytest.mark.parametrize("spellings", CONVERGING_GROUPS)
def test_normalized_form_is_itself_a_member(spellings):
    """The canonical form is a real URI of the same scheme, not a synthetic id."""
    normalized = normalize_uri(spellings[0])
    assert "://" in normalized


# Pairs that must NOT converge. Each is a case where an over-eager rule would
# merge two genuinely different artifacts.
@pytest.mark.parametrize(
    "left,right,why",
    [
        (
            "hf://huggingface.co/models/org/ModelX",
            "hf://huggingface.co/models/org/modelx",
            "HF repo names are case-sensitive",
        ),
        (
            "hf://huggingface.co/models/org/repo",
            "hf://huggingface.co/datasets/org/repo",
            "a model and a dataset can share a name",
        ),
        (
            "hf://huggingface.co/models/org/repo",
            "hf://huggingface.co/models/org/repo/v2",
            "a non-default revision is part of the identity",
        ),
        (
            "s3://bucket/Some/Key",
            "s3://bucket/some/key",
            "S3 object keys are case-sensitive",
        ),
        (
            "lh://prod/ns/models/tbl/lbl",
            "lh://staging/ns/models/tbl/lbl",
            "prod and staging are different lakehouses",
        ),
        (
            "lh://prod/ns/models/tbl/lbl",
            "lh://prod/other/models/tbl/lbl",
            "the namespace distinguishes artifacts",
        ),
        (
            "lh://prod/ns/tables/mytable",
            "lh://prod/ns/datasets/mytable",
            "type is part of the lh path, so a table is not a dataset",
        ),
        (
            "lh://prod/ns/models/tbl/lbl/v2",
            "lh://prod/ns/models/tbl/lbl",
            "a real revision must not be stripped like the injected default",
        ),
        (
            "file:///tmp/out/model/",
            "file:///tmp/out/model",
            "a trailing slash means the directory's contents, not the directory",
        ),
        (
            "git+https://host.com/org/repoA",
            "git+https://host.com/org/repoB",
            "different repositories",
        ),
    ],
)
def test_distinct_artifacts_do_not_converge(left, right, why):
    """Two different artifacts keep two identities."""
    assert normalize_uri(left) != normalize_uri(right), why
    assert normalize_uri(left) and normalize_uri(right), "both must be identifiable"


def test_lh_namespace_containing_the_default_revision_is_preserved():
    """Only a *trailing* injected default is stripped.

    ``LhURI``'s injection guard is ``if not revision in uristr`` -- a substring
    test over the whole URI -- so a URI whose namespace merely contains
    ``granite-dot-build`` never receives the append and must not be mistaken for
    one that did. Stripping by substring instead of by trailing segment would
    corrupt this URI's namespace.
    """
    raw = "lh://prod/granite-dot-build.public/models/tbl/lbl"
    assert normalize_uri(raw) == raw


def test_lh_label_named_like_the_default_keeps_its_real_revision():
    """A label legitimately named ``granite-dot-build`` is not the injected one."""
    raw = "lh://prod/ns/models/tbl/granite-dot-build/v2"
    assert normalize_uri(raw) == raw


def test_mem_uri_is_byte_exact():
    """``mem://`` is an opaque key and must not be touched.

    Its value may itself be a URL, which path normalization corrupts -- the
    reason ``MemURI.custom_str`` exists.
    """
    raw = "mem://http://host:8000/some//path"
    assert normalize_uri(raw) == raw


def test_relative_file_uri_is_not_resolved():
    """A relative ``file:`` URI is kept verbatim, never resolved against a cwd.

    Resolving would make the function impure and its output machine-dependent.
    """
    assert normalize_uri("file:outputs/model") == "file:outputs/model"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not a uri",
        "/plain/absolute/path",
        "relative/path",
        "bogus-scheme://host/path",
        "https://example.com/some/model",  # not huggingface.co
        "hf://huggingface.co/models/onlyowner",  # too few segments
        "https://huggingface.co/org/repo/discussions/3",  # a page, not an artifact
        "https://huggingface.co/org/repo/commit/abc123",  # not a revision marker
        "https://huggingface.co/org/repo/tree",  # marker with no revision
        "s3://",  # no bucket
        "git+ssh://host.com/onlyowner",  # no repo
    ],
)
def test_unidentifiable_input_returns_empty(raw):
    """Anything without an identity rule is dropped, never guessed.

    Empty is the honest answer and is also the row's terminal marker, so the walk
    already stops on it. Falling back to the raw string would mint a node that
    silently fails to merge with the same artifact spelled any other way.
    """
    assert normalize_uri(raw) == ""


def test_normalize_uri_never_raises():
    """Totality: a malformed URI on one row must not fail a walk or a scan."""
    hostile = [
        None,
        "hf://",
        "lh://",
        "lh://prod",
        "lh://prod/ns/models",
        "file://",
        "cos://",
        "://",
        "hf://huggingface.co/models/" + "x" * 5000,
        "lh://prod/ns/models/tbl/lbl/" + "v" * 5000,
        "\x00\x01",
        "hf://huggingface.co/models/org/repo?q=1#frag",
    ]
    for raw in hostile:
        try:
            result = normalize_uri(raw)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - the assertion is the point
            pytest.fail(f"normalize_uri raised on {raw!r}: {exc!r}")
        assert isinstance(result, str)


def test_normalized_or_none_distinguishes_unidentifiable_from_empty():
    """The optional wrapper lets a caller count dropped endpoints."""
    assert normalized_or_none("") is None
    assert normalized_or_none("bogus://x") is None
    assert normalized_or_none("hf:///org/repo") == "https://huggingface.co/org/repo"


def test_hf_path_in_repo_keeps_its_revision():
    """A path inside a repo keeps the default revision, as ``custom_str`` does.

    Dropping it would make ``.../repo/main/config.json`` and
    ``.../repo/config.json`` two spellings whose round trip differs.
    """
    raw = "hf://huggingface.co/models/org/repo/main/config.json"
    assert normalize_uri(raw) == "https://huggingface.co/org/repo/tree/main/config.json"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("hf:///org/repo", "https://huggingface.co/org/repo"),
        ("hf:///datasets/org/ds", "https://huggingface.co/datasets/org/ds"),
        ("hf:///spaces/org/app", "https://huggingface.co/spaces/org/app"),
        ("hf:///org/repo/v2", "https://huggingface.co/org/repo/tree/v2"),
        (
            "hf:///buckets/org/b/some/key",
            "https://huggingface.co/buckets/org/b/some/key",
        ),
    ],
)
def test_hf_identity_is_the_web_url(raw, expected):
    """The HF identity is the ``https://huggingface.co`` URL, not ``hf://``.

    The web URL is the spelling other systems understand; ``hf://`` is
    granite.build's own scheme.
    """
    assert normalize_uri(raw) == expected


def test_hf_on_another_host_keeps_its_hf_form():
    """A private hub stays ``hf://``: no rule reads its web URL back as HF.

    Emitting ``https://<host>/...`` would make the identity fail to normalize
    to itself, so a walk seeded with it would find nothing.
    """
    raw = "hf://hub.example.com/models/org/repo"
    assert normalize_uri(raw) == raw
    assert normalize_uri(normalize_uri(raw)) == raw


def test_identity_fits_the_column_width():
    """A normalized URI stays within the 1024-char URI column.

    The row's ``source``/``target`` columns are ``String(1024)``. A silently
    truncated URI is the worst possible failure here: two different artifacts
    sharing a prefix would collapse into one node.
    """
    long_repo = "x" * 400
    raw = f"hf://huggingface.co/models/org/{long_repo}"
    assert len(normalize_uri(raw)) <= 1024


def test_over_long_uri_is_dropped_not_truncated():
    """A URI that would not fit the column is dropped.

    Truncation is the dangerous option: the DB would do it silently, and two
    artifacts sharing a long prefix would then collapse into one node. Losing the
    node is the lesser harm, and it is logged.
    """
    raw = "https://huggingface.co/org/" + "x" * MAX_LINEAGE_URI_LENGTH
    assert normalize_uri(raw) == ""


def test_uri_at_exactly_the_limit_is_kept():
    """The guard is a ceiling, not an off-by-one rejection."""
    prefix = "https://huggingface.co/org/"
    raw = prefix + "x" * (MAX_LINEAGE_URI_LENGTH - len(prefix))
    normalized = normalize_uri(raw)
    assert len(normalized) == MAX_LINEAGE_URI_LENGTH
    assert normalized == raw
