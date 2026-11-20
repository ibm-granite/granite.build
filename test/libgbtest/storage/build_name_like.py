# SPDX-License-Identifier: Apache-2.0

"""Shared checks for ``Like`` where-values on a build table, run against each SQL
backend. Patterns come from ``*`` wildcards via ``parse_name_filter``, as the
``/builds/`` ``name`` filter builds them, so the escaping is checked against each
dialect: matches are case-insensitive, ``%``/``_``/``\\`` are literal, and
``count`` agrees with ``get_by_where`` so paginated totals stay right."""

import pytest

from gbserver.api.utils import parse_name_filter
from gbserver.storage.storage import Like
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status

NAMES = [
    "granite-SFT-run",
    "sft_lora",
    "eval-100%",
    "evalX100",
    "back\\slash",
    "star*name",
    "other",
]

# (name wildcard, expected names)
WILDCARD_CASES = [
    ("*sft*", ["granite-SFT-run", "sft_lora"]),
    ("*SFT*", ["granite-SFT-run", "sft_lora"]),
    ("sft*", ["sft_lora"]),
    ("*run", ["granite-SFT-run"]),
    ("*100%*", ["eval-100%"]),
    ("*t_l*", ["sft_lora"]),
    ("*k\\s*", ["back\\slash"]),
    ("*k\\\\s*", ["back\\slash"]),
    ("*r\\*n*", ["star*name"]),
    ("*nomatch*", []),
]


def populate(storage) -> None:
    for name in NAMES:
        storage.add(
            StoredBuild.create(
                name=name,
                space_name="s",
                source_uri="",
                username="u",
                build_archive="dGVzdA==",
                status=Status.SUCCESS,
                targets=[],
                tags=[],
            )
        )


def names(storage, where) -> list[str]:
    return sorted(b.name for b in storage.get_by_where(where))


def check_wildcard(storage, wildcard, expected) -> None:
    like = parse_name_filter(wildcard)
    assert isinstance(like, Like), wildcard
    where = {"name": like}
    assert names(storage, where) == expected
    assert storage.count(where) == len(expected)


def check_combines_with_exact_filters(storage) -> None:
    assert names(storage, {"name": Like("%sft%"), "username": "u"}) == [
        "granite-SFT-run",
        "sft_lora",
    ]
    assert names(storage, {"name": Like("%sft%"), "username": "nobody"}) == []


def check_plain_string_is_still_exact(storage) -> None:
    assert names(storage, {"name": "sft"}) == []
    assert names(storage, {"name": "sft_lora"}) == ["sft_lora"]
    assert names(storage, {"name": "star*name"}) == ["star*name"]
    # The reviewer's case: an escaped star is an exact match on the literal name.
    assert names(storage, {"name": parse_name_filter("star\\*name")}) == ["star*name"]


def check_non_string_column_fails_fast(storage) -> None:
    # ValueError is the one error the read retry doesn't retry, so this is
    # immediate rather than ten backed-off attempts.
    with pytest.raises(ValueError, match="string column"):
        storage.get_by_where({"created_time": Like("%2026%")})
    with pytest.raises(ValueError, match="string column"):
        storage.count({"created_time": Like("%2026%")})
