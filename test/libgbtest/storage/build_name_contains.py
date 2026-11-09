"""Shared checks for ``Contains`` where-values on a build table, run against
each SQL backend: case-insensitive substring match on ``name``, ``%``/``_``
taken literally, and ``count`` agreeing with ``get_by_where`` so paginated
totals stay right."""

from gbserver.storage.storage import Contains
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status

NAMES = ["granite-SFT-run", "sft_lora", "eval-100%", "evalX100", "back\\slash", "other"]

# (term, expected names)
CONTAINS_CASES = [
    ("sft", ["granite-SFT-run", "sft_lora"]),
    ("SFT", ["granite-SFT-run", "sft_lora"]),
    ("100%", ["eval-100%"]),
    ("t_l", ["sft_lora"]),
    ("k\\s", ["back\\slash"]),
    ("nomatch", []),
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


def check_contains(storage, term, expected) -> None:
    where = {"name": Contains(term)}
    assert names(storage, where) == expected
    assert storage.count(where) == len(expected)


def check_combines_with_exact_filters(storage) -> None:
    assert names(storage, {"name": Contains("sft"), "username": "u"}) == [
        "granite-SFT-run",
        "sft_lora",
    ]
    assert names(storage, {"name": Contains("sft"), "username": "nobody"}) == []


def check_plain_string_is_still_exact(storage) -> None:
    assert names(storage, {"name": "sft"}) == []
    assert names(storage, {"name": "sft_lora"}) == ["sft_lora"]
