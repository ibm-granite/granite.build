"""GbserverSource keeps its public methods on the class.

This exists because of a real regression that every other test missed. Adding
module-level helper functions in the middle of the class body silently ended the
class: the methods after the insertion point were re-parented as nested functions
inside a helper, so `GbserverSource` lost six public methods.

Nothing caught it. The file still parsed, `black` still formatted it, and 172
tests still passed — because those tests replace the source with an `AsyncMock`
and never touch the real class. It surfaced only as
`'GbserverSource' object has no attribute 'list_builds_for_dp_scan'` at runtime,
against a live database.

So this asserts the shape of the class rather than any behaviour. It is cheap and
it is the only thing standing between a mis-indented edit and a broken analytics
page.
"""

import inspect

from gb_ui_backend.services.gbserver_source import GbserverSource

# Every method a caller outside this module relies on. Adding to this list is
# expected; a name silently disappearing from the class is the bug.
REQUIRED_METHODS = (
    "list_builds",
    "list_builds_for_dp_scan",
    "get_build",
    "get_status_chart",
    "get_failed_builds",
    "get_build_events",
    "get_leaderboard",
    "close",
)


def test_public_methods_are_present_on_the_class():
    missing = [m for m in REQUIRED_METHODS if not hasattr(GbserverSource, m)]
    assert not missing, (
        f"GbserverSource is missing {missing}. A module-level `def` inserted inside "
        "the class body ends the class and re-parents everything after it — check "
        "indentation around recent edits."
    )


def test_required_methods_are_coroutines():
    """These are all awaited by callers; a plain function would fail at runtime."""
    for name in REQUIRED_METHODS:
        if name == "close":
            continue
        fn = getattr(GbserverSource, name)
        assert inspect.iscoroutinefunction(fn), f"{name} is not async"


def test_module_helpers_did_not_land_inside_the_class():
    """The helpers belong at module scope, not as attributes of the class.

    The inverse of the regression: if a helper is reachable via the class, it was
    defined inside the class body and the split happened somewhere unexpected.
    """
    for helper in (
        "_truncation_warning",
        "_yaml_from_archive_b64",
        "_yaml_from_json_blob",
    ):
        assert not hasattr(
            GbserverSource, helper
        ), f"{helper} is an attribute of GbserverSource; it should be module-level"
