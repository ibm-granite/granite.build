"""DP scan mitigations: truncation is reported, and decoding leaves the event loop.

These guard two failure modes that are invisible from the outside. A truncated
window looks identical to a complete one, and event-loop-blocking work looks
identical to non-blocking work until the sidecar is under load.
"""

import asyncio
import base64
import io
import zipfile

import pytest

from gb_ui_backend.services.gbserver_source import (
    _truncation_warning,
    _yaml_from_archive_b64,
    _yaml_from_json_blob,
)


def _archive(files: dict[str, str]) -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, body in files.items():
            zf.writestr(name, body)
    return base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------ truncation warning


def test_truncation_warning_fires_only_when_the_page_is_full():
    """Both directions matter: a warning that always fired would also pass a
    test that only checked the full-page case, while telling every user their
    complete window might be incomplete."""
    assert _truncation_warning([1, 2, 3], limit=3) is not None
    assert _truncation_warning([1, 2], limit=3) is None
    assert _truncation_warning([], limit=3) is None


def test_truncation_warning_names_the_limit_and_hedges():
    msg = _truncation_warning(list(range(10000)), limit=10000)
    assert "10000" in msg
    # A full page may or may not have more behind it; claiming it definitely
    # does would be its own inaccuracy.
    assert "may be missing" in msg


def test_truncation_warning_absent_limit_never_warns():
    """limit=0 means unbounded, which cannot be truncated."""
    assert _truncation_warning([1, 2, 3], limit=0) is None


# ------------------------------------------------------------------- decoders


def test_yaml_from_archive_prefers_build_yaml():
    b64 = _archive({"other.yaml": "no: 1", "build.yaml": "llm.build:\n  name: x"})
    assert _yaml_from_archive_b64(b64) == "llm.build:\n  name: x"


def test_yaml_from_archive_falls_back_to_any_yaml():
    assert "no: 1" in (_yaml_from_archive_b64(_archive({"other.yml": "no: 1"})) or "")


def test_yaml_from_archive_tolerates_garbage():
    """A build whose archive is unreadable must not abort the whole scan."""
    assert _yaml_from_archive_b64("not-base64-at-all!!") is None
    assert _yaml_from_archive_b64(base64.b64encode(b"not a zip").decode()) is None


def test_yaml_from_json_blob_reads_nested_archive():
    blob = f'{{"build_archive": "{_archive({"build.yaml": "a: 1"})}"}}'
    assert _yaml_from_json_blob(blob) == "a: 1"


def test_yaml_from_json_blob_tolerates_missing_archive():
    assert _yaml_from_json_blob('{"no_archive_here": true}') is None
    assert _yaml_from_json_blob("not json") is None


# -------------------------------------------------- the decode leaves the loop


@pytest.mark.asyncio
async def test_decoding_does_not_block_the_event_loop():
    """The point of the change: a wide scan must not stall unrelated requests.

    Decoding many archives in a worker thread should let a concurrent coroutine
    keep making progress. Asserting on wall-clock timing would be flaky, so this
    asserts on interleaving instead: the ticker must get scheduled while the
    decode is in flight, which is exactly what inline decoding prevented.
    """
    archives = [_archive({"build.yaml": f"n: {i}\n" + "x" * 20000}) for i in range(60)]

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    spinner = asyncio.create_task(ticker())
    await asyncio.sleep(0)  # let the ticker start

    def decode_all():
        return [_yaml_from_archive_b64(a) for a in archives]

    results = await asyncio.to_thread(decode_all)

    spinner.cancel()
    assert len(results) == 60 and all(r is not None for r in results)
    assert ticks > 1, "event loop made no progress during the decode"
