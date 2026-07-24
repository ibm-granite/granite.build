# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from services.datasets.tus_rendezvous import UploadRendezvous


async def test_incomplete_until_all_expected_present():
    rv = UploadRendezvous()
    done = await rv.record(
        "d1", role="train", expects=["train", "validation"], path="/t"
    )
    assert done is False  # validation still missing
    done = await rv.record(
        "d1", role="validation", expects=["train", "validation"], path="/v"
    )
    assert done is True


async def test_autosplit_single_file_completes_immediately():
    rv = UploadRendezvous()
    done = await rv.record("d1", role="source", expects=["source"], path="/s")
    assert done is True


async def test_paths_returned_for_group():
    rv = UploadRendezvous()
    await rv.record("d1", role="train", expects=["train", "validation"], path="/t")
    await rv.record("d1", role="validation", expects=["train", "validation"], path="/v")
    paths = rv.paths("d1")
    assert paths == {"train": "/t", "validation": "/v"}


async def test_claim_finalize_is_once_only():
    rv = UploadRendezvous()
    await rv.record("d1", role="source", expects=["source"], path="/s")
    first = await rv.claim_finalize("d1")
    second = await rv.claim_finalize("d1")
    assert first is True and second is False


async def test_release_finalize_allows_reclaim():
    # release_finalize (called after a FAILED finalize) drops the claim so a
    # retried completion can re-claim. A subsequent successful claim then stays
    # once-only.
    rv = UploadRendezvous()
    await rv.record("d1", role="source", expects=["source"], path="/s")
    assert await rv.claim_finalize("d1") is True
    await rv.release_finalize("d1")
    # Retry can re-claim now that the failed claim was released.
    assert await rv.claim_finalize("d1") is True
    # ...and is once-only again thereafter.
    assert await rv.claim_finalize("d1") is False


async def test_clear_drops_file_state_but_finalize_stays_once_only():
    # clear() releases the per-group file/lock state, but the finalize claim must
    # PERSIST so a retried completion of an already-finalized group (duplicate
    # PATCH / proxy retry) cannot finalize a second time.
    rv = UploadRendezvous()
    await rv.record("d1", role="source", expects=["source"], path="/s")
    assert await rv.claim_finalize("d1") is True
    rv.clear("d1")
    # A duplicate completion after clear: records again, but claim stays denied.
    await rv.record("d1", role="source", expects=["source"], path="/s2")
    assert await rv.claim_finalize("d1") is False
