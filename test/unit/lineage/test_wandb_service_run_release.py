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

"""Unit tests for WandBLineageService run release on failure paths.

``self._runs`` holds the open wandb runs for this process, and the watcher is a
long-lived daemon. A failure that leaves a run in there without finishing it leaks
that run and its background sync thread for the life of the process; a partial run
left on the module-global ``wandb.run`` would additionally be picked up by the next
``wandb.init`` in the same process. These tests assert both leak paths release the
run: a failure inside ``wandb.init()``, and a failure after the run was opened.

Note this is *not* about reusing an id. Run ids are random uuids now, so a retry
never presents the same id twice — the release still matters, for the leak.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbserver.lineage.wandb_service import WandBLineageService


def _service() -> WandBLineageService:
    """Construct the service without running __init__ (no wandb.login)."""
    service = WandBLineageService.__new__(WandBLineageService)
    service._runs = {}
    return service


def _event(event_type: str = "START") -> dict:
    return {
        "run": {"runId": "run-1", "facets": {}},
        "job": {"name": "job-1", "facets": {}, "namespace": "ns"},
        "eventType": event_type,
        "inputs": [],
        "outputs": [],
    }


def _make_run(run_id: str = "run-1") -> MagicMock:
    run = MagicMock()
    run.settings = SimpleNamespace(mode="online")
    run.tags = []
    run.config = MagicMock()
    run.summary = {}
    # A real run carries its id, and the init-failure path matches on it to tell
    # *this* call's partial run from an unrelated one parked on wandb.run.
    run.id = run_id
    return run


def test_init_failure_releases_partially_created_run():
    """A run that init() created before raising is finished, not left in use.

    This is the observed production loop: attempt 1 fails with CommError
    ("previously created and deleted"), leaving the run registered; attempt 2
    then fails with "run ID ... is in use" forever.
    """
    service = _service()
    leaked = _make_run()

    with (
        patch.object(service, "_init_run", side_effect=RuntimeError("boom")),
        patch("gbserver.lineage.wandb_service.wandb") as wandb_mod,
    ):
        wandb_mod.run = leaked
        with pytest.raises(RuntimeError):
            service._get_run("run-1", "job-1")

    leaked.finish.assert_called_once()
    # Nothing cached, so the retry does a fresh init rather than reusing a
    # dead handle.
    assert service._runs == {}


def test_init_failure_with_no_partial_run_is_a_clean_reraise():
    """When init() left nothing behind there is nothing to finish."""
    service = _service()

    with (
        patch.object(service, "_init_run", side_effect=RuntimeError("boom")),
        patch("gbserver.lineage.wandb_service.wandb") as wandb_mod,
    ):
        wandb_mod.run = None
        with pytest.raises(RuntimeError):
            service._get_run("run-1", "job-1")

    assert service._runs == {}


def test_init_failure_does_not_finish_an_unrelated_global_run():
    """A run on wandb.run that is not ours must be left alone.

    ``wandb.run`` is a module global. The cleanup used to finish whatever sat
    there with ``exit_code=1``, so a failed init for one id would corrupt an
    unrelated run that happened to be open — marking a healthy run failed. Only
    the id we tried to open may be released.
    """
    service = _service()
    stranger = _make_run("someone-elses-run")

    with (
        patch.object(service, "_init_run", side_effect=RuntimeError("boom")),
        patch("gbserver.lineage.wandb_service.wandb") as wandb_mod,
    ):
        wandb_mod.run = stranger
        with pytest.raises(RuntimeError):
            service._get_run("run-1", "job-1")

    stranger.finish.assert_not_called()
    assert service._runs == {}


def _hide_id(run: MagicMock) -> None:
    """Make the id missing entirely."""
    del run.id


def _break_id(run: MagicMock) -> None:
    """Make reading the id raise, as a decorated property can.

    Real wandb's ``Run.id`` is a property, so a read can fail with something other
    than AttributeError — ``getattr(run, "id", None)`` only defaults on the latter.
    """
    type(run).id = property(
        lambda _self: (_ for _ in ()).throw(AssertionError("id unavailable"))
    )


@pytest.mark.parametrize(
    "make_unreadable", [_hide_id, _break_id], ids=["absent", "raises"]
)
def test_init_failure_ignores_a_global_run_with_no_readable_id(make_unreadable):
    """An unreadable id counts as not-ours: releasing the wrong run corrupts it.

    Failing to release ours only leaves the id in use until restart, so this is
    the safer direction to fail in. Unreadable covers both an id that is absent and
    one whose read raises; the latter also must not escape, because this read happens
    inside the init-failure handler ahead of the bare ``raise``, so an error here
    would replace the init failure that is the real diagnostic.
    """
    service = _service()
    unreadable = _make_run()
    make_unreadable(unreadable)

    try:
        with (
            patch.object(service, "_init_run", side_effect=RuntimeError("boom")),
            patch("gbserver.lineage.wandb_service.wandb") as wandb_mod,
        ):
            wandb_mod.run = unreadable
            # The original init error surfaces, not any error from the id read.
            with pytest.raises(RuntimeError, match="boom"):
                service._get_run("run-1", "job-1")
    finally:
        # Set on the type by _break_id, so it outlives this instance.
        if "id" in vars(type(unreadable)):
            del type(unreadable).id

    unreadable.finish.assert_not_called()
    assert service._runs == {}


def test_init_failure_teardown_error_does_not_mask_original():
    """A finish() error during cleanup must not replace the real exception."""
    service = _service()
    leaked = _make_run()
    leaked.finish.side_effect = RuntimeError("teardown exploded")

    with (
        patch.object(service, "_init_run", side_effect=ValueError("original")),
        patch("gbserver.lineage.wandb_service.wandb") as wandb_mod,
    ):
        wandb_mod.run = leaked
        with pytest.raises(ValueError, match="original"):
            service._get_run("run-1", "job-1")


def test_mid_event_failure_releases_the_open_run():
    """A failure after the run opened releases it before re-raising."""
    service = _service()
    run = _make_run()
    service._runs["run-1"] = run

    with (
        patch.object(service, "_get_run", return_value=run),
        patch.object(service, "_register_artifacts", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError):
            service.emit_event(_event())

    run.finish.assert_called_once()
    assert service._runs == {}


def test_terminal_event_is_not_released_twice():
    """COMPLETE already finished the run; the failure path must not re-finish.

    The logger.info after the terminal branch is patched to raise so the except
    block runs with the run already finished and popped.
    """
    service = _service()
    run = _make_run()

    with (
        patch.object(service, "_get_run", return_value=run),
        patch.object(service, "_register_artifacts"),
        patch(
            "gbserver.lineage.wandb_service.logger.info",
            side_effect=RuntimeError("boom"),
        ),
    ):
        with pytest.raises(RuntimeError):
            service.emit_event(_event("COMPLETE"))

    # Exactly one finish(): the COMPLETE one, not a second from cleanup.
    run.finish.assert_called_once_with()
    assert service._runs == {}


def test_key_error_before_run_opens_releases_nothing():
    """A malformed event fails before any run exists."""
    service = _service()

    with patch.object(service, "_get_run") as get_run:
        with pytest.raises(KeyError):
            service.emit_event({"run": {}, "job": {}})
        get_run.assert_not_called()

    assert service._runs == {}


def test_release_run_is_a_noop_for_unknown_id():
    service = _service()
    service._release_run("never-opened")
    assert service._runs == {}


def test_init_uses_resume_never():
    """``resume="never"`` so a reused id fails loudly instead of appending.

    Run ids are random uuids, so a resume can never legitimately happen. Under
    ``resume="allow"`` a uuid collision — or a bug that reused an id — would
    silently append to an existing run, corrupting that run's lineage with no
    error. "never" turns it into a visible failure.
    """
    service = _service()
    with patch("gbserver.lineage.wandb_service.wandb") as wandb_mock:
        wandb_mock.init.return_value = _make_run()
        service._init_run("run-1", "job-1")

    _args, kwargs = wandb_mock.init.call_args
    assert kwargs["resume"] == "never"
