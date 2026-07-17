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

"""Tests for resolve_monitor_config: inline passthrough + monitor-library `ref`
resolution (space://monitors/<name>) with monitor→monitor parent chains,
overlay/append, same-type enforcement, cycle detection, and isolation."""

from pathlib import Path
from typing import Self

import pytest
import yaml

from gbcommon.uri.space import SpaceURI
from gbserver.build.targetsteprun import (
    _reset_monitor_file_cache,
    resolve_monitor_config,
)
from gbserver.types.stepconfig import StepMonitorConfig

# A base skypilot monitor: the standard artifact convention + a default profile.
_BASE = {
    "type": "skypilot_monitor",
    "config": {
        "poll_interval_seconds": 900,
        "log_retrieval": {"mode": "on_completion", "interval_seconds": 15},
        "event_configs": [
            {
                "event_type": "newartifact_in_environment_event",
                "line_regex": "LLMB_ARTIFACT_ID:.+LLMB_ARTIFACT_PATH:.+",
            }
        ],
    },
}


def _write_monitor(root: Path, name: str, data: dict) -> None:
    """Write ``<root>/monitors/<name>/monitor.yaml`` with ``data``."""
    d = root / "monitors" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "monitor.yaml").write_text(yaml.safe_dump(data))


@pytest.fixture
def monitor_library(tmp_path: Path):
    """Build a temp monitor library and point SpaceURI at it for the test.

    Yields the library root; restores the prior SpaceURI base URIs on exit.
    """
    _write_monitor(tmp_path, "skypilot", _BASE)
    # endpoint: same type via parent ref, overrides only the poll interval
    _write_monitor(
        tmp_path,
        "skypilot-fast",
        {"ref": "space://monitors/skypilot", "config": {"poll_interval_seconds": 30}},
    )
    # a different type, for the same-type-violation test
    _write_monitor(
        tmp_path,
        "dockerish",
        {"type": "docker_log", "config": {"event_configs": []}},
    )
    # a monitor that wrongly references a different-type parent
    _write_monitor(
        tmp_path,
        "crosstype",
        {"type": "skypilot_monitor", "ref": "space://monitors/dockerish"},
    )
    # a reference cycle a -> b -> a
    _write_monitor(tmp_path, "cyc_a", {"ref": "space://monitors/cyc_b"})
    _write_monitor(tmp_path, "cyc_b", {"ref": "space://monitors/cyc_a"})

    prev = getattr(SpaceURI._thread_local, "base_uris", None)
    prev_secrets = getattr(SpaceURI._thread_local, "space_secrets", None)
    SpaceURI.set_baseuris(base_uris=[tmp_path.as_uri()], space_secrets={})
    # Isolate the thread-local monitor-file cache between cases so a prior test's
    # parse can't be served for this test's freshly-written library.
    _reset_monitor_file_cache()
    try:
        yield tmp_path
    finally:
        SpaceURI.set_baseuris(
            base_uris=prev if prev is not None else ["file:"],
            space_secrets=prev_secrets or {},
        )


class TestResolveMonitorConfig:
    """resolve_monitor_config across inline and monitor-library refs."""

    def test_inline_passthrough(self: Self) -> None:
        """An inline monitor returns its own (type, config) unchanged."""
        m_type, cfg = resolve_monitor_config(
            StepMonitorConfig(type="docker_log", config={"a": 1})
        )
        assert m_type == "docker_log"
        assert cfg == {"a": 1}

    def test_inline_extra_event_configs_rejected(self: Self) -> None:
        """An inline monitor (no ref) setting extra_event_configs raises.

        extra_event_configs only appends to a referenced monitor's rules; on an
        inline monitor it has no base to append to and would be silently dropped
        downstream, so the resolver rejects it at config time.
        """
        with pytest.raises(ValueError, match="extra_event_configs"):
            resolve_monitor_config(
                StepMonitorConfig(
                    type="log_monitor",
                    config={"extra_event_configs": [{"event_type": "message_event"}]},
                )
            )

    def test_ref_to_monitor_file(self: Self, monitor_library) -> None:
        """A step ref loads the monitor file's (type, config)."""
        m_type, cfg = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/skypilot")
        )
        assert m_type == "skypilot_monitor"
        assert cfg["poll_interval_seconds"] == 900
        assert len(cfg["event_configs"]) == 1

    def test_monitor_parent_chain_merge(self: Self, monitor_library) -> None:
        """An endpoint monitor deep-merges over its parent (child wins)."""
        m_type, cfg = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/skypilot-fast")
        )
        assert m_type == "skypilot_monitor"
        assert cfg["poll_interval_seconds"] == 30  # child override
        assert cfg["log_retrieval"]["interval_seconds"] == 15  # inherited
        assert len(cfg["event_configs"]) == 1  # inherited artifact event

    def test_step_overlay_and_append(self: Self, monitor_library) -> None:
        """Step overlay overrides a knob and extra_event_configs appends."""
        status = {"event_type": "workload_status_event", "line_regex": "RUN"}
        _, cfg = resolve_monitor_config(
            StepMonitorConfig(
                ref="space://monitors/skypilot",
                config={
                    "poll_interval_seconds": 5,
                    "extra_event_configs": [status],
                },
            )
        )
        assert cfg["poll_interval_seconds"] == 5
        assert "extra_event_configs" not in cfg
        assert len(cfg["event_configs"]) == 2
        assert cfg["event_configs"][-1] == status

    def test_null_base_event_configs_with_extra(self: Self, monitor_library) -> None:
        """A base monitor written as ``event_configs:`` (null) + an overlay's
        extra_event_configs must not crash (list(None) TypeError); the extra
        rules become the resolved event_configs.
        """
        _write_monitor(
            monitor_library,
            "nullevents",
            {"type": "skypilot_monitor", "config": {"event_configs": None}},
        )
        status = {"event_type": "workload_status_event", "line_regex": "RUN"}
        _, cfg = resolve_monitor_config(
            StepMonitorConfig(
                ref="space://monitors/nullevents",
                config={"extra_event_configs": [status]},
            )
        )
        assert cfg["event_configs"] == [status]

    def test_nested_monitor_yaml_picks_shallowest(
        self: Self, monitor_library
    ) -> None:
        """A stray nested monitor.yaml must not make resolution nondeterministic;
        the canonical top-level file (shallowest) is chosen regardless of glob
        order."""
        _write_monitor(
            monitor_library,
            "nested",
            {"type": "skypilot_monitor", "config": {"poll_interval_seconds": 1}},
        )
        # A deeper monitor.yaml with a different value; must be ignored.
        deep = monitor_library / "monitors" / "nested" / "sub"
        deep.mkdir(parents=True, exist_ok=True)
        (deep / "monitor.yaml").write_text(
            yaml.safe_dump(
                {"type": "skypilot_monitor", "config": {"poll_interval_seconds": 999}}
            )
        )
        _, cfg = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/nested")
        )
        assert cfg["poll_interval_seconds"] == 1  # top-level wins, not 999

    def test_monitor_file_fetched_once_and_memoized(
        self: Self, monitor_library, monkeypatch
    ) -> None:
        """Resolving the same ref twice fetches (syncs) the monitor file once.

        resolve_monitor_config runs twice per step launch; the thread-local cache
        must avoid a second clone/copy for the same (uri, space).
        """
        import gbserver.build.targetsteprun as tsr

        real_asset = tsr.Asset
        sync_calls = {"n": 0}

        class CountingAsset:
            def __init__(self, uri: str) -> None:
                self._inner = real_asset(uri)

            def sync(self, dest=None, force: bool = False):
                sync_calls["n"] += 1
                return self._inner.sync(dest=dest, force=force)

        monkeypatch.setattr(tsr, "Asset", CountingAsset)
        first = resolve_monitor_config(StepMonitorConfig(ref="space://monitors/skypilot"))
        second = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/skypilot")
        )
        assert sync_calls["n"] == 1  # second resolve served from cache
        assert first[0] == second[0] == "skypilot_monitor"
        assert first[1] == second[1]

    def test_overlay_event_configs_rejected(self: Self, monitor_library) -> None:
        """An overlay that sets event_configs (vs extra_event_configs) raises.

        merge_dicts replaces lists wholesale, so allowing this would silently drop
        the referenced monitor's artifact rules; the resolver rejects it instead.
        """
        with pytest.raises(ValueError, match="event_configs"):
            resolve_monitor_config(
                StepMonitorConfig(
                    ref="space://monitors/skypilot",
                    config={"event_configs": [{"event_type": "message_event"}]},
                )
            )

    def test_same_type_violation_raises(self: Self, monitor_library) -> None:
        """A monitor referencing a different-type parent raises."""
        with pytest.raises(ValueError, match="same type|type"):
            resolve_monitor_config(StepMonitorConfig(ref="space://monitors/crosstype"))

    def test_cycle_raises(self: Self, monitor_library) -> None:
        """A reference cycle raises rather than recursing forever."""
        with pytest.raises(ValueError, match="cycle"):
            resolve_monitor_config(StepMonitorConfig(ref="space://monitors/cyc_a"))

    def test_unknown_ref_raises(self: Self, monitor_library) -> None:
        """Referencing a monitor with no monitor.yaml raises."""
        with pytest.raises(ValueError):
            resolve_monitor_config(
                StepMonitorConfig(ref="space://monitors/does-not-exist")
            )

    def test_no_mutation_of_loaded_file(self: Self, monitor_library) -> None:
        """Overrides must not mutate the on-disk monitor definition."""
        resolve_monitor_config(
            StepMonitorConfig(
                ref="space://monitors/skypilot",
                config={
                    "poll_interval_seconds": 1,
                    "extra_event_configs": [{"event_type": "message_event"}],
                },
            )
        )
        again_type, again = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/skypilot")
        )
        assert again["poll_interval_seconds"] == 900
        assert len(again["event_configs"]) == 1
