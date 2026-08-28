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

import re
from pathlib import Path
from typing import Self

import pytest
import yaml

from gbcommon.uri.space import SpaceURI
from gbserver.build.build import _step_monitor_ref_errors
from gbserver.build.targetsteprun import (
    _reset_monitor_file_cache,
    resolve_monitor_config,
)
from gbserver.types.stepconfig import (
    StepEnvironmentTypeConfig,
    StepLauncherConfig,
    StepMonitorConfig,
)

# A base skypilot monitor: the standard artifact convention + a default profile.
_BASE = {
    "type": "skypilot_monitor",
    "config": {
        "poll_interval_seconds": 900,
        "log_retrieval": {"mode": "on_completion", "interval_seconds": 15},
        "event_configs": [
            {
                "event_type": "newartifact_in_environment_event",
                # Dual-accept form: markers are standardized on GB_, with the
                # legacy LLMB_ prefix still recognized for compatibility.
                "line_regex": "(?:GB_|LLMB_)ARTIFACT_ID:.+(?:GB_|LLMB_)ARTIFACT_PATH:.+",
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

    def test_typeless_inline_monitor_raises(self: Self) -> None:
        """resolve never returns a None type: a typeless inline monitor (validator
        bypassed via model_construct) raises instead of returning (None, ...),
        which downstream would only trip the -O-strippable run-time assert.
        """
        m = StepMonitorConfig.model_construct(type=None, ref=None, config={})
        with pytest.raises(ValueError, match="type"):
            resolve_monitor_config(m)

    def test_typeless_ref_chain_raises(self: Self, monkeypatch) -> None:
        """A ref chain that resolves to no type at any level raises (so build
        validation surfaces it) rather than returning (None, ...)."""
        import gbserver.build.targetsteprun as tsr

        monkeypatch.setattr(
            tsr,
            "_load_monitor_file",
            lambda uri: StepMonitorConfig.model_construct(
                type=None, ref=None, config={}
            ),
        )
        with pytest.raises(ValueError, match="type"):
            resolve_monitor_config(StepMonitorConfig(ref="space://monitors/x"))

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

    def test_three_level_chain_compounds_overlays(self: Self, monitor_library) -> None:
        """A 3-level chain (leaf -> mid -> skypilot) compounds the recursive merge:
        scalar override takes the deepest value, nested-dict keys deep-merge across
        all levels, and extra_event_configs append base-first at each level.
        """
        e_mid = {"event_type": "workload_status_event", "line_regex": "MID"}
        e_leaf = {"event_type": "workload_status_event", "line_regex": "LEAF"}
        # mid: ref skypilot; override poll + log_retrieval.interval; append e_mid.
        _write_monitor(
            monitor_library,
            "skypilot-mid",
            {
                "ref": "space://monitors/skypilot",
                "config": {
                    "poll_interval_seconds": 300,
                    "log_retrieval": {"interval_seconds": 45},
                    "extra_event_configs": [e_mid],
                },
            },
        )
        # leaf: ref mid; override poll again + log_retrieval.mode; append e_leaf.
        _write_monitor(
            monitor_library,
            "skypilot-leaf",
            {
                "ref": "space://monitors/skypilot-mid",
                "config": {
                    "poll_interval_seconds": 60,
                    "log_retrieval": {"mode": "periodic"},
                    "extra_event_configs": [e_leaf],
                },
            },
        )
        m_type, cfg = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/skypilot-leaf")
        )
        assert m_type == "skypilot_monitor"  # type from the grandparent root
        assert cfg["poll_interval_seconds"] == 60  # deepest (leaf) scalar wins
        # log_retrieval deep-merges across all three levels: mode from leaf,
        # interval from mid; neither is the grandparent's (on_completion / 15).
        assert cfg["log_retrieval"] == {"mode": "periodic", "interval_seconds": 45}
        # extra_event_configs append base-first: [grandparent artifact, mid, leaf].
        assert len(cfg["event_configs"]) == 3
        assert cfg["event_configs"][1] == e_mid
        assert cfg["event_configs"][2] == e_leaf
        assert "extra_event_configs" not in cfg

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

    def test_nested_monitor_yaml_picks_shallowest(self: Self, monitor_library) -> None:
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
        first = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/skypilot")
        )
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

    def test_fetch_failure_classified_by_base_uri_locality(
        self: Self, monkeypatch
    ) -> None:
        """A failed fetch is a plain ValueError for a local-only space (dangling ref,
        fail fast) but a MonitorFetchError when a remote (git) base may have failed
        transiently. Uses a fake Asset so no real clone/network happens.
        """
        import gbserver.build.targetsteprun as tsr

        class _FakeAsset:
            def __init__(self, uri: str) -> None:
                pass

            def sync(self, dest=None, force: bool = False):
                raise RuntimeError("boom")

        monkeypatch.setattr(tsr, "Asset", _FakeAsset)
        prev = getattr(SpaceURI._thread_local, "base_uris", None)
        try:
            SpaceURI.set_baseuris(base_uris=["file:///tmp/space"], space_secrets={})
            _reset_monitor_file_cache()
            with pytest.raises(ValueError) as local_exc:
                tsr._load_monitor_file("space://monitors/x")
            assert not isinstance(local_exc.value, tsr.MonitorFetchError)

            SpaceURI.set_baseuris(
                base_uris=["git+ssh://example.com/o/repo.git@main"], space_secrets={}
            )
            _reset_monitor_file_cache()
            with pytest.raises(tsr.MonitorFetchError):
                tsr._load_monitor_file("space://monitors/x")
        finally:
            SpaceURI.set_baseuris(
                base_uris=prev if prev is not None else ["file:"], space_secrets={}
            )
            _reset_monitor_file_cache()


@pytest.fixture
def builtin_monitor_space():
    """Point SpaceURI at the real shipped ``builtins/`` monitor library.

    Unlike ``monitor_library`` (which synthesizes a temp library), this resolves
    ``space://monitors/<name>`` against the monitor.yaml files actually shipped in
    the repo, so a content regression in one is caught. Restores prior base URIs.
    """
    import gbserver.build.space as space_mod

    builtins = Path(space_mod.__file__).parent.parent / "builtins"
    prev = getattr(SpaceURI._thread_local, "base_uris", None)
    prev_secrets = getattr(SpaceURI._thread_local, "space_secrets", None)
    SpaceURI.set_baseuris(base_uris=[builtins.as_uri()], space_secrets={})
    _reset_monitor_file_cache()
    try:
        yield builtins
    finally:
        SpaceURI.set_baseuris(
            base_uris=prev if prev is not None else ["file:"],
            space_secrets=prev_secrets or {},
        )
        _reset_monitor_file_cache()


class TestBuiltinLsfMonitor:
    """The shipped builtins/monitors/lsf/monitor.yaml resolves as expected."""

    def test_lsf_monitor_resolves_to_bsub_type_and_rules(
        self: Self, builtin_monitor_space
    ) -> None:
        """`ref: space://monitors/lsf` yields the bsub_monitor type with exactly the
        three standard rules (artifact PATH, artifact STATE, step-metadata) and no
        inert poll/log_retrieval keys (which have no LSF runtime consumer)."""
        m_type, cfg = resolve_monitor_config(
            StepMonitorConfig(ref="space://monitors/lsf")
        )
        assert m_type == "bsub_monitor"
        assert "poll_interval_seconds" not in cfg
        assert "log_retrieval" not in cfg
        rules = {e["event_type"]: e for e in cfg["event_configs"]}
        event_types = [e["event_type"] for e in cfg["event_configs"]]
        assert event_types == [
            "NEWARTIFACT_IN_ENVIRONMENT_EVENT",  # env:// PATH
            "NEWARTIFACT_IN_ENVIRONMENT_EVENT",  # mem:// STATE
            "STEP_METADATA_UPDATE_EVENT",
        ]
        # Artifact rules are unanchored (raw-log forward-compat); the provenance
        # rule is `^`-anchored so it cannot be injected mid-line.
        assert rules["STEP_METADATA_UPDATE_EVENT"]["line_regex"].startswith("^")
        for e in cfg["event_configs"]:
            if e["event_type"] == "NEWARTIFACT_IN_ENVIRONMENT_EVENT":
                assert not e["line_regex"].startswith("^")

    @pytest.mark.parametrize("prefix", ["GB_", "LLMB_"])
    def test_lsf_artifact_rules_match_wrapper_emission(
        self: Self, builtin_monitor_space, prefix: str
    ) -> None:
        """The LSF monitor's artifact rules must match the marker lines the LSF
        wrapper actually emits.

        Regression guard for a producer/consumer prefix drift: llmb_lsf_wrapper.sh
        emits ``GB_ARTIFACT_ID:… GB_ARTIFACT_PATH:…`` while the monitor once matched
        only ``LLMB_``. That drift silently registered zero artifacts (no error, no
        log). Both the standardized ``GB_`` and legacy ``LLMB_`` prefixes must match,
        and field extraction must recover the id and path.
        """
        _, cfg = resolve_monitor_config(StepMonitorConfig(ref="space://monitors/lsf"))
        rules = cfg["event_configs"]
        path_rule = rules[0]  # env:// PATH artifact
        state_rule = rules[1]  # mem:// STATE artifact

        # Mirror the wrapper's emitted lines (see llmb_lsf_wrapper.sh:183,188).
        path_line = f"{prefix}ARTIFACT_ID:outputs {prefix}ARTIFACT_PATH:/work/outputs"
        state_line = f"{prefix}ARTIFACT_ID:model {prefix}ARTIFACT_STATE:ready"

        assert re.search(path_rule["line_regex"], path_line)
        assert re.search(state_rule["line_regex"], state_line)

        fields = {f["field_name"]: f for f in path_rule["event_fields"]}
        binding_id = re.search(fields["binding_id"]["field_regex"], path_line)
        path = re.search(fields["path"]["field_regex"], path_line)
        assert binding_id and binding_id.group(0) == "outputs"
        assert path and path.group(0) == "/work/outputs"

    def test_lsf_step_overlay_appends_push_event(
        self: Self, builtin_monitor_space
    ) -> None:
        """A step (e.g. lhpush) that refs the LSF monitor and appends its own
        ARTIFACT_PUSHED_EVENT via extra_event_configs keeps the three base rules."""
        push = {
            "event_type": "ARTIFACT_PUSHED_EVENT",
            "line_regex": r"Pushed\sURI:\s.+",
        }
        _, cfg = resolve_monitor_config(
            StepMonitorConfig(
                ref="space://monitors/lsf",
                config={"extra_event_configs": [push]},
            )
        )
        assert len(cfg["event_configs"]) == 4
        assert cfg["event_configs"][-1] == push
        assert "extra_event_configs" not in cfg


def _env_cfg(monitors: dict) -> StepEnvironmentTypeConfig:
    """Build a StepEnvironmentTypeConfig with the given monitors map."""
    return StepEnvironmentTypeConfig(
        launchers={"l": StepLauncherConfig(type="x")}, monitors=monitors
    )


class TestStepMonitorRefErrors:
    """_step_monitor_ref_errors: build-creation-time monitor-ref resolution.

    Restores fail-fast — a bad monitor ref is reported at validation, not step-run.
    """

    def test_valid_ref_no_error(self: Self, monitor_library) -> None:
        """A launcher-selected monitor with a resolvable ref yields no error."""
        env_cfg = _env_cfg({"m": StepMonitorConfig(ref="space://monitors/skypilot")})
        launcher = StepLauncherConfig(type="x", monitors=["m"])
        assert _step_monitor_ref_errors(env_cfg, launcher) == ([], [])

    def test_dangling_local_ref_is_fatal(self: Self, monitor_library) -> None:
        """A dangling *local* ref is a fatal error (fail fast), not a warning.

        The fixture's base_uris are file:// (local), so a missing monitor can't be
        transient — it must invalidate the build.
        """
        env_cfg = _env_cfg(
            {"m": StepMonitorConfig(ref="space://monitors/does-not-exist")}
        )
        launcher = StepLauncherConfig(type="x", monitors=["m"])
        errs, warns = _step_monitor_ref_errors(env_cfg, launcher)
        assert len(errs) == 1 and "does-not-exist" in errs[0]
        assert warns == []

    def test_launcher_names_undefined_monitor_errors(
        self: Self, monitor_library
    ) -> None:
        """A launcher naming a monitor absent from env_cfg.monitors is flagged."""
        env_cfg = _env_cfg({})
        launcher = StepLauncherConfig(type="x", monitors=["missing"])
        errs, warns = _step_monitor_ref_errors(env_cfg, launcher)
        assert len(errs) == 1 and "missing" in errs[0]
        assert warns == []

    def test_cycle_and_crosstype_refs_error(self: Self, monitor_library) -> None:
        """Ref cycles and cross-type refs are fatal (structural) errors."""
        env_cfg = _env_cfg(
            {
                "cyc": StepMonitorConfig(ref="space://monitors/cyc_a"),
                "xtype": StepMonitorConfig(ref="space://monitors/crosstype"),
            }
        )
        launcher = StepLauncherConfig(type="x", monitors=["cyc", "xtype"])
        errs, warns = _step_monitor_ref_errors(env_cfg, launcher)
        assert len(errs) == 2 and warns == []

    def test_launcher_with_no_monitors_no_error(self: Self, monitor_library) -> None:
        """A launcher that selects no monitors produces no errors."""
        env_cfg = _env_cfg({"m": StepMonitorConfig(ref="space://monitors/skypilot")})
        assert _step_monitor_ref_errors(env_cfg, StepLauncherConfig(type="x")) == (
            [],
            [],
        )

    def test_monitor_fetch_error_is_warning_not_error(
        self: Self, monitor_library, monkeypatch
    ) -> None:
        """A ``MonitorFetchError`` (transient/remote fetch) becomes a WARNING, not a
        build-invalidating error, so a network blip can't turn a valid build
        INVALID. (The local→ValueError / remote→MonitorFetchError classification is
        covered in TestResolveMonitorConfig.)
        """
        import gbserver.build.targetsteprun as tsr

        def boom(monitor, _seen=None):
            raise tsr.MonitorFetchError("Cannot fetch monitor for ref 'x': net down")

        monkeypatch.setattr(tsr, "resolve_monitor_config", boom)
        env_cfg = _env_cfg({"m": StepMonitorConfig(ref="space://monitors/skypilot")})
        launcher = StepLauncherConfig(type="x", monitors=["m"])
        errs, warns = _step_monitor_ref_errors(env_cfg, launcher)
        assert errs == []  # not fatal — must not invalidate the build
        assert len(warns) == 1 and "retry at run time" in warns[0]
