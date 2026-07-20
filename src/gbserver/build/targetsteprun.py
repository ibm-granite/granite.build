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

"""
The target step run.
"""

import asyncio
import dataclasses
import shutil
import tempfile
import threading
import traceback
from asyncio import Queue, TaskGroup
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Self, Tuple
from urllib.parse import urlparse

import yaml

from gbcommon.uri.space import SpaceURI
from gbserver.asset.asset import Asset
from gbserver.build.run import Run
from gbserver.build.target import Target
from gbserver.build.targetstep import TargetStep
from gbserver.types.buildconfig import BuildTargetStepConfig
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.constants import USE_LESS_COMPUTE_ON_DRY_RUN
from gbserver.types.status import STATUS_TO_ICON, Status
from gbserver.types.stepconfig import StepLauncherConfig, StepMonitorConfig
from gbserver.utils.filesystem import (
    fill_templates_in_dir,
    find_files_shallowest_first,
    merge_dicts,
    sync_or_copy,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_objtemplate

logger = get_logger(__name__)


MONITOR_FILE_NAME = "monitor.yaml"


class MonitorFetchError(ValueError):
    """A monitor ref could not be *fetched* (transport failure), as opposed to
    being structurally invalid.

    Raised only when the monitor's space resolves against a **remote** base URI
    (e.g. a git-hosted space) and the fetch fails — a failure that may be
    transient (network). Callers that validate refs ahead of time (build-creation
    validation) can treat this as non-fatal (don't permanently invalidate the
    build) and let it retry at run time, while still failing fast on structurally
    bad refs (dangling local ref, cycle, cross-type, missing monitor.yaml), which
    raise a plain ``ValueError``. Subclasses ``ValueError`` so existing
    ``except ValueError`` run-time call sites keep catching it.
    """


def _all_local_base_uris(base_uris: Tuple[str, ...]) -> bool:
    """Whether every space base URI is local (``file://`` or a bare path).

    A fetch failure against local-only bases can't be transient — the monitor
    simply isn't there (a dangling/typo'd ref), so it should fail fast.
    """
    return all(urlparse(b).scheme in ("", "file") for b in base_uris)


def _monitor_fetch_error(
    uri_str: str, base_uris: Tuple[str, ...], detail: str
) -> ValueError:
    """Build the error for a failed monitor fetch, classified by base-URI locality.

    Local-only spaces → plain ``ValueError`` (dangling ref, fail fast). A space
    with any remote base → :class:`MonitorFetchError` (possibly transient).
    """
    msg = f"Cannot fetch monitor for ref '{uri_str}': {detail}"
    return (
        ValueError(msg) if _all_local_base_uris(base_uris) else MonitorFetchError(msg)
    )


# Thread-local memoization of parsed monitor-library files. resolve_monitor_config
# runs twice per step launch (create loop + _run) and once per ref-chain level, and
# each fetch of a git-backed monitor is a clone+copy — so caching the parsed result
# fetches a monitor at most once per thread/space. Keyed on (uri, space base_uris):
# base_uris are thread-local per build, so builds using different spaces never share
# entries and the same space:// name can't collide across spaces. Thread-local (like
# Step's per-thread cache dir) means no locking and no cross-thread staleness.
_MONITOR_FILE_CACHE = threading.local()


def _reset_monitor_file_cache() -> None:
    """Clear this thread's memoized monitor-file parses.

    Test hook for isolating the thread-local :data:`_MONITOR_FILE_CACHE` between
    cases (the cache otherwise persists for the thread's lifetime). Not used in
    production — a real build never needs to invalidate mid-run.
    """
    _MONITOR_FILE_CACHE.__dict__.pop("entries", None)


def _load_monitor_file(uri_str: str) -> StepMonitorConfig:
    """Load and parse a monitor-library entry referenced by a ``space://`` URI.

    Mirrors how steps resolve (``space://steps/<name>`` → a directory containing
    ``step.yaml``): ``space://monitors/<name>`` resolves — via the space
    resolver — to a directory whose ``monitor.yaml`` holds the monitor. The
    shipped ``builtins/monitors/`` tree is a base URI (the default); a
    configurations-level ``monitors/`` tree overrides by name. A monitor.yaml
    has the same ``{type, ref, config}`` shape as a :class:`StepMonitorConfig`,
    so a monitor may itself reference a parent monitor.

    The resolved directory is materialized locally via :class:`Asset` (the same
    fetch path steps use), so a monitor ref works for every space backend that
    steps support — including git-hosted spaces, whose base URI resolves to a
    ``git+ssh://`` / ``git://`` URI rather than a local ``file://`` path.

    Threading / event loop: this resolves ``space://`` against **thread-local**
    ``SpaceURI.base_uris`` (and uses the thread-local cache below), so it must run
    on the per-build thread and is intentionally **not** offloaded to a worker
    thread — a worker would lack that space context. For a git-backed space the
    ``Asset.sync()`` clone blocks that (event-loop) thread, exactly as the
    step.yaml materialization in :class:`Step` and the ``copytree``/template work
    in ``TargetStepRun.__init__`` already do. In practice the run-time resolves are
    cache hits: ``Build.__validate_step_monitors`` pre-resolves every step's
    monitors during ``__setup`` on this same thread (build creation), warming the
    cache. The only cold clone on the loop is a resume (validation skipped) with a
    git-backed monitor *override*; local ``builtins/monitors/*`` refs are a cheap
    file copy.

    Args:
        uri_str: Monitor URI, e.g. ``space://monitors/skypilot``.

    Returns:
        The parsed monitor as a :class:`StepMonitorConfig`.

    Raises:
        ValueError: If the URI cannot be fetched or contains no ``monitor.yaml``.
    """
    # Return a memoized parse when this (uri, space) was already fetched on this
    # thread — avoids the redundant clone+copy on the second resolve per launch
    # and across ref-chain levels. base_uris scope the space:// resolution.
    base_uris = tuple(getattr(SpaceURI._thread_local, "base_uris", ()) or ())
    cache: Dict = _MONITOR_FILE_CACHE.__dict__.setdefault("entries", {})
    cache_key = (uri_str, base_uris)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch the referenced monitor directory into a temp dir, read the single
    # monitor.yaml, then delete the temp dir before returning. Asset.sync()
    # resolves the space:// URI against the space base_uris and pulls the result,
    # handling file:// (builtins / local space) and git-backed spaces uniformly —
    # exactly as Step does for step.yaml. resolve_monitor_config runs multiple
    # times per step launch (and once per ref-chain level), so the temp dir MUST
    # be cleaned up immediately — otherwise a long-lived gbserver leaks one dir
    # per call and grows unbounded.
    monitor_dir = Path(tempfile.mkdtemp())
    try:
        try:
            # sync() returns the dest on success, or None when pull() reports
            # failure without raising (e.g. a failed copy). Guard the None so a
            # failed fetch surfaces as an error rather than a later, less obvious
            # one (empty dir -> "no monitor.yaml", or a raw TypeError). Classify
            # the failure: a local-only space can't fail transiently (missing =
            # dangling ref, fail fast -> ValueError), while a remote (git) fetch
            # may be a transient network issue (-> MonitorFetchError).
            synced = Asset(uri_str).sync(dest=monitor_dir)
        except Exception as exc:
            raise _monitor_fetch_error(uri_str, base_uris, str(exc)) from exc
        if synced is None:
            raise _monitor_fetch_error(uri_str, base_uris, "fetch reported failure")
        # Locate monitor.yaml deterministically (shallowest first): a directory
        # source may nest it one level under the sync dest (dest/<name>/…), and
        # glob order is filesystem-dependent. Shared with step.yaml resolution.
        files = find_files_shallowest_first(monitor_dir, MONITOR_FILE_NAME)
        if not files:
            raise ValueError(
                f"Monitor ref '{uri_str}' resolved to '{monitor_dir}' but no "
                f"'{MONITOR_FILE_NAME}' was found there."
            )
        if len(files) > 1:
            logger.warning(
                "Monitor ref '%s' matched %d '%s' files under '%s'; using the "
                "shallowest (%s).",
                uri_str,
                len(files),
                MONITOR_FILE_NAME,
                monitor_dir,
                files[0],
            )
        monitor_path = Path(files[0])
        try:
            with open(monitor_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except OSError as exc:
            raise ValueError(
                f"Cannot read monitor for ref '{uri_str}' at '{monitor_path}': {exc}"
            ) from exc
    finally:
        shutil.rmtree(monitor_dir, ignore_errors=True)
    parsed = StepMonitorConfig.model_validate(data)
    cache[cache_key] = parsed
    return parsed


def resolve_monitor_config(
    monitor_config: StepMonitorConfig, _seen: Optional[Tuple[str, ...]] = None
) -> Tuple[Optional[str], Dict]:
    """Resolve a monitor entry (inline, or a ``ref`` to a monitor-library file).

    Inline entries (no ``ref``) return ``(type, deepcopy(config))``; they may not
    set ``extra_event_configs`` (there is no referenced base to append to — an
    inline monitor lists all its rules directly under ``event_configs``). A ``ref``
    names a monitor file (``space://monitors/<name>``); its own parent
    ``ref`` chain is resolved recursively, merging each ``config`` overlay over
    the inherited config via ``merge_dicts`` — nested dicts merge, but scalars
    and **lists replace wholesale** (child wins). To *extend* the referenced
    monitor's ``event_configs`` (rather than replace them), an overlay uses the
    reserved ``extra_event_configs`` list, which is appended to the inherited
    ``event_configs`` at each level. Setting ``event_configs`` directly in an
    overlay is **rejected**: because lists replace, it would silently drop the
    referenced monitor's artifact rules (and the ``binding`` payload downstream
    consumers dereference). The referring entry's ``type`` (when set) must equal
    the referenced chain's ``type`` — a monitor may only reference another of the
    same type. All configs are deepcopied so a resolution never mutates a shared
    loaded file.

    Args:
        monitor_config: The monitor entry to resolve (from a step's ``monitors``
            map or a monitor-library file).
        _seen: Internal — ref URIs already on the current chain, in traversal
            order, for cycle detection.

    Returns:
        A ``(type, config)`` tuple: the resolved monitor type and the merged,
        pre-template config dict (``extra_event_configs`` consumed, not present).

    Raises:
        ValueError: On an unreadable/non-local ref, a reference cycle, a
            cross-type reference, an overlay that sets ``event_configs`` directly
            (use ``extra_event_configs`` to append instead), an inline monitor
            that sets ``extra_event_configs`` (put rules in ``event_configs``), or
            a monitor/ref chain that resolves to no ``type``. The returned type is
            therefore never ``None``.
    """
    if not monitor_config.ref:
        config = deepcopy(monitor_config.config or {})
        if "extra_event_configs" in config:
            # extra_event_configs only appends to a *referenced* monitor's
            # event_configs; an inline monitor has no inherited base, so the key
            # would be dropped downstream (swallowed by the monitor method's
            # **kwargs) and its rules silently never register. Reject at config
            # time instead of failing invisibly at run time.
            raise ValueError(
                "Monitor sets 'extra_event_configs' but has no 'ref'; that key "
                "only appends to a referenced monitor's rules. For an inline "
                "monitor, put the rules directly in 'event_configs'."
            )
        if monitor_config.type is None:
            # Defense-in-depth: a validated StepMonitorConfig always has type or
            # ref (check_type_or_ref), but guard a bypassed/relaxed config so we
            # never return a None type (which downstream would only trip the
            # -O-strippable assert in TargetStepRun._run, with a worse message).
            raise ValueError(
                "Monitor resolves to no 'type' and has no 'ref'; an inline "
                "monitor must set 'type'."
            )
        return monitor_config.type, config

    seen = _seen or ()
    if monitor_config.ref in seen:
        # Report the ordered traversal path with the repeated ref appended so the
        # cycle is obvious (e.g. "a -> b -> a"); seen is kept in visit order.
        cycle = " -> ".join((*seen, monitor_config.ref))
        raise ValueError(
            f"Monitor ref cycle detected at '{monitor_config.ref}' (chain: {cycle})"
        )

    parent = _load_monitor_file(monitor_config.ref)
    base_type, base_config = resolve_monitor_config(
        parent, seen + (monitor_config.ref,)
    )
    if monitor_config.type and base_type and monitor_config.type != base_type:
        raise ValueError(
            f"Monitor ref '{monitor_config.ref}' has type '{base_type}', which "
            f"differs from the referring monitor's type '{monitor_config.type}';"
            " a monitor may only reference another of the same type."
        )

    overlay = deepcopy(monitor_config.config or {})
    extra_event_configs = overlay.pop("extra_event_configs", [])
    if "event_configs" in overlay:
        # merge_dicts replaces lists wholesale, so an overlay event_configs would
        # discard the referenced monitor's artifact rules (and their binding
        # payload) rather than extend them — a silent, hard-to-spot breakage.
        # Appending is the only sensible overlay operation; enforce it.
        raise ValueError(
            f"Monitor ref '{monitor_config.ref}' overlay sets 'event_configs', "
            "which would replace the referenced monitor's rules wholesale and "
            "drop its artifact bindings. Use 'extra_event_configs' to append "
            "rules, or define an inline monitor to replace them entirely."
        )
    merged = merge_dicts(base_config, overlay)
    if extra_event_configs:
        # `or []` guards a base monitor written with `event_configs:` (null) —
        # merged.get returns None (key present), and list(None) would TypeError.
        merged["event_configs"] = list(merged.get("event_configs") or []) + list(
            extra_event_configs
        )
    resolved_type = monitor_config.type or base_type
    if resolved_type is None:
        # No level of the ref chain set a type. check_type_or_ref forces the
        # refless root to have one, so this is unreachable for validated files —
        # but raise rather than return None so a bypassed/relaxed chain fails here
        # (surfaced by build validation) instead of at run time.
        raise ValueError(
            f"Monitor ref '{monitor_config.ref}' resolves to no monitor type; a "
            "monitor in its ref chain must set 'type'."
        )
    return resolved_type, merged


TARGETRUNS_KEY = "targetruns"
BINDINGS_KEY = "bindings"
RUN_METADATA_KEY = "run_metadata"
SETUP_CONFIG = "setup_config"
CONFIG_KEY = "config"
LAUNCHER_CONFIG = "launcher_config"
LAUNCHER_KEY = "launchers"
MONITOR_CONFIG = "monitor_config"
ENVIRONMENT_CONFIG = "environment_config"


class TargetStepRun(Run):
    """Run of a single build target step."""

    targetrun_id: str
    launch_id: str

    def __init__(
        self: Self,
        target: Target,
        targetstep: TargetStep,
        targetrun_id: str,
        event_q: Queue,
        additional_targetsteps_queue: Optional[Queue] = None,
        bindings: Optional[Dict] = None,
        setup_config: dict = None,  # type: ignore[assignment]
        dry_run: bool = False,
    ) -> None:
        try:
            self.target = target
            self.targetrun_id = targetrun_id
            self.launch_id = ""
            super().__init__(
                entity=targetstep,
                event_q=event_q,
                base_dir=targetstep.build_workspace_dir / TARGETRUNS_KEY,
                dry_run=dry_run,
            )
            self.bindings = bindings

            # Full config (without RUN_METADATA and BINDINGS) and MERGED unfilled step-default + step + build config from targetstep
            self.full_config = deepcopy(targetstep.full_config)

            # if step.yaml exists -> merging is already done in targetstep -> do a deepcopy only -> no need to merge again
            merged_step_build_config_unfilled = {}
            if targetstep.is_step_file_exists:
                # unfilled step config + build config
                merged_step_build_config_unfilled = deepcopy(
                    targetstep._merged_step_build_config_unfilled
                )

            # run-specific data
            targetsteprun_runtime_config = {
                BINDINGS_KEY: self.bindings,
                RUN_METADATA_KEY: dataclasses.asdict(self.get_runmetadata()),
                SETUP_CONFIG: setup_config,
            }

            # add the runtime config into the full config
            self.full_config.update(targetsteprun_runtime_config)

            logger.info(
                "FULL CONFIG RUN_METADATA_KEY: %s", self.full_config[RUN_METADATA_KEY]
            )

            logger.info("FULL CONFIG STEP NAME: %s", self.full_config["step"])

            # At this stage, full config CONTAINS RUN_METADATA and BINDINGS + MERGED unfilled step-default + step + build config from targetstep

            # ====== Final filling of config templates after runtime data into the full config ======
            filled_inner = fill_objtemplate(
                merged_step_build_config_unfilled,
                self.full_config,
                strict=False,
                skip_keys={"field_value_template"},
            )
            if isinstance(filled_inner, dict):
                if self.dry_run:
                    logger.info("dry_run: setting mock to True")
                    filled_inner_gb = filled_inner.get("gb", {})
                    filled_inner_gb["mock"] = True
                    filled_inner["gb"] = filled_inner_gb
                    if USE_LESS_COMPUTE_ON_DRY_RUN:
                        if "compute_config" in filled_inner:
                            filled_inner_compute_config = filled_inner["compute_config"]
                            if "num_nodes" in filled_inner_compute_config:
                                if filled_inner_compute_config["num_nodes"] > 2:
                                    filled_inner_compute_config["num_nodes"] = 2
                            if "num_gpus_per_node" in filled_inner_compute_config:
                                if filled_inner_compute_config["num_gpus_per_node"] > 2:
                                    filled_inner_compute_config["num_gpus_per_node"] = 2
                            if "num_cpus_per_node" in filled_inner_compute_config:
                                if (
                                    filled_inner_compute_config["num_cpus_per_node"]
                                    > 16
                                ):
                                    filled_inner_compute_config["num_cpus_per_node"] = (
                                        16
                                    )
                            filled_inner["compute_config"] = filled_inner_compute_config
            else:
                logger.warning(
                    "filled_inner is not a dict: %s %s",
                    type(filled_inner),
                    filled_inner,
                )

            self.full_config[CONFIG_KEY] = filled_inner

            env_config = (
                targetstep.step_environment_config
            )  # returns object of type StepEnvironmentConfig

            logger.info("STEP ENVIRONMENT CONFIG: %s", env_config)
            logger.info(
                "FULL CONFIG ENVIRONMENT CONFIG : %s",
                self.full_config[ENVIRONMENT_CONFIG],
            )
            env_type = targetstep.env_type

            # Populate launcher and monitor config here - different launcher for different runs

            #  --- Launcher config ---
            launchers = env_config.launchers or {}
            if not launchers:
                raise ValueError(f"No launchers found in environment '{env_type}'")

            # Prefer the launcher resolved from the build YAML by targetstep.
            launcher_name = targetstep.launcher_name
            if not launcher_name:
                launcher_name = getattr(env_config, "default_launcher", None)
            if not launcher_name:
                launcher_names = sorted(launchers.keys())
                if not launcher_names:
                    raise ValueError(
                        f"No launchers available in environment '{env_type}'"
                    )
                launcher_name = launcher_names[0]

            if launcher_name not in launchers:
                raise ValueError(
                    f"Failed to find the launcher '{launcher_name}' in {list(launchers.keys())}"
                )

            launcher_cfg = deepcopy(
                launchers[launcher_name]
            )  # returns a StepLauncherConfig object
            filled_launcher_config = fill_objtemplate(
                launcher_cfg, self.full_config, strict=True
            )
            targetstep.launcher = StepLauncherConfig(**filled_launcher_config)

            self.full_config[LAUNCHER_CONFIG] = filled_launcher_config.get(
                CONFIG_KEY, {}
            )
            logger.info(
                "FULL CONFIG LAUNCHER CONFIG: %s", self.full_config[LAUNCHER_CONFIG]
            )

            # --- Monitor Config ---
            # Shared launcher->monitor selection (StepEnvironmentTypeConfig)
            # keeps run-time and build-validation selection identical.
            monitor_pairs, missing_monitors = env_config.select_launcher_monitors(
                launcher_cfg
            )
            if missing_monitors:
                raise ValueError(
                    f"Launcher '{launcher_name}' requires monitor(s) "
                    f"{missing_monitors}, but they are not defined in the "
                    "environment config."
                )
            self.monitors = dict(monitor_pairs)

            filled_monitor_configs = {}
            for name, monitor in self.monitors.items():
                # Resolve any monitor-library reference (ref + overlay/append)
                # to a concrete (type, config) before rendering templates.
                # Normally a cache hit (build-creation validation pre-warms the
                # per-thread monitor cache); it blocks the loop only on a cold
                # cache (e.g. resume) for a git-backed monitor space — consistent
                # with the copytree/template work already done inline here. See
                # _load_monitor_file for the thread-affinity rationale.
                monitor_type, monitor_config = resolve_monitor_config(monitor)
                filled_monitor_config = fill_objtemplate(
                    monitor_config,
                    self.full_config,
                    strict=True,
                    skip_keys={"field_value_template"},
                )
                filled_monitor_configs[name] = {
                    "type": monitor_type,
                    "config": filled_monitor_config,
                }

            new_monitor_configs = {}  # type: ignore[var-annotated]
            for monitor_name, monitor_info in filled_monitor_configs.items():
                monitor_type = monitor_info["type"] or "unknown"
                new_monitor_configs.setdefault(monitor_type, []).append(
                    {
                        "name": monitor_name,
                        "config": monitor_info["config"],
                    }
                )

            self.full_config[MONITOR_CONFIG] = new_monitor_configs
            logger.info(
                "FULL CONFIG MONITOR CONFIG: %s", self.full_config[MONITOR_CONFIG]
            )
            logger.info("FULL CONFIG's CONFIG: %s", self.full_config[CONFIG_KEY])

            # Copy the shared merged_step_dir to a per-run temp directory.
            # fill_templates_in_dir destructively renders Jinja expressions in
            # files; without a copy, a second run of the same step (e.g. from a
            # repeated checkpoint event) finds already-rendered content containing
            # literal {{ }} from monitor field_value_templates and fails.
            source_path = targetstep.merged_step_dir
            temp_path = Path(tempfile.mkdtemp())
            shutil.copytree(source_path, temp_path, dirs_exist_ok=True)

            # Populate merged directory path to pass to the launch
            # in order to copy this final step folder to pod
            self.full_config["merged_dir_path"] = temp_path
            ignore_paths = [
                temp_path / p.relative_to(source_path)
                for p in targetstep.ignore_paths_final_fill
            ]
            self.temp_path = temp_path

            logger.info("Ignoring %d paths during template fill", len(ignore_paths))
            self.ignore_paths = ignore_paths

            fill_templates_in_dir(
                temp_path, self.full_config, ignore_paths=ignore_paths, strict=True
            )

            step_default_yaml = temp_path / "step_default.yaml"
            if step_default_yaml.exists():
                step_default_yaml.unlink()
                logger.info("Removed step_default.yaml before final sync")

            sync_or_copy(str(temp_path) + "/", self.dir, delete=False)
            logger.info("==== Final Step Folder ======: %s", str(temp_path))

        except Exception as e:
            current_err = f"Build `{self.build_id}` Target Step `{self.targetrun_id}` failed on creation."
            full_err_stack = traceback.format_exc()
            status = Status.FAILED
            icon = STATUS_TO_ICON[status]
            msg = f"{icon}  {current_err} error:\n```\n{full_err_stack}\n```\n"
            # logger.error("%s", msg) # TODO: is this necessary?
            run_metadata = self.get_runmetadata()
            payload = BuildEventStatusPayload(status=status, msg=msg)
            fail_event = BuildEvent(
                run_metadata=run_metadata,
                type=BuildEventType.STATUS_EVENT,
                payload=payload,
            )
            self.dispatch_event(fail_event)
            raise ValueError(current_err) from e

    async def _run(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> Any:
        self_entity = self.entity
        assert isinstance(self_entity, TargetStep)
        logger.info("self.full_config: %s", self.full_config)

        build_config = self_entity.config
        if (
            isinstance(build_config, BuildTargetStepConfig)
            and build_config.retry_enabled is not None
        ):
            self.full_config["retry_enabled"] = build_config.retry_enabled
        if (
            isinstance(build_config, BuildTargetStepConfig)
            and build_config.retry_transparently is not None
        ):
            self.full_config["retry_transparently"] = build_config.retry_transparently

        async with TaskGroup() as tg:
            self.launch_task = self_entity.environment.launch(
                launcher_type=self_entity.launcher.type,
                task_group=tg,
                targetsteprun_asset_dir=self.dir,
                setup_ids=list(self.target.setup_ids.keys()),
                **self.full_config,
            )
            launch_id = self.launch_task.launch_id  # type: ignore[attr-defined]
            assert (
                isinstance(launch_id, str) and launch_id != ""
            ), f"invalid launch_id: {launch_id}"
            self.launch_id = launch_id
            monitor_tasks = set()
            # Shared launcher->monitor selection (StepEnvironmentTypeConfig) —
            # same rule as TargetStepRun.__init__ and build-creation validation.
            monitor_pairs, missing_monitors = (
                self_entity.step_environment_config.select_launcher_monitors(
                    self_entity.launcher
                )
            )
            if missing_monitors:
                raise ValueError(
                    f"launcher requires monitor(s) {missing_monitors} not defined "
                    "in the environment config"
                )
            for monitor, monitor_config in monitor_pairs:
                # Resolve any monitor-library reference (ref + overlay/append)
                # to a concrete (type, config) — inline monitors pass through.
                monitor_type, resolved_config = resolve_monitor_config(monitor_config)
                assert (
                    monitor_type is not None
                ), f"monitor '{monitor}' has no resolved type"
                # Render Jinja templates in the monitor config against the full
                # config before passing it to the monitor. fill_objtemplate is
                # non-mutating, so the rendered copy stored in
                # full_config[MONITOR_CONFIG] above never reaches here — without
                # this the monitor receives literal "{{ config.* }}" strings (e.g.
                # log_retrieval.mode). Mirrors the launcher-config fill;
                # field_value_template is skipped so per-log-line templates stay
                # literal for later rendering.
                filled_monitor_config = fill_objtemplate(
                    resolved_config,
                    self.full_config,
                    strict=True,
                    skip_keys={"field_value_template"},
                )
                monitor_tasks.add(
                    self_entity.environment.monitor(
                        type=monitor_type,
                        launch_id=self.launch_id,
                        task_group=tg,
                        event_q=self.event_q,
                        entityrun_metadata=self.get_runmetadata(),
                        build_id=self.build_id,
                        **filled_monitor_config,
                    )
                )
            await asyncio.gather(*monitor_tasks)
            await self.launch_task

    async def _cleanup(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        """Tear down environment resources (e.g. sky.down) after the step finishes or is cancelled.

        Calls the environment's cleanup function directly (e.g. cleanup_skypilot)
        rather than going through environment.cleanup(). This is intentional:
        environment.cleanup() wraps the work in asyncio.ensure_future which creates
        a task that gets abandoned when the build finishes — the build does not wait
        for fire-and-forget futures. By calling cleanup_fn directly, the await blocks
        on asyncio.to_thread(sky.down) which runs in a real OS thread and cannot be
        cancelled, guaranteeing the cluster is torn down.

        The caller (Run.run finally block) uses Task.uncancel() to ensure this
        coroutine can await without CancelledError being raised immediately.
        """
        logger.info(
            "TargetStepRun._cleanup %s start (launch_id=%s)", self.id, self.launch_id
        )
        self_entity = self.entity
        assert isinstance(self_entity, TargetStep)
        if not self.launch_id:
            logger.warning(
                "TargetStepRun._cleanup %s: no launch_id set, skipping environment cleanup",
                self.id,
            )
            return
        env = self_entity.environment
        launch_type = self_entity.launcher.type
        if launch_type in env.cleanup_types:
            logger.info("TargetStepRun._cleanup %s: calling cleanup directly", self.id)
            cleanup_fn = env.cleanup_types[launch_type]
            await cleanup_fn(env, launch_id=self.launch_id)
        else:
            logger.info(
                "TargetStepRun._cleanup %s: no cleanup for launch_type=%s",
                self.id,
                launch_type,
            )
        logger.info("TargetStepRun._cleanup %s end", self.id)

    def get_runmetadata(self: Self) -> EntityRunMetadata:
        self_entity = self.entity
        assert isinstance(self_entity, TargetStep)
        return EntityRunMetadata(
            build_id=self.build_id,
            username=self_entity.username,
            type=type(self_entity).__name__,
            target_name=self_entity.target_name,
            targetrun_id=self.targetrun_id,
            targetsteprun_id=self.id,
            targetstep_uri=self_entity.step_uri,
            target_step_index=self_entity.target_step_index,
        )
