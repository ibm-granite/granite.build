# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

# from gbcli import client
import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, List

from services import db_service
from utils import get_gb_binary, get_gb_token, is_gb_enabled, run_command

from .yaml_service import YAMLManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", logging.INFO))

# gbcli is an optional IBM dependency. Importing it (and running
# configureGBWorkingEnv()) at module scope makes gb_service — and therefore the
# whole API — un-importable in environments without it installed (OSS /
# GB-disabled). It is loaded lazily on the first GB API call instead. All public
# GB methods are gated by is_gb_enabled(), so _load_gbcli() is only reached when
# GB is actually configured. The gbcli symbols are bound to module globals on
# first load so the method bodies below reference them by bare name unchanged.
_gbcli_loaded = False


def _load_gbcli() -> None:
    """Lazily import gbcli and bind its symbols to this module's globals.

    Idempotent: the import + configureGBWorkingEnv() side effect runs once.
    """
    global _gbcli_loaded
    global GBSERVER_BUILD_API, BUILD_LOGALL_PAGE_SIZE, get_user_token
    global get_build_status_with_targets_runs, get_build_events
    global gbserver_cancel_build, run_logquery
    global get_current_epoch, change_timestamp_by_days, process_target_runs_to_json
    if _gbcli_loaded:
        return

    from gbcli.utils.cli_config import configureGBWorkingEnv

    configureGBWorkingEnv()

    # These gbcli imports must follow configureGBWorkingEnv() so the working env
    # is set up before the modules are loaded (same ordering as the original
    # module-scope imports, now expressed as ordering within this function).
    from gbcli.services.service_build import process_target_runs_to_json
    from gbcli.utils.gbconstants import BUILD_LOGALL_PAGE_SIZE, GBSERVER_BUILD_API
    from gbcli.utils.gbcredentials import get_user_token
    from gbcli.utils.gbserver import cancel_build as gbserver_cancel_build
    from gbcli.utils.gbserver import (
        get_build_events,
        get_build_status_with_targets_runs,
    )
    from gbcli.utils.log_query import run_logquery
    from gbcli.utils.utils import change_timestamp_by_days, get_current_epoch

    _gbcli_loaded = True


class GBService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.db = db_service.Database()
        self._status_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl: int = int(os.getenv("GB_STATUS_CACHE_TTL", "45"))
        self._max_cache_size: int = int(os.getenv("GB_STATUS_CACHE_MAX_SIZE", "50"))

    async def get_gb_logs_by_job_id(self, job_id: str, fetch_all: bool = False):
        if not is_gb_enabled():
            return {"message": "GB Token not found"}
        try:
            task = await self.db.get_task_by_job_id(job_id=job_id, type="TUNING")
            if task is not None and (
                task["build_id"] is not None or task["pr_url"] is not None
            ):
                build_id = task.get("build_id", task.get("pr_url"))
                return await asyncio.to_thread(
                    self._fetch_gb_logs, str(build_id), fetch_all
                )
            else:
                return {"message": "job not found"}
        except Exception as e:
            logger.error(
                f"Exception occured in get_gb_logs_by_job_id: {e}", exc_info=True
            )
            return "Something went wrong"

    async def get_gb_logs(self, build_id: str, fetch_all: bool = False):
        if not is_gb_enabled():
            return {"message": "GB Token not found"}
        try:
            return await asyncio.to_thread(
                self._fetch_gb_logs, str(build_id), fetch_all
            )
        except Exception as e:
            logger.error(f"Exception occured in get_gb_logs: {e}", exc_info=True)
            return "Something went wrong"

    # Safety cap on the time-window pagination loop to prevent runaway fetches.
    _MAX_GB_LOG_PAGES = 50

    def _fetch_gb_logs(self, build_id: str, fetch_all: bool = False) -> list:
        """Fetch build logs directly from gbserver log query API (sync, runs in thread pool).

        When fetch_all is False (default), returns only the first page (up to
        BUILD_LOGALL_PAGE_SIZE lines) — preserves the legacy fast path.
        When fetch_all is True, paginates via time-window advancement (advance
        start_epoch_in_s to the last log's timestamp) and dedupes by logId,
        matching the approach used by gbcli's service_admin.
        """
        _load_gbcli()
        user_token = get_user_token()
        if not user_token:
            raise Exception("GB credentials not found. Please login.")

        current_epoch = get_current_epoch()
        start_epoch = change_timestamp_by_days(current_epoch, 7)

        def extract_lines(log_entries):
            out = []
            for log_entry in log_entries:
                try:
                    log_json = json.loads(log_entry["text"])
                    log_msg = log_json.get("log")
                    if log_msg is not None:
                        out.append(log_msg)
                    else:
                        out.append("<null>")
                except (json.JSONDecodeError, KeyError):
                    out.append(log_entry.get("text", ""))
            return out

        if not fetch_all:
            response = run_logquery(
                user_token,
                start_epoch_in_s=start_epoch,
                end_epoch_in_s=current_epoch,
                page_size=BUILD_LOGALL_PAGE_SIZE,
                page_index=0,
                sort="asc",
                build_id=build_id,
            )
            if not response or not response.get("logs"):
                return []
            return extract_lines(response["logs"])

        log_lines: list = []
        seen_log_ids: set = set()
        window_start = start_epoch  # seconds
        pages = 0
        while pages < self._MAX_GB_LOG_PAGES:
            response = run_logquery(
                user_token,
                start_epoch_in_s=window_start,
                end_epoch_in_s=current_epoch,
                page_size=BUILD_LOGALL_PAGE_SIZE,
                page_index=0,
                sort="asc",
                build_id=build_id,
            )
            pages += 1
            entries = (response or {}).get("logs") or []
            if not entries:
                break

            new_entries = []
            last_timestamp_ms = None
            for entry in entries:
                log_id = entry.get("logId")
                if log_id is not None:
                    if log_id in seen_log_ids:
                        continue
                    seen_log_ids.add(log_id)
                new_entries.append(entry)
                ts = entry.get("timestamp")
                if ts is not None and (
                    last_timestamp_ms is None or ts > last_timestamp_ms
                ):
                    last_timestamp_ms = ts

            log_lines.extend(extract_lines(new_entries))

            # If this page didn't fill, we've drained the window — stop.
            if len(entries) < BUILD_LOGALL_PAGE_SIZE:
                break
            # Advance the time window. Log timestamps are in ms; run_logquery
            # expects seconds. Overlap is handled by logId dedup above.
            if last_timestamp_ms is None:
                break
            next_window_start = int(last_timestamp_ms / 1000)
            if next_window_start <= window_start:
                next_window_start = window_start + 1
            window_start = next_window_start

        if pages >= self._MAX_GB_LOG_PAGES:
            logger.warning(
                "Hit GB logs pagination safety cap (%d pages) for build_id=%s",
                self._MAX_GB_LOG_PAGES,
                build_id,
            )

        return log_lines

    async def get_gb_status(self, build_id) -> Dict[str, Any]:
        """Get GB build status via direct gbcli API calls (no subprocess)."""
        if not is_gb_enabled():
            return {"message": "GB Token not found"}

        cache_key = str(build_id)

        # Fast path: return cached result if still fresh
        with self._cache_lock:
            cached = self._status_cache.get(cache_key)
            if cached and (time.monotonic() - cached["timestamp"]) < self._cache_ttl:
                logger.debug("GB status cache hit for build_id: %s", build_id)
                return cached["result"]

        # Slow path: fetch directly from gbserver API
        try:
            resp = await asyncio.to_thread(self._fetch_gb_status, str(build_id))

            with self._cache_lock:
                self._status_cache[cache_key] = {
                    "result": resp,
                    "timestamp": time.monotonic(),
                }
                if len(self._status_cache) > self._max_cache_size:
                    self._cleanup_stale_entries()

            return resp
        except Exception as e:
            logger.error(f"Exception occured in get_gb_status: {e}", exc_info=True)
            return {"error": str(e)}

    def _fetch_gb_status(self, build_id: str) -> Dict[str, Any]:
        """Fetch build status directly from gbserver API (sync, runs in thread pool)."""
        _load_gbcli()
        user_token = get_user_token()

        # 1. Fetch build status with target runs
        raw = get_build_status_with_targets_runs(
            user_token, build_id, GBSERVER_BUILD_API
        )
        status_data = raw["status"]

        # 2. Build details dict (matches CLI JSON output shape)
        build_details = {
            "build_id": status_data["build"]["uuid"],
            "name": status_data["build"]["name"],
            "started_at": status_data["build"]["created_time"],
            "updated_at": status_data["build"]["updated_time"],
            "status": status_data["build"]["status"],
            "source_pr": status_data["build"]["source_uri"],
            "description": status_data["build"]["description"],
        }

        # 3. Process targets — sanitize None started_at values that
        #    cause fromisoformat() to fail inside gbcli
        target_runs = status_data.get("target_runs", [])
        for tr in target_runs:
            if tr.get("target", {}).get("started_at") is None:
                tr["target"]["started_at"] = "0001-01-01T00:00:00"
            for step in tr.get("steps", []):
                if step.get("started_at") is None:
                    step["started_at"] = "0001-01-01T00:00:00"
        targets = process_target_runs_to_json(target_runs)

        # 4. Fetch build events
        build_history = []
        try:
            events_resp = get_build_events(build_id, user_token, GBSERVER_BUILD_API)
            build_history = [
                {
                    "time": event["build_event"]["timestamp"],
                    "description": event["build_event"]["payload"]
                    .get("msg", "")
                    .replace("`", ""),
                }
                for event in events_resp.get("events", [])
                if event["build_event"]["payload"].get("msg", "") != ""
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch build events for {build_id}: {e}")

        return {
            "details": build_details,
            "targets": targets,
            "build_history": build_history,
        }

    async def cancel_gb_build(self, build_id: str) -> Dict[str, Any]:
        """Cancel a GB build via direct API call (no subprocess)."""
        if not is_gb_enabled():
            return {"message": "GB Token not found"}
        try:
            result = await asyncio.to_thread(self._cancel_build, str(build_id))
            return result
        except Exception as e:
            logger.error(f"Exception occured in cancel_gb_build: {e}", exc_info=True)
            raise

    def _cancel_build(self, build_id: str) -> Dict[str, Any]:
        """Cancel a build directly via gbserver API (sync, runs in thread pool)."""
        _load_gbcli()
        user_token = get_user_token()
        return gbserver_cancel_build(build_id, user_token, GBSERVER_BUILD_API)

    def invalidate_status_cache(self, build_id: str) -> None:
        """Force-invalidate a cached GB status entry."""
        with self._cache_lock:
            self._status_cache.pop(str(build_id), None)

    def _cleanup_stale_entries(self) -> None:
        """Remove stale cache entries. Must be called while _cache_lock is held."""
        now = time.monotonic()
        stale_threshold = self._cache_ttl * 3
        stale_keys = [
            k
            for k, v in self._status_cache.items()
            if (now - v["timestamp"]) > stale_threshold
        ]
        for k in stale_keys:
            del self._status_cache[k]
        if stale_keys:
            logger.debug("Cleaned up %d stale GB status cache entries", len(stale_keys))

    def cleanup_stale_cache(self) -> None:
        """Remove cache entries older than 3x TTL to prevent memory leaks."""
        with self._cache_lock:
            self._cleanup_stale_entries()

    def load_yaml(self):
        try:
            yaml_file = YAMLManager("autotune-test/build.yaml")
            config = yaml_file.read_yaml()
            print(config["granite.build"]["targets"]["custom"]["inputs"])
            return config
            # with open(yaml_file, 'r') as file:
            #     config = yaml.safe_load(file) or {}
            #     return config
        except FileNotFoundError as e:
            config = {}
            logger.error(f"FileNotFoundError occured in load_yaml: {e}", exc_info=True)
            return f"Some Error {e}"
        except Exception as e:
            logger.error(f"Exception occured in load_yaml: {e}", exc_info=True)
            return f"Other Error {e}"

    def create_yaml(self, config):
        try:
            yaml_file = YAMLManager("api/autotune_yaml/new.yaml")
            yaml_file.create_empty_yaml()
            yaml_file.write_yaml(config)

            return {"message": "success", "config": config}
            # with open(yaml_file, 'r') as file:
            #     config = yaml.safe_load(file) or {}
            #     return config
        except FileNotFoundError as e:
            logger.error(
                f"FileNotFoundError occured in create_yaml: {e}", exc_info=True
            )
            return f"Some Error {e}"
        except Exception as e:
            logger.error(f"Exception occured in create_yaml: {e}", exc_info=True)
            return f"Other Error {e}"

    def login_gb(self):
        # is_gb_enabled() already requires the gb/llmb CLI on PATH, so a set
        # token with no binary lands here — warn actionably instead of letting
        # subprocess raise a bare FileNotFoundError.
        if not is_gb_enabled():
            if get_gb_token():
                logger.warning(
                    "GB_TOKEN is set but the Granite Build CLI (gb/llmb) is not "
                    "on PATH; GB features are disabled. Install the "
                    "'granite.build' extra or unset GB_TOKEN to silence this."
                )
            return
        try:
            binary = get_gb_binary()
            command = [binary, "auth", "login", "--token", get_gb_token()]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            logger.error(
                f"gb auth login failed (exit {e.returncode}): {e.stderr}",
                exc_info=True,
            )
            return f"Some Error {e}"
        except Exception as e:
            logger.error(f"Exception occured in login_gb: {e}", exc_info=True)
            return f"Some Error {e}"

    async def command_executor(self, command: List[str]):
        try:
            logger.debug("Start command_executor function")
            # Resolve gb vs its llmb alias; fall back to "gb" so the command
            # (and its error) is still coherent if the binary vanished post-check.
            binary = get_gb_binary() or "gb"
            cmd_str = binary + " " + " ".join(command)
            result = await run_command(cmd_str)
            logger.debug(f"Raw output: {result}")
            # Check if the command execution was successful (exit code 0 means success)
            if result.get("code") != 0:
                error_message = result.get("stderr", "No error output")
                logger.error(
                    f"Command execution failed with exit code {result.get('code')}"
                )
                logger.error(f"Error: {error_message}")

                raise Exception(error_message)

            logger.debug("End command_executor function")
            return result.get("stdout")

        except Exception as e:
            logger.error(f"command_executor exception: {str(e)}", exc_info=True)
            raise

    def sanitize_output(self, output):
        cleaned_output = re.sub(r"\x1b\[[0-9;]*m", "", output)
        cleaned_output = cleaned_output.strip()

        # Try to parse the cleaned output
        try:
            json_data = json.loads(cleaned_output)
            return {"status": "success", "data": json_data}

        except json.JSONDecodeError as j:
            logger.error(
                f"JSONDecodeError error occured in sanitize_output: {j}", exc_info=True
            )
            # If that fails, try to decode unicode escapes first
            try:
                decoded_output = cleaned_output.encode().decode("unicode_escape")
                cleaned_again = re.sub(r"\x1b\[[0-9;]*m", "", decoded_output)
                json_data = json.loads(cleaned_again)
                return {"status": "success", "data": json_data}

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"Error occured in sanitize_output: {e}", exc_info=True)
                return {
                    "status": "partial_success",
                    "message": output.strip().split("\n"),
                    "error": f"Could not parse as JSON: {str(e)}",
                }
