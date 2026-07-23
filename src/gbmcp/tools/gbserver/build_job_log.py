"""Standalone MCP tool for reading a build's real stdout/traceback log.

In standalone mode there is no gbserver REST file-browsing surface. Instead, a
standalone build writes its workload's combined stdout/stderr to a `job.log` on
the local filesystem, under the granite.build home directory:

    <GB_HOME_DIR or ~/.granite.build>/workdir/llm-build-<build_id>/
        target-<t>/target-run-<tr>/step-<s>/step-run-<sr>/launch-<l>/outputs/job.log

That `job.log` is the primary debug artifact: its success signature is
`workload script finished successfully` plus `INFERENCE_SUCCESS` / `^RESPONSE:`,
and its failure signature is `workload script failed, exit code: <N>`. This is
what an agent reads to see what actually happened when a build "succeeded but
did nothing" or failed.

Only registered in standalone mode (see the removal in `utils/lifespan.py`).
"""

import glob
import json
import os
import re

from fastmcp.tools import tool
from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


def tail_lines(path: str, lines: int) -> list[str]:
    """Return the last `lines` lines of `path` (trailing newlines stripped); [] if missing."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read().splitlines()
    except FileNotFoundError:
        return []
    return content[-lines:] if lines > 0 else content


def gb_home_dir() -> str:
    """The granite.build home dir that `workdir/` lives under.

    Overridden by `GB_HOME_DIR`; defaults to `~/.granite.build` (gbserver's own
    default).
    """
    return os.environ.get("GB_HOME_DIR", os.path.expanduser("~/.granite.build"))


def _safe_mtime(path: str) -> float:
    """mtime for sorting; if a job.log is rotated/removed between glob and sort,
    sort it last instead of crashing the whole call."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


@tool(
    description=(
        "Read a standalone build's job.log -- the real combined stdout/stderr of the "
        "build's workload (tracebacks, INFERENCE_SUCCESS / RESPONSE lines, 'workload "
        "script failed, exit code: N'). This is the standalone replacement for the "
        "build_files_* tools: it globs "
        "<GB_HOME_DIR or ~/.granite.build>/workdir/llm-build-<build_id>/**/outputs/job.log. "
        "A build has one job.log per step/launch; pass step to filter to /step-<step>/, "
        "otherwise the most-recently-modified job.log is tailed and the rest are listed "
        "in other_logs. Use this when a build 'succeeded but did nothing' or failed. "
        "Supply the full build UUID for a precise match. lines defaults to 80."
    )
)
def build_job_log(build_id: str, lines: int = 80, step: str | None = None) -> str:
    """Locate and tail a standalone build's job.log.

    Args:
        build_id: The build's UUID (as it appears in the `llm-build-<build_id>`
            workdir). Supply the full UUID for a precise match; a prefix may
            match multiple builds.
        lines: Number of trailing lines of the chosen job.log to return
            (default 80).
        step: If given, only consider job.logs under `/step-<step>/`.

    Returns:
        JSON: {"job_log_path": str|None, "tail": [...], "other_logs": [...],
        "message": str}. If no job.log is found, job_log_path is null and
        message explains where it looked.
    """
    build_id = build_id.strip()
    if not re.fullmatch(r"[0-9a-fA-F-]+", build_id):
        return json.dumps(
            {
                "job_log_path": None,
                "tail": [],
                "other_logs": [],
                "message": (
                    f"Invalid build_id {build_id!r}: expected a build UUID "
                    "(hex digits and hyphens only)."
                ),
            },
            indent=4,
        )
    home = gb_home_dir()
    build_root = os.path.join(home, "workdir", f"llm-build-{build_id}")
    pattern = os.path.join(build_root + "*", "**", "outputs", "job.log")
    matches = glob.glob(pattern, recursive=True)

    if step is not None:
        matches = [m for m in matches if f"{os.sep}step-{step}{os.sep}" in m]

    if not matches:
        step_note = f" under /step-{step}/" if step is not None else ""
        message = (
            f"No job.log found for build {build_id}{step_note}. Looked under "
            f"{build_root}*/**/outputs/job.log. The build may not have produced "
            "a workload log yet, the build_id may be wrong, or GB_HOME_DIR may "
            "differ from where gbserver wrote it."
        )
        logger.debug(f"build_job_log: no matches for pattern {pattern}")
        return json.dumps(
            {
                "job_log_path": None,
                "tail": [],
                "other_logs": [],
                "message": message,
            },
            indent=4,
        )

    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = matches[0]
    other_logs = matches[1:]
    tail = tail_lines(chosen, lines)

    message = f"Last {len(tail)} line(s) of {chosen}."
    if other_logs:
        message += (
            f" {len(other_logs)} other job.log(s) exist for this build "
            "(see other_logs; pass step= to select one)."
        )
    logger.debug(
        f"build_job_log: chose {chosen} ({len(other_logs)} others) for {build_id}"
    )
    return json.dumps(
        {
            "job_log_path": chosen,
            "tail": tail,
            "other_logs": other_logs,
            "message": message,
        },
        indent=4,
    )
