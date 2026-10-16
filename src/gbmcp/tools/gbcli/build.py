import json
import re
import shutil
import tempfile
from typing import Dict

from fastmcp.tools import tool
from fastmcp.utilities.logging import get_logger

from gbcli.client.client import GBClient
from gbmcp.utils.gbserver_errors import actionable_gbserver_errors
from gbmcp.utils.output_filter import apply_output_filters

logger = get_logger(__name__)


@tool(
    description="Return list of builds. show_all=True (default) returns all builds for the target user; without it only currently running builds are returned. Supports output filtering: grep, wc, head, tail."
)
@actionable_gbserver_errors
def build_list(
    space: str | None = None,
    all_space: bool | None = None,
    all_user: bool | None = None,
    show_all: bool = True,
    page_size: int | None = None,
    page_index: int | None = None,
    username: str | None = None,
    tags: list[str] | None = None,
    grep: str | None = None,
    wc: bool | None = None,
    head: int | None = None,
    tail: int | None = None,
) -> str:
    """Return gbcli build list output as JSON.

    Args:
        space: Filter builds by space name.
        all_space: If True, list builds across all spaces.
        all_user: If True, list builds for all users (not just the current user).
        show_all: If True (default), return all builds including completed ones;
            if False, return only currently running builds.
        page_size: Number of builds per page.
        page_index: Zero-based page index for paginated results.
        username: Filter builds by a specific username.
        tags: Filter builds by one or more tags.
        grep: Filter output lines by regex. Supports -Cn, -An, -Bn, -i, -v, -F, -w, -x, -c, -n, -o, -mN flags. Example: "-C2 FAILED".
        wc: If True, return only line and character count instead of full output.
            Use to gauge output size before fetching with head/tail/grep.
        head: Return only the first N lines. Mutually exclusive with tail.
        tail: Return only the last N lines. Mutually exclusive with head.

    Returns:
        JSON array of builds.
    """
    result = GBClient.Build(None).build_list(
        list_all=bool(all_user),
        show_done=bool(show_all),
        show_all=bool(show_all),
        all_spaces=bool(all_space),
        space=space,
        username=username,
        tags=tags,
        page_index=page_index,
        page_size=page_size,
    )
    logger.debug(f"build_list result: {result}")
    output = json.dumps(result, indent=4, default=str)
    return apply_output_filters(
        output, tool_name="build_list", grep=grep, wc=wc, head=head, tail=tail
    )


@tool(
    description="Return details of a build. Supports output filtering: grep, wc, head, tail."
)
@actionable_gbserver_errors
def build_describe(
    build_id: str,
    space: str | None = None,
    raw: bool | None = None,
    grep: str | None = None,
    wc: bool | None = None,
    head: int | None = None,
    tail: int | None = None,
) -> str:
    """Return gbcli build describe output as JSON.

    Args:
        build_id: The build's full UUID. Partial-ID resolution isn't available in
            standalone — pass the full UUID (use build_list to find it).
        space: Space name to scope the lookup.
        raw: If True, return the raw build definition (yaml) instead of parsed JSON.
        grep: Filter output lines by regex. Supports -Cn, -An, -Bn, -i, -v, -F, -w, -x, -c, -n, -o, -mN flags. Example: "-C2 status".
        wc: If True, return only line and character count instead of full output.
        head: Return only the first N lines. Mutually exclusive with tail.
        tail: Return only the last N lines. Mutually exclusive with head.

    Returns:
        JSON object with 'build' and 'targets' fields, or raw YAML/JSON string if raw=True.
    """
    result = GBClient.Build(None).build_describe(
        filename="",
        format="json",
        raw=bool(raw),
        build_id=build_id,
        id_format="uuid",
        space=space,
    )
    logger.debug(f"build_describe result: {result}")
    if raw:
        output = (
            result
            if isinstance(result, str)
            else json.dumps(result, indent=4, default=str)
        )
    else:
        targets, build = result
        output = json.dumps({"build": build, "targets": targets}, indent=4, default=str)
    return apply_output_filters(
        output, tool_name="build_describe", grep=grep, wc=wc, head=head, tail=tail
    )


@tool(
    description="Return current status of a build. Supports output filtering: grep, wc, head, tail."
)
@actionable_gbserver_errors
def build_status(
    build_id: str,
    grep: str | None = None,
    wc: bool | None = None,
    head: int | None = None,
    tail: int | None = None,
) -> str:
    """Return gbcli build status output as JSON.

    Args:
        build_id: The build's full UUID. Partial-ID resolution isn't available in
            standalone — pass the full UUID (use build_list to find it).
        grep: Filter output lines by regex. Supports -Cn, -An, -Bn, -i, -v, -F, -w, -x, -c, -n, -o, -mN flags.
        wc: If True, return only line and character count instead of full output.
        head: Return only the first N lines. Mutually exclusive with tail.
        tail: Return only the last N lines. Mutually exclusive with head.

    Returns:
        JSON object with 'details', 'targets', 'history', and 'error' fields.
    """
    details, targets, history, error = GBClient.Build(None).build_status(
        build_id=build_id,
        quiet=True,
        id_format="uuid",
        show_events=False,
        fetch_pr=False,
        result_format="json",
    )
    logger.debug(f"build_status details: {details}")
    logger.debug(f"build_status targets: {targets}")
    output = json.dumps(
        {"details": details, "targets": targets, "error": error},
        indent=4,
        default=str,
    )
    return apply_output_filters(
        output, tool_name="build_status", grep=grep, wc=wc, head=head, tail=tail
    )


@tool(
    description="Return log of a build. Supports output filtering: grep (regex with context), wc."
)
@actionable_gbserver_errors
def build_log(
    build_id: str,
    runner: bool | None = None,
    all_entries: bool | None = None,
    sort: str | None = None,
    page_size: int | None = None,
    page_index: int | None = None,
    head: int | None = None,
    tail: int | None = None,
    text: str | None = None,
    stream: str | None = None,
    build_step_id: str | None = None,
    build_step_name: str | None = None,
    grep: str | None = None,
    wc: bool | None = None,
) -> str:
    """Return gbcli build log output as text.

    Args:
        build_id: The build's full UUID. Partial-ID resolution isn't available in
            standalone — pass the full UUID (use build_list to find it).
        runner: If True, retrieve deeper build logs from the runner (e.g. step-level
            execution details). By default only the top-level build logs are returned.
        all_entries: If True, return all log entries (ignores pagination).
        sort: Sort order for log entries — 'asc' or 'desc'. Defaults to 'desc'; overridden by head/tail.
        page_size: Number of log entries per page. Overridden by head/tail.
        page_index: Zero-based page index for paginated results.
        head: Return the first N log entries (oldest first); overrides page_size and sets ascending sort.
        tail: Return the last N log entries (newest first); overrides page_size and sets descending sort.
        text: Filter log entries by matching text content (API-level substring filter).
        stream: Filter by log stream — 'stdout' or 'stderr'.
        build_step_id: Filter logs by a specific build step UUID.
        build_step_name: Filter logs by a specific build step name.
        grep: Post-process output by regex. Supports -Cn, -An, -Bn, -i, -v, -F, -w, -x, -c, -n, -o, -mN flags. Example: "-C3 -i error".
            Unlike text, grep supports regex and context lines.
        wc: If True, return only line and character count instead of full output.
    """
    if sum(x is not None for x in (head, tail, all_entries)) > 1:
        raise ValueError("head, tail, and all_entries are mutually exclusive")
    if stream is not None and stream not in ("stdout", "stderr"):
        raise ValueError(f"stream must be 'stdout' or 'stderr', got '{stream}'")
    if sort is not None and sort not in ("asc", "desc"):
        raise ValueError(f"sort must be 'asc' or 'desc', got '{sort}'")

    effective_sort = sort or "desc"
    if head is not None:
        effective_sort = "asc"
        page_size = head
    if tail is not None:
        effective_sort = "desc"
        page_size = tail
    if all_entries is not None:
        effective_sort = "asc"
        page_size = None

    def output_format_plain(logs):
        # Mirrors gbcli's renderer: each record's "text" is a JSON string; keep
        # the records that carry a "log" line. Contentless records (heartbeats or
        # status entries with empty text) are skipped; non-empty text that isn't
        # JSON is a real schema change and is allowed to surface rather than be
        # silently swallowed.
        log_entries = []
        for log in logs:
            text = log.get("text")
            if not text:
                continue
            log_line = json.loads(text).get("log")
            if log_line is None:
                continue
            log_entries.append(re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", log_line))
        return log_entries

    def echo_callback(callback_event: str, callback_args: Dict):
        match callback_event:
            case "display_logs":
                logs = callback_args.get("logs", [])
                log_entries = output_format_plain(logs)
                final_results.extend(log_entries)
            case _:
                pass

    final_results = []
    logs = GBClient.Build(None).build_log(
        build_id=build_id,
        id_format="uuid",
        runner=runner,
        all=bool(all_entries),
        sort=effective_sort,
        page_size=page_size,
        page_index=page_index,
        stream=stream,
        text=text,
        build_step_id=build_step_id,
        build_step_name=build_step_name,
        callback=echo_callback if all_entries else None,
    )

    if not all_entries:
        log_entries = output_format_plain(logs)
        if effective_sort == "desc":
            log_entries.reverse()
        final_results = log_entries

    logger.debug(
        f"build_log result count: {len(final_results) if final_results else 0}"
    )
    output = json.dumps(final_results, indent=4, default=str)
    return apply_output_filters(output, tool_name="build_log", grep=grep, wc=wc)


@tool(description="Start a build from file content.")
@actionable_gbserver_errors
def build_start(
    file_content: str,
    space: str | None = None,
    params: list[str] | None = None,
    tags: list[str] | None = None,
    description: str | None = None,
) -> str:
    """Return gbcli build start output as JSON.

    Args:
        file_content: YAML content of the build file to start.
        space: Space name to submit the build to.
        params: List of parameter overrides in 'key=value' format.
        tags: List of tags to attach to the build.
        description: Description for the build.

    Returns:
        JSON with the newly created build ID.
    """
    tmp_dir = tempfile.mkdtemp(prefix="gbmcp-build-")
    try:
        tmp_file = f"{tmp_dir}/build.yml"
        with open(tmp_file, "w") as f:
            f.write(file_content)

        result = GBClient.Build(None).build_start(
            quiet=True,
            filename=tmp_file,
            space=space,
            params=params or [],
            tags=tags or [],
            description=description or "",
        )
        logger.debug(f"build_start result: {result}")
        return json.dumps(result, indent=4, default=str)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@tool(description="Cancel a build.")
@actionable_gbserver_errors
def build_cancel(build_id: str, space: str | None = None) -> str:
    """Return gbcli build cancel output as JSON.

    Args:
        build_id: The build's full UUID. Partial-ID resolution isn't available in
            standalone — pass the full UUID (use build_list to find it).
        space: Space name to scope the lookup.

    Returns:
        JSON object with the cancellation result.
    """
    result = GBClient.Build(None).build_cancel(
        build_id=build_id,
        id_format="uuid",
        space=space,
    )
    logger.debug(f"build_cancel result: {result}")
    return json.dumps(result, indent=4, default=str)
