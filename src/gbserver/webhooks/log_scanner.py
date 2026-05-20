"""Regex-based build log scanner for webhook LOG_EVENT generation.

Scans MESSAGE_EVENT payloads (already in-process via the BuildRunner event queue)
against subscriber-defined patterns. No external log backend access required.

Future enhancement: add cloud_logs mode using IBM Cloud Logs API search pattern
from gb_dashboard/cloud_logs.py for full pod stdout coverage.
"""

import re
from typing import Any, Dict, List

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def scan_log_lines(
    lines: List[str],
    pattern: str,
    start_line_number: int = 1,
) -> List[Dict[str, Any]]:
    """Scan log lines for regex matches and return structured results.

    Args:
        lines: List of log line strings to scan.
        pattern: Regex pattern to match against each line.
        start_line_number: Line number offset for the first line in the batch.

    Returns:
        List of match dicts with keys: line, line_number, matched_pattern.
        Returns empty list if pattern is invalid regex.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        logger.warning("[LogScanner] Invalid regex pattern '%s': %s", pattern, e)
        return []

    matches = []
    for i, line in enumerate(lines):
        if compiled.search(line):
            matches.append(
                {
                    "line": line,
                    "line_number": start_line_number + i,
                    "matched_pattern": pattern,
                }
            )

    return matches
