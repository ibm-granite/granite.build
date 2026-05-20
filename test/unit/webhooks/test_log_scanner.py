"""Unit tests for the webhook log scanner module."""

from gbserver.webhooks.log_scanner import scan_log_lines


class TestLogScanner:
    """Tests for scan_log_lines function."""

    def test_scan_finds_matching_lines(self):
        """Verify regex matches return correct lines and line numbers."""
        lines = [
            "2026-05-20 INFO Starting step",
            "2026-05-20 ERROR Connection refused",
            "2026-05-20 INFO Retrying...",
            "2026-05-20 ERROR Timeout exceeded",
        ]
        matches = scan_log_lines(lines, pattern=r"(?i)error")
        assert len(matches) == 2
        assert matches[0]["line"] == "2026-05-20 ERROR Connection refused"
        assert matches[0]["line_number"] == 2
        assert matches[1]["line"] == "2026-05-20 ERROR Timeout exceeded"
        assert matches[1]["line_number"] == 4

    def test_scan_returns_empty_for_no_matches(self):
        """Verify empty list returned when no lines match the pattern."""
        lines = ["INFO all good", "INFO still good"]
        matches = scan_log_lines(lines, pattern=r"FATAL")
        assert matches == []

    def test_scan_with_invalid_regex_returns_empty(self):
        """Verify invalid regex gracefully returns empty list."""
        lines = ["some text"]
        matches = scan_log_lines(lines, pattern=r"[invalid")
        assert matches == []

    def test_scan_includes_pattern_in_result(self):
        """Verify each match dict includes the matched_pattern field."""
        lines = ["Traceback (most recent call last):"]
        matches = scan_log_lines(lines, pattern=r"Traceback")
        assert matches[0]["matched_pattern"] == r"Traceback"

    def test_scan_with_custom_start_line_number(self):
        """Verify start_line_number offsets the reported line numbers."""
        lines = ["ERROR something"]
        matches = scan_log_lines(lines, pattern=r"ERROR", start_line_number=50)
        assert matches[0]["line_number"] == 50
