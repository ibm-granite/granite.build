# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure reward static-analysis helpers."""

from __future__ import annotations

from autotunex.services.reward.static_analysis import (
    analyze_security,
    check_syntax,
    validate_function,
)


def test_check_syntax_accepts_valid_code() -> None:
    ok, errors, tree = check_syntax("def compute_score(a, b):\n    return 1.0\n")

    assert ok is True
    assert errors == []
    assert tree is not None


def test_check_syntax_reports_a_syntax_error() -> None:
    ok, errors, tree = check_syntax("def broken(:\n")

    assert ok is False
    assert tree is None
    assert errors and "Syntax error" in errors[0]


def test_analyze_security_flags_a_blocked_import() -> None:
    _, _, tree = check_syntax("import os\n")
    assert tree is not None

    issues = analyze_security(tree)

    assert any("os" in issue for issue in issues)


def test_analyze_security_flags_a_dunder_escape() -> None:
    _, _, tree = check_syntax("def f(a, b):\n    return a.__class__\n")
    assert tree is not None

    issues = analyze_security(tree)

    assert any("__class__" in issue for issue in issues)


def test_validate_function_accepts_two_args() -> None:
    _, _, tree = check_syntax("def compute_score(data_source, solution_str):\n    return 1.0\n")
    assert tree is not None

    found, sig_ok, errors = validate_function(tree, "compute_score")

    assert found is True
    assert sig_ok is True
    assert errors == []


def test_validate_function_rejects_too_few_args() -> None:
    _, _, tree = check_syntax("def compute_score(only_one):\n    return 1.0\n")
    assert tree is not None

    found, sig_ok, _errors = validate_function(tree, "compute_score")

    assert found is True
    assert sig_ok is False


def test_validate_function_reports_missing_function() -> None:
    _, _, tree = check_syntax("def other(a, b):\n    return 1.0\n")
    assert tree is not None

    found, _sig_ok, _errors = validate_function(tree, "compute_score")

    assert found is False
