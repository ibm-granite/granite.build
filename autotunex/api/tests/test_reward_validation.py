# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import ast

from services.reward_validation import (
    _analyze_security,
    _check_syntax,
    _execute_safely,
    _validate_function,
)


def test_check_syntax_valid_and_invalid():
    ok, errors, tree = _check_syntax("def compute_score(d, s):\n    return 1.0\n")
    assert ok is True and errors == [] and tree is not None
    bad_ok, bad_errors, bad_tree = _check_syntax("def f(:")
    assert bad_ok is False and bad_errors and bad_tree is None


def test_analyze_security_blocks_forbidden_import():
    issues = _analyze_security(ast.parse("import os\n"))
    assert issues  # non-empty => flagged os as a blocked module


def test_analyze_security_clean_for_safe_code():
    issues = _analyze_security(
        ast.parse("def compute_score(d, s):\n    return len(s)\n")
    )
    assert issues == []


def test_validate_function_arity():
    tree = ast.parse("def compute_score(data_source, solution_str):\n    return 1.0\n")
    found, sig_valid, errors = _validate_function(tree, "compute_score")
    assert found is True and sig_valid is True and errors == []
    found2, sig2, errs2 = _validate_function(tree, "missing")
    assert found2 is False


def test_execute_safely_runs_safe_function():
    code = "def compute_score(data_source, solution_str):\n    return 1.0\n"
    out = _execute_safely(
        code, "compute_score", {"data_source": "x", "solution_str": "y"}
    )
    assert out["executed"] is True
    assert out["error"] is None
