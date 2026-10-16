# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Pure, in-process static analysis of a reward function's source.

Three independent phases — syntax, security, signature — run before any
sandboxed execution is even attempted. Each is a pure function over an AST so
it needs no subprocess and no I/O, and each is ported from the 2025 validator
(``_check_syntax``, ``_analyze_security``, ``_validate_function``) with the
leading underscore dropped since these are now this module's public surface.
"""

from __future__ import annotations

import ast

from autotunex.services.reward.constants import BLOCKED_BUILTINS, BLOCKED_DUNDERS, BLOCKED_MODULES


def check_syntax(code: str) -> tuple[bool, list[str], ast.AST | None]:
    """Parse ``code`` and report whether it is syntactically valid Python.

    Returns:
        A ``(is_valid, errors, tree)`` triple; ``tree`` is ``None`` on failure.
    """
    try:
        tree = ast.parse(code)
        return True, [], tree
    except SyntaxError as exc:
        return False, [f"Syntax error at line {exc.lineno}: {exc.msg}"], None


def analyze_security(tree: ast.AST) -> list[str]:
    """Walk ``tree`` for blocked imports, builtin calls, and dunder access.

    This is the security boundary's static half: anything it flags here never
    reaches the sandbox at all, regardless of what the sandbox's own import
    whitelist would additionally catch at runtime.

    Returns:
        Human-readable issue strings, one per finding; empty if none.
    """
    issues: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_MODULES or top.startswith("_"):
                    issues.append(f"Forbidden import: '{alias.name}' (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BLOCKED_MODULES or top.startswith("_"):
                    issues.append(f"Forbidden import from: '{node.module}' (line {node.lineno})")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                issues.append(f"Forbidden call: '{node.func.id}()' (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr in BLOCKED_DUNDERS:
            issues.append(f"Forbidden attribute access: '.{node.attr}' (line {node.lineno})")
    return issues


def validate_function(tree: ast.AST, function_name: str) -> tuple[bool, bool, list[str]]:
    """Check that ``function_name`` is defined with the required 2-argument arity.

    The reward contract is ``(data_source, solution_str, ...)``; extra kwargs or
    defaults are fine, but fewer than two positional parameters cannot satisfy
    the caller.

    Returns:
        A ``(found, signature_valid, errors)`` triple.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            num_args = len(node.args.args)
            if num_args >= 2:
                return True, True, []
            return (
                True,
                False,
                [
                    f"Function '{function_name}' should accept at least 2 parameters "
                    f"(data_source, solution_str), found {num_args}"
                ],
            )
    return False, False, [f"Function '{function_name}' not found in the code"]
