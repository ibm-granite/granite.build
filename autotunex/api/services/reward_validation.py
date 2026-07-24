# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import ast
import builtins as builtins_module
import contextlib
import io as io_module
import json
import logging
import re
import threading
import time

logger = logging.getLogger(__name__)

# ── Reward function validation helpers ──────────────────────────────────────

BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "signal",
    "socket",
    "http",
    "urllib",
    "requests",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "shelve",
    "marshal",
    "importlib",
    "pathlib",
    "glob",
    "tempfile",
    "io",
    "builtins",
    "code",
    "codeop",
    "compileall",
    "py_compile",
    "pty",
    "pipes",
    "resource",
    "sysconfig",
    "platform",
    "webbrowser",
    "antigravity",
    "turtle",
}

BLOCKED_BUILTINS = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
}

BLOCKED_DUNDERS = {
    "__import__",
    "__builtins__",
    "__subclasses__",
    "__class__",
    "__bases__",
    "__globals__",
    "__code__",
    "__closure__",
    "__dict__",
    "__module__",
    "__qualname__",
}

SAFE_BUILTINS = {
    "abs",
    "all",
    "any",
    "ascii",
    "bin",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "id",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "True",
    "False",
    "None",
    "type",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AttributeError",
    "RuntimeError",
    "ZeroDivisionError",
    "StopIteration",
    "NotImplementedError",
}

DEFAULT_TEST_INPUTS = {
    "data_source": "Solve: What is the capital of France?",
    "solution_str": "The capital of France is Paris.",
}


def _check_syntax(code: str):
    """Parse code and return (is_valid, errors, ast_tree_or_None)."""
    try:
        tree = ast.parse(code)
        return True, [], tree
    except SyntaxError as e:
        return False, [f"Syntax error at line {e.lineno}: {e.msg}"], None


def _analyze_security(tree: ast.AST):
    """Walk AST and return list of human-readable security issue strings."""
    issues = []
    for node in ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_MODULES or top.startswith("_"):
                    issues.append(
                        f"Forbidden import: '{alias.name}' (line {node.lineno})"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in BLOCKED_MODULES or top.startswith("_"):
                    issues.append(
                        f"Forbidden import from: '{node.module}' (line {node.lineno})"
                    )

        # Block dangerous builtin calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_BUILTINS:
                issues.append(
                    f"Forbidden call: '{node.func.id}()' (line {node.lineno})"
                )

        # Block dunder attribute access
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_DUNDERS:
                issues.append(
                    f"Forbidden attribute access: '.{node.attr}' (line {node.lineno})"
                )
    return issues


def _validate_function(tree: ast.AST, function_name: str):
    """Check function exists and has correct arity. Returns (found, sig_valid, errors)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            num_args = len(node.args.args)
            # Accept 2+ positional args (extra kwargs/defaults are fine)
            if num_args >= 2:
                return True, True, []
            else:
                return (
                    True,
                    False,
                    [
                        f"Function '{function_name}' should accept at least 2 parameters "
                        f"(data_source, solution_str), found {num_args}"
                    ],
                )
    return False, False, [f"Function '{function_name}' not found in the code"]


def _execute_safely(
    code: str, function_name: str, test_inputs: dict, timeout_seconds: int = 5
):
    """Execute reward function in restricted namespace with timeout."""
    import collections
    import functools
    import itertools
    import math
    import string

    safe_builtins_dict = {}
    for name in SAFE_BUILTINS:
        if hasattr(builtins_module, name):
            safe_builtins_dict[name] = getattr(builtins_module, name)

    # Python internals needed by exec() to define functions/classes.
    # User-level calls to __import__ are already blocked by the AST security check,
    # but the runtime itself needs these to execute def statements and comprehensions.
    safe_builtins_dict["__build_class__"] = builtins_module.__build_class__
    safe_builtins_dict["__name__"] = "__main__"

    # Provide a restricted __import__ that only allows our pre-approved safe modules
    ALLOWED_EXEC_MODULES = {
        "math",
        "re",
        "json",
        "string",
        "collections",
        "functools",
        "itertools",
        "typing",
        "dataclasses",
        "enum",
        "decimal",
        "fractions",
        "statistics",
        "operator",
        "copy",
        "numbers",
        "abc",
        "textwrap",
        "difflib",
        "unicodedata",
    }

    def _restricted_import(name, *args, **kwargs):
        if name.split(".")[0] in ALLOWED_EXEC_MODULES:
            return builtins_module.__import__(name, *args, **kwargs)
        raise ImportError(f"Import of '{name}' is not allowed")

    safe_builtins_dict["__import__"] = _restricted_import

    restricted_globals = {
        "__builtins__": safe_builtins_dict,
        "math": math,
        "re": re,
        "json": json,
        "string": string,
        "collections": collections,
        "functools": functools,
        "itertools": itertools,
    }

    stdout_capture = io_module.StringIO()
    results_holder = [None]
    error_holder = [None]

    # Support both single test case (dict) and multiple test cases (list of dicts)
    if isinstance(test_inputs, list):
        test_cases = test_inputs
    else:
        test_cases = [test_inputs]

    def target():
        try:
            compiled = compile(code, "<reward_function>", "exec")
            exec(compiled, restricted_globals)
            fn = restricted_globals.get(function_name)
            if fn is None:
                error_holder[0] = (
                    f"Function '{function_name}' not found after execution"
                )
                return
            case_results = []
            with contextlib.redirect_stdout(stdout_capture):
                for i, case in enumerate(test_cases[:10]):  # Limit to 10 cases
                    if not isinstance(case, dict):
                        case_results.append(
                            {
                                "case": i + 1,
                                "inputs": case,
                                "return_value": None,
                                "return_type": None,
                                "error": f"Test case {i + 1} must be a JSON object, got {type(case).__name__}",
                            }
                        )
                        continue
                    try:
                        kwargs = dict(case)
                        rv = fn(**kwargs)
                        serializable = isinstance(
                            rv, (int, float, str, bool, list, dict, type(None))
                        )
                        case_results.append(
                            {
                                "case": i + 1,
                                "inputs": case,
                                "return_value": rv if serializable else str(rv),
                                "return_type": type(rv).__name__,
                                "error": None,
                            }
                        )
                    except Exception as e:
                        case_results.append(
                            {
                                "case": i + 1,
                                "inputs": case,
                                "return_value": None,
                                "return_type": None,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
            results_holder[0] = case_results
        except Exception as e:
            error_holder[0] = f"{type(e).__name__}: {e}"

    start = time.time()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    elapsed_ms = round((time.time() - start) * 1000, 1)

    if thread.is_alive():
        return {
            "executed": False,
            "results": [],
            "stdout": "",
            "error": f"Execution timed out after {timeout_seconds}s",
            "execution_time_ms": elapsed_ms,
        }

    if error_holder[0]:
        return {
            "executed": True,
            "results": [],
            "stdout": stdout_capture.getvalue()[:2000],
            "error": error_holder[0],
            "execution_time_ms": elapsed_ms,
        }

    return {
        "executed": True,
        "results": results_holder[0] or [],
        "stdout": stdout_capture.getvalue()[:2000],
        "error": None,
        "execution_time_ms": elapsed_ms,
    }


# ── End reward function validation helpers ──────────────────────────────────
