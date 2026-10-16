# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""The reward sandbox's security contract: blocklists, whitelists, and limits.

Ported verbatim from the 2025 in-process validator
(``api/services/reward_validation.py``) — these member sets *are* the security
boundary for untrusted reward code, so they are not "tuned", only carried
forward. The limits (``MAX_*``, ``DEFAULT_*``) are new: the 2025 code had no
code-size cap and an un-killable thread-based timeout.
"""

from __future__ import annotations

BLOCKED_MODULES: set[str] = {
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
"""Top-level modules a reward function may never import (or import-from)."""

BLOCKED_BUILTINS: set[str] = {
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
"""Builtin names a reward function may never call directly."""

BLOCKED_DUNDERS: set[str] = {
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
"""Dunder attribute names a reward function may never access (sandbox escapes)."""

SAFE_BUILTINS: set[str] = {
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
"""The whitelist of builtins the sandbox namespace exposes to reward code."""

ALLOWED_EXEC_MODULES: set[str] = {
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
    # NOTE: ``operator`` is deliberately NOT allowed. Its ``attrgetter`` /
    # ``methodcaller`` are ``getattr``/method-call equivalents that take the
    # attribute name as a *string*, so they bypass both the removed ``getattr``
    # builtin and the AST dunder blocklist (which only sees ``ast.Attribute``
    # nodes, never string constants) — a reliable ``object.__subclasses__()``
    # -> ``__globals__`` -> ``__import__`` escape. Do not re-add it.
    "copy",
    "numbers",
    "abc",
    "textwrap",
    "difflib",
    "unicodedata",
}
"""Modules the sandbox's restricted ``__import__`` allows a reward function to import."""

MAX_CODE_BYTES = 50_000
"""Reward source is rejected above this size, before any parsing is attempted."""

MAX_TEST_CASES = 10
"""Only the first N test cases are executed per validation request."""

MAX_STDOUT_CHARS = 2_000
"""Captured stdout is truncated to this many characters in the response."""

DEFAULT_TIMEOUT_SECONDS = 5
"""Wall-clock seconds before the sandbox child is hard-killed."""

DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
"""Default ``RLIMIT_AS`` cap applied to the sandbox child (best-effort on macOS)."""
