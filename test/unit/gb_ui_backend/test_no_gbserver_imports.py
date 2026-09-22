"""``gb_ui_backend`` must not import ``gbserver``.

``granite-build-analytics`` ships ``gb_ui_backend`` and ``gbcommon`` and
deliberately **not** ``gbserver``, so any import of it works in this tree and
fails wherever that distribution is installed.

That is not hypothetical: `services/gbserver_source.py` imported
``check_zip_safe`` from ``gbserver.utils.archive``, and in the hosted deployment
the Data Processing scan died with ``No module named 'gbserver'``. It went
unnoticed for a long time because the call site swallowed the ImportError, so the
archive silently never decoded and the page reported no datasets rather than an
error.

A static check, because an import error only shows up when the code path runs, in
an environment nobody develops in.
"""

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[3] / "src" / "gb_ui_backend"


def _source_files():
    for path in PACKAGE.rglob("*.py"):
        # Stale build artefacts vendor whole trees; they are not shipped.
        if "build/lib" in path.as_posix():
            continue
        yield path


def test_no_module_imports_gbserver():
    offenders = []
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text())
        except (
            SyntaxError
        ):  # pragma: no cover - a broken file is another test's problem
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "gbserver" or alias.name.startswith("gbserver."):
                        offenders.append(
                            f"{path.name}:{node.lineno} import {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "gbserver" or mod.startswith("gbserver."):
                    offenders.append(f"{path.name}:{node.lineno} from {mod}")

    assert not offenders, (
        "gb_ui_backend imports gbserver, which granite-build-analytics does not "
        "ship — these will raise ModuleNotFoundError wherever it is installed:\n  "
        + "\n  ".join(offenders)
        + "\nMove the shared code into gbcommon instead."
    )
