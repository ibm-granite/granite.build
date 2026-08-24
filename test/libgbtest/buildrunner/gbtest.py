# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""``gbtest`` CLI: run a single buildtest.yaml via the pytest harness.

Installed as a console script via ``[project.scripts]`` in pyproject.toml::

    gbtest path/to/buildtest.yaml [extra pytest args...]

``main()`` invokes ``pytest.main(...)`` against the sibling
``gbtest_runner.py`` (which defines ``TestYamlRunnerCli``) with
``--buildtest-yaml`` set to the supplied path.  Any extra args after the
YAML path are forwarded to pytest (e.g. ``-k test_runner_cancellation`` to
pick a single test method, ``-vv`` for more verbose output).

This module is intentionally kept free of test-infrastructure imports so
that ``main()`` can run pytest in-process without prematurely loading
``lib.test_utils`` (which would freeze ``GBSERVER_GITHUB_TOKEN`` to its
pre-secrets value before ``pytest_sessionstart`` fires).  The heavy imports
live in ``gbtest_runner.py`` and are only triggered when pytest collects
that file — i.e. AFTER sessionstart.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytest


def _split_build_yaml_flag(extra: List[str]) -> Tuple[Optional[str], List[str]]:
    """Pull a ``-f <path>`` / ``--build-yaml <path>`` override out of the run args.

    ``pytest`` has no ``-f`` option, so the gbtest CLI translates it into the
    ``--build-yaml`` pytest option (see ``main``). Returns
    ``(override_path_or_None, remaining_args)``.

    Raises:
        ValueError: if ``-f`` / ``--build-yaml`` is given without a path (e.g. a
            trailing bare ``-f``) — a usage error rather than a silent no-op.
    """
    override: Optional[str] = None
    rest: List[str] = []
    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg in ("-f", "--build-yaml"):
            if i + 1 < len(extra):
                override = extra[i + 1]
                i += 2
                continue
            raise ValueError(f"{arg} requires a build.yaml path")
        if arg.startswith("--build-yaml="):
            override = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1
    return override, rest


def _render_main(args: list[str]) -> int:
    """``gbtest render <build.yaml> [-o out]`` — print/write a skeleton buildtest.yaml.

    Kept out of ``main`` so the generator (and its gbcommon deps) is imported
    lazily, preserving this module's import-light property.
    """
    import argparse

    from libgbtest.buildrunner.buildtest_gen import generate_skeleton

    parser = argparse.ArgumentParser(prog="gbtest render")
    parser.add_argument("build_yaml")
    parser.add_argument("-o", "--out")
    ns = parser.parse_args(args)
    bp = Path(ns.build_yaml)
    if not bp.is_file():
        sys.stderr.write(f"gbtest render: not a file: {bp}\n")
        return 1
    text = generate_skeleton(bp)
    if ns.out:
        Path(ns.out).write_text(text, encoding="utf-8")
        sys.stderr.write(f"wrote {ns.out}\n")
    else:
        sys.stdout.write(text)
    return 0


def main() -> int:
    """Entry point for the ``gbtest`` console script.

    Returns:
        Exit code from pytest (0 for success, non-zero for failure).
    """
    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: gbtest path/to/buildtest.yaml [extra pytest args...]\n"
            "       gbtest render path/to/build.yaml [-o out]\n"
        )
        return 2
    if sys.argv[1] == "render":
        if len(sys.argv) < 3:
            sys.stderr.write("Usage: gbtest render path/to/build.yaml [-o out]\n")
            return 2
        return _render_main(sys.argv[2:])
    yaml_path = Path(sys.argv[1]).resolve()
    if not yaml_path.is_file():
        sys.stderr.write(f"gbtest: not a file: {yaml_path}\n")
        return 1
    extra = sys.argv[2:]
    try:
        build_yaml_override, extra = _split_build_yaml_flag(extra)
    except ValueError as e:
        sys.stderr.write(f"gbtest: {e}\n")
        return 2
    runner_module = Path(__file__).resolve().parent / "gbtest_runner.py"
    args = ["-s", str(runner_module), f"--buildtest-yaml={yaml_path}", *extra]
    if build_yaml_override:
        args.append(f"--build-yaml={Path(build_yaml_override).resolve()}")
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
