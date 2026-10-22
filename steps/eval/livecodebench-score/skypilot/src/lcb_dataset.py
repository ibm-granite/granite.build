#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Point LiveCodeBench's dataset loader at a local copy of the dataset.

LiveCodeBench loads its benchmark by Hugging Face name, hard-coded in
``lcb_runner/benchmarks/code_generation.py``::

    load_dataset("livecodebench/code_generation_lite", split="test",
                 version_tag=release_version, trust_remote_code=True)

There is no option for a local path. A granite.build step receives the dataset
as a declared ``hf://`` input, already downloaded by a separate hfpull step, and
runs offline. So :func:`install` wraps ``datasets.load_dataset`` and maps that one
name to the downloaded directory. Every other name, and every other argument,
passes through unchanged, and LiveCodeBench itself is not modified.

The directory must be a copy of the dataset repo: the loading script
(``code_generation_lite.py``) plus the JSONL files the release needs.

The loader is pointed at the **script file**, not the directory. Given a
directory, ``datasets`` looks at the repo's ``README.md`` first: the real repo's
card has YAML front matter, and with it present ``datasets`` treats the directory
as plain JSON data and ignores the script, failing with ``JsonConfig ... doesn't
have a 'version_tag' key``. A directory copied by hfpull carries that README; a
hand-made copy without one loads either way. Found on the first GPU run.

It also fixes memory use, in two places (measured on ``release_v6``, a window of
six problems, Linux, ``python:3.12-slim``):

* **Building the Arrow cache** on first load. ``datasets`` writes rows in batches of
  1000 by default, so it holds every row of a JSONL file at once; single rows are
  up to 93 MB. That peaked at 14.6 GB resident. :func:`install` passes
  ``writer_batch_size=1`` (:data:`WRITER_BATCH_SIZE`), which peaked at 0.74 GB and
  was no slower. Later loads read the cache and peak at about 0.17 GB.
* **Decoding.** LiveCodeBench builds a ``CodeGenerationProblem`` for every row,
  which decodes that row's compressed private test cases, and only then applies
  the date window (measured: 21.9 GB on macOS). :func:`install` replaces
  ``load_code_generation_dataset`` with one that filters the raw rows by
  ``contest_date`` first and decodes only the rows it keeps. The result is the
  same list of problems, in the same order; only the order of the two operations
  changes.

Call :func:`install` before LiveCodeBench imports its benchmark module, or call
it after: it patches the name the module bound at import time as well.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

#: The dataset name LiveCodeBench hard-codes for the code generation scenario.
HF_NAME = "livecodebench/code_generation_lite"

#: The loading script a usable dataset directory must contain.
LOADING_SCRIPT = "code_generation_lite.py"

#: Rows per Arrow write while ``datasets`` builds its cache. Its default (1000)
#: buffers whole JSONL files whose rows reach 93 MB each; see the module docstring.
WRITER_BATCH_SIZE = 1


def check_dataset_dir(path: str) -> Path:
    """Return ``path`` as a directory that holds the dataset's loading script.

    :param path: the downloaded dataset directory.
    :raises ValueError: when it is missing, not a directory, or has no loading script.
    """
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"dataset directory not found: {path}")
    if not (p / LOADING_SCRIPT).is_file():
        raise ValueError(
            f"{path} has no {LOADING_SCRIPT}; expected a copy of the {HF_NAME} dataset repo"
        )
    return p


def install(dataset_dir: str) -> None:
    """Make ``load_dataset(HF_NAME, ...)`` read ``dataset_dir`` instead of the Hub.

    Idempotent: installing twice wraps the original loader once.

    :param dataset_dir: the downloaded dataset directory (see the module docstring).
    """
    import datasets

    # The script itself, not the directory: see the module docstring.
    local = str(check_dataset_dir(dataset_dir) / LOADING_SCRIPT)
    real: Callable[..., Any] = getattr(
        datasets.load_dataset, "_lcb_real", datasets.load_dataset
    )

    def load_dataset(path: Any, *args: Any, **kwargs: Any) -> Any:
        if path == HF_NAME:
            path = local
            kwargs.setdefault("writer_batch_size", WRITER_BATCH_SIZE)
        return real(path, *args, **kwargs)

    load_dataset._lcb_real = real  # type: ignore[attr-defined]
    datasets.load_dataset = load_dataset

    # LiveCodeBench did `from datasets import load_dataset`, so its module holds its
    # own reference to the original function. Patch that one too.
    import lcb_runner.benchmarks.code_generation as code_generation

    code_generation.load_dataset = load_dataset
    code_generation.load_code_generation_dataset = _windowed_loader(code_generation)

    # Other LiveCodeBench modules bound the original function by name at import time
    # (`from lcb_runner.benchmarks import load_code_generation_dataset`). Rebind it in
    # every one already imported, so the scorer and the generate step both get it.
    import sys

    for name, module in list(sys.modules.items()):
        if (
            name.startswith("lcb_runner")
            and getattr(module, "load_code_generation_dataset", None) is not None
        ):
            if module is not code_generation:
                module.load_code_generation_dataset = (
                    code_generation.load_code_generation_dataset
                )


def _windowed_loader(code_generation: Any) -> Callable[..., Any]:
    """Return a ``load_code_generation_dataset`` that filters before it decodes.

    Same signature and same result as LiveCodeBench's own; see the module docstring.
    """
    from datetime import datetime

    def load_code_generation_dataset(
        release_version: str = "release_v1",
        start_date: Any = None,
        end_date: Any = None,
    ) -> list:
        ds = code_generation.load_dataset(
            HF_NAME, split="test", version_tag=release_version, trust_remote_code=True
        )
        lo = (
            datetime.strptime(start_date, "%Y-%m-%d")
            if start_date is not None
            else None
        )
        hi = datetime.strptime(end_date, "%Y-%m-%d") if end_date is not None else None
        if lo is not None or hi is not None:

            def keep(contest_date: str) -> bool:
                d = datetime.fromisoformat(contest_date)
                return (lo is None or lo <= d) and (hi is None or d <= hi)

            ds = ds.filter(keep, input_columns="contest_date")
        problems = [code_generation.CodeGenerationProblem(**p) for p in ds]
        print(f"Loaded {len(problems)} problems")
        return problems

    return load_code_generation_dataset


def install_from_env(var: str = "LCB_DATASET_DIR") -> None:
    """Call :func:`install` with the directory named by environment variable ``var``.

    :raises ValueError: when the variable is unset or empty.
    """
    value = os.environ.get(var, "").strip()
    if not value:
        raise ValueError(f"{var} is not set")
    install(value)
