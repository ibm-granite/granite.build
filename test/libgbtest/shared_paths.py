# SPDX-License-Identifier: Apache-2.0

"""Paths on a test cluster's shared ``/shared`` volume, for build tests that clean up.

Build tests whose steps write to a fixed ``/shared`` path clear it before they run, so
a re-run doesn't find the previous run's output. The removal itself is cluster
specific (``libgbtest.kube.remove_from_shared`` for kind,
``integration.ibm.buildrunner.skypilot.slurm_local.cluster.remove_from_shared`` for the
Docker SLURM cluster); the path check, and the lookup of a test's directory from its
fixture, are shared here, so they can't drift apart.
"""

import os
from pathlib import Path
from typing import Any

import yaml

SHARED_ROOT = "/shared"


def require_below_shared(path: str) -> str:
    """``path``, normalized, if it is strictly below ``/shared/``.

    :param path: An absolute path.
    :returns: The normalized path.
    :raises ValueError: For ``/shared`` itself, a relative path, or one that
        leaves ``/shared`` or is spelled unusually (``..``, ``//``, ``.``).
    """
    norm = os.path.normpath(path)
    if not norm.startswith(SHARED_ROOT + "/") or norm != path.rstrip("/"):
        raise ValueError(f"not a path strictly below {SHARED_ROOT}/: {path!r}")
    return norm


def step_output_dir(build_yaml: Path, target: str, step_index: int, *keys: str) -> str:
    """The directory holding a step's output, from the step's config in ``build_yaml``.

    Reads the output path at ``granite.build.targets.<target>.steps[<step_index>]
    .config.<keys...>`` and returns its parent: the test's own directory, which also
    holds whatever the step leaves beside its output. Read from the fixture, so test
    and build.yaml can't drift apart.

    :param build_yaml: The fixture's build.yaml.
    :param target: The target name.
    :param step_index: The step's position in the target.
    :param keys: Keys under the step's ``config`` leading to the output path.
    :returns: The parent directory, checked by :func:`require_below_shared`.
    """
    build = yaml.safe_load(Path(build_yaml).read_text(encoding="utf-8"))
    value: Any = build["granite.build"]["targets"][target]["steps"][step_index][
        "config"
    ]
    for key in keys:
        value = value[key]
    return require_below_shared(os.path.dirname(value))
