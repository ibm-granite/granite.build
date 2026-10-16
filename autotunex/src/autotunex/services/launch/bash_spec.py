# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Pure assembly of a granite.build local-bash ``build.yaml`` from a job.

Ports the 2025 ``utilites.granite_build_bash.BashLocalBuild``: the
``space://environments/bash`` + ``space://steps/autotune`` shape used to run
AutoTune locally (MPS/CPU on Apple Silicon). HuggingFace models only.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml

from autotunex.services.launch.protocols import LaunchContext

_DEFAULT_SERVER_URL = "http://localhost:8001"
"""Fallback base URL a local-bash worker reports back to when no callback is set."""


def build_bash_spec(
    ctx: LaunchContext,
    *,
    fm_tune_root: str | None,
    fm_tune_ref: str | None,
    fm_tune_extra: str,
    backend: str,
    callback_url: str | None,
    output_uri_root: str,
    num_nodes: int = 1,
) -> str:
    """Return the local-bash ``build.yaml`` text for ``ctx``.

    HuggingFace-only model: emitted as ``hf:///<model>`` (three slashes, the
    local-bash convention for a bare HF repo id). The dataset is referenced by
    ``ctx.dataset_uri`` (the dataset's ``artifact_url``) verbatim — the builder is
    scheme-agnostic: an ``hf://`` artifact is pulled by gbserver, and, in
    standalone mode, a local ``file://`` directory (what upload persists there;
    see the 2026-08-25 standalone-dataset-upload spec) is mounted by the same-host
    gbserver. A caller must ensure it is set.
    ``config_data`` is embedded verbatim as ``autotune-config`` (deep-copied so the
    caller's object is never mutated). No reward-function injection here — mirror
    2025 (the ``local`` runner handles online RL).

    Args:
        ctx: the launch inputs assembled from the job and its snapshot.
        fm_tune_root: ``FM_TUNE_ROOT`` for ``bash.env`` (the fm-tune checkout/repo).
        fm_tune_ref: ``FM_TUNE_REF`` for ``bash.env`` (the fm-tune branch/tag/commit
            to check out; ``None`` leaves the repo's default branch, emitted as null).
        fm_tune_extra: ``FM_TUNE_EXTRA`` for ``bash.env`` (extras to install).
        backend: ``BACKEND`` for ``bash.env`` (``mlx`` on Apple Silicon, or ``torch``).
        callback_url: base URL a worker reports back to; emitted as
            ``AUTOTUNEX_SERVER_URL``, defaulting to :data:`_DEFAULT_SERVER_URL` when unset.
        output_uri_root: root URI the run's artifacts are written under; the output
            is emitted as ``<root>/autotune_<job-id>/``. Pass an **absolute** URI
            (e.g. ``file:///data/artifacts``): gbserver resolves a *relative* ``file:``
            URI against its own process CWD, which in the container is the ephemeral
            image path — so a relative root silently strands artifacts off the ``/data``
            volume. The registry anchors this under ``settings.artifact_dir``.
        num_nodes: ``compute_config.num_nodes`` for the step (local runs use 1).

    Returns:
        The rendered ``build.yaml`` document as text.
    """
    config_data = copy.deepcopy(ctx.config_data)
    env = {
        "FM_TUNE_ROOT": fm_tune_root,
        "FM_TUNE_REF": fm_tune_ref,
        "FM_TUNE_EXTRA": fm_tune_extra,
        "BACKEND": backend,
        "NO_AUTOTUNE": str(not ctx.autotune).lower(),
        "RUN_NAME": ctx.experiment_name,
        "JOB_ID": str(ctx.job_id),
        "AUTOTUNEX_SERVER_URL": callback_url or _DEFAULT_SERVER_URL,
    }
    output_uri = f"{output_uri_root.rstrip('/')}/autotune_{ctx.job_id}/"
    build: dict[str, Any] = {
        "granite.build": {
            "name": f"custom-model-{ctx.experiment_name}",
            "version": "0.0.1",
            "targets": {
                "custom": {
                    "environment_uri": "space://environments/bash",
                    "inputs": {
                        "model": {"uri": f"hf:///{ctx.model}"},
                        "dataset_files": {"uri": ctx.dataset_uri},
                    },
                    "outputs": {"custom": {"uri": output_uri}},
                    "steps": [
                        {
                            "step_uri": "space://steps/autotune",
                            "config": {
                                "bash": {"env": env},
                                "compute_config": {"num_nodes": num_nodes},
                                "autotune-config": config_data,
                            },
                        }
                    ],
                }
            },
        }
    }
    return yaml.dump(build, default_flow_style=False, sort_keys=False)
