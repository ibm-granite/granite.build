# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Pure assembly of a granite.build ``build.yaml`` from a job.

Ports the 2025 ``gb_runner`` submission path: ``GraniteBuild.create_default_build``
(the ``targets.custom`` tree) and ``build_start_cmd`` (the ``python main.py``
invocation), rendered under the top-level ``granite.build`` key exactly as the
cluster's ``llmb build start -f`` expects. The whole per-trial configuration is
embedded verbatim as a ``k8s.additional_files`` literal block, not flattened into
CLI flags.

External model-catalogue inputs are deliberately omitted — this service launches only
HuggingFace and custom-path models (``model_source``), so there is no
``model_to_tune`` input and no ``lh://`` output. If the installed cluster's schema
differs, that change is localized to this module — the seam and runner do not move
(see the design's risk section).
"""

from __future__ import annotations

import copy
from typing import Any

import yaml

from autotunex.services.launch._shared import (
    REWARD_FUNCTION_PATH,
    BlockStringDumper,
    inject_reward_function,
    num_gpus_per_node,
    start_command,
)
from autotunex.services.launch.protocols import LaunchContext

__all__ = ["REWARD_FUNCTION_PATH", "build_spec"]

_ENVIRONMENT_URI = "space://environments/{{ space.variables.DEFAULT_ENVIRONMENT }}"
"""Space environment binding injected verbatim into every build (create_default_build)."""

_STEP_URI = "space://steps/custom_code"

# Compute defaults ported from gb_runner.update_compute_config: CPUs and node
# count are fixed (32 CPUs errored the cluster at 64), only num_gpus_per_node is
# derived from the configuration.
_NUM_CPUS_PER_NODE = 32
_NUM_NODES = 1
_TOTAL_MEMORY_PER_NODE = "256Gi"


def build_spec(
    ctx: LaunchContext,
    *,
    runtime_image: str,
    trainer_repo: str,
    trainer_ref: str,
    output_uri_root: str,
    callback_url: str | None,
) -> str:
    """Return the ``build.yaml`` text for ``ctx`` as a granite.build submission.

    The configuration is embedded verbatim as an ``additional_files`` literal block
    at ``/tmp/<config_name>.yaml`` (spaces in the name become underscores, and the
    same path is used for ``--config_file`` so the two never diverge). For online RL
    (a reward function is present) the config's ``reward_function_path`` — and
    ``reward_function_name`` when named — are rewritten to point at the injected
    :data:`REWARD_FUNCTION_PATH`, matching the 2025 runner. ``config_data`` is
    deep-copied first so the caller's object is never mutated.

    Args:
        ctx: the launch inputs assembled from the job and its snapshot.
        runtime_image: container image the cluster runs the build in (``k8s.image``).
        trainer_repo: trainer source repository (``custom_code_config.github_url``).
        trainer_ref: branch/tag/commit checked out in ``setup_command``.
        output_uri_root: root URI for run artifacts; the output is
            ``<root>/autotunex_<first-8-of-job-id>/``.
        callback_url: base URL a cluster worker reports back to; emitted as
            ``--autotunex_server_url`` only when set.
    """
    config_data = copy.deepcopy(ctx.config_data)

    config_file_path = f"/tmp/{ctx.config_name.replace(' ', '_')}.yaml"
    additional_files = inject_reward_function(config_data, ctx)

    embedded_config = yaml.dump(
        config_data, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper
    )
    files = {config_file_path: embedded_config, **additional_files}

    setup_command = (
        f'git checkout {trainer_ref} && pip install -e ".[full]" && pip list && nvidia-smi'
    )
    output_uri = f"{output_uri_root.rstrip('/')}/autotunex_{ctx.job_id.hex[:8]}/"

    build: dict[str, Any] = {
        "granite.build": {
            "name": f"autotunex-{ctx.experiment_name}",
            "targets": {
                "custom": {
                    "environment_uri": _ENVIRONMENT_URI,
                    "inputs": {"dataset_files": {"uri": ctx.dataset_uri}},
                    "outputs": {"custom": {"uri": output_uri}},
                    "steps": [
                        {
                            "step_uri": _STEP_URI,
                            "config": {
                                "custom_code_config": {
                                    "github_url": trainer_repo,
                                    "setup_command": setup_command,
                                    "start_command": start_command(
                                        ctx,
                                        config_file=config_file_path,
                                        callback_url=callback_url,
                                        cuda_home="/usr/local/cuda-13.0",
                                    ),
                                    "dir_to_save": ".",
                                },
                                "k8s": {"image": runtime_image, "additional_files": files},
                                "compute_config": {
                                    "num_gpus_per_node": num_gpus_per_node(config_data),
                                    "num_cpus_per_node": _NUM_CPUS_PER_NODE,
                                    "num_nodes": _NUM_NODES,
                                    "total_memory_per_node": _TOTAL_MEMORY_PER_NODE,
                                },
                            },
                        }
                    ],
                }
            },
        }
    }
    return yaml.dump(build, Dumper=BlockStringDumper, default_flow_style=False, sort_keys=False)
