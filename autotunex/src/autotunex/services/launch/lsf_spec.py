# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Pure assembly of a granite.build LSF/SkyPilot ``build.yaml`` from a job.

Ports the ``recipes/autotunex/lsf/tune-4gpu`` recipe — the LSF counterpart of the
custom_code build — expressed with the ``space://steps/autotunex-tune`` step, a
``skypilot:`` config block, an ``env://`` output on the shared filesystem, and
GPUs declared under ``launcher_config.resources.accelerators``. Reward-function
injection, the ``main.py`` start command, the ``num_gpus_per_node`` derivation and
the literal-block YAML dumper are shared with the custom_code builder via
``services.launch._shared``. HuggingFace and custom-path models only.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml

from autotunex.services.launch._shared import (
    BlockStringDumper,
    inject_reward_function,
    num_gpus_per_node,
    start_command,
)
from autotunex.services.launch.protocols import LaunchContext

_STEP_URI = "space://steps/autotunex-tune"
_OUTPUT_URI = "env://{{ binding.path }}"
"""Output on the shared FS: the step writes under ``$GB_BUILD_WORKDIR`` and the
``env://`` push is a no-op, so results stay where the cluster wrote them."""


def build_lsf_spec(
    ctx: LaunchContext,
    *,
    environment_uri: str,
    image: str,
    venv_path: str,
    cuda_home: str,
    trainer_repo: str,
    trainer_ref: str,
    cluster: str,
    queue: str | None,
    accelerators: str | None,
    memory: str | None,
    num_cpus_per_node: int,
    total_memory_per_node: str,
    poll_interval_seconds: int,
    callback_url: str | None,
) -> str:
    """Return the LSF/SkyPilot ``build.yaml`` text for ``ctx``.

    The configuration is embedded verbatim as a ``skypilot.additional_files``
    literal block at ``/tmp/<config_name>.yaml`` (spaces become underscores; the
    same path is passed to ``--config_file``). For online RL a reward function is
    injected and its config paths rewritten (see
    :func:`~autotunex.services.launch._shared.inject_reward_function`).
    ``config_data`` is deep-copied first so the caller's object is never mutated.

    ``accelerators``, ``queue`` (emitted as ``zone``) and ``memory`` are omitted
    from ``launcher_config.resources`` when ``None`` — omitting ``accelerators``
    yields a 0-GPU build. ``num_gpus_per_node`` in ``compute_config`` is derived
    from the config for parity with the custom_code build; on LSF the authoritative
    GPU request is ``accelerators``.

    Args:
        ctx: launch inputs assembled from the job and its snapshot.
        environment_uri: granite.build space environment (``environment_uri``).
        image: runtime container image (``skypilot.image``).
        venv_path: ``skypilot.venv_path`` the runtime image installs deps into.
        cuda_home: CUDA toolkit path exported in the start command.
        trainer_repo: trainer source repo (``custom_code_config.github_url``).
        trainer_ref: branch/tag/commit (``github_ref`` and the ``git checkout``).
        cluster: SkyPilot/LSF cluster (``resources.cluster``).
        queue: LSF queue, emitted as ``resources.zone`` when set.
        accelerators: SkyPilot accelerators string; omitted when ``None`` (0-GPU).
        memory: ``resources.memory``; omitted when ``None``.
        num_cpus_per_node: ``compute_config.num_cpus_per_node``.
        total_memory_per_node: ``compute_config.total_memory_per_node``.
        poll_interval_seconds: the step's poll and log-retrieval intervals.
        callback_url: base URL a worker reports back to; emitted as
            ``--autotunex_server_url`` only when set.
    """
    config_data = copy.deepcopy(ctx.config_data)

    config_file_path = f"/tmp/{ctx.config_name.replace(' ', '_')}.yaml"
    reward_files = inject_reward_function(config_data, ctx)
    embedded_config = yaml.dump(
        config_data, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper
    )
    additional_files = {config_file_path: embedded_config, **reward_files}

    setup_command = (
        f'git checkout {trainer_ref} && pip install -e ".[full]" && pip list && nvidia-smi'
    )

    resources: dict[str, str] = {"cluster": cluster}
    if accelerators:
        resources["accelerators"] = accelerators
    if queue:
        resources["zone"] = queue
    if memory:
        resources["memory"] = memory

    build: dict[str, Any] = {
        "granite.build": {
            "name": f"autotunex-{ctx.experiment_name}",
            "targets": {
                "autotunex-tune": {
                    "environment_uri": environment_uri,
                    "inputs": {"dataset_files": {"uri": ctx.dataset_uri}},
                    "outputs": {"checkpoint": {"uri": _OUTPUT_URI, "type": "model"}},
                    "steps": [
                        {
                            "step_uri": _STEP_URI,
                            "config": {
                                "poll_interval_seconds": poll_interval_seconds,
                                "log_retrieval_interval_seconds": poll_interval_seconds,
                                "custom_code_config": {
                                    "github_url": trainer_repo,
                                    "github_ref": trainer_ref,
                                    "setup_command": setup_command,
                                    "start_command": start_command(
                                        ctx,
                                        config_file=config_file_path,
                                        callback_url=callback_url,
                                        cuda_home=cuda_home,
                                    ),
                                    "dir_to_save": "$OUTPUT_PATH",
                                },
                                "skypilot": {
                                    "image": image,
                                    "venv_path": venv_path,
                                    "additional_files": additional_files,
                                },
                                "compute_config": {
                                    "num_gpus_per_node": num_gpus_per_node(config_data),
                                    "num_cpus_per_node": num_cpus_per_node,
                                    "num_nodes": 1,
                                    "total_memory_per_node": total_memory_per_node,
                                },
                                "launcher_config": {"resources": resources},
                            },
                        }
                    ],
                }
            },
        }
    }
    return yaml.dump(build, Dumper=BlockStringDumper, default_flow_style=False, sort_keys=False)
