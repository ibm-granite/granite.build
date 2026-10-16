# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""TuningLauncher selection from settings, mirroring ``storage/registry.py``."""

from __future__ import annotations

import functools
from collections.abc import Callable

from autotunex.core.config import Settings
from autotunex.services.launch.bash_spec import build_bash_spec
from autotunex.services.launch.llmb import LlmbBuildCanceller, LlmbTuningLauncher
from autotunex.services.launch.lsf_spec import build_lsf_spec
from autotunex.services.launch.protocols import BuildCanceller, LaunchContext, TuningLauncher
from autotunex.services.launch.spec import build_spec


def get_tuning_launcher(settings: Settings) -> TuningLauncher:
    """Return the launcher chosen by ``job_backend``, with its spec builder bound.

    Only called when a backend is configured (``get_job_runner`` returns the
    no-op runner for ``none``); the ``none`` branch raises defensively.

    The spec shape is selected by ``gb_environment``: under ``"standalone"`` the
    choice is three-way — the LSF/SkyPilot builder (``build_lsf_spec``) when
    ``lsf_cluster`` is set, else the local-bash builder (``build_bash_spec``, which
    needs no cluster inputs) — while any other value binds the custom_code builder
    (``build_spec``). The custom_code and LSF inputs are each guaranteed present by
    ``Settings._validate_job_backend`` at startup — the ``None`` checks here narrow
    the type and fail loudly if that invariant is ever bypassed.
    """
    if settings.job_backend != "llmb":
        raise ValueError(f"No launcher for job_backend={settings.job_backend!r}.")
    spec_builder: Callable[[LaunchContext], str]
    if settings.gb_environment == "standalone":
        if settings.lsf_cluster:
            # LSF (SkyPilot) variant — inputs guaranteed present by
            # Settings._validate_job_backend; the None checks narrow the type.
            if (
                settings.lsf_environment_uri is None
                or settings.lsf_image is None
                or settings.job_trainer_repo is None
            ):
                raise ValueError("job_backend='llmb' (LSF) is missing required build settings.")
            spec_builder = functools.partial(
                build_lsf_spec,
                environment_uri=settings.lsf_environment_uri,
                image=settings.lsf_image,
                venv_path=settings.lsf_venv_path,
                cuda_home=settings.lsf_cuda_home,
                trainer_repo=settings.job_trainer_repo,
                trainer_ref=settings.job_trainer_ref,
                cluster=settings.lsf_cluster,
                queue=settings.lsf_queue,
                accelerators=settings.lsf_accelerators,
                memory=settings.lsf_memory,
                num_cpus_per_node=settings.lsf_num_cpus_per_node,
                total_memory_per_node=settings.lsf_total_memory_per_node,
                poll_interval_seconds=settings.lsf_poll_interval_seconds,
                callback_url=settings.job_callback_url,
            )
        else:
            spec_builder = functools.partial(
                build_bash_spec,
                fm_tune_root=settings.bash_fm_tune_root,
                fm_tune_ref=settings.bash_fm_tune_ref,
                fm_tune_extra=settings.bash_fm_tune_extra,
                backend=settings.bash_backend,
                callback_url=settings.job_callback_url,
                # Anchor the run's artifacts under artifact_dir as an ABSOLUTE
                # file:// URI. gbserver resolves a relative file: URI against its
                # own CWD (the ephemeral image path in the container), which would
                # strand artifacts off the /data volume; an absolute URI is honored
                # verbatim. The bash environment only writes file://, so this is not
                # reused for job_output_uri_root (which may be a non-file scheme).
                output_uri_root=settings.artifact_dir.resolve().as_uri(),
            )
    else:
        if (
            settings.job_runtime_image is None
            or settings.job_trainer_repo is None
            or settings.job_output_uri_root is None
        ):
            raise ValueError("job_backend='llmb' (custom_code) is missing required build settings.")
        spec_builder = functools.partial(
            build_spec,
            runtime_image=settings.job_runtime_image,
            trainer_repo=settings.job_trainer_repo,
            trainer_ref=settings.job_trainer_ref,
            output_uri_root=settings.job_output_uri_root,
            callback_url=settings.job_callback_url,
        )
    return LlmbTuningLauncher(
        llmb_command=settings.llmb_command,
        spec_dir=settings.job_spec_dir,
        token_env=settings.gb_token_env,
        tags=settings.gb_tags,
        spec_builder=spec_builder,
    )


def get_build_canceller(settings: Settings) -> BuildCanceller:
    """Return the build canceller for the configured backend.

    Only ``job_backend="llmb"`` has one; every other backend raises, exactly as
    ``get_tuning_launcher`` does. The canceller needs only the CLI command and the
    token env var — no spec inputs — so no ``gb_environment`` branching applies.
    """
    if settings.job_backend != "llmb":
        raise ValueError(f"No build canceller for job_backend={settings.job_backend!r}.")
    return LlmbBuildCanceller(llmb_command=settings.llmb_command, token_env=settings.gb_token_env)
