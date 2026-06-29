#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for _copy_basestep_scaffold: the gbstep base-step scaffold ships every
backend's launch templates side by side, and only the active environment's dir
should survive the copy. Crucially, env_type may arrive capitalized ("Lsf"),
lowercase ("lsf"), or as a backend-equivalent alias ("kubernetes" -> K8s), so
the prune must normalize all of them — otherwise it deletes the dir it was meant
to keep (notably helm-charts, which bundles the gbstepbase library subchart). An
unknown env type must prune nothing rather than delete every backend's dir."""

from pathlib import Path

import pytest

from gbserver.build.targetstep import _copy_basestep_scaffold

# Top-level backend template dirs shipped by the gbstep scaffold.
BASH = "bash_scripts"
LSF = "lsf_scripts"
HELM = "helm-charts"
ALL_BACKEND_DIRS = {BASH, LSF, HELM}


def _kept_dirs(tmp_path: Path) -> set:
    """Return the set of backend template dirs present under tmp_path."""
    return {d.name for d in tmp_path.iterdir() if d.is_dir()} & ALL_BACKEND_DIRS


@pytest.mark.parametrize(
    "env_type, expected_kept",
    [
        # Known backends that read no scaffold dir keep only bash (both
        # lsf_scripts and helm-charts pruned — the bash double-render fix).
        ("Bash", {BASH}),
        ("bash", {BASH}),
        ("Docker", {BASH}),
        ("Skypilot", {BASH}),
        ("Skypilot_managed", {BASH}),
        ("Runpod", {BASH}),
        # Lsf keeps lsf_scripts; helm-charts is pruned.
        ("Lsf", {BASH, LSF}),
        # K8s keeps helm-charts; lsf_scripts is pruned.
        ("K8s", {BASH, HELM}),
        # Lowercase env_type (from an environment_configs dict key) must still
        # keep the right dir — this is the casing bug guard.
        ("lsf", {BASH, LSF}),
        ("k8s", {BASH, HELM}),
        # "kubernetes" is a real-world alias for the K8s backend; it must keep
        # helm-charts (the regression that broke gbstepbase resolution).
        ("kubernetes", {BASH, HELM}),
        ("Kubernetes", {BASH, HELM}),
        # An unknown env type prunes nothing — never delete a dir a backend may
        # need just because we don't recognize the type.
        ("totally-unknown", ALL_BACKEND_DIRS),
    ],
)
def test_copy_basestep_scaffold_prunes_inactive_backends(
    tmp_path, env_type, expected_kept
):
    _copy_basestep_scaffold(tmp_path, env_type)
    assert _kept_dirs(tmp_path) == expected_kept
    # bash_scripts is never pruned (it carries no double-render templates and is
    # the generic launch scaffold).
    assert (tmp_path / BASH).is_dir()
    # step_default.yaml (a non-backend scaffold file) is always copied.
    assert (tmp_path / "step_default.yaml").is_file()


@pytest.mark.parametrize("env_type", ["K8s", "k8s", "kubernetes", "Kubernetes"])
def test_copy_basestep_scaffold_keeps_gbstepbase_subchart_for_k8s(tmp_path, env_type):
    """The K8s backend's helm-charts dir bundles the gbstepbase library subchart
    that every step's appwrapper.yaml includes via `gbstepbase.app`. If the prune
    drops helm-charts (e.g. for the "kubernetes" alias), the helm render fails
    with `no template "gbstepbase.app"`. Lock in that the subchart survives."""
    _copy_basestep_scaffold(tmp_path, env_type)
    charts = list(tmp_path.glob(f"{HELM}/*/charts/gbstepbase"))
    assert charts, f"gbstepbase subchart missing under {HELM} for env_type={env_type!r}"
