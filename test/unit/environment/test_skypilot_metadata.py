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

"""Tests for SkyPilot build-tracking metadata rendering.

Exercises the pure helpers in ``_skypilot_metadata`` that turn a step's
``run_metadata`` into (a) ``sky.Resources`` labels and (b) a single-line SLURM
``--comment`` string. No cluster required.
"""

from gbserver.environment._skypilot_metadata import (
    apply_slurm_comment_override,
    normalize_run_metadata,
    task_metadata_comment,
    task_metadata_labels,
)

# A fully-populated run_metadata dict (the shape produced by
# TargetStepRun.get_runmetadata().to_dict()).
_FULL_RUN_METADATA = {
    "build_id": "b1234567-89ab-cdef-0123-456789abcdef",
    "build_config_name": "my-build",
    "target_name": "My Target",
    "targetstep_uri": "space://steps/mystep",
    "targetrun_id": "tr-0001",
    "targetsteprun_id": "tsr-0001",
    # Fields not tracked should be ignored.
    "username": "alice",
    "target_hash": "deadbeef",
}


def test_labels_full_metadata():
    """All tracked fields become label-safe key/value pairs."""
    labels = task_metadata_labels(_FULL_RUN_METADATA)
    assert labels == {
        "gb-build-id": "b1234567-89ab-cdef-0123-456789abcdef",
        "gb-build-name": "my-build",
        "gb-target-name": "my-target",  # slugified (space -> -, lowercased)
        "gb-step-uri": "space-steps-mystep",  # :// and / collapse to -
        "gb-target-id": "tr-0001",
        "gb-step-id": "tsr-0001",
    }


def test_comment_full_metadata():
    """All tracked fields render as key=value, joined by ';', in table order."""
    comment = task_metadata_comment(_FULL_RUN_METADATA)
    assert comment == (
        "build_id=b1234567-89ab-cdef-0123-456789abcdef;"
        "build_name=my-build;"
        "target=My_Target;"  # whitespace -> _, case preserved
        "step_uri=space://steps/mystep;"  # :// and / preserved
        "target_id=tr-0001;"
        "step_id=tsr-0001"
    )


def test_empty_and_missing_fields_skipped():
    """Blank or absent fields appear in neither channel."""
    run_metadata = {
        "build_id": "b1",
        "build_config_name": "",  # empty -> skipped
        "target_name": "   ",  # whitespace-only -> skipped
        # targetstep_uri / targetrun_id / targetsteprun_id absent -> skipped
    }
    assert task_metadata_labels(run_metadata) == {"gb-build-id": "b1"}
    assert task_metadata_comment(run_metadata) == "build_id=b1"


def test_empty_metadata_yields_empty():
    """No tracked values -> empty label dict and empty comment string."""
    assert task_metadata_labels({}) == {}
    assert task_metadata_comment({}) == ""


def test_comment_has_no_newline():
    """The SLURM comment must be single-line (SkyPilot schema rejects \\n)."""
    run_metadata = {"target_name": "line one\nline two", "build_id": "b1"}
    comment = task_metadata_comment(run_metadata)
    assert "\n" not in comment
    assert "target=line_one_line_two" in comment


def test_comment_delimiters_neutralized_in_values():
    """A value containing ';' or '=' can't forge or truncate comment fields."""
    run_metadata = {"build_config_name": "a;target=evil", "build_id": "b1"}
    comment = task_metadata_comment(run_metadata)
    # The injected "target=evil" must NOT survive as a parseable extra field:
    # ';' -> ',' and '=' -> '-' keep the value in its own slot.
    assert comment == "build_id=b1;build_name=a,target-evil"
    # Exactly the two real fields remain when splitting on the delimiters.
    keys = [pair.split("=", 1)[0] for pair in comment.split(";")]
    assert keys == ["build_id", "build_name"]


def test_label_value_truncated_to_63_chars():
    """Long values are slugified and capped at the k8s label limit."""
    run_metadata = {"target_name": "x" * 100}
    label = task_metadata_labels(run_metadata)["gb-target-name"]
    assert len(label) <= 63
    assert set(label) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")


def test_step_uri_preserved_in_comment():
    """A full step uri keeps its :// and / in the comment channel."""
    run_metadata = {"targetstep_uri": "space://steps/foo/bar"}
    assert task_metadata_comment(run_metadata) == "step_uri=space://steps/foo/bar"


def test_apply_slurm_comment_override_sets_nested_key():
    """The override helper writes slurm.sbatch_options.comment in place."""
    overrides: dict = {"docker": {"run_options": ["--gpus", "all"]}}
    apply_slurm_comment_override(overrides, _FULL_RUN_METADATA)
    assert overrides["slurm"]["sbatch_options"]["comment"] == task_metadata_comment(
        _FULL_RUN_METADATA
    )
    # Existing keys are left untouched.
    assert overrides["docker"] == {"run_options": ["--gpus", "all"]}


def test_apply_slurm_comment_override_noop_when_empty():
    """No comment -> no slurm section added to the overrides."""
    overrides: dict = {}
    apply_slurm_comment_override(overrides, {})
    assert overrides == {}


def test_normalize_run_metadata_passthrough_dict():
    """A dict is returned unchanged (same object)."""
    run_metadata = {"build_id": "b1"}
    assert normalize_run_metadata(run_metadata) is run_metadata


def test_normalize_run_metadata_calls_to_dict():
    """A non-dict object with to_dict() is converted via it."""

    class _Meta:
        def to_dict(self):
            return {"build_id": "b1"}

    assert normalize_run_metadata(_Meta()) == {"build_id": "b1"}


def test_normalize_run_metadata_none_and_other_yield_empty():
    """None (or anything without to_dict) normalizes to an empty dict."""
    assert normalize_run_metadata(None) == {}
    assert normalize_run_metadata("nope") == {}
