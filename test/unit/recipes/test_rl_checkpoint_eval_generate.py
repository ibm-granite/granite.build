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

"""Unit tests for the rl-checkpoint-eval build generator (issue #45).

The generator (``recipes/.../rl-checkpoint-eval/generate_build.py``) is a
standalone script, not part of the gbserver package, so it is loaded here by
path via importlib. These tests cover its pure, edge-case-prone functions —
the checkpoint schedule math, eval-set resolution, --param typing/dot-notation,
and the common-flag override shorthand — so a schedule or resolution regression
is caught in CI rather than only as a hung live BlueVela run.
"""

import importlib.util
import pathlib

import pytest

_GEN_PATH = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-350m"
    / "lsf"
    / "rl-checkpoint-eval"
    / "generate_build.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("rl_ckpt_eval_gen", _GEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()


def _params(total_episodes, save_freq, prompts=64, samples=16):
    return {
        "TOTAL_EPISODES": total_episodes,
        "NUM_UNIQUE_PROMPTS_ROLLOUT": prompts,
        "NUM_SAMPLES_PER_PROMPT_ROLLOUT": samples,
        "SAVE_FREQ": save_freq,
    }


# ─── compute_checkpoint_steps ─────────────────────────────────────────────────
class TestComputeCheckpointSteps:
    def test_shipped_smoke_default_two_checkpoints(self):
        # 20480 // (64*16) = 20 updates, SAVE_FREQ 10 -> steps 10, 20.
        assert gen.compute_checkpoint_steps(_params(20480, 10)) == [10, 20]

    def test_even_multiples_no_duplicate_final(self):
        # num_updates 12, SAVE_FREQ 3 -> 3,6,9,12 (12 is already a multiple).
        assert gen.compute_checkpoint_steps(_params(12288, 3)) == [3, 6, 9, 12]

    def test_final_update_appended_when_not_a_multiple(self):
        # num_updates 2, SAVE_FREQ 10 -> range() empty, final update appended.
        assert gen.compute_checkpoint_steps(_params(2048, 10)) == [2]

    def test_save_freq_one_every_update(self):
        assert gen.compute_checkpoint_steps(_params(3072, 1)) == [1, 2, 3]

    def test_zero_updates_raises(self):
        # TOTAL_EPISODES below one rollout batch -> 0 updates, nothing to save.
        with pytest.raises(ValueError, match="0 optimizer updates"):
            gen.compute_checkpoint_steps(_params(512, 1))

    def test_bad_denominator_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            gen.compute_checkpoint_steps(_params(1024, 1, prompts=0))

    def test_save_freq_below_one_raises(self):
        with pytest.raises(ValueError, match="SAVE_FREQ must be >= 1"):
            gen.compute_checkpoint_steps(_params(2048, 0))


# ─── resolve_eval_names ───────────────────────────────────────────────────────
_CATALOG = {
    "evals": {
        "a": {"category": "code"},
        "b": {"category": "math"},
        "c": {"category": "multilingual"},
        "bfcl": {"category": "bfcl"},
    },
    "sets": {"s": ["a", "b"], "full": ["a", "b", "c", "bfcl"]},
}


class TestResolveEvalNames:
    def test_set_expands(self):
        assert gen.resolve_eval_names({"EVAL_SETS": ["s"]}, _CATALOG) == ["a", "b"]

    def test_individual_evals(self):
        assert gen.resolve_eval_names({"EVAL_SETS": ["a", "bfcl"]}, _CATALOG) == [
            "a",
            "bfcl",
        ]

    def test_set_plus_individual_dedup_first_seen_order(self):
        # 's' -> a,b; then 'a' (dup, dropped); then 'bfcl'.
        assert gen.resolve_eval_names({"EVAL_SETS": ["s", "a", "bfcl"]}, _CATALOG) == [
            "a",
            "b",
            "bfcl",
        ]

    def test_string_shorthand_parsed_as_list(self):
        assert gen.resolve_eval_names({"EVAL_SETS": "[a, b]"}, _CATALOG) == ["a", "b"]

    def test_unknown_name_raises_with_known_list(self):
        with pytest.raises(ValueError, match="Unknown EVAL_SETS entry"):
            gen.resolve_eval_names({"EVAL_SETS": ["nope"]}, _CATALOG)

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match="empty selection"):
            gen.resolve_eval_names({"EVAL_SETS": []}, _CATALOG)


# ─── apply_override / add_key_value ───────────────────────────────────────────
class TestParamOverrides:
    def test_dot_notation_nests(self):
        data = {}
        gen.add_key_value(data, "a.b.c", "v")
        assert data == {"a": {"b": {"c": "v"}}}

    def test_yaml_typing_list(self):
        assert gen.apply_override({}, "EVAL_SETS=[x, y]") == {"EVAL_SETS": ["x", "y"]}

    def test_yaml_typing_int(self):
        assert gen.apply_override({}, "SAVE_FREQ=10") == {"SAVE_FREQ": 10}

    def test_missing_equals_raises(self):
        with pytest.raises(ValueError, match="key=value"):
            gen.apply_override({}, "NOEQUALS")


# ─── _flag_overrides ──────────────────────────────────────────────────────────
class _Args:
    """Minimal stand-in for the argparse Namespace _flag_overrides reads."""

    def __init__(self, **kw):
        for dest in gen.COMMON_FLAG_PARAMS:
            setattr(self, dest, kw.get(dest))


class TestFlagOverrides:
    def test_eval_sets_comma_shorthand_becomes_yaml_list(self):
        assert gen._flag_overrides(_Args(eval_sets="x,y")) == ["EVAL_SETS=[x, y]"]

    def test_eval_sets_explicit_list_passthrough(self):
        assert gen._flag_overrides(_Args(eval_sets="[x, y]")) == ["EVAL_SETS=[x, y]"]

    def test_unset_flags_omitted(self):
        assert gen._flag_overrides(_Args()) == []

    def test_scalar_flag_mapped_to_param_name(self):
        assert gen._flag_overrides(_Args(save_freq="10")) == ["SAVE_FREQ=10"]


# ─── _experiment_for_step (flat naming — sage-eval can't accept a "/") ────────
class TestExperimentForStep:
    def test_flat_suffix_no_slash(self):
        exp = gen._experiment_for_step(10)
        assert exp == "$${EXPERIMENT}-ckpt_10"
        assert "/" not in exp
