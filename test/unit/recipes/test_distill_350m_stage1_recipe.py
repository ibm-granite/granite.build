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

"""Unit tests for the distill-stage1 recipe.

Stage 1 is the real off-policy run, and it differs from distill-smoke in one
structural way: a `sources` target builds the corpus from the three SFT splits
before prep sees it. That target exists because of three facts about those files,
each of which fails SILENTLY if it is not handled:

* the rows are spelled ``conversations``, not ``messages``, and prep drops anything
  without ``messages`` as ``no_messages`` — all 5,081,504 of them;
* the splits are 84.86% / 13.98% / 1.17%, and prep's ``max_examples`` stops after N
  kept rows *in input order*, so a concatenation plus a cap yields pure ``general``;
* prep takes one path, not a directory (``prep_corpus.py:294``).

So most of what is asserted here is that those three are still handled, plus the
sizing invariants: the effective-batch product, and that the run is bounded by
epochs rather than by a step count that goes stale whenever the subset changes.
"""

import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-350m"
    / "lsf"
    / "distill-stage1"
)

_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")

# Measured on the cluster, 2026-09-22. The shares are what make a proportional
# sample necessary; the totals are what make the cost estimate real.
_SPLIT_ROWS = {"general": 4_311_969, "tools": 710_290, "rag": 59_245}
_TOTAL_ROWS = sum(_SPLIT_ROWS.values())

_TARGETS = [
    "sources",
    "align",
    "corpus",
    "train-gold",
    "export",
    "eval-transfer-baseline",
    "eval-transfer",
    "eval-bfcl",
]


def _params(**overrides):
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    params.update(overrides)
    return params


def _render(tmp_path, **overrides):
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(**overrides), str(tmp_path))
    )


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _config(rendered, target):
    return _targets(rendered)[target]["steps"][0]["config"]


@pytest.fixture(name="off")
def fixture_off(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False)


# ─── The parameter surface ─────────────────────────────────────────────────────


def test_every_marker_has_a_parameter():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert set(_MARKER.findall(contents)) - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert set(_params()) - set(_MARKER.findall(contents)) - {"INCLUDE_SFT"} == set()


def test_the_recipe_renders_the_stage1_graph(off):
    assert list(_targets(off)) == _TARGETS


def test_results_do_not_land_under_gbtest(off):
    """This run's checkpoints are the deliverable. The smoke tier lives under
    gbtest; a multi-hour run putting 40 GB checkpoints there is how scratch
    directories become load-bearing."""
    assert "gbtest" not in _params()["WORKDIR_ROOT"]


# ─── The sources target ────────────────────────────────────────────────────────


class TestSources:
    """The logic lives in the corpus-sources step and is tested there. What this
    recipe owns is the wiring: which splits, in which order, at what size."""

    def test_it_runs_the_corpus_sources_step(self, off):
        step = _targets(off)["sources"]["steps"][0]
        assert step["step_uri"] == "space://steps/distill/corpus-sources"

    def test_it_passes_the_three_splits_in_lineage_order(self, off):
        """Order names the per-split RNG streams, so it is part of the selection."""
        cfg = _config(off, "sources")["sources_config"]
        params = _params()
        assert cfg["sources"] == [
            params["SOURCE_GENERAL"],
            params["SOURCE_TOOLS"],
            params["SOURCE_RAG"],
        ]

    def test_the_size_and_seed_come_from_parameters(self, off):
        cfg = _config(off, "sources")["sources_config"]
        assert cfg["target_rows"] == _params()["TARGET_ROWS"]
        assert cfg["shuffle_seed"] == _params()["SHUFFLE_SEED"]
        assert cfg["output_dir"].endswith("/sources")

    def test_all_three_splits_are_declared_as_lineage_inputs(self, off):
        inputs = _targets(off)["sources"]["inputs"]
        assert set(inputs) == {"source_general", "source_tools", "source_rag"}
        for spec in inputs.values():
            assert spec["type"] == "dataset"
            assert spec["uri"].startswith("env://")

    def test_the_sources_are_the_raw_splits_not_the_tokenized_ones(self):
        """The tokenized directories hold 8192-token blocks with no `messages` field;
        they cannot be un-rendered, and prep measures every decision on rendered
        text."""
        for key in ("SOURCE_GENERAL", "SOURCE_TOOLS", "SOURCE_RAG"):
            path = _params()[key]
            assert path.endswith(".jsonl")
            assert "tokenized" not in path, path
            assert "granite-4.0-sft-datasets" in path

    def test_it_declares_the_output_the_step_registers(self, off):
        """An undeclared output makes the resolver drop the NEWARTIFACT event and the
        target completes with no output AND no error."""
        assert set(_targets(off)["sources"]["outputs"]) == {"corpus_source"}


class TestCorpusConsumesSources:
    def test_prep_reads_the_built_file_by_binding(self, off):
        """Not env://DATASET: prep takes ONE path (prep_corpus.py:294 refuses a
        directory), and the file does not exist until sources has run. A binding is
        also the ordering edge."""
        assert _targets(off)["corpus"]["inputs"]["source_dataset"] == {
            "binding": "sources.corpus_source"
        }
        dataset = _config(off, "corpus")["corpus_config"]["dataset"]
        assert dataset == "{{ bindings.source_dataset.binding.path }}"

    def test_prep_does_not_cap_the_row_count(self, off):
        """The cap belongs in sources, where it can be applied proportionally.
        max_examples stops after N kept rows in INPUT order, so a cap here would
        drop tools and rag entirely on an 84.86%-general corpus."""
        assert _config(off, "corpus")["corpus_config"]["max_examples"] == 0

    def test_a_held_out_split_exists_and_is_big_enough_to_evaluate(self, off):
        """eval-transfer draws EVAL_MAX_SAMPLES from the eval split; at
        eval_fraction 0 that split does not exist at all."""
        fraction = _config(off, "corpus")["corpus_config"]["eval_fraction"]
        assert fraction > 0
        held_out = _params()["TARGET_ROWS"] * fraction
        assert (
            held_out > _params()["EVAL_MAX_SAMPLES"]
        ), "the eval split is smaller than the number of samples the eval draws"


# ─── Sizing ────────────────────────────────────────────────────────────────────


class TestSizing:
    def test_the_effective_batch_is_the_product_not_the_factors(self, off):
        """Two nodes at grad_accum 3 and one node at grad_accum 6 are the SAME
        optimization at different wall-clock. Two nodes at grad_accum 6 is a
        different run. Asserting the product means a compensating edit passes and an
        uncompensated one fails."""
        p = _params()
        product = (
            p["GOLD_PER_DEVICE_TRAIN_BATCH_SIZE"]
            * p["GOLD_GRADIENT_ACCUMULATION_STEPS"]
            * p["GOLD_NUM_NODES"]
            * p["GOLD_NUM_GPUS"]
        )
        assert product == 96

    def test_the_run_is_bounded_by_epochs_not_by_a_step_count(self, off):
        """render_gold_config.py emits max_steps only when positive, and a positive
        value overrides num_train_epochs. At 0 the step count follows TARGET_ROWS
        automatically — otherwise every change to the subset needs a hand-recomputed
        step count, and a stale one silently covers a fraction of the corpus."""
        gold = _config(off, "train-gold")["gold_config"]
        assert int(gold["max_steps"]) == 0
        assert float(gold["num_train_epochs"]) > 0

    def test_checkpoints_arrive_often_enough_to_be_a_curve(self, off):
        """One end-point number is not a learning curve, and a preemptable multi-hour
        run needs something to resume from."""
        p = _params()
        steps = p["TARGET_ROWS"] / 96
        save = _config(off, "train-gold")["gold_config"]["save_steps"]
        assert 0 < save < steps / 4, f"{save} over ~{steps:.0f} steps is too coarse"

    def test_the_length_budget_is_the_one_the_baseline_was_tokenized_at(self, off):
        """8192 is sft-eval-full-dataset's MAX_SEQ_LEN. One value for prep, the
        trainer and both evals."""
        budget = _params()["MAX_LENGTH"]
        assert budget == 8192
        assert _config(off, "corpus")["corpus_config"]["max_length"] == budget
        assert _config(off, "train-gold")["gold_config"]["max_length"] == budget
        for target in ("eval-transfer", "eval-transfer-baseline"):
            assert _config(off, target)["eval_config"]["max_length"] == budget

    def test_the_target_row_count_is_a_real_fraction_of_the_corpus(self):
        """0 means the whole corpus, which is 52,932 steps and an estimated 365-729
        GPU-h. A default that large should be a decision, not an inheritance."""
        target = _params()["TARGET_ROWS"]
        assert 0 < target <= _TOTAL_ROWS
        assert target < _TOTAL_ROWS, "the default should be a subset, not a full epoch"


# ─── Inherited invariants that must not drift ──────────────────────────────────


class TestInheritedInvariants:
    def test_the_student_is_still_the_sft_checkpoint(self):
        assert _params()["STUDENT_MODEL"].endswith("/epoch_hf_2")

    def test_the_teacher_is_still_the_pinned_copy(self):
        assert _params()["TEACHER_MODEL"].endswith("-pinned")

    def test_chatml_is_not_required_and_the_template_is_granite_native(self, off):
        align = _config(off, "align")["align_config"]
        assert align["require_chatml"] is False
        assert "chatml" not in align["chat_template"].lower()

    def test_the_response_template_is_the_granite_role_marker(self, off):
        template = _config(off, "train-gold")["gold_config"]["response_template"]
        assert template == "<|start_of_role|>assistant<|end_of_role|>"
        assert "\\" not in template

    def test_it_is_off_policy_and_allocates_no_server(self, off):
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold["lmbda"]) == 0.0
        assert int(gold["vllm_num_servers"]) == 0

    def test_the_fused_jsd_kernel_stays_off(self, off):
        assert _config(off, "train-gold")["gold_config"]["use_liger_fused_jsd"] is False

    def test_the_learning_rate_suits_a_converged_student(self, off):
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold["learning_rate"]) <= 5e-6

    def test_documents_are_kept(self, off):
        assert _config(off, "corpus")["corpus_config"]["documents_policy"] == "keep"

    def test_no_sft_arm(self, off):
        assert _params()["INCLUDE_SFT"] is False
        assert "train-sft" not in _targets(off)


class TestThePinnedUpstreamCheckout:
    """Every distillation step reads its Python from a checkout THIS project controls.

    It used to be a shared, continuously-advancing tree, and that broke mid-build: the
    shared tree is advanced by its upstream author, and every time it moved, all six
    ported steps' pins stopped matching and the affected step exited 1. So the pin now
    names a small public repo (gb-steps-distillation) that only advances when this
    project deliberately bumps it.

    That lives in the STEP templates, identically across all six -- the source contract
    asserts those blocks are identical to each other, not that they hold any particular
    value, so a uniform change keeps it green. These tests assert the recipes do NOT
    re-specify it, because a per-recipe override is how five steps end up pointed at a
    tree their own pin rejects.
    """

    def test_no_target_overrides_the_shared_code_config(self, off):
        """The pin belongs to the step, once, for all six. A recipe-level override was
        the first attempt here and it left `corpus` reading the shared tree while
        `align` read the patched one -- which is precisely how build 1820703f failed."""
        for name in _targets(off):
            assert "code_config" not in _config(
                off, name
            ), f"{name} overrides code_config; the pin belongs in the step template"
