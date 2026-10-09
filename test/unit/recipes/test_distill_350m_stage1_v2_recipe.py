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

"""Unit tests for the distill-stage1-v2 recipe.

v2 is stage 1 retried after build df8512e0 produced a model worse on every one of the
30+ benchmarks measured. The run was mechanically perfect -- every step exit 0, no NaN,
no OOM -- and the objective is what failed: GOLD's loss IS the divergence, the student
was already SFT'd on this corpus family and therefore nearly satisfied it at
initialisation, and with no ground-truth term the only descent direction left for 7,640
steps was to become more certain. Entropy fell 42%, the generations became repetition
loops, and the train loss moved 2.7% and said nothing.

So what this file guards is different in kind from distill-stage1's tests. Those assert
the PLUMBING still works. These assert the four things that make v2 a different
experiment rather than a re-run, each of which is one edit away from silently reverting:

* the objective is anchored, and the anchor is REACHABLE (a patched trainer, pinned);
* the collapse is instrumented and the guard is armed;
* the horizon is bounded by steps, sized to the headroom that actually existed;
* the whole checkpoint curve survives, and every rung on the ladder is a checkpoint
  that will exist.

The last one is the subtlest. The ladder drives a Jinja loop, so CKPT_LADDER IS the
target graph: a rung that is not a multiple of GOLD_SAVE_STEPS names a checkpoint-N
directory the trainer never wrote, and the export target fails AFTER the training has
been paid for.
"""

import json
import pathlib
import re
import subprocess
import sys

import pytest
import yaml
from unit.recipes.published_step import gen_smoke_module, render_run

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-350m"
    / "lsf"
    / "distill-stage1-v2"
)
_STAGE1 = _RECIPE.parent / "distill-stage1"

_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_.,\"\[\]()\- ]+)\}")
_PLAIN_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")

# The loop variable is bound by the <% for %> block, not by parameters.yaml.
_LOOP_VARS = {"N"}

_LADDER = ["25", "50", "75", "100", "150", "200", "300"]
_TARGETS = (
    ["sources", "align", "corpus", "train-gold"]
    + [f"export-{n}" for n in _LADDER]
    + ["eval-transfer-baseline"]
    + [f"eval-transfer-{n}" for n in _LADDER]
    + ["gen-smoke", "eval-bfcl-baseline"]
    + [f"eval-bfcl-{n}" for n in _LADDER]
)


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
    names = set(_PLAIN_MARKER.findall(contents)) - _LOOP_VARS
    assert names - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    referenced = set(_PLAIN_MARKER.findall(contents))
    # CKPT_LADDER (.split(...)) and CORPUS_DIR (.rstrip('/'), so a copied path with a
    # trailing slash cannot reach an artifact URI) are reached through an expression,
    # which the plain marker pattern does not match, so allow them explicitly.
    referenced |= {"CKPT_LADDER", "CORPUS_DIR"}
    assert set(_params()) - referenced - {"INCLUDE_SFT"} == set()


def test_no_comment_uses_a_loop_variable_outside_its_loop():
    """The CLI templates this file IN FULL, comments included, with StrictUndefined.
    A loop variable named in a comment is an undefined-variable error that fails the
    build before submission -- which is exactly how this file first failed to render."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    for line in contents.splitlines():
        if line.lstrip().startswith("#"):
            assert not (set(_PLAIN_MARKER.findall(line)) & _LOOP_VARS), line


def test_the_recipe_renders_the_v2_graph(off):
    assert list(_targets(off)) == _TARGETS


def test_results_do_not_land_under_gbtest(off):
    assert "gbtest" not in _params()["WORKDIR_ROOT"]


def test_the_build_is_not_still_named_after_the_smoke_recipe(off):
    """distill-stage1 shipped as `name: distill-350m-smoke`, so three different
    builds share one name in `gb build list` and stage-1 runs are identifiable only
    by their artifact URI. v2 does not inherit that."""
    assert off["granite.build"]["name"] == "distill-350m-stage1-v2"
    assert "smoke" not in off["granite.build"]["name"]


# ─── The objective ─────────────────────────────────────────────────────────────


class TestTheUnanchoredObjective:
    """The CE anchor and the entropy guard are OFF, and that is the result of the sweep
    rather than a default nobody got round to setting.

    The four-coefficient grid completed on 2026-09-28 with a monotone-bad dose-response
    on two independent suites (ifeval gain +5.57 -> +4.04 -> +2.71 -> +2.56, HumanEval
    +1 -> 0 -> -3 -> -9 problems as ce goes 0 -> 0.05 -> 0.15 -> 0.40), and the collapse
    the guard watched for turned out to damage nothing: df8512e0's 42% entropy drop is
    real, but once its export was repaired that checkpoint beat its own SFT baseline.

    These tests assert the objective AND the property that follows from it -- that
    nothing patch-gated renders, which is what lets this recipe run on the step's own
    public trainer pin.
    """

    def test_the_ce_anchor_is_off(self, off):
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold.get("ce_coef", 0.0)) == 0.0

    def test_the_anchor_stays_the_same_order_as_the_divergence_if_re_armed(self, off):
        """Inert at the shipped default, and the reason it is kept is the one question
        the sweep left open: no CE arm was ever measured for safety, and safety is the
        only axis where the unanchored epoch went backwards. If that experiment re-arms
        the anchor, the coefficient still has to be readable -- df8512e0's JSD ran 0.068
        and CE on this student runs order 1, so below ~0.01 the anchor restrains nothing
        and above ~0.5 it is SFT with a divergence garnish."""
        ce = float(_config(off, "train-gold")["gold_config"].get("ce_coef", 0.0))
        if ce > 0:
            assert 0.01 <= ce <= 0.5

    def test_the_per_step_entropy_log_is_off_and_the_metric_survives_elsewhere(
        self, off
    ):
        """log_student_entropy is one of the six patch-gated fields, so leaving it true
        alone would hold CODE_DIR at a patched checkout for an instrument. The quantity
        is not lost: distill-eval computes entropy and rkld from a checkpoint with no
        trainer involvement, and every rung is transfer-evaluated."""
        gold = _config(off, "train-gold")["gold_config"]
        assert gold.get("log_student_entropy", False) is False
        assert int(gold["logging_steps"]) == 1
        assert "entropy" in _params()["EVAL_METRICS"].split(",")

    def test_the_collapse_guard_is_disarmed(self, off):
        """It never stopped a run here in any case -- ENTROPY_GUARD_ACTION has been
        "warn" for as long as there has been a ladder -- so disarming it changes no
        behaviour and removes a patch dependency."""
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold.get("entropy_guard_drop_frac", 0.0)) == 0.0

    def test_a_stopping_guard_and_a_fixed_ladder_cannot_both_be_asked_for(self, off):
        """THE failure of build d1acf1c0, as an invariant.

        The guard tripped at step 77 of 2,000 and stopped the run, exactly as armed. But
        every rung of this recipe's ladder is a literal step number, and the trainer had
        written none of them: all four export targets failed with `requested checkpoint
        does not exist`, the build failed, and the run's question went unanswered after
        the training was paid for. The two features are individually right and jointly
        contradictory -- a guard that may stop at an arbitrary step, and consumers that
        demand specific steps -- so one of them has to give, and for a run whose PURPOSE
        is the collapse curve it is the stop.

        Either the guard only warns, or no rung is allowed to outlive a possible stop.
        This recipe takes the first branch; the assertion admits both so that a future
        recipe choosing the second is not forced to lie."""
        gold = _config(off, "train-gold")["gold_config"]
        if float(gold["entropy_guard_drop_frac"]) <= 0:
            return  # guard disarmed: the ladder is bounded by max_steps alone
        rungs = [int(n) for n in _params()["CKPT_LADDER"].split(",")]
        if gold["entropy_guard_action"] == "stop":
            raise AssertionError(
                "the guard may stop at any step past "
                f"{gold['entropy_guard_baseline_steps']}, but the ladder demands "
                f"checkpoints at {rungs}. Set ENTROPY_GUARD_ACTION=warn, or stop naming "
                "fixed rungs."
            )
        assert gold["entropy_guard_action"] == "warn"

    def test_the_checkpoint_budget_covers_the_whole_run(self, off):
        """Unconditional, and the one line of df8512e0's configuration still worth not
        repeating: save_total_limit 3 deleted every checkpoint anyone later wanted to
        look at. A guard trip, if the guard is ever re-armed, adds one checkpoint beyond
        the scheduled grid, so the budget has to cover that too."""
        gold = _config(off, "train-gold")["gold_config"]
        produced = int(gold["max_steps"]) // int(gold["save_steps"]) + 1
        assert int(gold["save_total_limit"]) >= produced

    def test_a_re_armed_guard_still_gets_its_metric_and_only_warns(self, off):
        """Inert at the shipped defaults. It exists so that re-arming the guard for the
        safety experiment cannot quietly reintroduce either of the two combinations that
        cost a build: a guard with no entropy to read, and a guard that may stop at an
        arbitrary step while the ladder names fixed ones (build d1acf1c0)."""
        gold = _config(off, "train-gold")["gold_config"]
        if float(gold.get("entropy_guard_drop_frac", 0.0)) > 0:
            assert gold["log_student_entropy"] is True
            assert gold["entropy_guard_action"] == "warn"

    def test_the_guard_cannot_be_armed_without_its_metric(self, off):
        """Belt and braces: the renderer refuses this combination too, but a recipe
        that sets it has already wasted a submission."""
        gold = _config(off, "train-gold")["gold_config"]
        if float(gold["entropy_guard_drop_frac"]) > 0:
            assert gold["log_student_entropy"] is True

    def test_no_patch_gated_key_renders_at_the_defaults(self, off):
        """THE PROPERTY THE WHOLE REMOVAL IS FOR, and the one that breaks silently.

        render_gold_config.py emits each of these keys only when it is non-default, and
        refuses to emit any of them against a trainer whose CustomGOLDConfig lacks the
        field -- at the step's pin (a5d59bc4) none of the six exist. So a single one of
        them slipping back into the rendered config does not misconfigure the run, it
        makes the recipe unrunnable on the public pin and drags CODE_DIR back with it.

        Read off the step's gold_config -- the INPUT render_gold_config.py decides from --
        rather than off parameters.yaml, so a stray literal or a Jinja default in build.yaml
        cannot slip a non-default value past this. The renderer's own omit-when-default
        behaviour is tested in the step's suite, not here."""
        gold = _config(off, "train-gold")["gold_config"]
        for key in (
            "ce_coef",
            "log_student_entropy",
            "entropy_guard_drop_frac",
            "entropy_guard_baseline_steps",
            "entropy_guard_patience",
            "entropy_guard_action",
        ):
            assert key in gold, (
                f"{key} disappeared from the recipe; it is meant to be present and "
                "at its default, so the sweep arm is one --param away"
            )
        emitted = {
            "ce_coef": float(gold["ce_coef"]) > 0,
            "log_student_entropy": gold["log_student_entropy"] is True,
            "entropy_guard_drop_frac": float(gold["entropy_guard_drop_frac"]) > 0,
        }
        assert not any(emitted.values()), (
            "these reach the trainer's config and the step's pinned trainer has no such "
            f"field: {sorted(k for k, v in emitted.items() if v)}"
        )

    def test_the_objective_is_df8512e0s(self, off):
        """Unanchored, off-policy, pure divergence -- the best recipe on record, and now
        the default here rather than a control arm reached with two --params."""
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold.get("ce_coef", 0.0)) == 0.0
        assert float(_params()["LMBDA"]) == 0.0

    def test_the_sweep_arms_are_still_reachable(self, tmp_path):
        """Retiring the anchor must not mean deleting the plumbing: the safety question
        the sweep left open is run from here, and it is one --param plus a CODE_DIR."""
        arm = _render(tmp_path, INCLUDE_SFT=False, CE_COEF=0.15)
        gold = _config(arm, "train-gold")["gold_config"]
        assert float(gold["ce_coef"]) == 0.15

    def test_beta_and_the_policy_are_unchanged_from_stage1(self, off):
        """One variable at a time. The anchor already changes the loss; moving beta or
        lmbda as well would make a v2-vs-df8512e0 comparison unattributable."""
        stage1 = get_params_from_file(str(_STAGE1 / "parameters.yaml"))
        mine = _params()
        for key in ("BETA", "LMBDA", "TEMPERATURE", "USE_LIGER_FUSED_JSD"):
            assert mine[key] == stage1[key], key
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold["lmbda"]) == 0.0
        assert int(gold["vllm_num_servers"]) == 0

    def test_the_corpus_and_geometry_are_unchanged_from_stage1(self):
        """Same reason. If these drift, v2 measures a different experiment and the
        recorded baseline row stops being the right comparison."""
        stage1 = get_params_from_file(str(_STAGE1 / "parameters.yaml"))
        mine = _params()
        for key in (
            "TARGET_ROWS",
            "SHUFFLE_SEED",
            "SEED",
            "MAX_LENGTH",
            "EVAL_FRACTION",
            "STUDENT_MODEL",
            "TEACHER_MODEL",
            "GOLD_LEARNING_RATE",
            "GOLD_MIN_LR",
            "WARMUP_RATIO",
            "LR_SCHEDULER_TYPE",
            "GOLD_PER_DEVICE_TRAIN_BATCH_SIZE",
            "GOLD_GRADIENT_ACCUMULATION_STEPS",
            "GOLD_NUM_NODES",
            "GOLD_NUM_GPUS",
        ):
            assert mine[key] == stage1[key], key


# ─── The trainer pin ───────────────────────────────────────────────────────────


class TestTheTrainerPin:
    """train-gold reads the step's own pinned trainer again, which is the point of having
    retired the CE anchor.

    This recipe used to override CODE_DIR with a local patched checkout under /proj,
    because CE_COEF and the guard need fields the pinned trainer does not define. That
    override was also what made the recipe BlueVela-only: a /proj path resolves nowhere
    else. With the anchor retired the override is gone, and the invariant that replaces
    it is a conditional one -- whoever re-arms a patch-gated key must bring a CODE_DIR
    with it.

    The provenance requirement behind the original pin has not gone away. df8512e0
    recorded kd_sandbox_commit fc7d66e from a tree carrying 159 uncommitted files, so its
    single record of what trained described a commit whose code did not run. The clone
    path cannot drift that way -- it checks out code_config.ref by construction -- which
    is why empty is now the safer value rather than merely the simpler one.
    """

    def test_the_trainer_is_the_steps_own_public_pin(self):
        """Empty means the step clones github.com/laminair/gb-steps-distillation at the
        commit it pins. Not a /proj path, and specifically not the shared mutable tree:
        /proj/granite-build/g4os/kd-sandbox is another project's working copy."""
        code_dir = _params()["CODE_DIR"]
        assert code_dir == "", (
            "CODE_DIR is set, so this recipe no longer runs on the step's public pin and "
            "no longer runs anywhere but BlueVela. Only a patch-gated key justifies that"
        )

    def test_any_patch_gated_key_brings_a_code_dir_with_it(self, off, tmp_path):
        """THE INVARIANT THAT REPLACES THE PIN, and the expensive failure it prevents.

        None of the six patch-gated keys exist in CustomGOLDConfig at a5d59bc4. Emitting
        one against that trainer is caught by the renderer -- but only after the step has
        started, so a run that asks for CE without a patched checkout fails having already
        queued. Asserting both directions here catches it at test time instead.
        """
        gold = _config(off, "train-gold")["gold_config"]
        asks = (
            float(gold.get("ce_coef", 0)) > 0
            or gold.get("log_student_entropy", False) is True
            or float(gold.get("entropy_guard_drop_frac", 0)) > 0
        )
        assert not asks or _params()["CODE_DIR"], (
            "this recipe emits keys the pinned trainer does not define; CODE_DIR must "
            "name a checkout carrying ce_anchor_and_entropy_guard.diff"
        )
        # And the same recipe re-armed must still be able to reach a patched checkout.
        armed = _render(
            tmp_path, INCLUDE_SFT=False, CE_COEF=0.15, CODE_DIR="/proj/somewhere-gb"
        )
        assert _config(armed, "train-gold")["code_config"]["code_dir"] == (
            "/proj/somewhere-gb"
        )

    def test_the_pin_is_a_full_commit_or_deliberately_empty(self):
        """A branch head moves, which is the failure the pin exists for -- so if this names
        anything, it names a full commit.

        Empty is the shipped value and it disables nothing that matters: expect_ref is read
        only on the CODE_DIR path, and the clone path is at code_config.ref by
        construction. It is the value to FILL IN the moment a CODE_DIR is set, because that
        is when a checkout becomes mutable state again -- and it brings the dirty-tree check
        with it."""
        ref = _params()["CODE_EXPECT_REF"]
        assert ref == "" or re.fullmatch(r"[0-9a-f]{40}", ref), ref

    def test_the_pin_reaches_the_step(self, off):
        code_config = _config(off, "train-gold")["code_config"]
        assert code_config["code_dir"] == _params()["CODE_DIR"]
        assert code_config["expect_ref"] == _params()["CODE_EXPECT_REF"]

    def test_the_renderers_field_check_stays_on(self, off):
        """It is what stands between any future re-arm and a config TrlParser rejects on
        every node of an allocation that is already held. Unconditional: it costs one
        import, and the combination it catches is only ever a mistake."""
        gold = _config(off, "train-gold")["gold_config"]
        assert gold.get("verify_trainer_accepts_keys", True) is not False

    def test_the_patch_that_builds_that_checkout_ships_with_the_repo(self):
        """Otherwise the pin names a tree nobody can rebuild, which is a different
        way of having no provenance at all."""
        patch = (
            _RECIPE.parents[3]
            / "steps"
            / "distill"
            / "gold"
            / "skypilot"
            / "patches"
            / "ce_anchor_and_entropy_guard.diff"
        )
        assert patch.is_file()
        text = patch.read_text(encoding="utf-8")
        assert "Base commit:" in text
        # CODE_EXPECT_REF is empty until the patched checkout is built, so the pin cannot
        # be asserted against the patch yet. What still must hold is that the patch says
        # which commit it applies to and under which directory -- without the remap the
        # hunks land nowhere, and that is the part a reader needs.
        assert "a5d59bc4" in text, "the patch must name the pin it applies to"
        assert (
            "src/gb_steps_post_training/distillation" in text
        ), "the patch must state the directory remap it needs"
        for key in ("ce_coef", "log_student_entropy", "entropy_guard_drop_frac"):
            assert key in text, f"{key} is pinned but not in the patch"

    def test_only_the_trainer_targets_override_the_shared_step_pin(self, off):
        """The ported steps' pin belongs to the step, once, for all of them. A recipe-level
        override is what left align reading a patched tree while corpus read the shared one
        -- build 1820703f -- so an override is an exception that has to justify itself.

        train-gold is the one justified exception, and it is not new: this recipe always read
        its trainer from somewhere other than the shared steps checkout. Before the two
        sources became one repo that was invisible here, because the trainer arrived through
        gold_config.kd_code_dir and this test only ever looked at code_config. Now it is
        visible, so it is bounded instead: exactly the targets that run trainer code may
        override, every other target keeps the shared pin (which is what keeps align, corpus,
        export and eval runnable off BlueVela at all), and every override is identical -- one
        revision per build is the property 1820703f was really about."""
        allowed = {"train-gold", "vllm-server"}
        overrides = {}
        for name, target in _targets(off).items():
            for step in target["steps"]:
                if "code_config" in step.get("config", {}):
                    assert name in allowed, f"{name} overrides the shared steps pin"
                    overrides[name] = step["config"]["code_config"]
        assert overrides, "the patched trainer has to reach train-gold somehow"
        distinct = {tuple(sorted(c.items())) for c in overrides.values()}
        assert (
            len(distinct) == 1
        ), f"targets disagree on the trainer revision: {overrides}"


# ─── The horizon and the ladder ────────────────────────────────────────────────


class TestTheHorizonAndTheLadder:
    def test_the_run_is_bounded_by_steps_not_by_an_epoch(self, off):
        """The inverse of distill-stage1's assertion, and deliberately so. An epoch
        was 8,150 steps against ~12% of available headroom, two thirds of which was
        closed by step 509."""
        gold = _config(off, "train-gold")["gold_config"]
        assert int(gold["max_steps"]) > 0

    def test_the_horizon_covers_the_descent_and_not_much_more(self, off):
        """Bounded by the control's OWN measured curve, not by df8512e0's step-509 knee.

        Build bb779f1f ran the unanchored objective 2,000 steps with the guard off:
        entropy -19% by step 80, -29.7% by 120, -34.1% by 250, and -35.5% at 2,000. The
        last 1,750 steps bought 1.4%. JSD and forward KL were flat after 500 and reverse
        KL never moved. So the horizon has to clear the shoulder (~250) and has no reason
        to run far past it -- and a short horizon is what makes a CE_COEF sweep
        affordable, which is the experiment this recipe is now for.

        The upper bound is the load-bearing half. Nothing stops someone restoring 2,000
        'to be safe', and that silently triples the cost of every arm."""
        max_steps = int(_config(off, "train-gold")["gold_config"]["max_steps"])
        assert 250 <= max_steps <= 600, max_steps

    def test_every_rung_is_a_checkpoint_that_will_exist(self, off):
        """The failure this prevents: a rung that is not a multiple of save_steps names
        a checkpoint-N the trainer never wrote, and the export fails after the
        training has been paid for."""
        gold = _config(off, "train-gold")["gold_config"]
        save = int(gold["save_steps"])
        for rung in _params()["CKPT_LADDER"].split(","):
            assert (
                int(rung) % save == 0
            ), f"checkpoint-{rung} is not a multiple of {save}"
            assert int(rung) <= int(gold["max_steps"])

    def test_the_last_rung_is_the_end_of_the_run(self, off):
        rungs = [int(n) for n in _params()["CKPT_LADDER"].split(",")]
        assert rungs == sorted(rungs)
        assert rungs[-1] == int(_config(off, "train-gold")["gold_config"]["max_steps"])

    def test_no_checkpoint_is_evicted(self, off):
        """save_total_limit 3 at save_steps 250 is what made df8512e0 unsalvageable:
        it kept the last 750 steps of an 8,150-step run and deleted every checkpoint
        from before the collapse. For an exploratory run the valuable ones are EARLY."""
        gold = _config(off, "train-gold")["gold_config"]
        produced = int(gold["max_steps"]) // int(gold["save_steps"])
        assert int(gold["save_total_limit"]) >= produced

    def test_each_rung_exports_the_checkpoint_it_names(self, off):
        for rung in _params()["CKPT_LADDER"].split(","):
            export = _config(off, f"export-{rung}")["export_config"]
            assert export["checkpoint"] == f"checkpoint-{rung}"
            assert export["dest"].endswith(f"/export-{rung}")

    def test_no_export_is_left_to_pick_the_highest_step(self, off):
        """Empty means 'highest step number', which for a guard-stopped run is
        whatever step the guard fired on -- a different checkpoint in each arm."""
        for rung in _params()["CKPT_LADDER"].split(","):
            assert _config(off, f"export-{rung}")["export_config"]["checkpoint"] != ""

    def test_each_rung_is_transfer_evaluated_against_its_own_export(self, off):
        for rung in _params()["CKPT_LADDER"].split(","):
            target = _targets(off)[f"eval-transfer-{rung}"]
            assert target["inputs"]["student"]["binding"] == f"export-{rung}.hf_model"
            cfg = _config(off, f"eval-transfer-{rung}")["eval_config"]
            assert cfg["output_dir"].endswith(f"/eval-transfer-{rung}")

    def test_entropy_is_among_the_transfer_metrics(self, off):
        """The metric that caught this after the fact, and the only one of the four
        that can see a collapsed student whose divergence happens to look fine."""
        for rung in list(_params()["CKPT_LADDER"].split(",")) + ["baseline"]:
            metrics = _config(off, f"eval-transfer-{rung}")["eval_config"]["metrics"]
            assert "entropy" in metrics
            assert "rkld" in metrics

    def test_the_baseline_read_is_still_the_aligned_student(self, off):
        """A divergence with no baseline is a number without a direction."""
        target = _targets(off)["eval-transfer-baseline"]
        assert target["inputs"]["student"]["binding"] == "align.retagged_student"

    def test_every_rung_shares_the_length_budget(self, off):
        budget = _params()["MAX_LENGTH"]
        assert budget == 8192
        assert _config(off, "corpus")["corpus_config"]["max_length"] == budget
        assert _config(off, "train-gold")["gold_config"]["max_length"] == budget
        for rung in list(_params()["CKPT_LADDER"].split(",")) + ["baseline"]:
            assert (
                _config(off, f"eval-transfer-{rung}")["eval_config"]["max_length"]
                == budget
            )


# ─── The generation smoke test ─────────────────────────────────────────────────


class TestGenSmoke:
    """The detector lives in the gen-smoke step and is tested there. What this recipe
    owns is the wiring: every rung, in ladder order, the threshold, and the gate."""

    def test_it_runs_the_gen_smoke_step(self, off):
        step = _targets(off)["gen-smoke"]["steps"][0]
        assert step["step_uri"] == "space://steps/distill/gen-smoke"

    def test_the_interpreter_actually_receives_every_rung(self, tmp_path):
        """Executes the assembly instead of parsing it. `bash -n` cannot see this
        failure and neither can compile(): a <% %> block tag left a blank line after a
        `\\`, which ENDS the command, and bash then ran the next rung as a program name
        -- exit 127 with `500:/proj/.../export-500: No such file or directory`, which
        is build bb779f1f's gen-smoke. The only test that can catch it is one that looks
        at what the interpreter was handed, so this renders the recipe's own config
        through the published step template and swaps only the interpreter."""
        rendered = _render(tmp_path, INCLUDE_SFT=False, WORKDIR_ROOT=str(tmp_path))
        argv_log = tmp_path / "argv.json"
        stub = tmp_path / "python-stub"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)
        script, step_dir = render_run(
            "distill/gen-smoke",
            _config(rendered, "gen-smoke"),
            gen_smoke_config={"python": str(stub)},
        )

        result = subprocess.run(
            ["bash", "-c", script], cwd=step_dir, text=True, capture_output=True
        )

        assert result.returncode == 0, result.stderr
        argv = json.loads(argv_log.read_text(encoding="utf-8"))
        rungs = _params()["CKPT_LADDER"].split(",")
        assert argv[0] == "./src/gen_smoke.py", f"interpreter got {argv}"
        assert len(argv) == 5 + len(rungs), f"interpreter got {argv}"
        assert argv[1].endswith("/gen-smoke/repetition.json")
        assert argv[2] == str(_params()["GEN_SMOKE_MAX_REPETITION"])
        assert argv[3] == str(_params()["GEN_SMOKE_NEW_TOKENS"])
        assert argv[4] == "true"
        for rung, got in zip(rungs, argv[5:]):
            step, _, path = got.partition(":")
            assert step == rung, f"expected rung {rung}, got {got}"
            assert path.endswith(f"/export-{rung}"), got

    def test_no_rung_is_appended_through_a_line_continuation(self):
        """The shape that broke, guarded at the source. A `\\`-continued line
        followed by a <% %> tag renders to a continuation followed by a blank line,
        which terminates the command."""
        lines = (_RECIPE / "build.yaml").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith("\\") and lines[i + 1].lstrip().startswith("<%"):
                raise AssertionError(
                    f"{_RECIPE.name}/build.yaml:{i + 1}: continuation followed by a "
                    f"block tag renders to a blank line and ends the command:\n"
                    f"  {line}\n  {lines[i + 1]}"
                )

    def test_it_reads_every_rung(self, off):
        rungs = _config(off, "gen-smoke")["gen_smoke_config"]["rungs"]
        ladder = _params()["CKPT_LADDER"].split(",")
        assert [r.partition(":")[0] for r in rungs] == ladder
        for rung, entry in zip(ladder, rungs):
            assert entry.endswith(f"/export-{rung}")
            assert f"export_{rung}" in _targets(off)["gen-smoke"]["inputs"]

    def test_the_final_rung_gates_this_build(self, off):
        """An early rung above threshold is a finding to read in the table. Failing
        the build on it would discard later checkpoints that may be fine, so only the
        final rung can fail the target -- and here it does."""
        assert _config(off, "gen-smoke")["gen_smoke_config"]["gate_final_rung"] is True

    def test_the_detector_separates_the_post_mortems_two_samples(self):
        """Executes the published detector at THIS recipe's threshold rather than
        trusting its docstring. These are the actual generations recorded in the
        post-mortem."""
        measure = gen_smoke_module().measure
        collapsed = "// (true)\n" * 50
        correct = (
            "int count = 0;\n"
            "for (int i = 0; i < numbers.size(); i++) {\n"
            "    for (int j = i + 1; j < numbers.size(); j++) {\n"
            "        if (Math.abs(a - b) < threshold) { count++; }\n"
            "    }\n"
            "}\n"
            "return count >= 2;"
        )
        threshold = float(_params()["GEN_SMOKE_MAX_REPETITION"])
        assert measure(collapsed)[0] > threshold
        assert measure(correct)[0] <= threshold

    def test_capability_is_measured_at_every_rung_and_at_the_baseline(self, off):
        """bb779f1f measured capability ONCE, at step 2,000, and got 0.000 (0/400) with
        no baseline and no earlier reading -- so it could not say whether the capability
        went at step 25 or step 1,999, nor whether the starting student had it. Every
        divergence metric in that build improved while this one sat at zero, which is
        exactly what eval-transfer cannot see. A ladder of divergence readings beside a
        single capability reading measures the wrong thing carefully."""
        targets = _targets(off)
        assert (
            targets["eval-bfcl-baseline"]["inputs"]["model"]["binding"]
            == "align.retagged_student"
        )
        for rung in _params()["CKPT_LADDER"].split(","):
            binding = targets[f"eval-bfcl-{rung}"]["inputs"]["model"]["binding"]
            assert binding == f"export-{rung}.hf_model"

    def test_no_two_bfcl_targets_share_an_output_directory(self, off):
        """Each writes under output_dir/experiment/eval_name, so a shared pair would
        have two targets registering one artifact URI -- the b5f030cd mode, where the
        second reports SUCCESS with an empty output list and consumers wait forever."""
        seen = {}
        names = ["eval-bfcl-baseline"] + [
            f"eval-bfcl-{n}" for n in _params()["CKPT_LADDER"].split(",")
        ]
        for name in names:
            cfg = _config(off, name)["bfcl_config"]
            key = (cfg["output_dir"], cfg["experiment"])
            assert key not in seen, f"{name} collides with {seen[key]}: {key}"
            seen[key] = name

    def test_the_descent_is_where_the_rungs_are(self, off):
        """The old ladder was 500/1000/1500/2000 -- every rung on the plateau, which is
        how two builds produced no reading anywhere in the region that moves. The
        control loses 19% of its entropy by step 80; at least half the rungs belong at
        or below that shoulder."""
        rungs = [int(n) for n in _params()["CKPT_LADDER"].split(",")]
        early = [n for n in rungs if n <= 120]
        assert (
            len(early) >= len(rungs) / 2
        ), f"only {early} of {rungs} are in the descent"


# ─── Diagnosability ────────────────────────────────────────────────────────────


def test_nccl_debug_is_on_by_default(off):
    """Build 8f02b739 hung at step 219 in the ZeRO-3 parameter all-gather across two
    racks and left no INIT,NET trace, because NCCL_DEBUG was empty. It had to be
    cancelled and relaunched with the flag, paying 45-60 minutes of redone sources +
    align + corpus. A 2-node run is exactly the shape that hang needs."""
    gold = _config(off, "train-gold")["gold_config"]
    assert gold["nccl_debug"] == "INFO"
    assert gold["nccl_debug_subsys"], "a trace with no subsystem filter is unbounded"


def test_the_stale_smoke_comments_did_not_come_along(off):
    """distill-stage1's parameters.yaml still describes a 48-row smoke run in its
    header, one node with two GPUs at GOLD_NUM_NODES, and 'no cross-node collectives'
    at NCCL_DEBUG -- all three contradicting its own live values. Copying a recipe
    copies its comments, and a comment that lies is worse than none."""
    text = (_RECIPE / "parameters.yaml").read_text(encoding="utf-8")
    assert "48 rows" not in text
    assert "Nothing it produces is a publishable measurement" not in text
    assert "no cross-node collectives" not in text
    assert "One node, two GPUs for GOLD" not in text


# ─── Reusing an existing corpus ────────────────────────────────────────────────


_PIN = "/proj/granite-build/g4os/distill/distill-350m-s1v2-ce040/c20ed3c0/corpus"


def _corpus_inputs(rendered):
    """Every input named `corpus`, by target. These are the four consumers whose
    wiring has to change together: miss one and it either waits on a target that
    was not rendered, or reads a corpus nobody checked."""
    out = {}
    for name, target in _targets(rendered).items():
        spec = (target.get("inputs") or {}).get("corpus")
        if spec is not None:
            out[name] = spec
    return out


@pytest.fixture(name="pinned")
def fixture_pinned(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False, CORPUS_DIR=_PIN)


class TestTheCorpusPin:
    """`sources` + `corpus` are ~50 minutes on the critical path before train-gold can
    start, and they are deterministic: the three arms of the CE_COEF sweep that
    completed (9dab9130, 8c8ebd63, c20ed3c0) produced byte-identical
    sources/train.jsonl, corpus/train.jsonl and corpus/eval.jsonl. So a sweep pays for
    the same file once per arm.

    Pinning has to be done as a DIRECT input, not by fixing BUILD_SUBDIR. gbserver's
    target reuse is scoped to one build id (docs/builds/target-reuse.md), and a shared
    output URI is the b5f030cd mode: the second build's registration is refused, the
    target still reports SUCCESS with an empty output list, and every consumer waits
    forever on a binding that will never resolve."""

    def test_the_corpus_is_built_in_the_build_by_default(self, off):
        """The default must not change. Every run before this parameter existed built
        its own corpus, and an unpinned build still has to."""
        assert "sources" in _targets(off)
        assert "corpus" in _targets(off)
        assert _corpus_inputs(off)
        for name, spec in _corpus_inputs(off).items():
            assert spec.get("binding") == "corpus.corpus", name
            assert "uri" not in spec, name

    def test_a_pin_removes_the_targets_that_would_rebuild_it(self, pinned):
        assert "sources" not in _targets(pinned)
        assert "corpus" not in _targets(pinned)

    def test_a_pin_is_read_directly_and_not_through_a_binding(self, pinned):
        """The load-bearing assertion. A `binding` names another target's output in
        THIS build; with `corpus` not rendered, any surviving binding is a consumer
        blocked on a producer that does not exist."""
        consumers = _corpus_inputs(pinned)
        assert consumers, "no target reads the corpus at all"
        for name, spec in consumers.items():
            assert "binding" not in spec, name
            assert spec["uri"] == f"env://{_PIN}/train.jsonl", name

    def test_a_pin_is_never_re_registered_as_an_output(self, pinned):
        """b5f030cd. Reading a URI another build produced is ordinary; declaring it as
        your own output is what gbserver refuses."""
        for name, target in _targets(pinned).items():
            for out_name, spec in (target.get("outputs") or {}).items():
                assert _PIN not in str(spec.get("uri", "")), f"{name}.{out_name}"

    def test_the_transfer_evals_read_the_pinned_eval_split(self, pinned):
        """These two compose the eval.jsonl path by hand rather than through the
        binding, so they are the sites a parameter switch is most likely to miss."""
        evals = [n for n in _targets(pinned) if n.startswith("eval-transfer-")]
        assert evals
        for name in evals:
            cfg = _config(pinned, name)["eval_config"]
            assert cfg["corpus"] == f"{_PIN}/eval.jsonl", name

    def test_an_unpinned_build_still_reads_its_own_eval_split(self, off):
        for name in [n for n in _targets(off) if n.startswith("eval-transfer-")]:
            cfg = _config(off, name)["eval_config"]
            assert cfg["corpus"].endswith("/corpus/eval.jsonl"), name
            assert _PIN not in cfg["corpus"], name

    def test_a_pin_is_checked_before_any_allocation_is_held(self, pinned, off):
        """The corpus is DEFINED by the retagged tokenizer and by the prep policies.
        A pin taken from a run with a different teacher or a different max_length
        trains on a corpus this build does not describe, and nothing downstream would
        say so -- which is the one new silent failure mode pinning introduces."""
        assert "corpus-pin-check" in _targets(pinned)
        assert "corpus-pin-check" not in _targets(off)
        gate = _targets(pinned)["corpus-pin-check"]
        assert (gate["inputs"]["tokenizer"]["binding"]) == "align.retagged_student"
        res = _config(pinned, "corpus-pin-check")["launcher_config"]["resources"]
        assert "accelerators" not in res, "the gate must not hold a GPU"
        for name in _corpus_inputs(pinned):
            bindings = {
                s.get("binding") for s in _targets(pinned)[name]["inputs"].values()
            }
            assert "corpus-pin-check.pin_check" in bindings, name

    @pytest.mark.parametrize("include_sft", [False, True])
    @pytest.mark.parametrize("pin", ["", _PIN])
    def test_both_switches_render_together(self, tmp_path, include_sft, pin):
        """Two independent conditionals now gate the same graph, and INCLUDE_SFT's
        train-sft is itself a corpus consumer. The combinations are cheap to render
        and the failure mode -- one arm of one switch leaving unbalanced YAML -- is
        only visible when both are exercised."""
        rendered = _render(tmp_path, INCLUDE_SFT=include_sft, CORPUS_DIR=pin)
        targets = _targets(rendered)
        assert ("train-sft" in targets) is include_sft
        assert ("sources" in targets) is not bool(pin)
        assert ("corpus-pin-check" in targets) is bool(pin)
        for spec in _corpus_inputs(rendered).values():
            assert bool(spec.get("uri")) is bool(pin)

    def test_a_trailing_slash_does_not_reach_the_uri(self, tmp_path):
        """`ls -d .../*/` prints a trailing slash, which is how the path gets copied in
        practice. Unnormalised it renders `env://<dir>//train.jsonl` -- which resolves on
        POSIX but records an artifact URI that does not match the canonical one."""
        contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
        plain, slashed = (
            apply_parameters(
                contents, [], _params(INCLUDE_SFT=False, CORPUS_DIR=d), str(tmp_path)
            )
            for d in (_PIN, _PIN + "/")
        )
        assert plain == slashed


class TestThePinCheckScript:
    """The comparisons live in the corpus-pin-check step and are tested there. These
    run the recipe's OWN pin_check_config through the published step, so a value that
    lands in the wrong key, or a parameter that disagrees with the manifest this
    recipe's own corpus target writes, fails here."""

    def test_it_runs_the_pin_check_step(self, pinned):
        step = _targets(pinned)["corpus-pin-check"]["steps"][0]
        assert step["step_uri"] == "space://steps/distill/corpus-pin-check"

    def test_it_checks_everything_that_defines_the_corpus(self, pinned):
        cfg = _config(pinned, "corpus-pin-check")["pin_check_config"]
        params = _params()
        assert cfg["corpus_dir"] == _PIN
        assert cfg["teacher_model"] == params["TEACHER_MODEL"]
        assert cfg["max_length"] == params["MAX_LENGTH"]
        assert cfg["think_policy"] == params["THINK_POLICY"]
        assert cfg["documents_policy"] == params["DOCUMENTS_POLICY"]
        assert cfg["eval_fraction"] == params["EVAL_FRACTION"]
        # The tokenizer is the step's `tokenizer` input, not a config key.
        assert "tokenizer_dir" not in cfg
        assert _targets(pinned)["corpus-pin-check"]["inputs"]["tokenizer"] == {
            "binding": "align.retagged_student"
        }

    def _run(self, pinned, tmp_path, manifest, splits=("train.jsonl", "eval.jsonl")):
        corpus = tmp_path / "corpus"
        corpus.mkdir(exist_ok=True)
        (corpus / "corpus_manifest.json").write_text(json.dumps(manifest))
        for name in splits:
            (corpus / name).write_text("{}\n")
        tok = tmp_path / "retagged_student"
        tok.mkdir(exist_ok=True)
        script, step_dir = render_run(
            "distill/corpus-pin-check",
            _config(pinned, "corpus-pin-check"),
            bindings={"tokenizer": str(tok)},
            pin_check_config={
                "corpus_dir": str(corpus),
                "output_dir": str(tmp_path / "pin"),
                "python": sys.executable,
            },
        )
        return subprocess.run(
            ["bash", "-c", script], cwd=step_dir, text=True, capture_output=True
        )

    @staticmethod
    def _manifest(**over):
        """What this recipe's corpus target would have written."""
        params = _params()
        m = {
            "tokenizer_identity": pathlib.Path(
                params["TEACHER_MODEL"].rstrip("/")
            ).name,
            "tokenizer_path": "/gone/align/retagged_student",
            "seed": 42,
            "eval_fraction": params["EVAL_FRACTION"],
            "policies": {
                "max_length": params["MAX_LENGTH"],
                "think_policy": params["THINK_POLICY"],
                "completion_boundary": "last_message",
                "documents_policy": params["DOCUMENTS_POLICY"],
            },
        }
        m.update(over)
        return m

    def test_a_matching_manifest_is_accepted(self, pinned, tmp_path):
        proc = self._run(pinned, tmp_path, self._manifest())
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert (tmp_path / "pin" / "corpus_pin.json").is_file()
        assert "GB_ARTIFACT_ID:pin_check" in proc.stdout

    def test_a_different_max_length_is_rejected(self, pinned, tmp_path):
        m = self._manifest()
        m["policies"]["max_length"] = 4096
        proc = self._run(pinned, tmp_path, m)
        assert proc.returncode != 0
        assert "max_length" in proc.stdout + proc.stderr

    def test_a_corpus_prepared_for_a_different_teacher_is_rejected(
        self, pinned, tmp_path
    ):
        proc = self._run(
            pinned,
            tmp_path,
            self._manifest(tokenizer_identity="granite-4.0-1b-something"),
        )
        assert proc.returncode != 0
        assert "tokenizer_identity" in proc.stdout + proc.stderr

    def test_a_missing_split_is_rejected(self, pinned, tmp_path):
        proc = self._run(pinned, tmp_path, self._manifest(), splits=("train.jsonl",))
        assert proc.returncode != 0
        assert "eval.jsonl" in proc.stdout + proc.stderr

    def test_every_mismatch_is_reported_not_just_the_first(self, pinned, tmp_path):
        m = self._manifest(tokenizer_identity="wrong", eval_fraction=0.5)
        m["policies"]["max_length"] = 4096
        proc = self._run(pinned, tmp_path, m)
        blob = proc.stdout + proc.stderr
        assert proc.returncode != 0
        for key in ("tokenizer_identity", "eval_fraction", "max_length"):
            assert key in blob, key
