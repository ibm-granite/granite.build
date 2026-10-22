"""Best-epoch checkpointing for the HF SFT drivers.

Final training must persist the *best* epoch by ``eval_loss``, not the last step.
These tests pin the strategy resolution (pure, no cluster needed) and assert the
HF ``load_best_model_at_end`` preconditions that would otherwise fail only at
``TrainingArguments`` construction time inside a Ray worker — where the traceback
is far from the cause.
"""

import tempfile

import pytest

from autotune.trainers._checkpointing import (
    BEST_MODEL_GREATER_IS_BETTER,
    BEST_MODEL_METRIC,
    FINAL_SAVE_TOTAL_LIMIT,
    HPO_SAVE_TOTAL_LIMIT,
    resolve_checkpoint_strategy,
)

FINAL_CASES = [
    pytest.param(3, 100, id="multi_epoch"),
    pytest.param(1, 100, id="single_epoch_steps"),
    pytest.param(1, 3, id="single_epoch_tiny"),
    pytest.param(1, 1, id="single_epoch_one_step"),
]


class TestFinalTrainingTracksBestModel:
    """The whole point: final training keeps the best epoch, scored on eval loss."""

    @pytest.mark.parametrize("epochs,steps", FINAL_CASES)
    def test_load_best_model_is_on(self, epochs, steps):
        s = resolve_checkpoint_strategy(hpo_search=False, num_train_epochs=epochs, steps_per_epoch=steps)
        assert s.load_best_model_at_end is True

    @pytest.mark.parametrize("epochs,steps", FINAL_CASES)
    def test_metric_is_eval_loss_lower_is_better(self, epochs, steps):
        kw = resolve_checkpoint_strategy(
            hpo_search=False, num_train_epochs=epochs, steps_per_epoch=steps
        ).as_training_arguments()
        assert kw["metric_for_best_model"] == "eval_loss"
        # Loss decreases, so a *greater* value is not better. Getting this wrong
        # would select the single worst checkpoint.
        assert kw["greater_is_better"] is False

    @pytest.mark.parametrize("epochs,steps", FINAL_CASES)
    def test_eval_and_save_strategies_match(self, epochs, steps):
        # HF rejects load_best_model_at_end unless the two strategies agree.
        s = resolve_checkpoint_strategy(hpo_search=False, num_train_epochs=epochs, steps_per_epoch=steps)
        assert s.eval_strategy == s.save_strategy

    @pytest.mark.parametrize("epochs,steps", FINAL_CASES)
    def test_save_steps_multiple_of_eval_steps(self, epochs, steps):
        s = resolve_checkpoint_strategy(hpo_search=False, num_train_epochs=epochs, steps_per_epoch=steps)
        if s.save_strategy == "steps":
            assert s.eval_steps is not None and s.eval_steps >= 1
            assert s.save_steps % s.eval_steps == 0

    def test_multi_epoch_uses_epoch_boundaries(self):
        s = resolve_checkpoint_strategy(hpo_search=False, num_train_epochs=4, steps_per_epoch=50)
        assert s.save_strategy == "epoch"
        # No steps cadence on the epoch path, or HF would ignore the strategy.
        assert s.save_steps is None and s.eval_steps is None

    def test_single_epoch_falls_back_to_steps(self):
        # One epoch boundary gives nothing to choose between, so evaluate within it.
        s = resolve_checkpoint_strategy(hpo_search=False, num_train_epochs=1, steps_per_epoch=100)
        assert s.save_strategy == "steps"
        assert s.save_steps == 20 and s.eval_steps == 20

    def test_cadence_never_zero(self):
        # steps_per_epoch < SINGLE_EPOCH_CHECKPOINTS would floor-divide to 0,
        # which HF rejects.
        s = resolve_checkpoint_strategy(hpo_search=False, num_train_epochs=1, steps_per_epoch=2)
        assert s.save_steps >= 1

    @pytest.mark.parametrize("epochs,steps", FINAL_CASES)
    def test_keeps_several_checkpoints(self, epochs, steps):
        s = resolve_checkpoint_strategy(hpo_search=False, num_train_epochs=epochs, steps_per_epoch=steps)
        assert s.save_total_limit == FINAL_SAVE_TOTAL_LIMIT


class TestHpoDoesNotTrackBestModel:
    """HPO trials report metrics and persist no model, so best-tracking is off."""

    def test_multi_gpu_hpo_saves_once_per_epoch(self):
        # The save exists only so RayTrainReportCallback.on_save() fires.
        s = resolve_checkpoint_strategy(hpo_search=True, num_train_epochs=3, steps_per_epoch=100)
        assert s.load_best_model_at_end is False
        assert s.save_strategy == "epoch"
        assert s.save_total_limit == HPO_SAVE_TOTAL_LIMIT

    def test_single_gpu_hpo_saves_nothing(self):
        s = resolve_checkpoint_strategy(
            hpo_search=True, num_train_epochs=3, steps_per_epoch=100, hpo_save_strategy="no"
        )
        assert s.save_strategy == "no"
        assert s.load_best_model_at_end is False
        assert s.save_total_limit is None

    def test_no_best_metric_keys_when_off(self):
        # Passing metric_for_best_model without load_best_model_at_end would make
        # HF track a metric it never acts on.
        kw = resolve_checkpoint_strategy(
            hpo_search=True, num_train_epochs=3, steps_per_epoch=100
        ).as_training_arguments()
        assert "metric_for_best_model" not in kw
        assert "greater_is_better" not in kw

    def test_hpo_still_evaluates(self):
        # eval_loss is the HPO objective, so evaluation must happen regardless.
        for strategy in ("epoch", "no"):
            s = resolve_checkpoint_strategy(
                hpo_search=True, num_train_epochs=2, steps_per_epoch=10, hpo_save_strategy=strategy
            )
            assert s.eval_strategy == "epoch"


class TestAcceptedByRealTrainingArguments:
    """The resolved kwargs must survive HF's own validation, not just ours."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(dict(hpo_search=True, num_train_epochs=3, steps_per_epoch=100), id="hpo"),
            pytest.param(
                dict(hpo_search=True, num_train_epochs=3, steps_per_epoch=100, hpo_save_strategy="no"),
                id="hpo_no_save",
            ),
            pytest.param(dict(hpo_search=False, num_train_epochs=3, steps_per_epoch=100), id="final_epoch"),
            pytest.param(dict(hpo_search=False, num_train_epochs=1, steps_per_epoch=100), id="final_steps"),
        ],
    )
    def test_training_arguments_construct(self, kwargs):
        transformers = pytest.importorskip("transformers")
        strategy = resolve_checkpoint_strategy(**kwargs)
        with tempfile.TemporaryDirectory() as d:
            args = transformers.TrainingArguments(output_dir=d, **strategy.as_training_arguments())
        assert args.load_best_model_at_end is strategy.load_best_model_at_end
        if strategy.load_best_model_at_end:
            assert args.metric_for_best_model == BEST_MODEL_METRIC
            assert args.greater_is_better is BEST_MODEL_GREATER_IS_BETTER


class TestDriversUseTheSharedStrategy:
    """All three HF SFT drivers must route through the one definition."""

    @pytest.mark.parametrize(
        "module",
        [
            "autotune.trainers.driver_single",
            "autotune.trainers.driver_multi_hf_ds",
            "autotune.trainers.driver_multi_hf_fsdp",
        ],
    )
    def test_driver_imports_resolver(self, module):
        import importlib

        mod = importlib.import_module(module)
        assert hasattr(mod, "resolve_checkpoint_strategy")

    @pytest.mark.parametrize(
        "path",
        [
            "autotune/trainers/driver_single.py",
            "autotune/trainers/driver_multi_hf_ds.py",
            "autotune/trainers/driver_multi_hf_fsdp.py",
        ],
    )
    def test_no_hardcoded_false(self, path):
        # Regression: both multi-GPU drivers used to hardcode
        # load_best_model_at_end=False, silently saving the last step's weights.
        assert "load_best_model_at_end=False" not in open(path).read()
