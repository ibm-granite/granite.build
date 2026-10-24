"""Epoch step math: HF counts optimizer steps, not micro-batches.

The regression these pin is quiet and expensive. Every multi-GPU driver used to
compute ``steps_per_epoch = train_ds_size // (batch_size * num_workers)`` -- a
*micro*-batch count -- and feed it to ``max_steps``, which HF Trainer interprets
as optimizer steps. ``max_steps`` overrides ``num_train_epochs``, so a config
asking for 2 epochs at ``gradient_accumulation_steps: 4`` silently ran 8, and
nothing in the logs said so: the reported ``epoch`` just climbed past the
configured number, and the DP audit compared HF's ``max_steps`` against the
driver's own product of the same wrong number.

HPO made it worse rather than catching it: HPO trials run on ``driver_single``,
which sets no ``max_steps`` and therefore honoured ``num_train_epochs`` exactly,
so hyperparameters were selected under a schedule ``grad_accum`` times shorter
than the final training they were chosen for.
"""

import pathlib
import re

import pytest

from autotune.trainers._steps import EpochSteps, resolve_epoch_steps

DRIVERS = [
    "driver_single.py",
    "driver_single_trl.py",
    "driver_multi_hf_fsdp.py",
    "driver_multi_hf_ds.py",
    "driver_multi_trl_ds.py",
    "driver_multi_trl_fsdp.py",
]


class TestOptimizerStepsDivideByGradAccum:
    """The fix itself: optimizer steps == micro-batches // grad_accum."""

    @pytest.mark.parametrize("accum", [1, 2, 4, 8, 16])
    def test_optimizer_steps_scale_inversely_with_accum(self, accum):
        steps = resolve_epoch_steps(
            train_ds_size=1024,
            per_device_batch_size=2,
            gradient_accumulation_steps=accum,
            num_workers=1,
        )
        assert steps.micro_batches == 512
        assert steps.optimizer_steps == 512 // accum

    def test_micro_batches_account_for_world_size(self):
        steps = resolve_epoch_steps(
            train_ds_size=1024,
            per_device_batch_size=2,
            gradient_accumulation_steps=1,
            num_workers=4,
        )
        assert steps.micro_batches == 128

    def test_job_e_regression(self):
        """The observed failure: 1833 rows, bs 2, 2 ranks, grad_accum 4.

        The driver set max_steps=916 for a 2-epoch config; true optimizer steps
        per epoch are 114, so the run trained for 8 epochs. Logged epoch values
        across the four smoke jobs were 7.97/7.97/7.87/3.98 -- exactly
        num_train_epochs * grad_accum.
        """
        steps = resolve_epoch_steps(
            train_ds_size=1833,
            per_device_batch_size=2,
            gradient_accumulation_steps=4,
            num_workers=2,
        )
        assert steps.micro_batches == 458
        assert steps.optimizer_steps == 114
        assert steps.optimizer_steps * 2 == 228  # max_steps for 2 epochs, was 916


class TestMatchesHFDerivation:
    """Agree with ``transformers``' own ``num_update_steps_per_epoch``.

    HF computes ``max(len(train_dataloader) // gradient_accumulation_steps, 1)``.
    A driver-set ``max_steps`` has to match that or the requested epoch count is
    not what runs.
    """

    @pytest.mark.parametrize("rows", [1, 7, 64, 1000, 1833, 50_000])
    @pytest.mark.parametrize("accum", [1, 3, 4])
    @pytest.mark.parametrize("workers", [1, 2, 8])
    def test_matches_len_dataloader_floor_div(self, rows, accum, workers):
        bs = 2
        steps = resolve_epoch_steps(
            train_ds_size=rows,
            per_device_batch_size=bs,
            gradient_accumulation_steps=accum,
            num_workers=workers,
        )
        len_dataloader = max(1, rows // (bs * workers))
        assert steps.optimizer_steps == max(1, len_dataloader // accum)


class TestDegenerateInputs:
    """A dataset smaller than one global batch still has to train."""

    def test_clamps_to_one_step(self):
        steps = resolve_epoch_steps(
            train_ds_size=1,
            per_device_batch_size=32,
            gradient_accumulation_steps=16,
            num_workers=8,
        )
        assert steps.micro_batches == 1
        assert steps.optimizer_steps == 1

    @pytest.mark.parametrize("accum", [None, 0])
    def test_missing_or_zero_accum_is_one(self, accum):
        steps = resolve_epoch_steps(
            train_ds_size=100,
            per_device_batch_size=1,
            gradient_accumulation_steps=accum,
        )
        assert steps.gradient_accumulation_steps == 1
        assert steps.optimizer_steps == steps.micro_batches == 100

    def test_zero_workers_treated_as_one(self):
        steps = resolve_epoch_steps(train_ds_size=100, per_device_batch_size=1, num_workers=0)
        assert steps.micro_batches == 100

    def test_frozen(self):
        steps = resolve_epoch_steps(train_ds_size=10, per_device_batch_size=1)
        with pytest.raises(Exception):
            steps.optimizer_steps = 5  # type: ignore[misc]
        assert isinstance(steps, EpochSteps)


class TestSamplesPerRank:
    """The Ray Data sample cap must cover every optimizer step, not fewer."""

    def test_covers_max_steps(self):
        steps = resolve_epoch_steps(
            train_ds_size=1833,
            per_device_batch_size=2,
            gradient_accumulation_steps=4,
            num_workers=2,
        )
        epochs, bs = 2, 2
        needed = steps.optimizer_steps * epochs * 4 * bs  # max_steps * accum * bs
        assert steps.samples_per_rank(epochs, bs) == needed

    def test_scales_with_accum(self):
        """A grad_accum-blind budget starves the dataloader by that factor."""
        one = resolve_epoch_steps(train_ds_size=800, per_device_batch_size=2, gradient_accumulation_steps=1)
        four = resolve_epoch_steps(train_ds_size=800, per_device_batch_size=2, gradient_accumulation_steps=4)
        # Same rows either way: fewer, larger optimizer steps consume the same data.
        assert one.samples_per_rank(1, 2) == four.samples_per_rank(1, 2) == 800


class TestNoDriverComputesStepsItself:
    """Source-level pin: the old expression must not come back.

    The math lives in one place so a future driver cannot reintroduce the
    micro-batch/optimizer-step confusion, the same reason
    ``resolve_checkpoint_strategy`` is shared.
    """

    OLD_PATTERN = re.compile(r"steps_per_epoch\s*=\s*max\(\s*1\s*,\s*train_ds_size\s*//")

    @pytest.mark.parametrize("name", DRIVERS)
    def test_driver_delegates_to_resolve_epoch_steps(self, name):
        src = (pathlib.Path("autotune/trainers") / name).read_text()
        assert not self.OLD_PATTERN.search(src), (
            f"{name} computes steps_per_epoch inline again; use "
            "autotune.trainers._steps.resolve_epoch_steps so gradient "
            "accumulation is not dropped"
        )
        assert "resolve_epoch_steps" in src, f"{name} does not import the shared helper"

    # The single-GPU drivers set no max_steps; num_train_epochs bounds them.
    @pytest.mark.parametrize("name", [d for d in DRIVERS if not d.startswith("driver_single")])
    def test_max_steps_built_from_optimizer_steps(self, name):
        """The four drivers that set max_steps must feed it optimizer steps."""
        src = (pathlib.Path("autotune/trainers") / name).read_text()
        assert "max_steps=steps_per_epoch * num_train_epochs" in src
        assert "steps_per_epoch = epoch_steps.optimizer_steps" in src


class TestWorkerGetsTheResolvedAccum:
    """The worker must scale by the same accum the driver divided by.

    With ``gradient_accumulation_steps`` marked ``for_tuner: true`` it is left
    out of ``train_kwargs``, so the worker's ``train_loop_config`` had no key
    and ``max(1, None)`` in the Ray Data sample budget raised ``TypeError``.
    """

    @pytest.mark.parametrize("name", ["driver_multi_hf_fsdp.py", "driver_multi_hf_ds.py"])
    def test_train_loop_config_carries_resolved_accum(self, name):
        src = (pathlib.Path("autotune/trainers") / name).read_text()
        assert '"gradient_accumulation_steps": epoch_steps.gradient_accumulation_steps,' in src
