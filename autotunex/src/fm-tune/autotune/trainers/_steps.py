# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

"""Epoch step math shared by the drivers that set ``max_steps`` explicitly.

Every step-valued ``TrainingArguments`` knob HF Trainer accepts — ``max_steps``,
``logging_steps``, ``save_steps``/``eval_steps``, ``torch_empty_cache_steps`` —
counts **optimizer** steps, i.e. one step per ``gradient_accumulation_steps``
micro-batches. HF derives that itself from the dataloader
(``num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps``,
``transformers/trainer.py``), and a driver that computes its own step count has
to divide by ``gradient_accumulation_steps`` the same way.

A row count divided by ``per_device_train_batch_size * world_size`` yields
*micro*-batches per rank, so feeding that straight into ``max_steps`` overshoots
the requested schedule by exactly ``gradient_accumulation_steps``: a config
asking for 2 epochs at ``gradient_accumulation_steps: 4`` runs 8. That was the
behaviour of all four multi-GPU drivers, and it was invisible from the outside —
``max_steps`` silently wins over ``num_train_epochs`` in HF, the logged
``epoch`` just climbs past the configured number, and
``utils.assert_dp_sharding`` compared HF's ``max_steps`` against the driver's
own ``steps_per_epoch * epochs`` product, so it reproduced the same error and
never warned.

The consequences were worse than a longer run. HPO trials go through
``driver_single``, which sets no ``max_steps`` at all and therefore honoured
``num_train_epochs`` exactly, so hyperparameters were selected under a schedule
``gradient_accumulation_steps`` times shorter than the final training they were
chosen for. And the ``save_steps``/``eval_steps`` cadence for a single-epoch run
(``steps_per_epoch // 5``, see :mod:`autotune.trainers._checkpointing`) was
computed in micro-batches, so the five best-model candidates it is meant to
produce collapsed to one.

Both counts are exposed because the two are needed for different things: HF's
step knobs want :attr:`EpochSteps.optimizer_steps`, while a sample budget for an
iterable dataset (the Ray Data backend caps per-rank iteration so every rank
produces identical step counts) wants :attr:`EpochSteps.micro_batches`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochSteps:
    """Per-rank step counts for one epoch.

    ``micro_batches`` is how many batches a rank's dataloader yields;
    ``optimizer_steps`` is how many times the optimizer actually steps, which is
    what HF Trainer means by "steps" everywhere in ``TrainingArguments``.
    """

    micro_batches: int
    optimizer_steps: int
    gradient_accumulation_steps: int

    def samples_per_rank(self, num_train_epochs: int, per_device_batch_size: int) -> int:
        """Rows one rank must be able to draw to complete ``num_train_epochs``.

        Derived from ``optimizer_steps`` rather than ``micro_batches`` so it stays
        consistent with the ``max_steps`` the driver sets: an iterable dataset cut
        short of this leaves the last optimizer step starved.
        """
        micro = self.optimizer_steps * self.gradient_accumulation_steps
        return micro * per_device_batch_size * max(1, num_train_epochs)


def resolve_epoch_steps(
    *,
    train_ds_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int = 1,
    num_workers: int = 1,
) -> EpochSteps:
    """Compute per-rank micro-batch and optimizer-step counts for one epoch.

    Mirrors HF's own derivation so that a driver-set ``max_steps`` agrees with
    what HF would have computed from ``num_train_epochs`` alone: floor-divide the
    rows this rank sees by its batch size to get micro-batches, then floor-divide
    by ``gradient_accumulation_steps`` to get optimizer steps. Both are clamped
    to at least 1 so a dataset smaller than one global batch still trains.
    """
    accum = max(1, int(gradient_accumulation_steps or 1))
    per_rank_batch = max(1, int(per_device_batch_size)) * max(1, int(num_workers))
    micro_batches = max(1, int(train_ds_size) // per_rank_batch)
    return EpochSteps(
        micro_batches=micro_batches,
        optimizer_steps=max(1, micro_batches // accum),
        gradient_accumulation_steps=accum,
    )
