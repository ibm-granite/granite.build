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

"""Save/eval strategy selection for the HF SFT drivers.

The three HF SFT drivers (single-GPU, DeepSpeed, FSDP) all face the same
question: how often to evaluate, how often to checkpoint, and whether the model
that gets persisted is the *best* epoch or merely the *last* one.

Final training tracks the best checkpoint by ``eval_loss``, so the saved model is
the best epoch rather than whatever the last step happened to produce — eval loss
routinely turns back up after its minimum, and without this the run would persist
an overfitted model while a better one sat on disk.

HF's ``load_best_model_at_end`` imposes preconditions that are easy to violate
silently, so they are encoded here rather than restated in each driver:

- ``eval_strategy`` must equal ``save_strategy`` (and, for ``"steps"``,
  ``save_steps`` must be a multiple of ``eval_steps``). This helper always emits
  matching strategies, with ``save_steps == eval_steps`` on the steps path.
- ``metric_for_best_model`` must be set, with ``greater_is_better`` matching its
  direction — ``eval_loss`` is lower-is-better, hence ``False``.
- ``save_total_limit`` rotation exempts the best checkpoint, so the best survives
  even once it falls outside the newest N.

HPO trials deliberately do *not* track a best model: they report metrics and
write no model. Their ``save_strategy="epoch"`` exists only so Ray Train v2's
``RayTrainReportCallback.on_save()`` fires at least once — standalone
``train.report()`` calls are ignored by ``CheckpointManager``, so without a save
the trial's ``Result.metrics`` comes back empty. See the "Ray Train v2 metrics"
gotcha in CLAUDE.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

# The metric that selects the best checkpoint, and its direction. Eval loss is
# lower-is-better, so ``greater_is_better`` is False.
BEST_MODEL_METRIC = "eval_loss"
BEST_MODEL_GREATER_IS_BETTER = False

# Checkpoints retained during final training. HF exempts the best checkpoint from
# rotation, so the best one survives even outside the newest N.
FINAL_SAVE_TOTAL_LIMIT = 3

# HPO trials keep exactly one checkpoint — just enough for on_save() to fire.
HPO_SAVE_TOTAL_LIMIT = 1

# Evals/saves per epoch when training runs for a single epoch. Epoch-boundary
# checkpointing would otherwise give exactly one datapoint and nothing to choose
# between.
SINGLE_EPOCH_CHECKPOINTS = 5


@dataclass
class CheckpointStrategy:
    """Resolved save/eval strategy for one training run.

    ``load_best_model_at_end`` is True only for final training, where a model is
    actually persisted. Feed :meth:`as_training_arguments` straight into
    ``TrainingArguments(**kwargs)``.
    """

    save_strategy: str
    eval_strategy: str
    save_total_limit: Optional[int]
    load_best_model_at_end: bool
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_training_arguments(self) -> Dict[str, Any]:
        """The ``TrainingArguments`` kwargs this strategy implies.

        ``save_steps``/``eval_steps`` are included only on the steps path, and the
        best-model metric keys only when best-model tracking is on — passing
        ``metric_for_best_model`` without ``load_best_model_at_end`` would make HF
        track a metric it never acts on.
        """
        kwargs: Dict[str, Any] = {
            "save_strategy": self.save_strategy,
            "eval_strategy": self.eval_strategy,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
        }
        if self.save_steps is not None:
            kwargs["save_steps"] = self.save_steps
        if self.eval_steps is not None:
            kwargs["eval_steps"] = self.eval_steps
        if self.load_best_model_at_end:
            kwargs["metric_for_best_model"] = BEST_MODEL_METRIC
            kwargs["greater_is_better"] = BEST_MODEL_GREATER_IS_BETTER
        kwargs.update(self.extra)
        return kwargs


def resolve_checkpoint_strategy(
    *,
    hpo_search: bool,
    num_train_epochs: int,
    steps_per_epoch: int,
    hpo_save_strategy: str = "epoch",
    has_length: bool = True,
) -> CheckpointStrategy:
    """Pick the save/eval strategy for an SFT run.

    HPO trials save once per epoch purely so Ray Train sees a checkpoint-attached
    report; no best-model tracking, since no model is persisted.

    Final training tracks the best checkpoint by ``eval_loss``. Multi-epoch runs
    evaluate and save at each epoch boundary; a single-epoch run has only one such
    boundary, so it switches to a steps cadence
    (:data:`SINGLE_EPOCH_CHECKPOINTS` times per epoch) to give the best-model
    search something to choose between. ``save_steps == eval_steps`` keeps HF's
    "save_steps must be a multiple of eval_steps" check satisfied.

    ``hpo_save_strategy`` lets a driver that needs no HPO checkpoint at all pass
    ``"no"`` (the single-GPU driver, which reads metrics from
    ``trainer.state.log_history`` instead of a checkpoint-attached report).

    ``has_length=False`` is for a train dataset with no ``__len__`` (the Ray Data
    backend's ``RayShardIterableDataset``). HF cannot see epoch boundaries in an
    unsized iterator: with ``max_steps`` set it runs the whole schedule as one
    epoch (``num_train_epochs = sys.maxsize`` in ``Trainer.set_initial_training_values``),
    so ``on_epoch_end`` fires once, after the last step, and an ``"epoch"``
    strategy evaluates and saves exactly once. Multi-epoch final training then
    takes a steps cadence of ``steps_per_epoch`` instead -- one eval/save per
    epoch, which is what the epoch strategy would have done with a sized dataset.
    """
    if hpo_search:
        saves = hpo_save_strategy != "no"
        return CheckpointStrategy(
            save_strategy=hpo_save_strategy,
            eval_strategy="epoch",
            save_total_limit=HPO_SAVE_TOTAL_LIMIT if saves else None,
            load_best_model_at_end=False,
        )

    if num_train_epochs > 1 and not has_length:
        cadence = max(1, steps_per_epoch)
        return CheckpointStrategy(
            save_strategy="steps",
            eval_strategy="steps",
            save_total_limit=FINAL_SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
            save_steps=cadence,
            eval_steps=cadence,
        )

    if num_train_epochs > 1:
        return CheckpointStrategy(
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=FINAL_SAVE_TOTAL_LIMIT,
            load_best_model_at_end=True,
        )

    # Single epoch: evaluate and save several times within it, on a shared cadence.
    cadence = max(1, steps_per_epoch // SINGLE_EPOCH_CHECKPOINTS)
    return CheckpointStrategy(
        save_strategy="steps",
        eval_strategy="steps",
        save_total_limit=FINAL_SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        save_steps=cadence,
        eval_steps=cadence,
    )


def resolve_resume_checkpoint(training_output_dir: str, resume_flag: bool) -> Union[str, bool]:
    """Resolve the ``resume_from_checkpoint`` argument for ``Trainer.train()``.

    Returns the path of the latest ``checkpoint-<step>`` directory under
    ``training_output_dir`` when ``resume_flag`` is set and one exists, or
    ``False`` otherwise (HF Trainer treats ``False`` as "train from scratch").

    Pure / side-effect-free (only stats the filesystem) so it is unit-testable
    without Ray or Torch. Shared by the multi-GPU drivers (FSDP and DeepSpeed, HF
    and TRL); the single-GPU drivers do not wire up resume. ``utils.py``'s
    ``has_resumable_final_checkpoint`` mirrors this rule on the head node.

    Args:
        training_output_dir: The directory the Trainer writes checkpoints to.
        resume_flag: Whether this run was asked to resume.

    Returns:
        The latest checkpoint directory, or ``False`` to train from scratch.
    """
    if not resume_flag:
        return False
    if not os.path.isdir(training_output_dir):
        return False
    candidates = []
    for name in os.listdir(training_output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(training_output_dir, name)
        if not os.path.isdir(path):
            continue
        step_str = name[len("checkpoint-") :]
        if step_str.isdigit():
            candidates.append((int(step_str), path))
    if not candidates:
        return False
    # Highest step number is the most recent checkpoint.
    return max(candidates, key=lambda c: c[0])[1]
