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

"""verl-side training metrics: the in-memory step logger and what consumes it.

Deliberately imports neither torch nor verl at module level, so everything here
is unit-testable on any platform — verl is not installable on macOS, and the
driver module itself imports torch, hydra and verl on load.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from autotune.callbacks.metrics_emitter import (
    close_metrics_handler,
    emit_metric_row,
    finite,
    make_metrics_handler,
)

logger = logging.getLogger(__name__)

# The reporter for the trial currently training in this process, or None.
# Set by the verl driver around fit(); read by _InMemoryMetricsLogger.log.
_ACTIVE_REPORTER: VerlStepReporter | None = None


def set_active_reporter(reporter: VerlStepReporter | None) -> None:
    """Register (or, with None, clear) the reporter that receives each verl step."""
    global _ACTIVE_REPORTER
    _ACTIVE_REPORTER = reporter


class _InMemoryMetricsLogger:
    """In-memory logger that captures verl's per-step metrics without file I/O."""

    _all_steps = []

    def __init__(self):
        _InMemoryMetricsLogger._all_steps = []

    def log(self, data, step):
        clean = {"_step": step}
        for k, v in data.items():
            # torch.Tensor, duck-typed so this module never imports torch.
            if hasattr(v, "numel") and hasattr(v, "tolist"):
                clean[k] = v.item() if v.numel() == 1 else v.tolist()
            elif hasattr(v, "item"):
                # numpy: .item() raises on a multi-element array, which would crash fit().
                clean[k] = v.item() if getattr(v, "size", 1) == 1 else v.tolist()
            else:
                clean[k] = v
        _InMemoryMetricsLogger._all_steps.append(clean)

        reporter = _ACTIVE_REPORTER
        if reporter is not None:
            reporter.report(clean, step)

    def finish(self):
        pass

    @classmethod
    def collect(cls, rl_algorithm: str) -> Dict[str, Any]:
        """Aggregate captured metrics into a flat dict for fm-tune."""
        result = {}
        all_steps = cls._all_steps

        if not all_steps:
            return result

        last = all_steps[-1]

        # Reward metrics (last step)
        result["reward_mean"] = last.get("critic/score/mean", float("nan"))
        result["reward_max"] = last.get("critic/score/max", float("nan"))
        result["reward_min"] = last.get("critic/score/min", float("nan"))

        # Actor metrics (average)
        pg_losses = [s["actor/pg_loss"] for s in all_steps if "actor/pg_loss" in s]
        entropies = [s["actor/entropy"] for s in all_steps if "actor/entropy" in s]

        result["pg_loss"] = sum(pg_losses) / len(pg_losses) if pg_losses else float("nan")
        # verl 0.7.1 emits no separate PPO loss, so actor_loss reports the policy-gradient loss.
        result["actor_loss"] = result["pg_loss"]
        result["actor_entropy"] = sum(entropies) / len(entropies) if entropies else float("nan")

        # KL divergence
        kl_values = [s["actor/kl_loss"] for s in all_steps if "actor/kl_loss" in s]
        kl_reward = [s["actor/reward_kl_penalty"] for s in all_steps if "actor/reward_kl_penalty" in s]
        if kl_values:
            result["kl_divergence"] = sum(kl_values) / len(kl_values)
        elif kl_reward:
            result["kl_divergence"] = sum(kl_reward) / len(kl_reward)
        else:
            result["kl_divergence"] = float("nan")

        # Response length (last step)
        result["response_length_mean"] = last.get("response_length/mean", float("nan"))
        result["response_length_clip_ratio"] = last.get("response_length/clip_ratio", float("nan"))

        # Advantage metrics (last step)
        result["advantages_mean"] = last.get("critic/advantages/mean", float("nan"))
        result["returns_mean"] = last.get("critic/returns/mean", float("nan"))

        # PPO-specific critic metrics
        if rl_algorithm == "ppo":
            critic_losses = [s["critic/loss"] for s in all_steps if "critic/loss" in s]
            result["critic_loss"] = sum(critic_losses) / len(critic_losses) if critic_losses else float("nan")
            result["critic_values_mean"] = last.get("critic/values/mean", float("nan"))
            result["critic_vf_explained_var"] = last.get("critic/vf_explained_var", float("nan"))

        # Training progress
        result["global_steps"] = last.get("training/global_step", 0)
        result["epoch"] = last.get("training/epoch", 0)
        result["total_steps_logged"] = len(all_steps)

        # Throughput
        # verl 0.7.1 has no "perf/overall_tokens_per_second"; "perf/throughput" is tokens/s per GPU.
        result["tokens_per_second_per_gpu"] = last.get("perf/throughput", float("nan"))

        logger.info(f"[AutoTune] Collected {len(all_steps)} training steps from in-memory metrics logger")

        return result


def _last_hf_model_dir(ckpt_dirs):
    """Return ``actor/huggingface`` of the highest-step checkpoint that still has one.

    verl's ``max_actor_ckpt_to_keep`` rotation deletes ``global_step_N/actor``
    but leaves ``global_step_N/`` itself behind, so a glob alone also returns
    checkpoints whose weights are gone. Steps are compared as integers: a sorted
    glob puts ``global_step_99`` after ``global_step_100``.

    Args:
        ckpt_dirs: ``global_step_*`` directory paths, in any order.

    Returns:
        Path to the HF model dir, or None when no checkpoint holds one.
    """

    def _step_from_dir(d):
        try:
            return int(os.path.basename(d).split("global_step_")[-1])
        except (ValueError, IndexError):
            return -1

    for d in sorted(ckpt_dirs, key=_step_from_dir, reverse=True):
        hf_dir = os.path.join(d, "actor", "huggingface")
        if os.path.isdir(hf_dir):
            logger.info(f"[AutoTune] Using last checkpoint: {os.path.basename(d)}")
            return hf_dir
    return None


# verl 0.7.1 keys that fill training_metrics' fixed columns; every other numeric
# key goes to `extra`. There is no "loss" in RL — the policy-gradient loss is the
# honest occupant of that column (reward is in extra as critic/score/mean).
_COLUMN_KEYS = {
    "loss": "actor/pg_loss",
    "grad_norm": "actor/grad_norm",
    "learning_rate": "actor/lr",
}
_EPOCH_KEY = "training/epoch"
_GLOBAL_STEP_KEY = "training/global_step"  # same value as the global_step column
_STEP_KEY = "_step"  # added by _InMemoryMetricsLogger.log; not a verl metric


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(data: dict, key: str) -> float | None:
    """`data[key]` as a finite float, or None when absent, non-numeric or non-finite."""
    value = data.get(key)
    if not _is_number(value):
        return None
    return finite(float(value))


def build_verl_row(
    data: dict, step: int, trial_id: str | None, job_id: str | None, steps_per_epoch: int | None = None
) -> dict:
    """Map one verl step's metrics to a `training_metrics` row (the HF row's key set).

    With `steps_per_epoch`, `epoch` is fractional progress (`step / steps_per_epoch`,
    1.0 at the end of the first epoch), the same scale as HF's `state.epoch`, so
    epoch-based charts line up across SFT and RL trials. Without it, `epoch` falls
    back to verl's 0-based integer `training/epoch`. A step made only of `val-*`
    keys is verl's separate validation log call, so it is labelled `eval`;
    validation keys merged into a training step stay `train`.
    """
    # Unreachable with the driver's test_freq=-1 / val_before_train=False; kept for configs that enable validation.
    metric_keys = [k for k in data if k != _STEP_KEY]
    is_eval = bool(metric_keys) and all(str(k).startswith("val-") for k in metric_keys)
    consumed = set(_COLUMN_KEYS.values()) | {_EPOCH_KEY, _GLOBAL_STEP_KEY, _STEP_KEY}
    epoch = step / steps_per_epoch if steps_per_epoch else _number(data, _EPOCH_KEY)
    return {
        "job_id": job_id,
        "trial_id": trial_id,
        "global_step": int(step),
        "epoch": epoch,
        "loss": _number(data, _COLUMN_KEYS["loss"]),
        "grad_norm": _number(data, _COLUMN_KEYS["grad_norm"]),
        "learning_rate": _number(data, _COLUMN_KEYS["learning_rate"]),
        "split": "eval" if is_eval else "train",
        "extra": {k: finite(v) for k, v in data.items() if k not in consumed and _is_number(v)},
    }


def format_step_summary(data: dict, step: int) -> str:
    """One trial-log line for a verl step; a missing or non-finite field is omitted."""
    parts = []
    reward = _number(data, "critic/score/mean")
    if reward is not None:
        part = f"reward={reward:.3f}"
        low, high = _number(data, "critic/score/min"), _number(data, "critic/score/max")
        if low is not None and high is not None:
            part += f" (min {low:.3f}, max {high:.3f})"
        parts.append(part)
    kl = _number(data, "actor/kl_loss")
    if kl is None:
        kl = _number(data, "actor/reward_kl_penalty")
    fields = (
        ("pg_loss", _number(data, "actor/pg_loss"), ".3f"),
        ("kl", kl, ".3f"),
        ("entropy", _number(data, "actor/entropy"), ".3f"),
        ("resp_len", _number(data, "response_length/mean"), ".1f"),
        ("grad_norm", _number(data, "actor/grad_norm"), ".3g"),
        ("lr", _number(data, "actor/lr"), ".1e"),
    )
    parts.extend(f"{label}={value:{spec}}" for label, value, spec in fields if value is not None)
    return f"[AutoTune] verl step {step}: {' '.join(parts)}".rstrip()


class VerlStepReporter:
    """Emit one `training_metrics` row and one trial-log line per verl step.

    Runs in the Ray Tune trial process — verl's `Tracking` (and so
    `_InMemoryMetricsLogger.log`) lives there — where `bind_trial_id` has already
    bound the bridge log handler, so the summary line is trial-attributed.

    Args:
        trial_id: The Ray Tune trial id stamped onto every row.
        steps_per_epoch: verl steps per epoch; when set, rows carry fractional epoch progress.
    """

    def __init__(self, trial_id: str | None = None, steps_per_epoch: int | None = None) -> None:
        self._trial_id = trial_id
        self._steps_per_epoch = steps_per_epoch
        self._job_id = os.environ.get("AUTOTUNE_JOB_ID")
        self._handler = make_metrics_handler(self._job_id)

    def report(self, data: dict, step: int) -> None:
        # Best-effort, like TrainingMetricsCallback.on_log: never raise into verl's loop.
        try:
            row = build_verl_row(data, step, self._trial_id, self._job_id, self._steps_per_epoch)
            emit_metric_row(row, self._handler)
            logger.info(format_step_summary(data, step))
        except Exception:
            logger.debug("VerlStepReporter.report suppressed an error", exc_info=True)

    def close(self) -> None:
        """Release the bridge handler; safe to call more than once."""
        handler, self._handler = self._handler, None
        close_metrics_handler(handler)
