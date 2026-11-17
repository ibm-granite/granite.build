# SPDX-License-Identifier: Apache-2.0
"""Training entrypoint for the fms-hf-tuning step: upstream's sft_trainer.main()
plus a `Step: <n>/<max> loss=<x>` progress line.

`accelerate launch` runs this file on every rank (fms_hf_tuning_run.sh). Upstream
reads the job from $SFT_TRAINER_CONFIG_JSON_PATH. Its main() passes
additional_callbacks=None to train(), so the progress callback is added by
wrapping tuning.sft_trainer.train before main() looks it up. That keyword is part
of train()'s signature in the pinned fms-hf-tuning 3.2.2;
test_fms_hf_tuning_entry.py fails if a bump drops it.
"""

import functools
from typing import Any


def step_line(step: int, max_steps: int, logs: dict) -> str:
    """The progress line the step's `Step:\\s\\d+` monitor rule picks up.

    :param step: The global step.
    :param max_steps: The total steps; left out of the line when not positive.
    :param logs: The logged metrics; a numeric ``loss`` is appended.
    :returns: The progress line.
    """
    head = (
        f"Step: {step}/{max_steps}" if max_steps and max_steps > 0 else f"Step: {step}"
    )
    loss = logs.get("loss")
    return f"{head} loss={loss:.4f}" if isinstance(loss, (int, float)) else head


def progress_callback() -> Any:
    """Build the TrainerCallback that prints step_line() on world process zero.

    :returns: A transformers.TrainerCallback instance.
    """
    from transformers import (  # pylint: disable=import-outside-toplevel
        TrainerCallback,
    )

    class StepLogCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
        """Print step_line() for every log event that carries a loss."""

        # pylint: disable-next=unused-argument
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Print the progress line on world process zero."""
            if state.is_world_process_zero and logs and "loss" in logs:
                print(step_line(state.global_step, state.max_steps, logs), flush=True)

    return StepLogCallback()


def install(sft: Any) -> None:
    """Wrap sft.train so every call also gets the progress callback.

    :param sft: The tuning.sft_trainer module.
    """
    original = sft.train

    @functools.wraps(original)
    def train(*args, additional_callbacks=None, **kwargs):
        callbacks = [*(additional_callbacks or []), progress_callback()]
        return original(*args, additional_callbacks=callbacks, **kwargs)

    sft.train = train


def main() -> None:
    """Install the callback, then run upstream's main()."""
    import tuning.sft_trainer as sft  # pylint: disable=import-outside-toplevel

    install(sft)
    sft.main()


if __name__ == "__main__":
    main()
