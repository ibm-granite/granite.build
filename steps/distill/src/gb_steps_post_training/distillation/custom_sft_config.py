# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import dataclass, field

from trl import SFTConfig


@dataclass
class CustomSFTConfig(SFTConfig):
    r"""Custom configuration class that extends [`SFTConfig`] with a longer
    `ddp_timeout` default (1 hour instead of HF's 30 min).

    HF `TrainingArguments.ddp_timeout` is forwarded as
    `timedelta(seconds=ddp_timeout)` into `PartialState(...)`, so it
    controls the barrier timeout used by
    `PartialState().main_process_first()` — including the dataset
    preprocessing map chain in `gold/sft.py:_prepare_dataset`. A direct measurement
    fired NCCL's watchdog at exactly 1_800_000 ms on that barrier while
    rank 0 was still tokenizing 4.05 M examples at `max_length=16384`
    with `return_assistant_tokens_mask=True`. Raising the default to
    3600 s gives preprocessing a hard 1-hour SLA before the watchdog
    aborts.
    """

    ddp_timeout: int = field(default=3600)
