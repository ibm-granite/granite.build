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

# gsm8k_reward.py
#
# Custom reward function for GSM8K math problems.
# Compatible with verl 0.7.0's NaiveRewardManager which calls:
#   compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)
# and expects a scalar float (or dict with "score" key) in return.

import re
from typing import Optional, Union

# -------- parsing helpers --------

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _normalize_number_str(s) -> Optional[str]:
    """Normalize numeric string: remove commas, strip, and extract a number."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "")
    if _NUMBER_RE.fullmatch(s):
        return s
    m = _NUMBER_RE.search(s)
    return m.group(0) if m else None


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from a model response.
    Prefer GSM8K style: '#### <answer>'
    Fallback: last number in the completion.
    """
    if not text:
        return None
    # Prefer #### convention
    m = re.search(r"####\s*([-+$]?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return _normalize_number_str(m.group(1))
    # Fallback: last number
    nums = _NUMBER_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else None


# -------- core scoring --------

# Hyperparameters
CORRECT_REWARD = 1.0
WRONG_REWARD = -1.0
FORMAT_BONUS = 0.05
LENGTH_PENALTY_COEF = 1.0 / 4000.0
MAX_LENGTH_PENALTY = 0.2


def _score_one(
    response: str,
    gt: Union[str, int, float, None],
) -> float:
    """Score a single response against a ground truth answer."""
    gt_str = _normalize_number_str(gt)
    pred = extract_final_answer(response)

    used_hash = "####" in (response or "")
    bonus = FORMAT_BONUS if used_hash else 0.0

    # Length penalty (small, PPO-stabilizing)
    lp = 0.0
    if response:
        lp = -min(len(response) * LENGTH_PENALTY_COEF, MAX_LENGTH_PENALTY)

    if pred is None or gt_str is None:
        return WRONG_REWARD + bonus + lp

    return (CORRECT_REWARD if pred == gt_str else WRONG_REWARD) + bonus + lp


# -------- entry point for verl 0.7.0 --------


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Union[str, int, float, None] = None,
    extra_info: dict = None,
    **kwargs,
) -> float:
    """
    verl 0.7.0 custom reward function entrypoint.

    Called per-sample by NaiveRewardManager with:
        compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)

    Returns:
        float: scalar reward score for this single sample.
    """
    return _score_one(solution_str, ground_truth)
