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

from gsm8k_reward import compute_score

samples = [
    {"reward_model": {"ground_truth": "42"}},
    {"reward_model": {"ground_truth": "3.5"}},
]
responses = [
    "Reasoning... #### 42",
    "Reasoning... #### 3.50",
]

for sample, response in zip(samples, responses):
    print(compute_score(solution_str=response, ground_truth=sample["reward_model"]["ground_truth"]))
