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

"""TRL compatibility shim for verl's PPO critic workers.

verl 0.7.1 does ``from trl import AutoModelForCausalLMWithValueHead``.
In trl >= 0.29.0 the class moved to ``trl.experimental.ppo``.
Importing this module patches it back into the ``trl`` namespace.

This module is loaded inside Ray workers via verl's ``external_lib``
config key so the patch is applied before verl attempts the import.
"""

import trl

if not hasattr(trl, "AutoModelForCausalLMWithValueHead"):
    from trl.experimental.ppo import AutoModelForCausalLMWithValueHead

    trl.AutoModelForCausalLMWithValueHead = AutoModelForCausalLMWithValueHead
