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
"""Dataset-building and data-prep tools.

Each ``build_*_dataset`` module is runnable both as a script and as a module::

    python autotune/tools/build_factuality_dataset.py --help
    python -m autotune.tools.build_factuality_dataset --help

The modules therefore use absolute imports (``from autotune.tools.chat_utils
import ...``) rather than relative ones: a relative import needs a parent
package, which direct script execution does not provide.
"""
