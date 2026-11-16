# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Dataset-building and data-prep tools.

Each ``build_*_dataset`` module is runnable both as a script and as a module::

    python autotune/tools/build_factuality_dataset.py --help
    python -m autotune.tools.build_factuality_dataset --help

The modules therefore use absolute imports (``from autotune.tools.chat_utils
import ...``) rather than relative ones: a relative import needs a parent
package, which direct script execution does not provide.
"""
