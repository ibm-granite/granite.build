#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

"""Guard that the LSF/skypilot hfpull shell paths keep the #320 protections.

Those two paths shell out to ``hf download`` (gbcommon is not importable on the
worker), so the cross-process download lock and corrupt-cache self-healing are
reproduced in shell rather than inherited from ``HfURI.pull``. These are static
content checks so an accidental removal of that shell logic fails loudly instead
of silently regressing the shared-cache race back to the #320 behavior.
"""

from pathlib import Path

import gbserver

_STEPS = Path(gbserver.__file__).parent / "builtins" / "steps"
_LSF_HFPULL = _STEPS / "lsf" / "hfpull" / "lsf_scripts" / "hfpull" / "command.sh"
_SKY_HFPULL = _STEPS / "skypilot" / "hfpull" / "step.yaml"

# Markers that must be present for the shell path to serialize + self-heal in a
# way that mirrors gbcommon.uri.hf (HfURI.pull / SharedFileSystemLock).
_REQUIRED = [
    ".gb-hfpull-locks",  # the mkdir lock dir (matches _hfpull_lock_path)
    "hfpull_acquire_lock",  # best-effort cross-process serialization
    "GB_HFPULL_LOCK_TIMEOUT",  # bounded wait, same env var as the Python path
    "GB_HFPULL_FORCE",  # operator-forced re-pull, same env var
    "--force-download",  # corrupt-cache self-heal retry
    ".cache/huggingface/download",  # scratch-cache clear on repeated corruption
    "Consistency check failed",  # recoverable-error classification
]


def test_lsf_hfpull_has_lock_and_self_heal():
    text = _LSF_HFPULL.read_text()
    missing = [m for m in _REQUIRED if m not in text]
    assert not missing, f"LSF hfpull command.sh missing #320 protections: {missing}"


def test_skypilot_hfpull_has_lock_and_self_heal():
    text = _SKY_HFPULL.read_text()
    missing = [m for m in _REQUIRED if m not in text]
    assert not missing, f"skypilot hfpull step.yaml missing #320 protections: {missing}"
