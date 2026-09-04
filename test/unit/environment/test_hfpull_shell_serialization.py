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

import re
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
    ".breaking",  # no-rename removal fallback: single-winner break claim (#354)
]


def test_lsf_hfpull_has_lock_and_self_heal():
    text = _LSF_HFPULL.read_text()
    missing = [m for m in _REQUIRED if m not in text]
    assert not missing, f"LSF hfpull command.sh missing #320 protections: {missing}"


def test_skypilot_hfpull_has_lock_and_self_heal():
    text = _SKY_HFPULL.read_text()
    missing = [m for m in _REQUIRED if m not in text]
    assert not missing, f"skypilot hfpull step.yaml missing #320 protections: {missing}"


# The lock + self-heal logic is maintained as two shell copies (gbcommon is not
# importable on the LSF/skypilot workers). Sentinels delimit the region that
# must stay in lockstep so a fix to one copy can't silently skip the other.
_SENTINEL_START = ">>> gb-hfpull shared shell"
_SENTINEL_END = "<<< gb-hfpull shared shell"


def _shared_shell_code(path: Path) -> list[str]:
    """Executable (non-comment, non-blank) lines of the shared hfpull block.

    Comments and indentation are dropped so the two copies -- one at column 0
    (LSF), one indented inside a YAML block scalar (skypilot), each with its own
    prose/wrapping -- compare only on the shell that actually runs.
    """
    lines = path.read_text().splitlines()
    starts = [i for i, ln in enumerate(lines) if _SENTINEL_START in ln]
    ends = [i for i, ln in enumerate(lines) if _SENTINEL_END in ln]
    assert starts and ends, f"{path} is missing the shared-shell sentinels"
    body = lines[starts[0] + 1 : ends[0]]
    return [ln.strip() for ln in body if ln.strip() and not ln.strip().startswith("#")]


def test_lsf_and_skypilot_hfpull_shell_blocks_are_in_sync():
    """The two shell copies must carry identical executable logic.

    The per-file marker checks above can't catch drift: a fix to the LSF copy
    (e.g. the recoverable-error classification) that misses the skypilot copy
    would leave both markers present yet the paths silently divergent. Compare
    the executable lines directly so any such drift fails here.
    """
    lsf = _shared_shell_code(_LSF_HFPULL)
    sky = _shared_shell_code(_SKY_HFPULL)
    assert lsf, "no shared-shell code extracted from the LSF command.sh"
    assert lsf == sky, (
        "LSF and skypilot hfpull shell blocks have diverged; keep them in sync "
        f"(first diff near: {next((f'{a!r} != {b!r}' for a, b in zip(lsf, sky) if a != b), 'length mismatch')})"
    )


def test_shell_recoverable_regex_matches_python_source_of_truth():
    """The shell recoverable-error regex must equal the Python classifier's.

    The recoverable-error set lives in three copies (Python + the two shell
    scripts). The shell-to-shell sync test alone can't catch a change to the
    Python source of truth silently leaving the shell workers on the old
    classification -- the exact cross-boundary drift that would reopen the #320
    shared-cache race. Pin the shell regex to the Python pattern so all three
    move together.
    """
    from gbcommon.uri.hf import HF_RECOVERABLE_CACHE_ERROR_RE

    m = re.search(r"HFPULL_RECOVERABLE_RE='([^']*)'", _LSF_HFPULL.read_text())
    assert m, "HFPULL_RECOVERABLE_RE not found in LSF command.sh"
    assert m.group(1) == HF_RECOVERABLE_CACHE_ERROR_RE.pattern, (
        "shell HFPULL_RECOVERABLE_RE has drifted from Python "
        "HF_RECOVERABLE_CACHE_ERROR_RE; the shell workers would classify "
        "recoverable errors differently than HfURI.pull"
    )
