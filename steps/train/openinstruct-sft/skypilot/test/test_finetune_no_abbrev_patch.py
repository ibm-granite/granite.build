# Copyright LLM.build Authors
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

"""The build-time finetune.py patch turns off argparse's prefix matching.

The upstream stand-in below keeps the pinned finetune.py's `__main__` lines
(open-instruct 26de9960) and a minimal HfArgumentParser (transformers 5.4.0): it
forwards keyword arguments to argparse.ArgumentParser unchanged, and
parse_args_into_dataclasses() raises on any argument parse_known_args() left over.
test_finetune_flags_in_image.py runs the real HfArgumentParser, in the image.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

_PATCH = (
    pathlib.Path(__file__).resolve().parents[1] / "patches" / "finetune_no_abbrev.py"
)

_UPSTREAM = """import argparse
import json


class ArgumentParserPlus(argparse.ArgumentParser):
    def __init__(self, dataclass_types, **kwargs):
        super().__init__(**kwargs)
        self.add_argument("--learning_rate")
        self.add_argument("--output_dir")

    def parse_args_into_dataclasses(self):
        namespace, remaining_args = self.parse_known_args()
        if remaining_args:
            raise ValueError(
                f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}"
            )
        return vars(namespace), None


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    print(json.dumps(args))
"""


def _load():
    """Import patches/finetune_no_abbrev.py by path (it is not a package module).

    :returns: The loaded module; its ``patch(path)`` edits a file in place.
    """
    spec = importlib.util.spec_from_file_location("finetune_no_abbrev", _PATCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(path, *argv):
    """Run the stand-in at ``path`` as a script, with ``argv`` as its arguments.

    Writes ``run_<name>`` next to ``path`` (in the test's tmp_path): the stand-in
    plus a prelude defining FlatArguments/TokenizerConfig, which are only names in
    its parser call. ``path`` itself is left as patched.

    :param path: The stand-in finetune.py, patched or not.
    :param argv: Command-line arguments for it.
    :returns: The CompletedProcess, with text stdout/stderr captured; a non-zero
        exit is returned, not raised.
    """
    prelude = "FlatArguments = TokenizerConfig = None\n"
    script = path.with_name("run_" + path.name)
    script.write_text(prelude + path.read_text())
    return subprocess.run(
        [sys.executable, str(script), *argv],
        capture_output=True,
        text=True,
        check=False,
    )


def test_stand_in_expands_an_abbreviation_before_the_patch(tmp_path):
    # Pins the hazard the patch removes: argparse's default allow_abbrev=True
    # turns --learning_rat into --learning_rate.
    f = tmp_path / "finetune.py"
    f.write_text(_UPSTREAM)
    proc = _run(f, "--learning_rat", "1e-4")
    assert proc.returncode == 0, proc.stderr
    assert '"learning_rate": "1e-4"' in proc.stdout


def test_patched_parser_refuses_an_abbreviation(tmp_path):
    f = tmp_path / "finetune.py"
    f.write_text(_UPSTREAM)
    _load().patch(f)
    proc = _run(f, "--learning_rat", "1e-4")
    assert proc.returncode != 0
    assert (
        "Some specified arguments are not used by the HfArgumentParser: "
        "['--learning_rat', '1e-4']"
    ) in proc.stderr


def test_patched_parser_still_takes_full_flags(tmp_path):
    f = tmp_path / "finetune.py"
    f.write_text(_UPSTREAM)
    _load().patch(f)
    proc = _run(f, "--learning_rate", "1e-4", "--output_dir", "/o")
    assert proc.returncode == 0, proc.stderr
    assert '"learning_rate": "1e-4", "output_dir": "/o"' in proc.stdout


def test_patch_fails_loudly_when_upstream_changes(tmp_path):
    f = tmp_path / "finetune.py"
    f.write_text(_UPSTREAM.replace("TokenizerConfig))", "TokenizerConfig, Extra))"))
    with pytest.raises(SystemExit):
        _load().patch(f)


def test_patch_is_not_applied_twice(tmp_path):
    f = tmp_path / "finetune.py"
    f.write_text(_UPSTREAM)
    mod = _load()
    mod.patch(f)
    with pytest.raises(SystemExit):
        mod.patch(f)
