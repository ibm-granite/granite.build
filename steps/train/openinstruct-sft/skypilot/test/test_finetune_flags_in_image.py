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

"""Checks against the image's own Open-Instruct source and transformers.

They skip where that source is absent (the repo venv). Run them in the image,
with this step directory mounted, after a build or an Open-Instruct pin bump:

    podman run --rm --platform linux/amd64 --entrypoint bash \\
      -v "$PWD:/step:ro" <IMAGE_REF> \\
      -c 'python -m pytest -p no:cacheprovider /step/test/test_finetune_flags_in_image.py'

OPEN_INSTRUCT_HOME (default /opt/open-instruct) points them at another checkout.
"""

import ast
import dataclasses
import os
import pathlib
import re

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_OI_HOME = pathlib.Path(os.environ.get("OPEN_INSTRUCT_HOME", "/opt/open-instruct"))
_FINETUNE = _OI_HOME / "open_instruct" / "finetune.py"
_DATASETS = _OI_HOME / "open_instruct" / "dataset_transformation.py"

pytestmark = pytest.mark.skipif(
    not _FINETUNE.is_file(), reason=f"no Open-Instruct source at {_OI_HOME}"
)


def _fields(path: pathlib.Path, cls: str) -> dict[str, tuple[str, str]]:
    """Return the fields of dataclass ``cls`` defined in ``path``.

    :param path: A Python source file.
    :param cls: The class name.
    :returns: Field name -> (annotation source, default source; "" if none).
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {
                s.target.id: (
                    ast.unparse(s.annotation),
                    ast.unparse(s.value) if s.value else "",
                )
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            }
    raise AssertionError(f"no class {cls} in {path}")


def _options() -> set[str]:
    """Return finetune.py's flag names (underscore form) as its parser builds them.

    Every FlatArguments and TokenizerConfig field, plus the ``no_<field>``
    complement HfArgumentParser adds for a bool field that defaults to True.
    """
    fields = {
        **_fields(_FINETUNE, "FlatArguments"),
        **_fields(_DATASETS, "TokenizerConfig"),
    }
    options = set(fields)
    for name, (annotation, default) in fields.items():
        if annotation.startswith("bool") and re.search(r"(^|default=)True\b", default):
            options.add(f"no_{name}")
    return options


def _owned() -> set[str]:
    """Return the step-owned flags from step-template.yaml's ``owned`` list."""
    text = (_STEP_DIR / "step-template.yaml").read_text()
    return set(
        re.findall(r'"([a-z_0-9]+)"', re.findall(r"set owned = \[(.*?)\]", text)[0])
    )


def test_every_owned_flag_is_a_finetune_flag():
    # A stale owned entry protects nothing; a renamed flag would go unprotected.
    assert not _owned() - _options()


def test_no_finetune_flag_abbreviates_an_owned_one():
    # The template refuses an args key that is a proper prefix of an owned flag.
    # A real flag like that would be refused by mistake: exempt it in the
    # template, as hf-peft-sft's full_flags does.
    owned = _owned()
    clashes = sorted(
        (flag, o) for flag in _options() - owned for o in owned if o.startswith(flag)
    )
    assert not clashes


def test_patched_parser_refuses_abbreviations():
    transformers = pytest.importorskip("transformers")
    line = next(
        ln
        for ln in _FINETUNE.read_text().splitlines()
        if "parser = ArgumentParserPlus(" in ln
    )
    call = ast.parse(line.strip()).body[0].value
    kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}

    @dataclasses.dataclass
    class Args:
        learning_rate: float = 1.0

    parser = transformers.HfArgumentParser(Args, **kwargs)
    assert parser.parse_args_into_dataclasses(["--learning_rate", "0.5"]) == (
        Args(learning_rate=0.5),
    )
    with pytest.raises(ValueError, match="not used by the HfArgumentParser"):
        parser.parse_args_into_dataclasses(["--learning_rat", "0.5"])
