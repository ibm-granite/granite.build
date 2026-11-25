# SPDX-License-Identifier: Apache-2.0
"""Every config attribute the GOLD and SFT entrypoints read exists on the trl dataclasses.

gold.py and sft.py read `model_args.*`, `training_args.*` and `script_args.*` in their
`__main__` blocks, which no unit test runs. trl 1.x moved fields between these classes
(`trust_remote_code` from ModelConfig to SFTConfig), and the first anyone heard of it was
`AttributeError: 'ModelConfig' object has no attribute 'trust_remote_code'` on every rank
of the student-sft and logit-distill build tests (dev-676-5e931046). This reads the
attribute accesses from the source and checks each against the real classes, so a field
that moves or disappears in a trl bump fails here instead.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

trl = pytest.importorskip("trl")
pytest.importorskip("torch")

from gb_steps_post_training.distillation.custom_gold_config import (  # noqa: E402
    CustomGOLDConfig,
)
from gb_steps_post_training.distillation.custom_sft_config import (  # noqa: E402
    CustomSFTConfig,
)

_PKG = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "gb_steps_post_training"
    / "distillation"
)

# Which dataclass each name is bound to, per entrypoint (the TrlParser tuples in each file).
_BINDINGS = {
    "gold.py": {
        "model_args": trl.ModelConfig,
        "training_args": CustomGOLDConfig,
        "script_args": trl.ScriptArguments,
    },
    "sft.py": {
        "model_args": trl.ModelConfig,
        "training_args": CustomSFTConfig,
        "script_args": trl.ScriptArguments,
    },
}


def _attributes(cls) -> set[str]:
    return {f.name for f in dataclasses.fields(cls)} | set(dir(cls))


def _accesses(path: Path, names):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in names
        ):
            yield node.lineno, node.value.id, node.attr


@pytest.mark.parametrize("entrypoint", sorted(_BINDINGS))
def test_every_config_attribute_read_exists(entrypoint):
    bindings = _BINDINGS[entrypoint]
    missing = [
        f"{entrypoint}:{line} {name}.{attr} (not on {bindings[name].__name__})"
        for line, name, attr in _accesses(_PKG / entrypoint, bindings)
        if attr not in _attributes(bindings[name])
    ]
    assert not missing, "config attributes that do not exist:\n" + "\n".join(missing)


def test_the_check_would_have_caught_trust_remote_code():
    """The regression itself: trl 1.x's ModelConfig has no trust_remote_code."""
    assert "trust_remote_code" not in _attributes(trl.ModelConfig)
    assert "trust_remote_code" in _attributes(CustomSFTConfig)
    assert "trust_remote_code" in _attributes(CustomGOLDConfig)
