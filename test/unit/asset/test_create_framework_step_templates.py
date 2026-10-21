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

"""The create-framework-step skill's skeletons produce a step whose render test passes.

The skill writes a step from the code blocks in its ``references/templates.md``. That
step exists only after the skill runs, and PR CI never collects ``steps/``, so a broken
template would otherwise ship unnoticed. This test fills the templates in the way the
skill does, for a few input/output shapes, writes the step into a temp dir, and runs
the generated render test with pytest. A freshly generated step must pass, with only
the expected xfail (``run.sh`` is a stub until the user implements it).

Filling in a template follows the rules at the top of templates.md: ``<name>`` and
``<cfg>`` are substituted, and each line holding ``<in>``/``<out>`` (or the YAML block
it heads) is repeated once per input/output name.
"""

import pathlib
import re
import subprocess
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TEMPLATES_MD = (
    REPO_ROOT
    / ".claude"
    / "skills"
    / "create-framework-step"
    / "references"
    / "templates.md"
)
NAME, CFG = "text-norm", "text_norm"


def _section(md: str, heading: str, lang: str, index: int = 0) -> str:
    """Return the ``index``-th ``lang`` code block under a ``## `` heading.

    Args:
        md: the templates.md text.
        heading: the start of the ``## `` heading line, e.g. ``"## step-template.yaml"``.
        lang: the code fence language, e.g. ``"yaml"``.
        index: which matching block in the section.

    Returns:
        The code block's body.
    """
    start = md.index(heading)
    nxt = md.find("\n## ", start + 1)
    sec = md[start : nxt if nxt != -1 else len(md)]
    return re.findall(rf"```{lang}\n(.*?)```", sec, re.S)[index]


def _expand(text: str, inputs: list[str], outputs: list[str]) -> str:
    """Fill in a template the way the skill does.

    Args:
        text: one template's text.
        inputs: input names; each ``<in>`` line or block is repeated per name.
        outputs: output names; each ``<out>`` line or block is repeated per name.

    Returns:
        The filled-in text.
    """

    def indent(line: str) -> int:
        return len(line) - len(line.lstrip())

    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        token = "<in>" if "<in>" in line else "<out>" if "<out>" in line else None
        if token is None:
            out.append(line)
            i += 1
            continue
        end = i + 1
        if line.split("#")[0].rstrip().endswith(":"):  # a YAML key heading a block
            while (
                end < len(lines)
                and lines[end].strip()
                and indent(lines[end]) > indent(line)
            ):
                end += 1
        for name in inputs if token == "<in>" else outputs:
            out += [ln.replace(token, name) for ln in lines[i:end]]
        i = end
    return "\n".join(out).replace("<name>", NAME).replace("<cfg>", CFG) + "\n"


def _write_step(root: pathlib.Path, inputs: list[str], outputs: list[str]) -> None:
    """Write a public-image SkyPilot step from the templates into ``root``.

    Args:
        root: the step directory to create.
        inputs: input names.
        outputs: output names.
    """
    md = TEMPLATES_MD.read_text()
    (root / "src").mkdir(parents=True)
    (root / "test").mkdir()
    template = _expand(_section(md, "## step-template.yaml", "yaml"), inputs, outputs)
    (root / "step-template.yaml").write_text(template)
    for i, script in enumerate(("setup.sh", "run.sh")):
        stub = _section(md, "## src/setup.sh and src/run.sh", "bash", i)
        (root / "src" / script).write_text(_expand(stub, inputs, outputs))
    render = _section(md, "## test/test_<env>_<category>_<name>_render.py", "python")
    (root / "test" / "test_render.py").write_text(_expand(render, inputs, outputs))
    (root / ".gbignore").write_text(_section(md, "## .gbignore", "gitignore"))


SHAPES = {
    "one-in-one-out": (["docs"], ["tokens"]),
    "two-in-two-out": (["docs", "vocab"], ["tokens", "report"]),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_template_is_valid_yaml(tmp_path, shape):
    """The filled-in step-template.yaml parses, with the inputs and outputs named."""
    inputs, outputs = SHAPES[shape]
    _write_step(tmp_path, inputs, outputs)
    step = yaml.safe_load((tmp_path / "step-template.yaml").read_text())
    assert list(step["inputs"]["required"]) == inputs
    assert list(step["outputs"]["required"]) == outputs


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_generated_render_test_passes(tmp_path, shape):
    """The render test the skill writes passes on the fresh skeleton (one xfail)."""
    _write_step(tmp_path, *SHAPES[shape])
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert "1 xfailed" in summary, summary


def _patterns(text: str) -> list[str]:
    """Return a .gbignore's glob lines, without comments or blanks."""
    return [
        ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")
    ]


STEP_SOURCES = sorted(
    mk.parent
    for mk in (REPO_ROOT / "steps").glob("**/Makefile")
    if "space" not in mk.parts and "common.mk" in mk.read_text()
)


def test_there_are_step_sources():
    """Guard against the glob silently matching nothing."""
    assert STEP_SOURCES


@pytest.mark.parametrize(
    "step", STEP_SOURCES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_every_step_source_has_the_skeleton_gbignore(step):
    """Each step source ships a .gbignore with the same patterns the skill writes.

    common.mk only copies a step's .gbignore and refuses a step without one, so every
    step source needs the file. A step that needs a different list may diverge; add it
    to an allow-list here with a comment saying why.
    """
    gbignore = step / ".gbignore"
    assert gbignore.is_file(), f"{gbignore} is missing"
    template = _section(TEMPLATES_MD.read_text(), "## .gbignore", "gitignore")
    assert _patterns(gbignore.read_text()) == _patterns(template)
