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

"""steps/scripts/render_step_template.py: gb-include expansion, then ${IMAGE_REF}.

`make space` and `make publish-step` render every framework step through this
script (common.mk's render-step-template). A `{#- gb-include <path> -#}` line is
replaced by the file it names, each line indented like the marker; then the literal
${IMAGE_REF} is substituted exactly as the sed it replaced did.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "steps" / "scripts" / "render_step_template.py"

# The sed render common.mk used before this script, with the ref and template
# passed in the environment.
_OLD_SED = r"""ref=$(printf '%s' "$REF" | sed 's/[#&\\]/\\&/g'); sed "s#\${IMAGE_REF}#$ref#g" "$T" """


def _load():
    spec = importlib.util.spec_from_file_location("render_step_template", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rst = _load()


@pytest.fixture
def steps(tmp_path):
    """A stand-in steps/ root with train/foo/skypilot and train/shared."""
    root = tmp_path / "steps"
    (root / "train" / "foo" / "skypilot").mkdir(parents=True)
    (root / "train" / "shared").mkdir()
    return root


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _template(steps, body):
    return _write(steps / "train" / "foo" / "skypilot" / "step-template.yaml", body)


class TestExpand:
    def test_marker_is_replaced_at_its_indentation(self, steps):
        _write(
            steps / "train" / "shared" / "m.j2",
            "{%- macro a() -%}\n  x\n{%- endmacro %}\n",
        )
        t = _template(
            steps,
            "run: |\n    set -e\n    {#- gb-include ../../shared/m.j2 -#}\n    echo\n",
        )
        assert rst.expand(t, root=steps) == (
            "run: |\n    set -e\n    {%- macro a() -%}\n      x\n"
            "    {%- endmacro %}\n    echo\n"
        )

    def test_empty_lines_stay_empty(self, steps):
        _write(steps / "train" / "shared" / "m.j2", "a\n\nb\n")
        t = _template(steps, "  {#- gb-include ../../shared/m.j2 -#}\n")
        assert rst.expand(t, root=steps) == "  a\n\n  b\n"

    def test_path_is_relative_to_the_template(self, steps):
        _write(steps / "train" / "foo" / "skypilot" / "macros" / "v.j2", "local\n")
        t = _template(steps, "{#- gb-include macros/v.j2 -#}\n")
        assert rst.expand(t, root=steps) == "local\n"

    def test_a_template_without_markers_is_unchanged(self, steps):
        text = (
            "name: foo\n# a YAML comment may mention gb-include\n"
            "run: '{{ x }} ${HOME}'\n"
        )
        assert rst.expand(_template(steps, text), root=steps) == text


class TestRefuse:
    @staticmethod
    def _error(steps, body):
        with pytest.raises(rst.IncludeError) as exc:
            rst.expand(_template(steps, body), root=steps)
        return str(exc.value)

    def test_a_path_outside_steps(self, steps, tmp_path):
        _write(tmp_path / "x.j2", "x\n")
        msg = self._error(steps, "{#- gb-include ../../../../x.j2 -#}\n")
        assert "resolves outside" in msg and "step-template.yaml:1" in msg

    def test_a_symlink_out_of_steps(self, steps, tmp_path):
        _write(tmp_path / "x.j2", "x\n")
        (steps / "train" / "shared" / "link.j2").symlink_to(tmp_path / "x.j2")
        msg = self._error(steps, "{#- gb-include ../../shared/link.j2 -#}\n")
        assert "resolves outside" in msg

    def test_a_missing_file(self, steps):
        msg = self._error(steps, "{#- gb-include ../../shared/nope.j2 -#}\n")
        assert "no such file" in msg

    def test_a_nested_include(self, steps):
        _write(steps / "train" / "shared" / "inner.j2", "x\n")
        _write(steps / "train" / "shared" / "outer.j2", "{#- gb-include inner.j2 -#}\n")
        msg = self._error(steps, "{#- gb-include ../../shared/outer.j2 -#}\n")
        assert "do not nest" in msg

    @pytest.mark.parametrize(
        "line",
        [
            "{# gb-include ../../shared/m.j2 -#}",
            "{#- gb-include ../../shared/m.j2 #}",
            "\t{#- gb-include ../../shared/m.j2 -#}",
            "{#-  gb-include ../../shared/m.j2 -#}",
            "{#- gb-include ../../shared/m.j2 -#} trailing",
            "{#- gb-include -#}",
        ],
    )
    def test_a_malformed_marker(self, steps, line):
        _write(steps / "train" / "shared" / "m.j2", "x\n")
        msg = self._error(steps, f"a\n{line}\n")
        assert "malformed gb-include line" in msg and "step-template.yaml:2" in msg


class TestRender:
    def test_image_ref_is_replaced_in_included_text(self, steps):
        _write(steps / "train" / "shared" / "m.j2", "image: ${IMAGE_REF}\n")
        t = _template(steps, "a: ${IMAGE_REF}\n{#- gb-include ../../shared/m.j2 -#}\n")
        assert rst.render(t, "q.io/o/i:t", root=steps) == (
            "a: q.io/o/i:t\nimage: q.io/o/i:t\n"
        )

    @pytest.mark.parametrize(
        "ref", ["quay.io/o/i:t", "", "a#b", "a&b", "a\\b", "q.io/o/i@sha256:0f"]
    )
    def test_without_markers_it_matches_the_old_sed(self, steps, ref):
        t = _template(
            steps,
            'image_id: "docker:${IMAGE_REF}"\nrun: |\n'
            '  echo "${IMAGE_REF} ${HOME} $(hostname) {{ config.x }}"\n',
        )
        old = subprocess.run(
            ["sh", "-c", _OLD_SED],
            env={**os.environ, "REF": ref, "T": str(t)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert rst.render(t, ref, root=steps) == old


class TestCli:
    def test_writes_the_render(self, steps, tmp_path):
        t = _template(steps, "a: ${IMAGE_REF}\n")
        out = tmp_path / "step.yaml"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(t), str(out), "--image-ref", "r"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.read_text() == "a: r\n"

    def test_an_error_writes_nothing_and_exits_1(self, steps, tmp_path):
        # The CLI resolves includes against the real steps/ tree, so this tmp
        # path is outside it.
        _write(steps / "train" / "shared" / "m.j2", "x\n")
        t = _template(steps, "{#- gb-include ../../shared/m.j2 -#}\n")
        out = tmp_path / "step.yaml"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), str(t), str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert proc.stderr.startswith("render_step_template: ")
        assert "resolves outside" in proc.stderr
        assert not out.exists()
