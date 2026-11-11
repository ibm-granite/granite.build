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

"""Tests for mergekit_prepare.py's recipe-shape and binding checks, and its template-name fallback."""

import mergekit_prepare as mp
import pytest
from mergekit_test_helpers import _main


@pytest.mark.parametrize(
    "merge_config, message",
    [
        (
            {"merge_method": "linear", "modules": [{"models": [{"model": "model_1"}]}]},
            "merge_config.modules must be a mapping",
        ),
        (
            {"merge_method": "linear", "models": {"model": "model_1"}},
            "merge_config.models must be a list",
        ),
        (
            {"merge_method": "linear", "slices": "model_1"},
            "merge_config.slices must be a list",
        ),
    ],
)
def test_wrong_typed_recipe_section_refused(tmp_path, capsys, merge_config, message):
    """A list-valued modules (or a non-list models / slices) is a clean error, not a
    traceback from the reference walk."""
    rc, recipe = _main(tmp_path, merge_config, bound={"model_1": "/staged/m1"})
    assert rc == 1
    assert message in capsys.readouterr().err
    assert not recipe.exists()


def test_walk_refs_ignores_a_list_valued_modules():
    """The walk itself never assumes modules is a mapping (unbound_names runs it)."""
    assert not mp.unbound_names({"modules": [{"models": [{"model": "model_2"}]}]}, {})


@pytest.mark.parametrize(
    "merge_config, bound, message",
    [
        # The typo the review names: model1 for model_1 goes to the Hub otherwise.
        (
            {"merge_method": "linear", "models": [{"model": "model1"}]},
            {"model_1": "/staged/m1"},
            "input(s) model_1 is bound, but merge_config never names it",
        ),
        (
            {"merge_method": "linear", "models": [{"model": "model_1"}]},
            {"model_1": "/staged/m1", "base_model": "/staged/base"},
            "input(s) base_model is bound",
        ),
        (
            {"merge_method": "linear", "models": [{"model": "Qwen/Qwen2.5-0.5B"}]},
            {"model_1": "/staged/m1", "model_2": "/staged/m2"},
            "input(s) model_1, model_2 are bound",
        ),
    ],
)
def test_bound_but_unused_input_refused(tmp_path, capsys, merge_config, bound, message):
    rc, recipe = _main(tmp_path, merge_config, bound=bound)
    assert rc == 1
    assert message in capsys.readouterr().err
    assert not recipe.exists()


@pytest.mark.parametrize(
    "merge_config",
    [
        # Each place a reference can name an input, other than the top-level string
        # forms render time also sees: the input counts as used.
        {"merge_method": "linear", "models": [{"model": {"model": "model_1"}}]},
        {
            "merge_method": "linear",
            "models": [{"model": "x/y"}],
            "tokenizer_source": "model_1",
        },
        {
            "merge_method": "linear",
            "models": [{"model": "x/y"}],
            "tokenizer": {"source": "model_1"},
        },
        {"merge_method": "linear", "models": [{"model": "x/y+model_1"}]},
        {
            "merge_method": "linear",
            "modules": {"m": {"models": [{"model": "model_1"}]}},
        },
    ],
)
def test_a_reference_anywhere_counts_as_using_the_input(tmp_path, merge_config):
    assert _main(tmp_path, merge_config, bound={"model_1": "/staged/m1"})[0] == 0


def test_chat_template_names_fall_back_on_any_import_failure(monkeypatch, capsys):
    """A broken mergekit (e.g. a torch that fails with OSError on import in an
    mk_image) falls back to the pinned names with a note, not a traceback."""
    import builtins  # pylint: disable=import-outside-toplevel

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name.startswith("mergekit"):
            raise OSError("libtorch_cpu.so: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    assert mp.chat_template_names() == ["auto", *mp.FALLBACK_TEMPLATE_NAMES]
    assert "OSError: libtorch_cpu.so" in capsys.readouterr().err


def test_a_name_inside_a_hub_id_is_not_a_use(tmp_path, capsys):
    """Render's value search counts `org/model_1-x` (a later refusal on the node is
    documented); prepare's reference walk does not, so the bound input is unused."""
    rc, _ = _main(
        tmp_path,
        {"merge_method": "linear", "models": [{"model": "org/model_1-x"}]},
        bound={"model_1": "/staged/m1"},
    )
    assert rc == 1
    assert (
        "model_1 is bound, but merge_config never names it" in capsys.readouterr().err
    )


def test_the_bindings_form_counts_as_naming_the_input(tmp_path):
    """`{{ bindings.model_1.binding.path }}` renders to the staged path: that is a use."""
    staged = tmp_path / "m1"
    staged.mkdir()
    rc, recipe = _main(
        tmp_path,
        {"merge_method": "linear", "models": [{"model": f"{staged}/"}]},
        bound={"model_1": str(staged)},
    )
    assert rc == 0
    assert recipe.exists()


def test_an_unfilled_bindings_reference_is_refused(tmp_path, capsys):
    rc, recipe = _main(
        tmp_path,
        {
            "merge_method": "linear",
            "models": [
                {"model": "model_1"},
                {"model": "{{ model_2.binding.path }}"},
            ],
        },
        bound={"model_1": "/staged/m1"},
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "could not fill: '{{ model_2.binding.path }}'" in err
    assert not recipe.exists()
