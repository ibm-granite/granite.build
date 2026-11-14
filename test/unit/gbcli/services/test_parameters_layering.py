# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Layered parameter files for `--param-file` (issue #663).

The recipe's parameters.yaml next to build.yaml is the base layer; each
`--param-file` file, in the order given, overrides it. Top-level keys
only: a later layer replaces a key's value whole, never deep-merges it.
"""

from pathlib import Path

import pytest

from gbcli.services.service_build import get_params_from_file, load_layered_params


def _write(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return str(path)


@pytest.fixture
def recipe(tmp_path):
    build = tmp_path / "build.yaml"
    build.write_text("granite.build: {}\n", encoding="utf-8")
    _write(tmp_path / "parameters.yaml", "A: base\nB: base\nLIST: [1, 2, 3]\n")
    return str(build)


def test_recipe_file_is_base_layer(tmp_path, recipe):
    over = _write(tmp_path / "over.yaml", "B: over\n")
    assert load_layered_params(recipe, [over]) == {
        "A": "base",
        "B": "over",
        "LIST": [1, 2, 3],
    }


def test_recipe_file_alone(recipe):
    assert load_layered_params(recipe, None)["A"] == "base"
    assert load_layered_params(recipe, [])["A"] == "base"


def test_order_given_matters(tmp_path, recipe):
    one = _write(tmp_path / "one.yaml", "B: one\nC: one\n")
    two = _write(tmp_path / "two.yaml", "B: two\n")
    assert load_layered_params(recipe, [one, two])["B"] == "two"
    assert load_layered_params(recipe, [two, one])["B"] == "one"
    assert load_layered_params(recipe, [two, one])["C"] == "one"


def test_list_replaced_whole(tmp_path, recipe):
    over = _write(tmp_path / "over.yaml", "LIST: [9]\n")
    assert load_layered_params(recipe, [over])["LIST"] == [9]


def test_map_replaced_whole(tmp_path):
    build = _write(tmp_path / "build.yaml", "granite.build: {}\n")
    _write(tmp_path / "parameters.yaml", "M:\n  x: 1\n  y: 2\n")
    over = _write(tmp_path / "over.yaml", "M:\n  y: 3\n")
    assert load_layered_params(build, [over]) == {"M": {"y": 3}}


def test_no_recipe_file_one_layer_is_that_layer(tmp_path):
    build = _write(tmp_path / "build.yaml", "granite.build: {}\n")
    other = tmp_path / "elsewhere"
    other.mkdir()
    layer = _write(other / "p.yaml", "X: 1\nY: [a]\n")
    assert load_layered_params(build, [layer]) == {"X": 1, "Y": ["a"]}


def test_no_files_is_empty(tmp_path):
    build = _write(tmp_path / "build.yaml", "granite.build: {}\n")
    assert load_layered_params(build, None) == {}
    assert load_layered_params(build, ()) == {}


def test_missing_layer_names_path(tmp_path, recipe):
    missing = str(tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_layered_params(recipe, [missing])


def test_non_mapping_layer_errors(tmp_path, recipe):
    bad = _write(tmp_path / "bad.yaml", "- a\n- b\n")
    with pytest.raises(Exception, match="key-value pairs") as exc:
        load_layered_params(recipe, [bad])
    # The error names the offending file, not "parameters.yaml".
    assert str(exc.value).startswith(f"{bad}: ")
    assert "must be a mapping" in str(exc.value)


def test_get_params_from_file_comments_only_is_empty(tmp_path):
    f = _write(tmp_path / "c.yaml", "# nothing set here\n# FOO: 1\n")
    assert get_params_from_file(f) == {}


def test_get_params_from_file_empty_is_empty(tmp_path):
    f = _write(tmp_path / "e.yaml", "")
    assert get_params_from_file(f) == {}


def test_get_params_from_file_non_mapping_names_path(tmp_path):
    f = _write(tmp_path / "scalar.yaml", "just a string\n")
    with pytest.raises(Exception, match="scalar.yaml: parameters must be a mapping"):
        get_params_from_file(f)


def test_comments_only_layer_keeps_earlier_values(tmp_path, recipe):
    c = _write(tmp_path / "c.yaml", "# placeholder\n")
    assert load_layered_params(recipe, [c])["A"] == "base"


def test_relative_build_file_finds_recipe_file(tmp_path, recipe, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "over.yaml", "B: over\n")
    # A bare relative build.yaml: the recipe file is looked up in the cwd.
    assert load_layered_params("build.yaml", ["over.yaml"]) == {
        "A": "base",
        "B": "over",
        "LIST": [1, 2, 3],
    }


def test_recipe_file_via_symlink_applies_once_at_its_position(
    tmp_path, recipe, monkeypatch
):
    import gbcli.services.service_build as sb

    other = tmp_path / "elsewhere"
    other.mkdir()
    link = other / "linked.yaml"
    link.symlink_to(tmp_path / "parameters.yaml")
    over = _write(tmp_path / "over.yaml", "B: over\n")

    loaded = []
    real = sb.get_params_from_file

    def spy(path):
        loaded.append(Path(path).resolve())
        return real(path)

    monkeypatch.setattr(sb, "get_params_from_file", spy)
    merged = sb.load_layered_params(recipe, [over, str(link)])
    # The symlink is the recipe file: applied after `over`, and only once.
    assert merged["B"] == "base"
    assert loaded.count((tmp_path / "parameters.yaml").resolve()) == 1


def test_explicit_recipe_file_applies_at_its_position(tmp_path, recipe):
    recipe_params = str(tmp_path / "parameters.yaml")
    over = _write(tmp_path / "over.yaml", "B: over\n")
    # [over, recipe]: the recipe file is applied after `over`, so it wins.
    assert load_layered_params(recipe, [over, recipe_params])["B"] == "base"
    # [recipe, over]: same as the implicit base layer.
    assert load_layered_params(recipe, [recipe_params, over])["B"] == "over"


def test_explicit_recipe_file_not_applied_twice(tmp_path, recipe, monkeypatch):
    import gbcli.services.service_build as sb

    seen = []
    real = sb.get_params_from_file

    def spy(path):
        seen.append(Path(path).resolve())
        return real(path)

    monkeypatch.setattr(sb, "get_params_from_file", spy)
    recipe_params = tmp_path / "parameters.yaml"
    over = _write(tmp_path / "over.yaml", "B: over\n")
    # Pass the recipe file via a different spelling of the same path.
    sb.load_layered_params(recipe, [over, str(tmp_path / "." / "parameters.yaml")])
    assert seen.count(recipe_params.resolve()) == 1


def test_single_str_accepted(tmp_path, recipe):
    over = _write(tmp_path / "over.yaml", "A: over\n")
    assert load_layered_params(recipe, over)["A"] == "over"
