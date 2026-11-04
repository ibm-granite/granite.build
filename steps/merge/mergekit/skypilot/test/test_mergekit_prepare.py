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

"""Unit tests for src/mergekit_prepare.py: input-name substitution and args flags.

``make test`` puts ``src/`` on PYTHONPATH. The MergeKit round-trip tests import
mergekit and are skipped when it isn't installed; the rest need only PyYAML.
"""

import copy
import json
import pathlib

import mergekit_prepare as mp
import pytest
import yaml

BOUND = {"model_1": "/staged/m1", "base_model": "/staged/base"}


def _sub(config: dict) -> dict:
    """Substitute the BOUND inputs into a copy of ``config``."""
    return mp.walk_refs(
        copy.deepcopy(config),
        lambda r: mp.substitute_ref(r, BOUND),
        lambda p: BOUND.get(p, p),
    )


# --- substitute_ref ----------------------------------------------------------------


@pytest.mark.parametrize(
    "ref, expected",
    [
        ("model_1", "/staged/m1"),
        ("model_1@main", "/staged/m1@main"),
        ("base_model+my-org/my-lora", "/staged/base+my-org/my-lora"),
        ("Qwen/Qwen2.5-0.5B+model_1", "Qwen/Qwen2.5-0.5B+/staged/m1"),
        ("base_model@v1+model_1@v2", "/staged/base@v1+/staged/m1@v2"),
        ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B"),
        ("/node/path/model", "/node/path/model"),
        ("model_2", "model_2"),  # not bound: left for MergeKit
        ("model_10", "model_10"),  # not an input name
    ],
)
def test_substitute_ref(ref, expected):
    """Only parts naming a bound input change; @revision and +lora are kept."""
    assert mp.substitute_ref(ref, BOUND) == expected


# --- walk_refs: every field MergeKit reads as a model reference ----------------------


def test_walk_models_and_base_model():
    out = _sub(
        {
            "merge_method": "ties",
            "base_model": "base_model",
            "models": [
                {"model": "model_1", "parameters": {"weight": 0.5}},
                {"model": "HuggingFaceTB/SmolLM2-135M"},
            ],
        }
    )
    assert out["base_model"] == "/staged/base"
    assert [m["model"] for m in out["models"]] == [
        "/staged/m1",
        "HuggingFaceTB/SmolLM2-135M",
    ]
    assert out["models"][0]["parameters"] == {"weight": 0.5}


def test_walk_slices():
    out = _sub(
        {
            "merge_method": "slerp",
            "slices": [
                {
                    "sources": [
                        {"model": "base_model", "layer_range": [0, 30]},
                        {"model": "model_1", "layer_range": [0, 30]},
                    ],
                    "base_model": "model_1",
                }
            ],
        }
    )
    s = out["slices"][0]
    assert [src["model"] for src in s["sources"]] == ["/staged/base", "/staged/m1"]
    assert s["base_model"] == "/staged/m1"


def test_walk_modules():
    out = _sub(
        {
            "merge_method": "linear",
            "modules": {
                "text": {
                    "models": [{"model": "model_1"}],
                    "slices": [{"sources": [{"model": "base_model"}]}],
                }
            },
        }
    )
    assert out["modules"]["text"]["models"][0]["model"] == "/staged/m1"
    assert out["modules"]["text"]["slices"][0]["sources"][0]["model"] == "/staged/base"


@pytest.mark.parametrize("keyword", ["union", "base"])
def test_walk_tokenizer_keywords_untouched(keyword):
    out = _sub({"tokenizer_source": keyword, "tokenizer": {"source": keyword}})
    assert out["tokenizer_source"] == keyword
    assert out["tokenizer"]["source"] == keyword


def test_walk_tokenizer_models():
    out = _sub(
        {
            "tokenizer_source": "model_1",
            "tokenizer": {
                "source": "base_model",
                "tokens": {
                    "<a>": {"source": "model_1"},
                    "<b>": {"source": {"kind": "model_token", "model": "base_model"}},
                    "<c>": {"source": {"kind": "zero"}},
                },
            },
        }
    )
    assert out["tokenizer_source"] == "/staged/m1"
    tok = out["tokenizer"]
    assert tok["source"] == "/staged/base"
    assert tok["tokens"]["<a>"]["source"] == "/staged/m1"
    assert tok["tokens"]["<b>"]["source"]["model"] == "/staged/base"
    assert tok["tokens"]["<c>"]["source"] == {"kind": "zero"}


def test_walk_leaves_parameters_alone():
    """A parameter value that happens to equal an input name is not a model reference."""
    out = _sub(
        {
            "merge_method": "linear",
            "models": [{"model": "model_1", "parameters": {"filter": "model_1"}}],
            "parameters": {"note": "base_model"},
        }
    )
    assert out["models"][0]["parameters"] == {"filter": "model_1"}
    assert out["parameters"] == {"note": "base_model"}


def test_walk_mapping_form_refs():
    """MergeKit's mapping form of a reference is substituted like the string form."""
    out = _sub(
        {
            "merge_method": "linear",
            "models": [
                {
                    "model": {
                        "model": "model_1",
                        "override_architecture": "LlamaForCausalLM",
                    }
                },
                {
                    "model": {
                        "model": {"path": "base_model", "revision": "v1"},
                        "lora": "model_1",
                    }
                },
                {
                    "model": {
                        "model": {"path": "Qwen/Qwen2.5-0.5B"},
                        "lora": {"path": "base_model"},
                    }
                },
            ],
            "base_model": {"model": "base_model"},
        }
    )
    m = out["models"]
    assert m[0]["model"] == {
        "model": "/staged/m1",
        "override_architecture": "LlamaForCausalLM",
    }
    assert m[1]["model"] == {
        "model": {"path": "/staged/base", "revision": "v1"},
        "lora": "/staged/m1",
    }
    assert m[2]["model"] == {
        "model": {"path": "Qwen/Qwen2.5-0.5B"},
        "lora": {"path": "/staged/base"},
    }
    assert out["base_model"] == {"model": "/staged/base"}


@pytest.mark.parametrize(
    "model_path, expected",
    [
        # revision is literal and never touched, even when it holds '@' or '+'
        (
            {"path": "model_1", "revision": "a@b"},
            {"path": "/staged/m1", "revision": "a@b"},
        ),
        (
            {"path": "model_1", "revision": "x+y"},
            {"path": "/staged/m1", "revision": "x+y"},
        ),
        (
            {"path": "model_1", "revision": None},
            {"path": "/staged/m1", "revision": None},
        ),
        ({"path": "model_1", "revision": ""}, {"path": "/staged/m1", "revision": ""}),
        # path is literal: '@' and '+' in it are part of the name, not syntax
        ({"path": "model_1@x"}, {"path": "model_1@x"}),
        ({"path": "model_1+model_1"}, {"path": "model_1+model_1"}),
        ({"path": "model_2", "revision": "v1"}, {"path": "model_2", "revision": "v1"}),
    ],
)
def test_mapping_form_path_is_literal(model_path, expected):
    """MergeKit doesn't parse a {path, revision} ModelPath: match the whole path only."""
    out = _sub({"merge_method": "linear", "models": [{"model": {"model": model_path}}]})
    assert out["models"][0]["model"]["model"] == expected


@pytest.mark.parametrize(
    "ref, expected",
    [
        # A string ModelPath inside the mapping form: MergeKit splits '@' only, not '+'.
        ({"model": "model_1+model_2"}, {"model": "model_1+model_2"}),
        ({"model": "model_1@v2"}, {"model": "/staged/m1@v2"}),
        (
            {"model": "model_1", "lora": "base_model+x"},
            {"model": "/staged/m1", "lora": "base_model+x"},
        ),
        (
            {"model": "Qwen/Qwen2.5-0.5B", "lora": "base_model@r"},
            {"model": "Qwen/Qwen2.5-0.5B", "lora": "/staged/base@r"},
        ),
    ],
)
def test_mapping_form_string_path_splits_at_only(ref, expected):
    out = _sub({"merge_method": "linear", "models": [{"model": ref}]})
    assert out["models"][0]["model"] == expected


def test_mapping_form_refs_parse_in_mergekit():
    """The substituted mapping forms are what MergeKit expects (skipped without it)."""
    config = pytest.importorskip("mergekit.config")
    out = _sub(
        {
            "merge_method": "linear",
            "models": [
                {
                    "model": {
                        "model": "model_1",
                        "override_architecture": "LlamaForCausalLM",
                    }
                },
                {
                    "model": {
                        "model": {"path": "base_model", "revision": "v1"},
                        "lora": "model_1",
                    }
                },
            ],
        }
    )
    parsed = config.MergeConfiguration.model_validate(out)
    assert sorted(str(r) for r in parsed.referenced_models()) == [
        "/staged/base@v1+/staged/m1",
        "/staged/m1",
    ]


# --- render_args ---------------------------------------------------------------------


def test_render_args_values():
    assert mp.render_args(
        {
            "out-shard-size": "1B",
            "random-seed": 42,
            "write-model-card": False,
            "async-write": True,
            "num-threads": None,
        }
    ) == [
        "--out-shard-size",
        "1B",
        "--random-seed",
        "42",
        "--no-write-model-card",
        "--async-write",
    ]


@pytest.mark.parametrize("key", sorted(mp.OWNED_FLAGS))
def test_render_args_refuses_owned_flags(key):
    with pytest.raises(mp.PrepareError, match=f"args.{key} is not allowed"):
        mp.render_args({key: True})


@pytest.mark.parametrize("key", ["--out-shard-size", "no-copy-tokenizer", "a b", ""])
def test_render_args_refuses_malformed_keys(key):
    with pytest.raises(mp.PrepareError, match="must be a mergekit-yaml flag name"):
        mp.render_args({key: True})


@pytest.mark.parametrize("value", ["", "   "])
def test_render_args_refuses_empty_values(value):
    """An empty value would vanish in run.sh's line read and shift the next flag."""
    with pytest.raises(mp.PrepareError, match="args.lora-merge-dtype is empty"):
        mp.render_args({"lora-merge-dtype": value, "write-model-card": False})


def test_render_args_refuses_lists():
    with pytest.raises(mp.PrepareError, match="must be a scalar"):
        mp.render_args({"out-shard-size": ["1B"]})


def test_args_flag_names_are_real_mergekit_options():
    """Every flag the step itself passes is a real mergekit-yaml option."""
    run_yaml = pytest.importorskip("mergekit.scripts.run_yaml")
    names = set()
    for p in run_yaml.main.params:
        names.update(o.lstrip("-") for o in getattr(p, "opts", []))
        names.update(o.lstrip("-") for o in getattr(p, "secondary_opts", []))
    for flag in mp.OWNED_FLAGS:
        assert flag in names, flag
        if flag not in ("transformers-cache", "lora-merge-cache", "device", "gpu-rich"):
            assert f"no-{flag}" in names, flag


# --- main ----------------------------------------------------------------------------


def _main(tmp_path, merge_config, args=None, **flags):
    recipe = tmp_path / "merge.yaml"
    argv = [
        "--merge-config-json",
        json.dumps(merge_config),
        "--args-json",
        json.dumps(args or {}),
        "--recipe-out",
        str(recipe),
        "--expect-out",
        str(tmp_path / "expect"),
        "--input-model_1",
        "/staged/m1",
        "--input-base_model",
        "/staged/base",
    ]
    defaults = {
        "copy-tokenizer": "true",
        "lazy-unpickle": "true",
        "allow-crimes": "false",
        "trust-remote-code": "false",
    }
    for k, v in {**defaults, **flags}.items():
        argv += [f"--{k}", v]
    return mp.main(argv), recipe


def test_main_writes_recipe_and_prints_flags(tmp_path, capsys):
    rc, recipe = _main(
        tmp_path,
        {
            "merge_method": "linear",
            "models": [{"model": "model_1", "parameters": {"weight": 5e-06}}],
        },
        {"out-shard-size": "1B"},
    )
    assert rc == 0
    written = yaml.safe_load(recipe.read_text())
    assert written["models"][0]["model"] == "/staged/m1"
    # YAML, not JSON: PyYAML (YAML 1.1) must read the exponent back as a float.
    assert written["models"][0]["parameters"]["weight"] == 5e-06
    assert capsys.readouterr().out.splitlines() == [
        "--copy-tokenizer",
        "--lazy-unpickle",
        "--no-allow-crimes",
        "--no-trust-remote-code",
        "--out-shard-size",
        "1B",
    ]


def test_main_named_flags_off(tmp_path, capsys):
    rc, _ = _main(
        tmp_path,
        {"merge_method": "linear", "models": [{"model": "model_1"}]},
        **{"copy-tokenizer": "false", "allow-crimes": "true"},
    )
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert "--no-copy-tokenizer" in out and "--allow-crimes" in out


@pytest.mark.parametrize(
    "merge_config, args, flags, message",
    [
        ({}, {}, {}, "merge_config is required"),
        (
            {"merge_method": "linear", "models": [{"model": "model_1"}]},
            ["x"],
            {},
            "args must be a mapping",
        ),
        (
            {"merge_method": "linear", "models": [{"model": "model_1"}]},
            {"cuda": True},
            {},
            "args.cuda is not allowed",
        ),
        (
            {"merge_method": "linear", "models": [{"model": "model_1"}]},
            {},
            {"allow-crimes": "yes"},
            "allow_crimes must be",
        ),
        (
            {"merge_method": "linear", "models": [{"model": "model_1"}]},
            {"k": "a\nb"},
            {},
            "must not contain a newline",
        ),
        (
            {"merge_method": "linear", "models": [{"model": "model_1"}]},
            {"safe-serialization": False},
            {},
            "safe-serialization is not allowed",
        ),
        (
            {
                "merge_method": "linear",
                "models": [{"model": "model_1"}],
                "chat_template": "",
            },
            {},
            {},
            "chat_template is empty",
        ),
        (
            {
                "merge_method": "linear",
                "models": [{"model": "model_1"}],
                "chat_template": "{{ bos_token }}{{ eos_token }}",
            },
            {},
            {},
            "is not a template name MergeKit ships",
        ),
    ],
)
def test_main_errors(tmp_path, capsys, merge_config, args, flags, message):
    rc, recipe = _main(tmp_path, merge_config, args, **flags)
    assert rc == 1
    assert message in capsys.readouterr().err
    assert not recipe.exists()


def test_substituted_recipe_parses_in_mergekit(tmp_path):
    """The fixture's recipe, substituted, is a valid MergeKit config."""
    config = pytest.importorskip("mergekit.config")
    step = pathlib.Path(__file__).resolve().parents[1]
    fx = yaml.safe_load((step / "test-data" / "slurm" / "build.yaml").read_text())
    mc = fx["granite.build"]["targets"]["merge"]["steps"][0]["config"][
        "mergekit_config"
    ]
    rc, recipe = _main(tmp_path, mc["merge_config"])
    assert rc == 0
    parsed = config.MergeConfiguration.model_validate(
        yaml.safe_load(recipe.read_text())
    )
    assert sorted(str(r) for r in parsed.referenced_models()) == [
        "/staged/base",
        "/staged/m1",
    ]


def test_chat_template_needs_copy_tokenizer(tmp_path, capsys):
    """MergeKit only warns and drops the template when there is no tokenizer."""
    rc, recipe = _main(
        tmp_path,
        {
            "merge_method": "linear",
            "models": [{"model": "model_1"}],
            "chat_template": "chatml",
        },
        **{"copy-tokenizer": "false"},
    )
    assert rc == 1
    assert "needs copy_tokenizer: true" in capsys.readouterr().err
    assert not recipe.exists()


@pytest.mark.parametrize(
    "tokenizer_key",
    [
        {"tokenizer_source": "base"},
        {"tokenizer": {"source": "base"}},
        {"tokenizer": {}},  # MergeKit builds one: tokenizer is not None, source "union"
    ],
)
def test_chat_template_without_copy_tokenizer_ok_when_mergekit_builds_one(
    tmp_path, tokenizer_key
):
    """With tokenizer_source / tokenizer MergeKit builds and saves the tokenizer itself."""
    rc, recipe = _main(
        tmp_path,
        {
            "merge_method": "linear",
            "models": [{"model": "model_1"}],
            "chat_template": "chatml",
            **tokenizer_key,
        },
        **{"copy-tokenizer": "false"},
    )
    assert rc == 0
    assert yaml.safe_load(recipe.read_text())["chat_template"] == "chatml"


def test_chat_template_names_fallback_says_so(monkeypatch, capsys):
    """Without mergekit the pinned list is used, and a line on stderr says so."""
    import builtins  # pylint: disable=import-outside-toplevel

    real_import = builtins.__import__

    def no_mergekit(name, *args, **kwargs):
        if name.startswith("mergekit"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mergekit)
    assert mp.chat_template_names() == ["auto", *mp.FALLBACK_TEMPLATE_NAMES]
    assert "not importable" in capsys.readouterr().err


def test_chat_template_name_passes_through(tmp_path):
    rc, recipe = _main(
        tmp_path,
        {
            "merge_method": "linear",
            "models": [{"model": "model_1"}],
            "chat_template": "chatml",
        },
    )
    assert rc == 0
    assert yaml.safe_load(recipe.read_text())["chat_template"] == "chatml"


def _template_list(name: str) -> list:
    """A `{%- set <name> = [...] -%}` list from step-template.yaml's run: block."""
    step = pathlib.Path(__file__).resolve().parents[1]
    template = yaml.safe_load((step / "step-template.yaml").read_text())
    run = template["environment_configs"]["Skypilot"]["launchers"]["mergekit"][
        "config"
    ]["run"]
    line = next(ln for ln in run.splitlines() if f"set {name} = [" in ln)
    return yaml.safe_load(line[line.index("[") : line.rindex("]") + 1])


def test_merge_method_list_matches_mergekit_registry():
    """The template's method list is MergeKit's registry (skipped without mergekit).

    Lives here, not in the render test, because that one imports gbserver, which the
    mergekit test venv does not have.
    """
    registry = pytest.importorskip("mergekit.merge_methods.registry")
    names = sorted(m.spec.name for m in registry.registered_methods())
    assert sorted(_template_list("methods")) == names


def test_template_owned_flags_match_prepare():
    """The template refuses exactly the args keys mergekit_prepare.py refuses."""
    assert sorted(_template_list("owned")) == sorted(mp.OWNED_FLAGS)


# What gbserver's non-strict config fill leaves of real chat templates: {% %} blocks are
# dropped, unknown {{ var }} are written back as text (targetsteprun.py fill_objtemplate).
_LLAMA3 = (
    "{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}"
    "<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{{ eos_token }}"
)


@pytest.mark.parametrize(
    "value",
    [
        None,  # MergeKit's default: no template
        "chatml",
        "auto",
        _LLAMA3,  # survived the fill because the build wrapped it in {% raw %}
        "{%- for m in messages %}{{ m['content'] }}{%- endfor %}",
    ],
)
def test_chat_template_accepted(value):
    mp.check_chat_template(value)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "{{ bos_token }}{{ eos_token }}",  # what an unwrapped Llama-3 template becomes
        "<|im_start|>assistant\n",  # fragments with a newline, no {% left
        # Plain text an unwrapped template can leave, which MergeKit rejects inside
        # _copy_tokenizer's try (logging "The merge was still successful").
        "ASSISTANT:",
        "<|assistant|>",
        "doesnotexist",  # not a template MergeKit ships
        "x" * 300,
        ["chatml"],
    ],
)
def test_chat_template_mangled_or_wrong_type_refused(value):
    with pytest.raises(mp.PrepareError, match="raw"):
        mp.check_chat_template(value)


def test_chat_template_error_lists_the_names():
    with pytest.raises(mp.PrepareError) as exc:
        mp.check_chat_template("ASSISTANT:")
    for name in mp.chat_template_names():
        assert name in str(exc.value)


def test_chat_template_names_match_mergekit():
    """The pinned fallback list is MergeKit's shipped templates (skipped without it)."""
    resources = pytest.importorskip("importlib.resources")
    chat_templates = pytest.importorskip("mergekit._data.chat_templates")
    shipped = sorted(
        f.name[: -len(".jinja")]
        for f in resources.files(chat_templates).iterdir()
        if f.name.endswith(".jinja")
    )
    assert sorted(mp.FALLBACK_TEMPLATE_NAMES) == shipped
    assert mp.chat_template_names() == ["auto", *shipped]


def test_every_accepted_name_is_one_mergekit_resolves():
    """MergeKit's own _set_chat_template accepts every name the step accepts."""
    pytest.importorskip("mergekit")
    merge = pytest.importorskip("mergekit.merge")
    transformers = pytest.importorskip("transformers")

    class _Cfg:  # pylint: disable=too-few-public-methods  # what _set_chat_template reads
        chat_template = None

    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=__import__("tokenizers").Tokenizer(
            __import__("tokenizers").models.WordLevel(
                {"a": 0, "[UNK]": 1}, unk_token="[UNK]"
            )
        )
    )
    for name in mp.chat_template_names():
        if name == "auto":
            continue  # resolved from the input models' own templates, not a file
        cfg = _Cfg()
        cfg.chat_template = name
        merge._set_chat_template(tok, cfg)  # pylint: disable=protected-access
        assert "{%" in tok.chat_template, name


@pytest.mark.parametrize(
    "recipe_extra, expected",
    [
        ({}, False),
        ({"tokenizer_source": "base"}, True),
        ({"tokenizer": {"source": "union"}}, True),
        ({"tokenizer": {}}, True),
        ({"tokenizer_source": None}, False),
        ({"tokenizer": None}, False),
    ],
)
def test_recipe_builds_tokenizer_matches_mergekit_plan(recipe_extra, expected):
    """MergeKit's plan.py: tokenizer is not None, or tokenizer_source is not None."""
    assert mp.recipe_builds_tokenizer(recipe_extra) is expected


@pytest.mark.parametrize(
    "recipe_extra, copy_tokenizer, expected",
    [
        ({}, "true", ["tokenizer"]),
        ({}, "false", []),
        ({"chat_template": "chatml"}, "true", ["tokenizer", "chat_template"]),
        ({"tokenizer_source": "base"}, "false", ["tokenizer"]),
        # The exempted path: the output still has to carry the requested template.
        (
            {"tokenizer_source": "base", "chat_template": "auto"},
            "false",
            ["tokenizer", "chat_template"],
        ),
        (
            {"tokenizer": {}, "chat_template": "chatml"},
            "false",
            ["tokenizer", "chat_template"],
        ),
    ],
)
def test_expect_out_lists_what_run_sh_must_check(
    tmp_path, recipe_extra, copy_tokenizer, expected
):
    rc, _ = _main(
        tmp_path,
        {"merge_method": "linear", "models": [{"model": "model_1"}], **recipe_extra},
        **{"copy-tokenizer": copy_tokenizer},
    )
    assert rc == 0
    assert (tmp_path / "expect").read_text().split() == expected


@pytest.mark.parametrize(
    "merge_config, missing",
    [
        ({"models": [{"model": "model_1"}, {"model": "model_2"}]}, ["model_2"]),
        (
            {"base_model": "base_model", "models": [{"model": "model_1"}]},
            ["base_model"],
        ),
        ({"models": [{"model": "model_1+model_3@v1"}]}, ["model_3"]),
        ({"models": [{"model": "model_4@main"}]}, ["model_4"]),
        ({"models": [{"model": {"model": {"path": "model_2"}}}]}, ["model_2"]),
        ({"models": [{"model": {"model": "model_1", "lora": "model_3"}}]}, ["model_3"]),
        (
            {
                "slices": [
                    {"sources": [{"model": "model_2"}], "base_model": "base_model"}
                ]
            },
            ["base_model", "model_2"],
        ),
        (
            {
                "models": [
                    {"model": "model_1"},
                    {"model": "Qwen/Qwen2.5-0.5B"},
                    {"model": "/abs"},
                ]
            },
            [],
        ),
        ({"models": [{"model": "model_10"}, {"model": "my_model_2"}]}, []),
    ],
)
def test_unbound_names(merge_config, missing):
    """Only the five input names, in any reference field and form, when not bound."""
    assert mp.unbound_names(merge_config, {"model_1": "/staged/m1"}) == missing


def test_main_refuses_an_unbound_input_name(tmp_path, capsys):
    rc, recipe = _main(
        tmp_path,
        {
            "merge_method": "linear",
            "models": [{"model": "model_1"}, {"model": "model_2"}],
        },
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "model_2" in err and "not bound" in err
    assert not recipe.exists()


def test_unregistered_merge_method_refused_with_mergekit(tmp_path, capsys):
    """On the node, the installed mergekit's registry is the authority (as render time's
    pinned 0.1.5 list is, before provisioning)."""
    pytest.importorskip("mergekit.merge_methods.registry")
    rc, _ = _main(
        tmp_path, {"merge_method": "not_a_method", "models": [{"model": "model_1"}]}
    )
    assert rc == 1
    assert "not one of the installed mergekit's methods" in capsys.readouterr().err


def test_unregistered_merge_method_left_to_mergekit_without_it(monkeypatch, tmp_path):
    """Without mergekit importable (a direct run), there is no registry to check."""
    import builtins  # pylint: disable=import-outside-toplevel

    real_import = builtins.__import__

    def no_mergekit(name, *args, **kwargs):
        if name.startswith("mergekit"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_mergekit)
    assert mp.merge_method_names() == []
    rc, _ = _main(
        tmp_path, {"merge_method": "not_a_method", "models": [{"model": "model_1"}]}
    )
    assert rc == 0


def _fake_registry(monkeypatch, registry):
    """Make `from mergekit.merge_methods import registry` return `registry`."""
    import sys  # pylint: disable=import-outside-toplevel
    import types  # pylint: disable=import-outside-toplevel

    pkg = types.ModuleType("mergekit")
    sub = types.ModuleType("mergekit.merge_methods")
    pkg.merge_methods = sub
    if registry is not None:
        sub.registry = registry
    monkeypatch.setitem(sys.modules, "mergekit", pkg)
    monkeypatch.setitem(sys.modules, "mergekit.merge_methods", sub)
    if registry is not None:
        monkeypatch.setitem(sys.modules, "mergekit.merge_methods.registry", registry)
    else:
        monkeypatch.delitem(
            sys.modules, "mergekit.merge_methods.registry", raising=False
        )


def test_merge_method_names_on_an_old_mergekit(monkeypatch, tmp_path, capsys):
    """mergekit 0.1.4 and earlier have REGISTERED_MERGE_METHODS, not registered_methods()
    (mergekit_version and mk_image allow them): the check still works, and does not crash.
    """
    import types  # pylint: disable=import-outside-toplevel

    old = types.ModuleType("mergekit.merge_methods.registry")
    old.REGISTERED_MERGE_METHODS = {"linear": object(), "slerp": object()}
    _fake_registry(monkeypatch, old)
    assert mp.merge_method_names() == ["linear", "slerp"]
    assert (
        _main(tmp_path, {"merge_method": "linear", "models": [{"model": "model_1"}]})[0]
        == 0
    )
    rc, _ = _main(tmp_path, {"merge_method": "ties", "models": [{"model": "model_1"}]})
    assert rc == 1
    assert "not one of the installed mergekit's methods" in capsys.readouterr().err


def test_merge_method_names_on_an_unfamiliar_registry(monkeypatch, tmp_path):
    """A registry layout the step doesn't know, or one that fails to import (an mk_image
    with a broken torch), leaves merge_method to MergeKit rather than failing prepare.
    """
    import types  # pylint: disable=import-outside-toplevel

    _fake_registry(monkeypatch, types.ModuleType("mergekit.merge_methods.registry"))
    assert mp.merge_method_names() == []
    _fake_registry(monkeypatch, None)  # `import registry` raises ImportError
    assert mp.merge_method_names() == []
    rc, _ = _main(tmp_path, {"merge_method": "x", "models": [{"model": "model_1"}]})
    assert rc == 0


@pytest.mark.parametrize(
    "merge_config, message",
    [
        ({"models": [{"model": "model_1"}]}, "merge_method is required"),
        (
            {"merge_method": 1, "models": [{"model": "model_1"}]},
            "merge_method is required, as a non-empty string",
        ),
        ({"merge_method": "linear"}, "exactly one of models, slices or modules"),
        (
            {
                "merge_method": "linear",
                "models": [{"model": "model_1"}],
                "slices": [{"sources": []}],
            },
            "exactly one of models, slices or modules",
        ),
    ],
)
def test_recipe_shape_refused(tmp_path, capsys, merge_config, message):
    rc, recipe = _main(tmp_path, merge_config)
    assert rc == 1
    assert message in capsys.readouterr().err
    assert not recipe.exists()


def test_non_ascii_args_key_refused():
    with pytest.raises(mp.PrepareError, match="must be a mergekit-yaml flag name"):
        mp.render_args({"r\u00e1ndom-seed": 1})
