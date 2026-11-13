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

"""Render tests for the mergekit step's step-template.yaml.

Renders the launcher with ``gbserver.utils.template.fill_template(..., strict=True)``,
the fill ``targetsteprun.py`` applies to the launcher config (StrictUndefined and the
``raise_error`` global), so a validate rule that fails here fails a real build at step
creation. The rendered blocks are then run by bash with stub scripts that print their
argv, because bash is what splits and unquotes these words on the node. Mode 1 only.
"""

import copy
import json
import pathlib
import shutil
import subprocess

import pytest
import yaml

from gbserver.utils.template import fill_objtemplate, fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["mergekit"][
    "config"
]
_FIXTURE = yaml.safe_load(
    (_STEP_DIR / "test-data" / "slurm" / "build.yaml").read_text()
)
_FIXTURE_MK = _FIXTURE["granite.build"]["targets"]["merge"]["steps"][0]["config"][
    "mergekit_config"
]
# A quote in a path checks the q() escaping.
_BINDINGS = {
    "model_1": {"binding": {"path": "/staged/model_1 it's"}},
    "base_model": {"binding": {"path": "/staged/base_model"}},
}
_INPUTS = ("model_1", "model_2", "model_3", "model_4", "base_model")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


def _config(over=None, compute=None, launcher=None) -> dict:
    """The step's config with the fixture's mergekit_config, plus overrides.

    Args:
        over: keys merged into ``mergekit_config``.
        compute: replaces ``compute_config`` when given.
        launcher: set as ``launcher_config`` when given.

    Returns:
        The ``config`` the launcher is rendered with.
    """
    config = copy.deepcopy(_TEMPLATE["config"])
    config["mergekit_config"].update(copy.deepcopy(_FIXTURE_MK))
    config["mergekit_config"].update(over or {})
    if compute is not None:
        config["compute_config"] = compute
    if launcher is not None:
        config["launcher_config"] = launcher
    return config


def _render(
    block="run",
    over=None,
    bindings=None,
    compute=None,
    launcher=None,
    run_metadata=None,
) -> str:
    """Render one launcher block strictly."""
    ctx = {
        "config": _config(over, compute, launcher),
        "bindings": _BINDINGS if bindings is None else bindings,
    }
    if run_metadata is not None:
        ctx["run_metadata"] = run_metadata
    return fill_template(_LAUNCHER[block], ctx, strict=True)


def _render_error(over=None, compute=None, launcher=None) -> str:
    """The raise_error message a failing fill carries (fill_template wraps it)."""
    with pytest.raises(RuntimeError) as exc:
        _render("run", over, compute=compute, launcher=launcher)
    return str(exc.value.__cause__ or exc.value)


def _argv(block: str, tmp_path: pathlib.Path, **kw) -> list[str]:
    """Run a rendered block with stub scripts that print their argv.

    Args:
        block: ``"setup"`` or ``"run"``.
        tmp_path: scratch dir to run in, so the step tree is never touched.
        **kw: passed to :func:`_render`.

    Returns:
        The arguments the block passed to its ``src/`` script, one per item.
    """
    stub = '#!/usr/bin/env bash\nfor a in "$@"; do echo "ARG:$a"; done\n'
    (tmp_path / "src").mkdir(exist_ok=True)
    for name in ("setup.sh", "run.sh"):
        (tmp_path / "src" / name).write_text(stub)
    proc = subprocess.run(
        ["bash", "-c", _render(block, **kw)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("ARG:")]


def _value(argv: list[str], flag: str) -> str:
    """The argument following ``flag``."""
    return argv[argv.index(flag) + 1]


# --- rendering ---------------------------------------------------------------------


@pytest.mark.parametrize("block", ["setup", "run"])
def test_block_renders_and_is_valid_bash(block):
    """Each block renders strictly with the fixture config and bash can parse it."""
    rendered = _render(block)
    proc = subprocess.run(["bash", "-n"], input=rendered, text=True, check=False)
    assert proc.returncode == 0


def test_whole_launcher_fills_strictly():
    """Every string in the launcher config fills, as targetsteprun.py does it."""
    ctx = {"config": _config(), "bindings": _BINDINGS}
    filled = fill_objtemplate(_LAUNCHER, ctx, strict=True)
    assert filled["image_id"] == ""


def test_run_passes_bound_input_paths_verbatim(tmp_path):
    """A bound input reaches run.sh as one argument, quotes intact."""
    argv = _argv("run", tmp_path)
    assert _value(argv, "--input-model_1") == "/staged/model_1 it's"
    assert _value(argv, "--input-base_model") == "/staged/base_model"


def test_run_passes_empty_paths_for_unbound_inputs(tmp_path):
    """An unbound optional input the recipe doesn't use renders as an empty argument."""
    # A recipe that names only model_1 (the fixture's also names base_model, which must
    # then be bound: see test_render_and_prepare_agree).
    recipe = {"merge_method": "linear", "models": [{"model": "model_1"}]}
    argv = _argv(
        "run",
        tmp_path,
        over={"merge_config": recipe},
        bindings={"model_1": _BINDINGS["model_1"]},
    )
    for name in _INPUTS[1:]:
        assert _value(argv, f"--input-{name}") == ""


def test_run_passes_config_as_json_and_flags(tmp_path):
    """merge_config and args reach run.sh as JSON; the booleans as true/false."""
    argv = _argv("run", tmp_path, over={"args": {"out-shard-size": "1B"}})
    assert (
        json.loads(_value(argv, "--merge-config-json")) == _FIXTURE_MK["merge_config"]
    )
    assert json.loads(_value(argv, "--args-json")) == {"out-shard-size": "1B"}
    assert _value(argv, "--copy-tokenizer") == "true"
    assert _value(argv, "--lazy-unpickle") == "true"
    assert _value(argv, "--allow-crimes") == "false"
    assert _value(argv, "--trust-remote-code") == "false"
    assert _value(argv, "--venv") == "./venv"


def test_run_passes_the_target_step_run_id(tmp_path):
    """run.sh gets the run id that stays the same across retries of one step run."""
    argv = _argv("run", tmp_path, run_metadata={"targetsteprun_id": "tsr-abc"})
    assert _value(argv, "--run-id") == "tsr-abc"
    assert _value(_argv("run", tmp_path), "--run-id") == ""


def test_run_output_path_defaults_to_model(tmp_path):
    """With no output_path the model goes to ./model in the workdir."""
    argv = _argv("run", tmp_path, over={"output_path": ""})
    assert _value(argv, "--output-model") == "./model"
    argv = _argv("run", tmp_path)
    assert _value(argv, "--output-model") == _FIXTURE_MK["output_path"]


def test_setup_installs_the_pinned_version(tmp_path):
    """setup installs mergekit==<mergekit_version> from pip_index_url."""
    argv = _argv("setup", tmp_path, over={"mergekit_version": "0.1.5"})
    assert argv == [
        "--venv",
        "./venv",
        "--index-url",
        "https://pypi.org/simple",
        "--",
        "mergekit==0.1.5",
    ]


def test_mk_image_skips_setup_and_venv(tmp_path):
    """With mk_image set, setup installs nothing, run uses no venv, image_id is docker:."""
    over = {"mk_image": "quay.io/org/mergekit:1"}
    assert _argv("setup", tmp_path, over=over) == []
    assert _value(_argv("run", tmp_path, over=over), "--venv") == ""
    ctx = {"config": _config(over), "bindings": _BINDINGS}
    assert fill_template(_LAUNCHER["image_id"], ctx, strict=True) == (
        "docker:quay.io/org/mergekit:1"
    )


# --- validate (raise_error at step creation) ---------------------------------------


@pytest.mark.parametrize(
    "over, message",
    [
        ({"merge_config": {}}, "merge_config is required"),
        (
            {
                "merge_config": {
                    "merge_method": "nope",
                    "models": [{"model": "model_1"}],
                }
            },
            "merge_method must be one of",
        ),
        (
            {"merge_config": {"models": [{"model": "model_1"}]}},
            "merge_method is required",
        ),
        (
            {
                "merge_config": {
                    "merge_method": "linear",
                    "models": [{"model": "model_1"}],
                    "slices": [{"sources": [{"model": "model_1"}]}],
                }
            },
            "exactly one of models, slices or modules",
        ),
        ({"merge_config": {"merge_method": "linear"}}, "exactly one of models"),
        ({"args": ["out-shard-size"]}, "args must be a mapping"),
        ({"args": {"cuda": True}}, "args.cuda is set by the step"),
        ({"args": {"lora-merge-cache": "/x"}}, "args.lora-merge-cache is set by"),
        ({"args": {"trust-remote-code": True}}, "args.trust-remote-code is set by"),
        ({"args": {"safe-serialization": False}}, "args.safe-serialization is set by"),
        ({"allow_crimes": "yes"}, "allow_crimes must be true or false"),
        ({"lazy_unpickle": None}, "lazy_unpickle must be true or false"),
        ({"output_path": ["/a"]}, "output_path must be a string"),
        ({"mergekit_version": ""}, "set mergekit_config.mergekit_version"),
    ],
)
def test_validate_rejects_bad_config(over, message):
    """Each config rule fails the strict fill with its own message."""
    assert message in _render_error(over)


def test_validate_accepts_mk_image_without_a_version():
    """An image needs no mergekit_version: nothing is installed."""
    _render(over={"mergekit_version": "", "mk_image": "quay.io/org/mergekit:1"})


@pytest.mark.parametrize("where", ["compute", "launcher"])
def test_validate_rejects_more_than_one_node(where):
    """num_nodes must be 1 in both sources gbserver reads it from."""
    msg = _render_error(**{where: {"num_nodes": 2}})
    assert "num_nodes must be 1" in msg


@pytest.mark.parametrize("n", [1, 1.0, "1"])
def test_validate_accepts_one_node(n):
    """Whole-number forms gbserver itself accepts (skypilot.py's num_nodes resolver)."""
    _render(compute={"num_nodes": n}, launcher={"num_nodes": n})


@pytest.mark.parametrize("gpus", [0, 0.0, "0", "0.0", False, None, "", "abc"])
def test_validate_accepts_no_gpu_values(gpus):
    """What gbserver's float(value) > 0 reads as no GPU request (_warn_unrequested_gpus)."""
    _render(compute={"num_gpus_per_node": gpus})


@pytest.mark.parametrize("gpus", [1, 1.0, "1", "2", True])
def test_validate_rejects_gpu_counts_without_a_request(gpus):
    """What gbserver's float(value) > 0 counts as GPUs, true included (float(True) == 1)."""
    assert "issue #536" in _render_error(compute={"num_gpus_per_node": gpus})


def test_validate_rejects_gpus_without_accelerators():
    """A GPU count alone requests nothing (issue #536): refuse it."""
    msg = _render_error(compute={"num_gpus_per_node": 1})
    assert "issue #536" in msg


@pytest.mark.parametrize(
    "resources", [{"accelerators": "A10G:1"}, {"instance_type": "g5.xlarge"}]
)
def test_validate_accepts_gpus_with_a_gpu_request(resources):
    """Either GPU request gbserver honours (skypilot.py _warn_unrequested_gpus)."""
    _render(compute={"num_gpus_per_node": 1}, launcher={"resources": resources})


# --- run.sh ------------------------------------------------------------------------


def test_run_sh_registers_the_model_output():
    """src/run.sh prints the GB_ARTIFACT_ID marker for the declared output.

    Only code counts: comment lines are skipped, or a comment alone would pass.
    """
    code = "\n".join(
        line
        for line in (_STEP_DIR / "src" / "run.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    for name in _TEMPLATE["outputs"]["required"]:
        assert f"GB_ARTIFACT_ID:{name} " in code


# gbserver's non-strict fill of the step config (targetsteprun.py), then the strict
# launcher fill: what a build's chat_template really looks like when run.sh gets it.
_LLAMA3 = (
    "{{ bos_token }}{% for message in messages %}<|start_header_id|>"
    "{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
    "{% endfor %}{{ eos_token }}"
)


def _filled_by_gbserver(template: str) -> str:
    """What gbserver's non-strict config fill leaves of a build.yaml chat_template."""
    step_config = {"mergekit_config": {"merge_config": {"chat_template": template}}}
    filled = fill_objtemplate(step_config, {"config": {}}, strict=False)
    return filled["mergekit_config"]["merge_config"]["chat_template"]


def _chat_template_after_gbserver(template: str) -> str:
    """The chat_template run.sh receives for a build.yaml value ``template``."""
    step_config = {"mergekit_config": {"merge_config": {"chat_template": template}}}
    filled = fill_objtemplate(step_config, {"config": {}}, strict=False)
    over = {
        "merge_config": {
            **_FIXTURE_MK["merge_config"],
            **filled["mergekit_config"]["merge_config"],
        }
    }
    rendered = _render("run", over)
    arg = rendered.split("--merge-config-json '", 1)[1].split("' \\\n", 1)[0]
    return json.loads(arg.replace("'\"'\"'", "'"))["chat_template"]


def test_raw_wrapped_template_survives_gbserver_fill():
    assert (
        _chat_template_after_gbserver("{% raw %}" + _LLAMA3 + "{% endraw %}") == _LLAMA3
    )


def test_unwrapped_template_is_mangled_by_gbserver_fill_and_refused():
    """Unwrapped, the fill mangles it; both the render and the node then refuse it."""
    mp = pytest.importorskip("mergekit_prepare")
    mangled = _filled_by_gbserver(_LLAMA3)
    assert mangled != _LLAMA3
    over = {"merge_config": {**_FIXTURE_MK["merge_config"], "chat_template": mangled}}
    assert "chat_template" in _render_error(over)  # refused at step creation
    with pytest.raises(mp.PrepareError):
        mp.check_chat_template(mangled)  # and on the node


# The validate macro (render time) and mergekit_prepare.py (node) check the same config
# rules; the macro so a bad build fails before provisioning, prepare for a direct run.
# One table, both sides: they must agree on every case.
_BOTH = [
    # (label, merge_config overrides, mergekit_config overrides, bindings, ok)
    ("fixture", {}, {}, None, True),
    ("models and slices", {"models": [{"model": "model_1"}]}, {}, None, False),
    ("none of models/slices/modules", {"slices": None}, {}, None, False),
    ("no merge_method", {"merge_method": None}, {}, None, False),
    ("args non-ascii key", {}, {"args": {"r\u00e1ndom-seed": 1}}, None, False),
    ("args empty value", {}, {"args": {"lora-merge-dtype": ""}}, None, False),
    ("args no- key", {}, {"args": {"no-write-model-card": True}}, None, False),
    ("args dashed key", {}, {"args": {"--random-seed": 1}}, None, False),
    ("args list value", {}, {"args": {"out-shard-size": ["1B"]}}, None, False),
    (
        "args fine",
        {},
        {"args": {"out-shard-size": "1B", "write-model-card": False}},
        None,
        True,
    ),
    ("template name", {"chat_template": "chatml"}, {}, None, True),
    (
        "template mangled",
        {"chat_template": "{{ bos_token }}{{ eos_token }}"},
        {},
        None,
        False,
    ),
    ("template plain text", {"chat_template": "ASSISTANT:"}, {}, None, False),
    ("template empty", {"chat_template": ""}, {}, None, False),
    (
        "template raw-wrapped",
        {"chat_template": "{% for m in messages %}{{ m }}{% endfor %}"},
        {},
        None,
        True,
    ),
    (
        "template, no tokenizer",
        {"chat_template": "chatml"},
        {"copy_tokenizer": False},
        None,
        False,
    ),
    (
        "template, builds one",
        {"chat_template": "chatml", "tokenizer_source": "base"},
        {"copy_tokenizer": False},
        None,
        True,
    ),
    (
        "template, empty block",
        {"chat_template": "chatml", "tokenizer": {}},
        {"copy_tokenizer": False},
        None,
        True,
    ),
    (
        # model_2 is optional, so only the unbound-name check can refuse this (an
        # unbound required model_1 would fail on the launcher's own bindings lookup).
        "unbound optional model_2",
        {"models": [{"model": "model_1"}, {"model": "model_2"}], "slices": None},
        {},
        None,
        False,
    ),
    (
        # The review's typo: a bound model_1 the recipe never names.
        "bound model_1 named model1",
        {
            "models": [{"model": "model1"}, {"model": "base_model"}],
            "slices": None,
        },
        {},
        None,
        False,
    ),
    (
        "bound base_model unused",
        {"models": [{"model": "model_1"}], "slices": None, "base_model": None},
        {},
        None,
        False,
    ),
    (
        # Used only in a mapping-form reference: render can't see the reference, so it
        # must not refuse; prepare sees it and accepts.
        "base_model used only in mapping form",
        {
            "models": [{"model": "model_1"}, {"model": {"model": "base_model"}}],
            "slices": None,
            "base_model": None,
        },
        {},
        None,
        True,
    ),
    (
        # S1: a base_model KEY whose value is a Hub id doesn't make the bound base_model
        # input used. Render must refuse too, not leave it to the node.
        "bound base_model, key holds a Hub id",
        {
            "models": [{"model": "model_1"}],
            "slices": None,
            "base_model": "Qwen/Qwen2.5-0.5B",
        },
        {},
        None,
        False,
    ),
    (
        # S2: the bindings form, as gbserver's fill renders it for bound inputs.
        "bindings form, bound",
        {
            "models": [
                {"model": "/staged/model_1 it's"},
                {"model": "/staged/base_model"},
            ],
            "slices": None,
            "base_model": None,
        },
        {},
        None,
        True,
    ),
    (
        # ...and as the fill leaves it for an unbound one.
        "bindings form, unbound",
        {
            "models": [
                {"model": "model_1"},
                {"model": "base_model"},
                {"model": "{{ model_2.binding.path }}"},
            ],
            "slices": None,
            "base_model": None,
        },
        {},
        None,
        False,
    ),
    (
        "mapping-valued models",
        {"models": {"model": "model_1"}, "slices": None},
        {},
        None,
        False,
    ),
    (
        "list-valued modules",
        {"modules": [{"models": [{"model": "model_1"}]}], "slices": None},
        {},
        None,
        False,
    ),
]


def _prepare_verdict(merge_config: dict, mk: dict, bindings: dict) -> bool:
    mp = pytest.importorskip("mergekit_prepare")
    import tempfile  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as d:
        argv = [
            "--merge-config-json",
            json.dumps(merge_config),
            "--args-json",
            json.dumps(mk.get("args", {})),
            "--recipe-out",
            f"{d}/r.yaml",
            "--expect-out",
            f"{d}/e",
            "--copy-tokenizer",
            str(mk.get("copy_tokenizer", True)).lower(),
            "--lazy-unpickle",
            "true",
            "--allow-crimes",
            "false",
            "--trust-remote-code",
            "false",
        ]
        for name, b in bindings.items():
            argv += [f"--input-{name}", b["binding"]["path"]]
        return mp.main(argv) == 0


@pytest.mark.parametrize(
    "label, mc_over, mk_over, bindings, ok", _BOTH, ids=[c[0] for c in _BOTH]
)
def test_render_and_prepare_agree(label, mc_over, mk_over, bindings, ok):
    """Each case: the render-time macro and the node-side prepare give the same verdict."""
    bindings = _BINDINGS if bindings is None else bindings
    # An override of None removes the key (e.g. the fixture's slices, for a models recipe).
    mc = {**copy.deepcopy(_FIXTURE_MK["merge_config"]), **mc_over}
    mc = {k: v for k, v in mc.items() if v is not None}
    over = {"merge_config": mc, **mk_over}
    try:
        _render("run", over, bindings=bindings)
        rendered = True
    except RuntimeError:
        rendered = False
    assert rendered is ok, f"{label}: render {'accepted' if rendered else 'refused'}"
    assert _prepare_verdict(mc, mk_over, bindings) is ok, f"{label}: prepare disagrees"


@pytest.mark.parametrize(
    "mk_over, merge_method, ok",
    [
        ({}, "not_a_method", False),  # the step installs 0.1.5: the pinned list applies
        ({"mergekit_version": "0.1.6"}, "not_a_method", True),  # MergeKit checks it
        ({"mk_image": "quay.io/org/mergekit:1"}, "not_a_method", True),
        ({"mergekit_version": "0.1.6"}, "", False),  # empty is refused on any version
        ({"mergekit_version": "0.1.6"}, 1, False),  # so is a non-string
        ({"mk_image": "quay.io/org/mergekit:1"}, True, False),
    ],
)
def test_merge_method_list_applies_only_to_the_pinned_version(
    mk_over, merge_method, ok
):
    over = {
        "merge_config": {**_FIXTURE_MK["merge_config"], "merge_method": merge_method},
        **mk_over,
    }
    if ok:
        _render("run", over)
    else:
        assert "merge_method" in _render_error(over)


@pytest.mark.parametrize(
    "mk_over, ok",
    [
        ({}, False),
        ({"mk_image": "quay.io/org/mergekit:1"}, True),
        ({"mergekit_version": "0.2.0"}, True),
    ],
)
def test_template_name_list_applies_only_to_the_pinned_version(mk_over, ok):
    """A name newer than 0.1.5 is left to the node, which reads the installed list."""
    over = {
        "merge_config": {**_FIXTURE_MK["merge_config"], "chat_template": "newtemplate"},
        **mk_over,
    }
    if ok:
        _render("run", over)
    else:
        assert "chat_template" in _render_error(over)
