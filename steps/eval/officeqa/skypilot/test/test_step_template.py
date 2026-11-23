# SPDX-License-Identifier: Apache-2.0

"""Contract tests for step-template.yaml.

Every granite-evals family step under steps/eval/ shares this shape and differs only
in `name`, the family's image env var, and its config defaults. `benchmark` selects
which benchmark in the family a target runs; this suite checks every sibling as
well as this one.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
from libgbtest.step_templates import expanded_step_template

_HERE = Path(__file__).resolve().parent.parent
_STEPS = _HERE.parent.parent
# The granite-evals image families. steps/eval/ also holds eval-example, a
# pre-existing exemplar step that does not follow this contract.
_FAMILIES = (
    "bfcl",
    "bird",
    "ifbench",
    "judged",
    "lmeval",
    "mcpmark",
    "nemoskills",
    "officeqa",
    "swebench",
    "tau",
    "tbench",
)
_TEMPLATES = sorted(_STEPS / f / "skypilot" / "step-template.yaml" for f in _FAMILIES)


def _load(path):
    # As `make space` renders it: gb-include lines expanded.
    return expanded_step_template(path)


def _launcher(step):
    return step["environment_configs"]["Skypilot"]["launchers"]["granite"]["config"]


@pytest.fixture(params=_TEMPLATES, ids=lambda p: p.parent.parent.name)
def template(request):
    return request.param


@pytest.fixture
def step(template):
    return _load(template)


@pytest.fixture
def run_script(step):
    return _launcher(step)["run"]


def _as_shell(script):
    # Jinja comments (a gb-include's header) and tags out, expressions as a word.
    script = re.sub(r"\{#.*?#\}", " ", script, flags=re.S)
    script = re.sub(r"\{%.*?%\}", " ", script, flags=re.S)
    return re.sub(r"\{\{.*?\}\}", "X", script, flags=re.S)


def test_this_step_is_found():
    assert _HERE / "step-template.yaml" in _TEMPLATES


def test_name_matches_directory(template, step):
    assert step["name"] == template.parent.parent.name


def test_benchmark_is_a_required_config_field(step, run_script):
    assert step["config"]["benchmark"] == ""
    assert 'BENCHMARK="{{ config.benchmark }}"' in run_script


def test_type_is_a_real_step_type(step):
    from gbcommon.types.stepconfig import StepType

    assert StepType(step["type"]) == StepType.CUSTOM


@pytest.mark.parametrize("strip_optional", [False, True])
def test_run_script_is_valid_bash(run_script, strip_optional):
    script = run_script
    if strip_optional:
        script = re.sub(r"\{%\s*if.*?%\}.*?\{%\s*endif\s*%\}", "", script, flags=re.S)
    result = subprocess.run(
        ["bash", "-n"], input=_as_shell(script), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_script_hygiene(run_script):
    assert "${#" not in run_script  # opens a Jinja comment
    assert "set -o pipefail" in run_script  # the run is behind `| tee`
    assert "unset RANK WORLD_SIZE" in run_script  # vLLM under torch.distributed env


def test_image_is_chosen_at_runtime_not_built(step):
    # The family image is built by steps/eval/Makefile from the Dockerfile one
    # directory up from this template's skypilot/ dir, never by this step.
    assert not (_HERE / "Dockerfile").exists()
    assert (_HERE.parent / "Dockerfile.gpu").exists()
    assert "config.image" in _launcher(step)["image_id"]
    assert _launcher(step)["resources"] == {}


def test_outputs_match_markers(step, run_script):
    # results.json (phase all/score) and generation.json (phase generate); which one a
    # run emits depends on the phase, so both are optional.
    assert step["outputs"] == {
        "optional": {
            "granite_results": {"type": "dataset"},
            "granite_generation": {"type": "dataset"},
        }
    }
    assert set(re.findall(r"GB_ARTIFACT_ID:(\w+)", run_script)) == {
        "granite_results",
        "granite_generation",
    }
    assert run_script.count("GB_ARTIFACT_PATH:${RESULT_FILE}") == 2
    assert 'RESULT_FILE="$OUTPUT_DIR/generation.json"' in run_script


def test_inputs_allow_recipe_ordering_bindings(step):
    # The checkpoint is the config string model_path, and recipes bind only
    # ordering-only inputs (names vary by recipe); nothing is read through
    # bindings.
    assert step["inputs"] == {"allow_unknown": True}


def test_phase_defaults_to_one_job(step, run_script):
    assert step["config"]["phase"] == "all"
    assert '--phase "{{ config.phase }}"' in run_script


def test_monitor_honours_recipe_poll_keys(step):
    mon = step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]
    assert mon["ref"] == "space://monitors/skypilot"
    assert "config.poll_interval_seconds" in mon["config"]["poll_interval_seconds"]
    assert (
        "config.log_retrieval_interval_seconds"
        in mon["config"]["log_retrieval"]["interval_seconds"]
    )


def test_monitor_regex_is_keyed_on_the_configured_benchmark(step):
    mon = step["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"]
    regex = mon["config"]["extra_event_configs"][0]["line_regex"]
    assert "{{ config.benchmark }}" in regex
    assert "vLLM ready at" in regex


def test_every_config_key_reaches_the_script(step, run_script):
    for key in step["config"]:
        if key == "image":
            continue
        assert f"config.{key}" in run_script, key


def test_siblings_share_one_config_contract(step):
    reference = _load(_HERE / "step-template.yaml")
    assert set(step["config"]) == set(reference["config"])


# ---- hf_offline and the composed run dir, as gbserver renders the run script ----


def _run_until_output_dir(step, tmp_path, **config):
    """Render the run script with gbserver's strict renderer, cut it after the
    output dir is made, and run that head under bash (a stub granite-evals on
    PATH, a cwd holding a file a glob in an option would match). Returns the
    process; on success its stdout has OUT=<output dir>, OFF=<HF_HUB_OFFLINE>
    and one OPT=<word> line per option word."""
    from gbserver.utils.template import fill_template

    ctx = dict(step["config"], model_path="org/model-x")
    ctx["output_root"] = str(tmp_path / "out")
    ctx.update(config)
    script = fill_template(_launcher(step)["run"], {"config": ctx}, strict=True)
    head, made, _ = script.partition('mkdir -p "$OUTPUT_DIR"\n')
    assert made, "the run script no longer makes its output dir"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "granite-evals"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "tasks=globbed").touch()
    probe = (
        head
        + made
        + 'echo "OUT=$OUTPUT_DIR"; echo "OFF=${HF_HUB_OFFLINE:-}"\n'
        + 'for w in "${OPT_WORDS[@]}"; do echo "OPT=$w"; done\n'
    )
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path)}
    return subprocess.run(
        ["bash", "-c", probe], cwd=cwd, env=env, capture_output=True, text=True
    )


def _out(proc):
    assert proc.returncode == 0, proc.stderr
    return {
        k: v
        for k, _, v in (ln.partition("=") for ln in proc.stdout.splitlines())
        if k in ("OUT", "OFF")
    }


def _opts(proc):
    return [ln[4:] for ln in proc.stdout.splitlines() if ln.startswith("OPT=")]


def test_hf_offline_without_hf_home_is_refused(step, tmp_path):
    proc = _run_until_output_dir(step, tmp_path, benchmark="b", hf_offline="1")
    assert proc.returncode == 1
    assert "hf_offline needs hf_home" in proc.stderr
    assert not (tmp_path / "out").exists()


def test_hf_offline_with_hf_home_runs_offline(step, tmp_path):
    cache = tmp_path / "cache"
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="b", hf_offline="1", hf_home=str(cache)
    )
    assert _out(proc)["OFF"] == "1"
    assert cache.is_dir()


def test_online_by_default(step, tmp_path):
    proc = _run_until_output_dir(step, tmp_path, benchmark="b")
    assert _out(proc)["OFF"] == ""


@pytest.mark.parametrize("options", ["", "limit_langs=en judge_model=self"])
def test_a_curated_benchmark_keeps_its_own_dir(step, tmp_path, options):
    # No selection option: $OUTPUT_ROOT/$RUN_NAME/<benchmark>, as before the bridges.
    proc = _run_until_output_dir(step, tmp_path, benchmark="mmlu-x", options=options)
    assert _out(proc)["OUT"] == str(tmp_path / "out" / "model-x" / "mmlu-x")


@pytest.mark.parametrize(
    ("options", "suffix"),
    [
        ("tasks=hellaswag,arc_easy", "lm-eval-hellaswag-arc_easy"),
        ("domains=airline,retail", "lm-eval-airline-retail"),
        ("service=filesystem task_suite=easy", "lm-eval-filesystem"),
        ("categories=web_search,memory", "lm-eval-web_search-memory"),
        ("benchmark=math-500", "lm-eval-math-500"),
    ],
)
def test_a_bridge_selection_suffixes_the_dir(step, tmp_path, options, suffix):
    proc = _run_until_output_dir(step, tmp_path, benchmark="lm-eval", options=options)
    assert _out(proc)["OUT"] == str(tmp_path / "out" / "model-x" / suffix)


@pytest.mark.parametrize(
    ("options", "suffix"),
    [
        ("tasks=glob*", "lm-eval-glob_"),
        ("tasks=../../etc/x", "lm-eval-.._.._etc_x"),
        ("tasks=a/b,c?d", "lm-eval-a_b-c_d"),
    ],
)
def test_a_selection_is_one_literal_path_segment(step, tmp_path, options, suffix):
    proc = _run_until_output_dir(step, tmp_path, benchmark="lm-eval", options=options)
    out = _out(proc)["OUT"]
    assert out == str(tmp_path / "out" / "model-x" / suffix)
    assert Path(out).parent == tmp_path / "out" / "model-x"


def test_options_are_split_but_never_globbed(step, tmp_path):
    # The cwd holds a file named tasks=globbed; an unquoted for would expand
    # tasks=glob* to it. The words reach the harness call as written.
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="lm-eval", options="tasks=glob*  metric=it's"
    )
    assert _opts(proc) == ["tasks=glob*", "metric=it's"]
    assert "${OPT_WORDS[@]}" in _launcher(step)["run"].split("OPTS=()", 1)[1]


@pytest.mark.parametrize(
    ("benchmark", "dataset", "suffix"),
    [
        (
            "swebench",
            "SWE-bench/SWE-bench_Multilingual",
            "swebench-SWE-bench_Multilingual",
        ),
        ("terminal-bench", "harborframework/terminal-bench-2.0", "terminal-bench-2.0"),
        ("swebench-verified", "SWE-bench/SWE-bench_Verified", "swebench-verified"),
    ],
)
def test_a_dataset_pinned_bridge_takes_the_dataset_basename(
    step, tmp_path, benchmark, dataset, suffix
):
    # Only the two dataset-pinned bridges; a curated id with a dataset swap keeps its dir.
    proc = _run_until_output_dir(step, tmp_path, benchmark=benchmark, dataset=dataset)
    assert _out(proc)["OUT"] == str(tmp_path / "out" / "model-x" / suffix)
