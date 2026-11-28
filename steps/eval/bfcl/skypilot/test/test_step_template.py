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


def test_output_location_is_composable(step, run_script):
    # Either a full output_dir, or output_root + run_name composed per benchmark,
    # run_name defaulting to the model_path basename (the same defaulting as
    # served_model_name); with neither set the run fails fast. Recipes bind
    # output_root + run_name.
    assert step["config"]["output_dir"] == ""
    assert step["config"]["output_root"] == ""
    assert step["config"]["run_name"] == ""
    assert 'OUTPUT_DIR="{{ config.output_dir }}"' in run_script
    assert 'OUTPUT_ROOT="{{ config.output_root }}"' in run_script
    assert 'RUN_NAME="{{ config.run_name }}"' in run_script
    assert 'RUN_NAME="$(basename "{{ config.model_path }}")"' in run_script


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


def _run_until_output_dir(step, tmp_path, env_extra=None, **config):
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
        + 'echo "DOFF=${HF_DATASETS_OFFLINE:-}"\n'
        + 'for w in "${OPT_WORDS[@]}"; do echo "OPT=$w"; done\n'
    )
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path)}
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", "-c", probe], cwd=cwd, env=env, capture_output=True, text=True
    )


def _out(proc):
    assert proc.returncode == 0, proc.stderr
    return {
        k: v
        for k, _, v in (ln.partition("=") for ln in proc.stdout.splitlines())
        if k in ("OUT", "OFF", "DOFF")
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
    assert _out(proc)["DOFF"] == "1"  # datasets' own offline switch too
    assert cache.is_dir()


def test_online_by_default(step, tmp_path):
    proc = _run_until_output_dir(step, tmp_path, benchmark="b")
    assert _out(proc)["OFF"] == ""


@pytest.mark.parametrize("word", ["0", "false", "False", "no", "off", " "])
def test_hf_offline_off_words_stay_online(step, tmp_path, word):
    # A word, not a truth value: "0" and "false" are non-empty strings.
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="b", hf_offline=word, hf_home=str(tmp_path / "c")
    )
    assert _out(proc)["OFF"] == ""


@pytest.mark.parametrize("word", ["1", "true", "TRUE", "yes", "on"])
def test_hf_offline_on_words_run_offline(step, tmp_path, word):
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="b", hf_offline=word, hf_home=str(tmp_path / "c")
    )
    assert _out(proc)["OFF"] == "1"


@pytest.mark.parametrize("word", ["maybe", "y", "n", "2", "enabled"])
def test_hf_offline_rejects_other_words(step, tmp_path, word):
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="b", hf_offline=word, hf_home=str(tmp_path / "c")
    )
    assert proc.returncode == 1
    assert "hf_offline must be" in proc.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "options", ["", "max_retries=3 max_failed_frac=0.1 request_timeout=60"]
)
def test_a_curated_benchmark_keeps_its_own_dir(step, tmp_path, options):
    # No option that can change results: $OUTPUT_ROOT/$RUN_NAME/<benchmark>, as
    # before the bridges. Operational options (retries, timeouts, the failure
    # threshold) do not count.
    proc = _run_until_output_dir(step, tmp_path, benchmark="mmlu-x", options=options)
    assert _out(proc)["OUT"] == str(tmp_path / "out" / "model-x" / "mmlu-x")


def test_a_curated_benchmarks_options_name_its_dir(step, tmp_path):
    # A curated recipe that passes options gets a dir named after them, so two
    # runs with different options never resume into one dir.
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="mmlu-x", options="limit_langs=en judge_model=self"
    )
    name = Path(_dir(proc)).name
    assert name == "mmlu-x@judge_model-self@limit_langs-en"


HYDRA_SAFE = re.compile(r"[A-Za-z0-9._@+-]+")
"""The run-dir name's characters: a subset of what a Hydra 1.3.8 override
value takes unquoted (NeMo-Skills passes paths as ++output_file=<path>;
= , ~ ( ) and the like break its parser or make a sweep), with no shell glob
character and no % (URI handling could decode it)."""


def _dir(proc):
    """The composed output dir, checked as a Hydra override value: its name
    within the charset, and, where hydra-core is importable (the nemoskills
    images; any venv with it), parsed exactly as ns parses ++output_file."""
    out = _out(proc)["OUT"]
    name = Path(out).name
    assert HYDRA_SAFE.fullmatch(name), name
    try:
        from hydra.core.override_parser.overrides_parser import OverridesParser
    except ImportError:
        return out
    path = f"{out}/output-rs0.jsonl"
    override = OverridesParser.create().parse_overrides([f"++output_file={path}"])[0]
    assert not override.is_sweep_override(), path
    assert override.value() == path, (override.value(), path)
    return out


@pytest.mark.parametrize(
    ("options", "name"),
    [
        ("tasks=hellaswag,arc_easy", "lm-eval@tasks-hellaswag+2Carc_easy"),
        ("domains=airline,retail", "lm-eval@domains-airline+2Cretail"),
        (
            "service=filesystem task_suite=easy",
            "lm-eval@service-filesystem@task_suite-easy",
        ),
        (
            "task_suite=easy service=filesystem",
            "lm-eval@service-filesystem@task_suite-easy",
        ),
        ("categories=web_search,memory", "lm-eval@categories-web_search+2Cmemory"),
        ("benchmark=math-500 split=test", "lm-eval@benchmark-math-500@split-test"),
        ("tasks=x exclude=y", "lm-eval@tasks-x@exclude-y"),
        ("instances=^apache__druid", "lm-eval@instances-+5Eapache__druid"),
        ("tasks=a tasks=b", "lm-eval@tasks-b"),  # the harness reads the last
        ("tasks=hellaswag metric=acc_norm", "lm-eval@tasks-hellaswag@metric-acc_norm"),
        ("tasks=a@b", "lm-eval@tasks-a+40b"),  # an @ in a value is escaped
        (
            "domains=airline task_split=test",
            "lm-eval@task_split-test@domains-airline",
        ),
        (
            "tasks=gsm8k thinking=on mode=chat num_fewshot=5 "
            "gen_kwargs=temperature=0.6 metric=exact_match,strict-match",
            "lm-eval@tasks-gsm8k@gen_kwargs-temperature+3D0.6"
            "@metric-exact_match+2Cstrict-match@mode-chat@num_fewshot-5@thinking-on",
        ),
        ("tasks=x thinking=on thinking=off", "lm-eval@tasks-x@thinking-off"),
        ("tasks=x max_retries=3 max_failed_frac=0.1", "lm-eval@tasks-x"),
        (
            "tasks=x judge_base_url=http://h:1/v1 user_base_url=http://u "
            "search_mcp_url=http://m critpt_api_url=http://c judge_api_key_env=K",
            "lm-eval@tasks-x",
        ),  # endpoint addresses
        ("tasks=x base_url_mode=y", "lm-eval@tasks-x@base_url_mode-y"),
        (
            "a=0 split=10 instances=9 exclude=8 tasks=7 categories=6 domains=5 "
            "task_split=4 task_suite=3 service=2 benchmark=1",
            "lm-eval@benchmark-1@service-2@task_suite-3@task_split-4@domains-5"
            "@categories-6@tasks-7@exclude-8@instances-9@split-10@a-0",
        ),
        ("tasks=x a-b=1 c*=2", "lm-eval@tasks-x@a+2Db-1@c+2A-2"),
        (
            "tasks=x step_limit=50 temperature=0.7",
            "lm-eval@tasks-x@step_limit-50@temperature-0.7",
        ),
    ],
)
def test_a_selection_names_the_dir(step, tmp_path, options, name):
    # Every selection value in a fixed key order, then every other option
    # that can change results, sorted by key (operational ones do not count).
    proc = _run_until_output_dir(step, tmp_path, benchmark="lm-eval", options=options)
    assert _dir(proc) == str(tmp_path / "out" / "model-x" / name)


@pytest.mark.parametrize(
    "option",
    [
        "judge_workers=8",
        "judge_parallel=2",
        "concurrency=3",
        "llm_timeout=60",
        "judge_timeout=30",
        "judge_retries=5",
        "env_start_timeout=90",
        "env_poll=1",
        "env_url=http://e:1",
        "env_port=2000",
        "max_judge_invalid_frac=0.2",
    ],
)
def test_an_operational_option_leaves_the_dir_alone(step, tmp_path, option):
    # Worker counts, the timeout and retries of a call whose failure is left
    # out of the score and retried on resume, the environment's address and a
    # failure threshold: none changes a reported score.
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="lm-eval", options=f"tasks=x {option}"
    )
    assert _dir(proc) == str(tmp_path / "out" / "model-x" / "lm-eval@tasks-x")


@pytest.mark.parametrize(
    "option, word",
    [("task_timeout=600", "task_timeout-600"), ("max_errors=3", "max_errors-3")],
)
def test_a_limit_that_is_scored_names_the_dir(step, tmp_path, option, word):
    # mcpatlas scores a task that runs out of time as its timeout error, and
    # tau2-bench ends a simulation after max_errors tool errors: both change
    # scores, so they are not operational.
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="lm-eval", options=f"tasks=x {option}"
    )
    assert _dir(proc) == str(tmp_path / "out" / "model-x" / f"lm-eval@tasks-x@{word}")


def test_distinct_selections_never_share_a_dir(step, tmp_path):
    # Values that a lossy mapping would merge (a / or a glob character into
    # _, a comma into -), and a bridge selection that names a curated id's
    # dir (mcpmark service=filesystem vs the curated mcpmark-filesystem).
    runs = [
        ("lm-eval", "tasks=a_b"),
        ("lm-eval", "tasks=a*b"),
        ("lm-eval", "tasks=a?b"),
        ("lm-eval", "tasks=a/b"),
        ("lm-eval", "tasks=a,b"),
        ("lm-eval", "tasks=a-b"),
        ("lm-eval", "tasks=a+2Cb"),
        ("lm-eval", "tasks=a+b"),
        ("lm-eval", "tasks=a domains=b"),
        ("lm-eval", "tasks=a@domains-b"),
        ("lm-eval", "tasks=a-b"),
        ("lm-eval", "tasks=a@b"),
        ("lm-eval", "tasks=a+40b"),
        ("lm-eval", "tasks=a thinking=on"),
        ("lm-eval", "tasks=a thinking=off"),
        ("lm-eval", "tasks=a a-b=1"),
        ("lm-eval", "tasks=a a=b-1"),
        ("mcpmark", "service=filesystem"),
        ("mcpmark", "service=filesystem task_suite=easy"),
        ("mcpmark-filesystem", ""),
        ("tau3", "domains=airline"),
        ("tau3-airline", ""),
    ]
    dirs = {}
    for i, (benchmark, options) in enumerate(runs):
        sub = tmp_path / str(i)
        sub.mkdir()
        proc = _run_until_output_dir(step, sub, benchmark=benchmark, options=options)
        out = Path(_dir(proc))
        assert out.parent == sub / "out" / "model-x"
        dirs.setdefault((benchmark, options), set()).add(out.name)
    names = [name for names in dirs.values() for name in names]
    assert all(len(names) == 1 for names in dirs.values())  # deterministic
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize(
    ("options", "name"),
    [
        ("tasks=glob*", "lm-eval@tasks-glob+2A"),
        ("tasks=../../etc/x", "lm-eval@tasks-..+2F..+2Fetc+2Fx"),
        ("tasks=a/b,c?d", "lm-eval@tasks-a+2Fb+2Cc+3Fd"),
        ("tasks=x=y~z", "lm-eval@tasks-x+3Dy+7Ez"),
        ("tasks=café", "lm-eval@tasks-caf+C3+A9"),
    ],
)
def test_a_selection_is_one_literal_path_segment(step, tmp_path, options, name):
    proc = _run_until_output_dir(step, tmp_path, benchmark="lm-eval", options=options)
    out = _dir(proc)
    assert out == str(tmp_path / "out" / "model-x" / name)
    assert Path(out).parent == tmp_path / "out" / "model-x"


def test_options_are_split_but_never_globbed(step, tmp_path):
    # The cwd holds a file named tasks=globbed; an unquoted for would expand
    # tasks=glob* to it. The words reach the harness call as written.
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="lm-eval", options="tasks=glob*  metric=it's"
    )
    assert _opts(proc) == ["tasks=glob*", "metric=it's"]
    assert "${OPT_WORDS[@]}" in _launcher(step)["run"].split("OPTS=()", 1)[1]


def test_options_over_several_lines_are_all_read(step, tmp_path):
    # A YAML block scalar spreads options over lines; every word reaches the
    # harness call and the run dir, not only the first line's.
    proc = _run_until_output_dir(
        step,
        tmp_path,
        benchmark="lm-eval",
        options="tasks=a metric=m\n  exclude=b\r\n\tsplit=c\n",
    )
    assert _opts(proc) == ["tasks=a", "metric=m", "exclude=b", "split=c"]
    assert Path(_dir(proc)).name == "lm-eval@tasks-a@exclude-b@split-c@metric-m"


def test_a_long_selection_is_shortened_but_stays_distinct(step, tmp_path):
    names = []
    for i, last in enumerate("xy"):
        sub = tmp_path / str(i)
        sub.mkdir()
        tasks = ",".join(f"task_{n}" for n in range(40)) + last
        proc = _run_until_output_dir(
            step, sub, benchmark="lm-eval", options=f"tasks={tasks}"
        )
        names.append(Path(_dir(proc)).name)
    for name in names:
        assert len(name) == 180 + 3 + 16 and name[180:183] == "@h-"
    assert names[0][:180] == names[1][:180] and names[0] != names[1]


NS_MATH_500_URL = (
    "https://github.com/openai/prm800k/raw/main/prm800k/math_splits/test.jsonl"
)
NS_MATH_500_SHA = (
    "sha256:35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
)


@pytest.mark.parametrize(
    ("benchmark", "dataset", "revision", "options", "name"),
    [
        (
            "swebench",
            "SWE-bench/SWE-bench_Multilingual",
            "846e647b9f33c0b51b739d005d13d85493c9af09",
            "instances=^apache__druid",
            "swebench@instances-+5Eapache__druid"
            "@dataset-SWE-bench+2FSWE-bench_Multilingual"
            "@dataset_revision-846e647b9f33c0b51b739d005d13d85493c9af09",
        ),
        (
            "terminal-bench",
            "harborframework/terminal-bench-2.1",
            "",
            "",
            "terminal-bench@dataset-harborframework+2Fterminal-bench-2.1@dataset_revision-",
        ),
        (
            "nemoskills",
            "https://example.org/t.jsonl",
            "sha256:00",
            "benchmark=math-500",
            "nemoskills@benchmark-math-500@dataset-https+3A+2F+2Fexample.org+2Ft.jsonl"
            "@dataset_revision-sha256+3A00",
        ),
        (
            "swebench-verified",
            "SWE-bench/SWE-bench_Verified",
            "x",
            "",
            "swebench-verified@dataset-SWE-bench+2FSWE-bench_Verified@dataset_revision-x",
        ),
        (
            "swebench-verified",
            "",
            "abc123",
            "",
            "swebench-verified@dataset-@dataset_revision-abc123",
        ),
    ],
)
def test_a_dataset_pinned_bridge_names_its_pin(
    step, tmp_path, benchmark, dataset, revision, options, name
):
    # The data is a selection too: a dataset-pinned bridge's (the bridge's
    # terminal-bench-2.1 is not the curated terminal-bench-2.1 dir), and a
    # curated id's dataset or revision swap.
    proc = _run_until_output_dir(
        step,
        tmp_path,
        benchmark=benchmark,
        dataset=dataset,
        dataset_revision=revision,
        options=options,
    )
    assert _dir(proc) == str(tmp_path / "out" / "model-x" / name)


def test_ns_math_500s_dir_is_hydra_safe_and_stable(step, tmp_path):
    # The shipped nemoskills bridge recipe: its pin is long, so the name is
    # cut and hashed, but it starts with the readable selection, is the same
    # for every render (generate and score share it), and parses as a Hydra
    # override value.
    names = set()
    for phase in ("generate", "score"):
        sub = tmp_path / phase
        sub.mkdir()
        proc = _run_until_output_dir(
            step,
            sub,
            benchmark="nemoskills",
            dataset=NS_MATH_500_URL,
            dataset_revision=NS_MATH_500_SHA,
            options="benchmark=math-500",
            phase=phase,
        )
        names.add(Path(_dir(proc)).name)
    (name,) = names
    assert name.startswith(
        "nemoskills@benchmark-math-500@dataset-https+3A+2F+2Fgithub.com"
    )
    assert name[180:183] == "@h-"


@pytest.mark.parametrize("workdir", [True, False])
def test_a_relative_output_dir_is_anchored(step, tmp_path, workdir):
    # A relative output_dir resolves against $GB_BUILD_WORKDIR (the build's
    # workdir), else the job's cwd -- never left relative to wherever the
    # harness later changes directory.
    wd = tmp_path / "wd"
    extra = {"GB_BUILD_WORKDIR": str(wd)} if workdir else {}
    proc = _run_until_output_dir(
        step, tmp_path, env_extra=extra, benchmark="b", output_dir="rel/x"
    )
    base = wd if workdir else tmp_path / "cwd"
    assert _out(proc)["OUT"] == str(base / "rel" / "x")
    assert (base / "rel" / "x").is_dir()


@pytest.mark.parametrize(
    ("seed", "suffix"), [("0", ""), ("", ""), ("1", "@seed-1"), ("42", "@seed-42")]
)
def test_a_non_default_seed_names_the_dir(step, tmp_path, seed, suffix):
    # Another seed samples (and orders shuffled docs) differently: its own dir.
    proc = _run_until_output_dir(
        step, tmp_path, benchmark="lm-eval", options="tasks=x thinking=on", seed=seed
    )
    assert Path(_dir(proc)).name == "lm-eval@tasks-x@thinking-on" + suffix


def test_a_seed_alone_names_a_curated_dir(step, tmp_path):
    proc = _run_until_output_dir(step, tmp_path, benchmark="mmlu-x", seed="7")
    assert Path(_dir(proc)).name == "mmlu-x@seed-7"
