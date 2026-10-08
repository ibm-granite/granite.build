#!/usr/bin/env python3

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

"""Unit tests for the distill-checkpoint-eval recipe.

This recipe runs a full epoch of off-policy GOLD and evaluates nine checkpoints on all
27 benchmarks. distill-stage1-v2 asks which OBJECTIVE; this asks which CHECKPOINT, and
it is the shape of the graph rather than any single value that makes it able to answer.

So what this file guards is the graph:

* the fanout is MID-RUN. Every export is gated on its own checkpoint_<N> artifact, not
  on the single final `checkpoint` that distill-stage1-v2's exports bind. That is one
  word per target, it is invisible in a rendered YAML unless you look for it, and
  getting it wrong silently turns a 13-hour overlapped run into a serial one.
* every rung will EXIST. A rung that is not a multiple of GOLD_SAVE_STEPS names a
  checkpoint directory the trainer never writes, and because the bindings are mid-run
  the failure is not a late error but a target that never dispatches at all.
* the roll-up can actually be assembled. combined-export derives each column key with
  re.search(r"ckpt_(\\d+)"), so a per-rung CSV whose name does not carry a NUMERIC step
  lands in a column called ckpt_-1 and the table is silently wrong.
* the instruments do not act. gen-smoke reports and exits 0, the entropy guard warns
  rather than stops, and nothing is gated on a divergence number -- because the whole
  epoch is the measurement and a reading that truncated it would destroy it. This is
  the deliberate difference from distill-stage1-v2, where gen-smoke DOES gate, so it
  is exactly the kind of thing a later copy-paste would quietly restore.
"""

import json
import pathlib
import re
import subprocess

import pytest
import yaml
from unit.recipes.published_step import render_run

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-350m"
    / "lsf"
    / "distill-checkpoint-eval"
)
_V2 = _RECIPE.parent / "distill-stage1-v2"
_CATALOG_SRC = _RECIPE.parent / "rl-checkpoint-eval" / "eval-catalog.yaml"

_PLAIN_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")

# Bound by the <% for %> blocks, not by parameters.yaml.
_LOOP_VARS = {"N", "e", "k", "v", "name", "CATS", "SAGE_NAMES"}

_LADDER = ["1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "8150"]
_CATEGORIES = ["code", "general", "math", "safety", "multilingual", "bfcl"]


def _params(**overrides):
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    params.update(overrides)
    return params


def _render(tmp_path, **overrides):
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(**overrides), str(tmp_path))
    )


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _config(rendered, target):
    return _targets(rendered)[target]["steps"][0]["config"]


def _sage_names(**overrides):
    cats = (
        str(_params(**overrides)["EVAL_CATEGORIES"]).replace(" ", "").lower().split(",")
    )
    return [e["name"] for e in _params()["EVAL_CATALOG"] if e["category"] in cats]


@pytest.fixture(name="built")
def fixture_built(tmp_path):
    return _render(tmp_path)


# ─── The parameter surface ─────────────────────────────────────────────────────


def test_every_marker_has_a_parameter():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    names = set(_PLAIN_MARKER.findall(contents)) - _LOOP_VARS
    assert names - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    referenced = set(_PLAIN_MARKER.findall(contents))
    # Reached through an expression rather than a plain marker: CKPT_LADDER via
    # .split(","), CORPUS_DIR via .rstrip("/"), EVAL_CATEGORIES/EVAL_CATALOG via the
    # <% set %> preamble, and SAGE_BATCH_SIZE/SAGE_MAX_LENGTH as the fallback of a
    # `| default(...)` on a catalog entry's optional override.
    referenced |= {
        "CKPT_LADDER",
        "CORPUS_DIR",
        "EVAL_CATEGORIES",
        "EVAL_CATALOG",
        "SAGE_BATCH_SIZE",
        "SAGE_MAX_LENGTH",
    }
    assert set(_params()) - referenced - {"INCLUDE_SFT"} == set()


def test_no_comment_uses_a_template_variable_outside_its_scope():
    """The CLI templates this file IN FULL, comments included, with StrictUndefined, so
    a marker in a comment is evaluated like any other. A loop variable named there is
    an undefined-variable error, and a marker spelled out to EXPLAIN markers is a
    syntax error -- which is how this file first failed to render."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    for line in contents.splitlines():
        if line.lstrip().startswith("#"):
            assert "$${" not in line, line


def test_it_renders_in_every_switch_combination(tmp_path):
    for overrides in (
        {},
        {"INCLUDE_SFT": True},
        {"CORPUS_DIR": "/proj/somewhere/corpus"},
        {"EVAL_CATEGORIES": "bfcl"},
        {"EVAL_CATEGORIES": "math,general"},
        {"CKPT_LADDER": "2,4"},
    ):
        rendered = _render(tmp_path, **overrides)
        assert _targets(rendered), overrides


# ─── The mid-run fanout: the property the recipe exists for ───────────────────


def test_every_export_is_gated_on_its_own_mid_run_checkpoint(built):
    """THE load-bearing edge.

    distill-stage1-v2's exports bind train-gold.checkpoint -- the single artifact the
    step emits when the run ENDS -- so nothing exports until everything has trained.
    These bind train-gold.checkpoint_<N>, which the step's watcher emits as soon as
    rung N is complete, so export-1000 runs ~11 hours before the epoch finishes and
    its 27 evals run beside training that is still going.

    Reverting this is a one-word edit. Nothing else in the build would change, no
    target would fail, and the only symptom would be a run that took as long as
    df8512e0 did.
    """
    for rung in _LADDER:
        binding = _targets(built)[f"export-{rung}"]["inputs"]["checkpoint_dir"][
            "binding"
        ]
        assert binding == f"train-gold.checkpoint_{rung}"


def test_v2_is_the_thing_this_differs_from(tmp_path):
    """Guards the claim above by checking it is actually a difference: if v2 ever moves
    to mid-run bindings too, this test fails and the comments here need rewriting."""
    v2 = yaml.safe_load(
        apply_parameters(
            (_V2 / "build.yaml").read_text(encoding="utf-8"),
            [],
            get_params_from_file(str(_V2 / "parameters.yaml")),
            str(tmp_path),
        )
    )
    v2_export = next(
        t
        for name, t in v2["granite.build"]["targets"].items()
        if name.startswith("export-")
    )
    assert "train-gold.checkpoint" in str(v2_export["inputs"])
    assert "train-gold.checkpoint_" not in str(v2_export["inputs"])


def test_train_gold_turns_the_watcher_on_and_declares_every_rung(built):
    cfg = _config(built, "train-gold")
    # Off (the step default) every checkpoint_<N> output goes unproduced and every
    # export waits forever for an artifact that never arrives.
    assert cfg["emit_checkpoint_artifacts"] is True
    assert int(cfg["checkpoint_watch_interval_seconds"]) > 0
    outputs = _targets(built)["train-gold"]["outputs"]
    for rung in _LADDER:
        assert f"checkpoint_{rung}" in outputs
    # The final single-checkpoint artifact stays declared: it is the one part of the
    # step's output contract that does not depend on the watcher having worked.
    assert "checkpoint" in outputs


def test_each_rung_declares_a_distinct_artifact_uri(built):
    """Nine outputs pointing at one shared parent directory would make nine
    registrations of one URI. gbserver refuses the duplicate, the target still reports
    SUCCESS with an EMPTY output list, and every consumer waits forever -- the
    b5f030cd mode. Each artifact is therefore the checkpoint dir itself, which is why
    the export names it as an absolute path."""
    for rung in _LADDER:
        cfg = _config(built, f"export-{rung}")["export_config"]
        assert cfg["checkpoint"] == "{{ bindings.checkpoint_dir.binding.path }}"
        # Never empty: empty means "highest step number", which across nine concurrent
        # exports of one directory would export the same model nine times.
        assert cfg["checkpoint"]


def test_the_watcher_detection_latency_is_bounded_by_both_intervals():
    """The watcher prints a line immediately; the line reaches the build only when the
    monitor next pulls the logs. A 900s log-retrieval interval (the step's own default)
    would leave a rung unnoticed for a quarter hour of a 16-H100 allocation."""
    params = _params()
    assert int(params["LOG_RETRIEVAL_INTERVAL_SECONDS"]) <= 300
    assert int(params["CKPT_WATCH_INTERVAL_SECONDS"]) <= 300


# ─── The ladder: every rung must exist ────────────────────────────────────────


def test_it_runs_a_full_epoch_not_a_step_budget():
    params = _params()
    # 0 => bounded by the epoch. A positive max_steps overrides num_train_epochs in
    # the trainer, which is what v2 wants and what this recipe must not have.
    assert int(params["GOLD_MAX_STEPS"]) == 0
    assert float(params["GOLD_NUM_TRAIN_EPOCHS"]) == 1.0


def test_every_rung_but_the_last_is_a_multiple_of_the_save_grid():
    params = _params()
    save_steps = int(params["GOLD_SAVE_STEPS"])
    rungs = [int(n) for n in str(params["CKPT_LADDER"]).split(",")]
    for rung in rungs[:-1]:
        assert rung % save_steps == 0, rung
    # The last rung is the trainer's final checkpoint, written at the last step
    # whatever the grid says -- so it is NOT required to be a multiple, and on this
    # corpus it is not one.
    assert rungs[-1] > rungs[-2]


def test_the_ladder_is_the_1000_step_grid_plus_the_final_checkpoint():
    assert str(_params()["CKPT_LADDER"]).split(",") == _LADDER


def test_no_checkpoint_is_evicted_before_its_export_reads_it():
    """save_total_limit 3 is what made df8512e0 unsalvageable: it retained 7750, 8000
    and 8150 of 8,150 and deleted every checkpoint from the healthy region."""
    params = _params()
    rungs = str(params["CKPT_LADDER"]).split(",")
    assert int(params["GOLD_SAVE_TOTAL_LIMIT"]) >= len(rungs)


# ─── Full-eval, per rung ──────────────────────────────────────────────────────


def test_the_inlined_catalog_matches_the_shared_one():
    """The 26 sage evals are transcribed from rl-checkpoint-eval's eval-catalog.yaml.
    A transcription is only as good as the check on it, and this one was wrong in ten
    fields on the first attempt -- gb_script names and result subpaths that look
    guessable and are not."""
    shared = yaml.safe_load(_CATALOG_SRC.read_text(encoding="utf-8"))["evals"]
    mine = {e["name"]: e for e in _params()["EVAL_CATALOG"]}
    expected = {n: v for n, v in shared.items() if v["category"] != "bfcl"}
    assert set(mine) == set(expected)
    for name, ref in expected.items():
        got = mine[name]
        for key in ("category", "gb_script", "image_id", "output_subpath"):
            assert got[key] == ref[key], (name, key)
        overrides = ref.get("overrides") or {}
        assert got.get("batch_size") == overrides.get("batch_size"), name
        assert got.get("max_length") == overrides.get("max_length"), name
        env = dict(overrides.get("extra_env") or {})
        bcb = env.pop("OE_EVAL_BCB_API_URL", None)
        assert dict(got.get("extra_env") or {}) == env, name
        # The BCB URL cannot travel as catalog data: a marker inside a parameter VALUE
        # is emitted literally, never re-rendered. The flag selects it instead.
        assert bool(got.get("needs_bcb_url")) is (bcb is not None), name


def test_the_catalog_plus_bfcl_is_the_full_eval_suite():
    shared = yaml.safe_load(_CATALOG_SRC.read_text(encoding="utf-8"))
    assert len(_params()["EVAL_CATALOG"]) + 1 == len(shared["sets"]["full-eval"])


def test_every_rung_is_crossed_with_every_selected_eval(built):
    names = _sage_names()
    assert len(names) == 26
    for rung in _LADDER:
        for name in names:
            assert f"eval-{name}-ck{rung}" in _targets(built)
        assert f"eval-bfcl-ck{rung}" in _targets(built)


def test_every_eval_measures_an_exported_model(built):
    """Never the raw checkpoint. The export is what prunes the resumable-run state,
    normalises tokenizer.json and pins tokenizer_class, so a benchmark run against the
    unexported directory would measure a model nobody can download."""
    for name in _sage_names() + ["bfcl"]:
        for rung in _LADDER:
            target = _targets(built)[f"eval-{name}-ck{rung}"]
            assert target["inputs"]["model"]["binding"] == f"export-{rung}.hf_model"


def test_no_experiment_name_contains_a_slash(built):
    """sage builds a per-eval job-script FILENAME out of the experiment name
    (sage-<experiment>-<eval>.sh), so a "/" turns it into a nested path and its
    open(..., "w") raises FileNotFoundError."""
    for name in _sage_names():
        for rung in _LADDER:
            cfg = _config(built, f"eval-{name}-ck{rung}")["sage_eval_config"]
            assert "/" not in cfg["experiment"], cfg["experiment"]


def test_the_bcb_url_reaches_the_only_eval_that_needs_it(built):
    """A marker inside a catalog VALUE would be emitted literally, so the template
    writes it. If this regresses, olmes-bigcodebench fails on all nine rungs."""
    url = _params()["OE_EVAL_BCB_API_URL"]
    for rung in _LADDER:
        env = _config(built, f"eval-olmes-bigcodebench-ck{rung}")["sage_eval_config"][
            "extra_env"
        ]
        assert env["OE_EVAL_BCB_API_URL"] == url
    # And nowhere else.
    for name in _sage_names():
        if name == "olmes-bigcodebench":
            continue
        env = (
            _config(built, f"eval-{name}-ck1000")["sage_eval_config"]["extra_env"] or {}
        )
        assert "OE_EVAL_BCB_API_URL" not in env, name


def test_per_eval_overrides_survive_the_loop(built):
    """The override columns are the reason the catalog is data rather than 26 identical
    targets; an override silently lost is an eval measured at the wrong budget."""
    ifeval = _config(built, "eval-olmes-ifeval-ck1000")["sage_eval_config"]
    java = _config(built, "eval-multiple-java-ck1000")["sage_eval_config"]
    cruxeval = _config(built, "eval-olmes-cruxeval-ck1000")["sage_eval_config"]
    params = _params()
    assert int(ifeval["batch_size"]) == 30
    assert int(cruxeval["batch_size"]) == int(params["SAGE_BATCH_SIZE"])
    assert int(java["max_length"]) == 512
    assert int(cruxeval["max_length"]) == int(params["SAGE_MAX_LENGTH"])
    assert java["extra_env"] == {"MULTIPLE_LANG": "java", "MAX_LENGTH": "512"}


def test_the_eval_poll_interval_is_read_by_the_monitor(built):
    """poll_interval_seconds is a TOP-LEVEL step config key (sage-eval/step.yaml reads
    config.poll_interval_seconds | default(900)). rl-checkpoint-eval's generator nests
    it under sage_eval_config, where it is silently ignored."""
    cfg = _config(built, "eval-olmes-gsm8k-ck1000")
    assert "poll_interval_seconds" in cfg
    assert "poll_interval_seconds" not in cfg["sage_eval_config"]


def test_the_allocation_and_the_harness_agree_on_gpu_count(built):
    """Two independent numbers say how many GPUs an eval gets: EVAL_ACCELERATORS, which
    is what LSF allocates, and sage_eval_config.num_gpus, which is what the harness
    shards over. A mismatch does not fail -- it either wastes an allocated GPU or has
    the harness address one that is not there."""
    params = _params()
    allocated = int(str(params["EVAL_ACCELERATORS"]).split(":")[1])
    assert allocated == int(params["SAGE_NUM_GPUS"])
    for rung in _LADDER:
        cfg = _config(built, f"eval-olmes-gsm8k-ck{rung}")
        assert int(cfg["sage_eval_config"]["num_gpus"]) == allocated
        resources = cfg["launcher_config"]["resources"]
        assert resources["accelerators"] == params["EVAL_ACCELERATORS"]
        assert resources["zone"] == params["EVAL_QUEUE"]
    # BFCL states its own, in both generate and evaluate phases.
    bfcl = _config(built, "eval-bfcl-ck1000")["bfcl_config"]
    assert int(bfcl["num_gpus_generate"]) == int(params["BFCL_NUM_GPUS"])
    assert int(bfcl["num_gpus_evaluate"]) == int(params["BFCL_NUM_GPUS"])


def test_every_sage_result_uri_is_where_the_harness_writes(built):
    """The declared output URI is <output_dir>/<experiment>/<output_subpath>, which is
    where sage actually puts results. A drift here means the target reports SUCCESS with
    an artifact pointing at nothing, and the rung's exporter then finds no CSV rows."""
    params = _params()
    catalog = {e["name"]: e for e in params["EVAL_CATALOG"]}
    for name in _sage_names():
        for rung in _LADDER:
            cfg = _config(built, f"eval-{name}-ck{rung}")["sage_eval_config"]
            uri = _targets(built)[f"eval-{name}-ck{rung}"]["outputs"][
                "sage_eval_results"
            ]["uri"]
            expected = (
                f"env://{cfg['output_dir']}/{cfg['experiment']}/"
                f"{catalog[name]['output_subpath']}"
            )
            assert uri == expected, (name, rung, uri)


def test_bfcl_runs_the_full_breadth():
    """full-eval runs BFCL at test_categories all. v2 runs it at `simple` as a cheap
    per-rung probe -- a row measured that way does not mean the same thing as the
    recorded baseline row."""
    assert _params()["BFCL_TEST_CATEGORIES"] == "all"


def test_evals_ride_the_preemptable_queue_and_training_does_not(built):
    """The evals are short, idempotent and independently retryable. A 13-hour epoch is
    none of those things."""
    params = _params()
    assert params["EVAL_QUEUE"] == "preemptable"
    assert params["EXPORT_QUEUE"] == "preemptable"
    assert params["QUEUE"] == "normal"
    assert (
        _config(built, "train-gold")["launcher_config"]["resources"]["zone"] == "normal"
    )


# ─── Aggregation ──────────────────────────────────────────────────────────────


def test_each_rungs_exporter_waits_for_that_rungs_evals_only(built):
    """Per-rung rather than one exporter at the end, so rung 1000's CSV is written
    while rung 5000 is still evaluating and a preempted eval delays only its own rung.
    """
    names = _sage_names()
    for rung in _LADDER:
        gates = _targets(built)[f"export-sage-ck{rung}"]["inputs"]
        assert len(gates) == len(names)
        for gate in gates.values():
            assert gate["binding"].endswith(f"-ck{rung}.sage_eval_results")
        bfcl = _targets(built)[f"export-bfcl-ck{rung}"]["inputs"]["gate_bfcl"]
        assert bfcl["binding"] == f"eval-bfcl-ck{rung}.bfcl_results"


def test_every_per_rung_csv_name_carries_a_numeric_step(built):
    """combined-export derives each column key with re.search(r"ckpt_(\\d+)") and files
    a non-matching name under column ckpt_-1. So the rung has to stay numeric in the
    FILENAME, whatever it is called elsewhere."""
    for rung in _LADDER:
        for target, output in (
            (f"export-sage-ck{rung}", "sage_export_csv"),
            (f"export-bfcl-ck{rung}", "bfcl_export_csv"),
        ):
            uri = _targets(built)[target]["outputs"][output]["uri"]
            found = re.search(r"ckpt_(\d+)-", uri)
            assert found and found.group(1) == rung, uri


def test_the_roll_up_waits_for_every_rung_of_both_eval_kinds(built):
    gates = _targets(built)["export-combined"]["inputs"]
    assert len(gates) == 2 * len(_LADDER)
    bindings = {g["binding"] for g in gates.values()}
    for rung in _LADDER:
        assert f"export-sage-ck{rung}.sage_export_csv" in bindings
        assert f"export-bfcl-ck{rung}.bfcl_export_csv" in bindings


def test_the_roll_up_is_a_single_output_file(built):
    uri = _targets(built)["export-combined"]["outputs"]["combined_csv"]["uri"]
    assert uri.endswith("/combined.csv")
    cfg = _config(built, "export-combined")["combined_export_config"]
    assert cfg["output_csv"].endswith("/combined.csv")
    assert cfg["sage_input_dir"] and cfg["bfcl_input_dir"]


def test_a_trimmed_suite_still_produces_the_roll_up(tmp_path):
    """The exporters for an unselected eval kind are dropped rather than left gating on
    targets that were never emitted, and combined-export is told which side is absent.
    """
    built = _render(tmp_path, EVAL_CATEGORIES="bfcl")
    assert not [t for t in _targets(built) if t.startswith("export-sage-ck")]
    cfg = _config(built, "export-combined")["combined_export_config"]
    assert cfg["sage_input_dir"] == ""
    assert cfg["bfcl_input_dir"]
    assert len(_targets(built)["export-combined"]["inputs"]) == len(_LADDER)

    built = _render(tmp_path, EVAL_CATEGORIES="math")
    assert not [t for t in _targets(built) if t.startswith("export-bfcl-ck")]
    cfg = _config(built, "export-combined")["combined_export_config"]
    assert cfg["bfcl_input_dir"] == ""
    assert cfg["sage_input_dir"]


def test_every_category_is_selectable(tmp_path):
    for category in _CATEGORIES:
        built = _render(tmp_path, EVAL_CATEGORIES=category)
        expected = [
            e["name"] for e in _params()["EVAL_CATALOG"] if e["category"] == category
        ]
        for name in expected:
            assert f"eval-{name}-ck1000" in _targets(built)
        if category == "bfcl":
            assert "eval-bfcl-ck1000" in _targets(built)
        else:
            assert "eval-bfcl-ck1000" not in _targets(built)


# ─── The instruments do not act ───────────────────────────────────────────────


def test_gen_smoke_reports_without_gating(built):
    """distill-stage1-v2's gen-smoke fails the target when the final rung is
    degenerate, because there it protects a downstream consumer from picking up a
    collapsed model. This recipe has no such consumer: it measures a whole epoch,
    every rung is priced by full-eval regardless, and failing here would only remove
    evidence. It still measures and still writes the table: that is the deliverable."""
    step = _targets(built)["gen-smoke"]["steps"][0]
    assert step["step_uri"] == "space://steps/distill/gen-smoke"
    assert step["config"]["gen_smoke_config"]["gate_final_rung"] is False


def test_gen_smoke_hands_the_interpreter_every_rung(built, tmp_path):
    """Neither `bash -n` nor compile() catches the real failure here: a <% %> block tag
    left a blank line after a `\\`, which ends the command, and bash then ran the next
    rung as a program name. That is build bb779f1f, and `bash -n` was green
    throughout. This renders the recipe's own config through the published step
    template and swaps only the interpreter."""
    argv_log = tmp_path / "argv.json"
    stub = tmp_path / "python-stub"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"json.dump(sys.argv[1:], open({str(argv_log)!r}, 'w'))\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script, step_dir = render_run(
        "distill/gen-smoke",
        _config(built, "gen-smoke"),
        gen_smoke_config={"python": str(stub), "output_dir": str(tmp_path / "gs")},
    )

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=step_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    argv = json.loads(argv_log.read_text(encoding="utf-8"))
    assert argv[4] == "false"
    rungs = [a.partition(":") for a in argv[5:]]
    assert [r[0] for r in rungs] == _LADDER
    for rung, _, path in rungs:
        assert path.endswith(f"/export-{rung}")


def test_nothing_can_stop_this_run_early():
    """A guard that may stop at an arbitrary step and a ladder that names fixed ones
    cannot both have their way. On build d1acf1c0 the guard stopped at step 77 of 2,000
    and all four export targets failed with `requested checkpoint does not exist`. On a
    run whose whole point is the epoch, that is the one outcome worth ruling out.

    It is now ruled out twice over: the guard is disarmed (the completed CE sweep retired
    it -- the 42% entropy collapse it watched for turned out not to damage the model), and
    even re-armed it may only warn."""
    params = _params()
    assert float(params["ENTROPY_GUARD_DROP_FRAC"]) == 0
    assert params["ENTROPY_GUARD_ACTION"] == "warn"


def test_the_objective_is_unanchored_and_needs_no_patched_trainer():
    """This recipe measures WHERE in an epoch the gains land, and the epoch it measures is
    df8512e0's: pure divergence, lmbda 0, no CE anchor. Keeping it that way is also what
    keeps it runnable on the step's own public trainer pin -- none of the six patch-gated
    fields exist in CustomGOLDConfig at a5d59bc4, so any one of them being set both changes
    the objective under measurement and drags CODE_DIR back to a BlueVela-only /proj path.
    """
    params = _params()
    assert float(params["CE_COEF"]) == 0
    assert params["LOG_STUDENT_ENTROPY"] is False
    assert float(params["ENTROPY_GUARD_DROP_FRAC"]) == 0
    assert float(params["LMBDA"]) == 0
    assert params["CODE_DIR"] == ""


def test_entropy_is_still_measured_without_the_trainers_help():
    """The per-step entropy log went off with the patch dependency; the quantity did not
    go away. distill-eval computes entropy and reverse KL from a checkpoint with no
    trainer involvement, and this recipe evaluates every rung -- which is the resolution
    that actually matters for the question it asks."""
    assert set(_params()["EVAL_METRICS"].split(",")) >= {"entropy", "rkld"}


def test_nothing_downstream_is_gated_on_a_divergence_reading(built):
    """eval-transfer and gen-smoke are leaves. If anything ever binds their outputs,
    a bad number starts blocking the capability measurement that is the deliverable."""
    instruments = {f"eval-transfer-{rung}" for rung in _LADDER} | {"gen-smoke"}
    for name, target in _targets(built).items():
        for key, spec in (target.get("inputs") or {}).items():
            binding = spec.get("binding", "")
            producer = binding.split(".")[0]
            assert producer not in instruments, (name, key, binding)


def test_there_is_no_baseline_eval_target(built):
    """The after-SFT row is already recorded; re-deriving it would spend a GPU to
    reproduce a number we hold. distill-stage1-v2 has eval-transfer-baseline and
    eval-bfcl-baseline; this deliberately does not."""
    assert not [t for t in _targets(built) if t.endswith("-baseline")]
    assert len([t for t in _targets(built) if t.startswith("eval-transfer-")]) == len(
        _LADDER
    )


# ─── The graph as a whole ─────────────────────────────────────────────────────


def test_the_target_count_is_what_the_readme_says(built):
    assert len(_targets(built)) == 285


def test_the_corpus_front_end_is_unchanged_from_v2(tmp_path):
    """sources, align and corpus are v2's, verbatim, because the corpus this trains on
    has to be the same corpus -- a rung here is only comparable to a rung there if the
    data and the tokenizer that measured it are identical."""
    v2 = yaml.safe_load(
        apply_parameters(
            (_V2 / "build.yaml").read_text(encoding="utf-8"),
            [],
            get_params_from_file(str(_V2 / "parameters.yaml")),
            str(tmp_path),
        )
    )
    built = _render(tmp_path)
    v2_params = get_params_from_file(str(_V2 / "parameters.yaml"))

    def normalise(config, params):
        # Two values legitimately differ and neither is part of the corpus definition:
        # RUN_NAME, which namespaces this build's paths, and the log-scrape cadence,
        # which this recipe raises to 120s because its training target runs for 13
        # hours. Everything else must match byte for byte.
        text = yaml.safe_dump(config).replace(params["RUN_NAME"], "RUN")
        return text.replace(
            f"log_retrieval_interval_seconds: {params['LOG_RETRIEVAL_INTERVAL_SECONDS']}",
            "log_retrieval_interval_seconds: CADENCE",
        )

    for name in ("sources", "align", "corpus"):
        assert normalise(_config(built, name), _params()) == normalise(
            v2["granite.build"]["targets"][name]["steps"][0]["config"], v2_params
        ), name


def test_pinning_a_corpus_swaps_bindings_for_direct_reads(tmp_path):
    """Cross-build reuse is intentionally unsupported, so a pinned corpus has to be a
    direct uri input; the ordering edge a binding provided is carried by
    corpus-pin-check instead, so nothing reads a pinned corpus before it is checked."""
    built = _render(tmp_path, CORPUS_DIR="/proj/somewhere/corpus")
    assert "sources" not in _targets(built)
    assert "corpus" not in _targets(built)
    assert "corpus-pin-check" in _targets(built)
    inputs = _targets(built)["train-gold"]["inputs"]
    assert inputs["corpus"]["uri"] == "env:///proj/somewhere/corpus/train.jsonl"
    assert inputs["corpus_pin"]["binding"] == "corpus-pin-check.pin_check"


def test_a_trailing_slash_on_a_pinned_corpus_cannot_reach_an_artifact_uri(tmp_path):
    built = _render(tmp_path, CORPUS_DIR="/proj/somewhere/corpus/")
    uri = _targets(built)["train-gold"]["inputs"]["corpus"]["uri"]
    assert uri == "env:///proj/somewhere/corpus/train.jsonl"


def test_retries_reuse_what_already_succeeded(built):
    """A fault three targets deep must not re-train. With 285 targets and a 13-hour
    training target, re-running the graph from scratch is not a recovery strategy."""
    retries = built["granite.build"]["retries"]
    assert retries["target_reuse_enabled"] is True


def test_resume_and_ib_hca_are_off_by_default_and_reach_the_step(tmp_path):
    step = _config(_render(tmp_path), "train-gold")
    assert step["resume_from_checkpoint_dir"] == ""
    assert step["resume_emit_seeded"] is True
    assert step["gold_config"]["nccl_ib_hca"] == ""

    src = "/proj/x/builds/b/runs/r/checkpoints/distill-350m-ckpt-eval_node2"
    step = _config(
        _render(
            tmp_path,
            RESUME_FROM_CHECKPOINT_DIR=src,
            NCCL_IB_HCA="^=mlx5_1,mlx5_6,mlx5_8",
            RESUME_EMIT_SEEDED="false",
            CKPT_LADDER="2000,2500",
        ),
        "train-gold",
    )
    assert step["resume_from_checkpoint_dir"] == src
    assert step["gold_config"]["nccl_ib_hca"] == "^=mlx5_1,mlx5_6,mlx5_8"
    assert step["resume_emit_seeded"] is False
