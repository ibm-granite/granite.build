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

"""Unit tests for the distill-onpolicy-v2 recipe.

This recipe has not run. What it guards against, therefore, is not regression but the
set of on-policy mistakes that are already documented as having happened once, every one
of which produces a run that COMPLETES and reports a plausible loss:

* serving the teacher instead of the student -- a different algorithm, not a crash;
* allocating a server and then generating locally, because --use_vllm defaulted False
  (build d77546a9, which logged not one request against an idle 8-GPU server);
* a server whose address never reaches the trainer, which is what the in-allocation
  split does: it passes no --vllm_server_host;
* an immediate-EOS collapse, where the teacher scores empty completions as rollouts;
* a stranded SERVICE allocation, because LSF clusters never autostop and teardown is
  gated on an artifact a crashed trainer never emits.

It also asserts the recipe is stage 1 v2 plus on-policy and nothing else, because the
question it exists to answer is where the sequences come from -- not what the objective
is, which stage 1 v2 already changed.
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

_LSF = pathlib.Path(__file__).resolve().parents[3] / "recipes" / "granite4-350m" / "lsf"
_RECIPE = _LSF / "distill-onpolicy-v2"
_OFFPOLICY = _LSF / "distill-stage1-v2"

_PLAIN_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")
_LOOP_VARS = {"N"}


def _params(path=_RECIPE, **overrides):
    params = get_params_from_file(str(path / "parameters.yaml"))
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


@pytest.fixture(name="on")
def fixture_on(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False)


def test_every_marker_has_a_parameter():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert (set(_PLAIN_MARKER.findall(contents)) - _LOOP_VARS) - set(_params()) == set()


def test_every_parameter_is_referenced():
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    # Both are reached through an expression -- CKPT_LADDER via .split(...), CORPUS_DIR
    # via .rstrip('/') -- which the plain marker pattern does not match.
    referenced = set(_PLAIN_MARKER.findall(contents)) | {"CKPT_LADDER", "CORPUS_DIR"}
    assert set(_params()) - referenced - {"INCLUDE_SFT"} == set()


def test_the_graph_has_a_server_a_trainer_and_a_teardown(on):
    names = list(_targets(on))
    assert names.index("vllm-server") < names.index("train-gold")
    assert "teardown" in names


class TestTheServerServesTheStudent:
    def test_the_served_model_is_a_binding_not_a_parameter(self, on):
        """The copy being served and the copy being trained have to be the same
        weights. A binding makes that structural; two parameters agreeing is a
        convention that survives until someone edits one of them."""
        cfg = _config(on, "vllm-server")["vllm_config"]
        # The step reads its declared `model` input; the retired config key is gone.
        assert "model_path" not in cfg
        assert _targets(on)["vllm-server"]["inputs"]["model"] == {
            "binding": "align.retagged_student"
        }

    def test_it_is_not_the_teacher(self, on):
        """The silent trap: on-policy has the STUDENT generate and the teacher score
        those generations. Serving the teacher is a different algorithm that runs to
        completion and reports a plausible loss."""
        model = _targets(on)["vllm-server"]["inputs"]["model"]
        assert _params()["TEACHER_MODEL"] not in str(model)
        assert "uri" not in model

    def test_the_server_window_covers_the_trainers_prompts(self, on):
        """A max_model_len below the trainer's max_length means a prompt the trainer
        accepts is one the server refuses, mid-run."""
        server = _config(on, "vllm-server")["vllm_config"]
        trainer = _config(on, "train-gold")["gold_config"]
        assert int(server["max_model_len"]) >= int(trainer["max_length"])

    def test_the_url_travels_over_mem_not_env(self, on):
        """env:// runs the payload through filesystem-path normalisation and mangles
        http://host:8001 into /http:/host:8001."""
        outputs = _targets(on)["vllm-server"]["outputs"]
        assert outputs["vllm_url"]["uri"].startswith("mem://")
        assert "type" not in outputs["vllm_url"]

    def test_the_trainer_gates_on_the_servers_health(self, on):
        """The binding resolves only once the server answers /health, so it is the
        ordering primitive as well as the address."""
        assert _targets(on)["train-gold"]["inputs"]["vllm"]["binding"] == (
            "vllm-server.vllm_url"
        )


class TestTheTrainerActuallyUsesTheServer:
    def test_the_external_url_is_wired(self, on):
        """Its presence is what selects the external-server path over the
        in-allocation split, which passes no --vllm_server_host at all."""
        gold = _config(on, "train-gold")["gold_config"]
        # The URL is the step's optional mem:// `vllm` input, read by the step itself;
        # the retired config key must not come back.
        assert _targets(on)["train-gold"]["inputs"]["vllm"] == {
            "binding": "vllm-server.vllm_url"
        }
        assert "vllm_server_url" not in gold
        assert gold["vllm_mode"] == "server"

    def test_one_external_server_is_declared(self, on):
        """At 0 the renderer refuses the URL outright: the on-policy keys would not be
        emitted and the run would train off-policy while a server sat idle."""
        assert int(_config(on, "train-gold")["gold_config"]["vllm_num_servers"]) == 1

    def test_the_student_actually_generates(self, on):
        """lmbda 0 with a server allocated is the d77546a9 shape: a held allocation
        that is never asked for anything. The renderer refuses it."""
        assert float(_config(on, "train-gold")["gold_config"]["lmbda"]) > 0

    def test_weights_are_synced_every_step(self, on):
        """A stale served copy generates from a model that no longer exists, which is
        off-policy training paying on-policy's costs."""
        assert int(_config(on, "train-gold")["gold_config"]["vllm_sync_frequency"]) >= 1

    def test_the_lmbda_schedule_is_valid_and_paired(self, on):
        gold = _config(on, "train-gold")["gold_config"]
        assert gold["lmbda_schedule"] in ("constant", "linear")
        if gold["lmbda_schedule"] == "linear":
            assert 0.0 <= float(gold["lmbda_init"]) <= 1.0
            assert float(gold["lmbda_init"]) < float(gold["lmbda"]), "a ramp must rise"


class TestTheGenerationFloor:
    """There is none, and 0 is the state of the PINNED CODE rather than a preference.

    The floor has two halves and the step's pin (a5d59bc4) carries neither, checked against
    the public repo on 2026-09-28. The trainer half needs
    CustomGOLDConfig.min_completion_length, which is absent -- and unlike the CE anchor keys,
    ce_anchor_and_entropy_guard.diff does not add it, so no patched checkout carries it
    either. The server half, which is where the sampling decision is actually taken, needs
    run_vllm_serve.py to read GOLD_MIN_TOKENS and patch SamplingParams.min_tokens; that file
    at the pin is 108 lines and mentions none of the three.

    So a non-zero value buys two pre-allocation refusals -- render_gold_config's field check
    and vllm-server's own GOLD_MIN_TOKENS grep -- rather than a floor.

    WHAT IS UNMITIGATED: an immediate EOS. An empty completion is a rollout the teacher scores
    as if it were one, and the run becomes an expensive way to score nothing. Nothing in the
    pinned code reports it, so gen-smoke on the first rung is the read that has to replace it.
    """

    def test_neither_side_asks_for_a_floor_the_pin_cannot_deliver(self, on):
        """Asserted as a value rather than tolerated either way: raising it is only correct
        once both halves land upstream and the pin is bumped, and the cost of getting that
        wrong is an arm that trains on empty completions with nothing saying so."""
        assert int(_params()["MIN_COMPLETION_LENGTH"]) == 0
        assert (
            int(_config(on, "train-gold")["gold_config"]["min_completion_length"]) == 0
        )
        assert (
            int(_config(on, "vllm-server")["vllm_config"]["min_completion_length"]) == 0
        )

    def test_the_two_sides_agree(self, on):
        assert int(
            _config(on, "train-gold")["gold_config"]["min_completion_length"]
        ) == (int(_config(on, "vllm-server")["vllm_config"]["min_completion_length"]))

    def test_the_step_exports_it_to_the_server_process(self):
        """The recipe key is inert unless the step turns it into GOLD_MIN_TOKENS. Kept while
        the floor is 0 because this is the half that has to survive for a future pin bump to
        be a one-line change -- and because it is what makes vllm-server's refusal loud
        rather than silent if someone raises the floor too early."""
        step = (
            _RECIPE.parents[3]
            / "configurations"
            / "assets"
            / "environments"
            / "skypilot"
            / "steps"
            / "distill"
            / "vllm-server"
            / "step.yaml"
        ).read_text(encoding="utf-8")
        assert "GOLD_MIN_TOKENS" in step
        assert "min_completion_length" in step


class TestTheAllocationIsReleased:
    def test_teardown_exists_and_is_gated_on_training_finishing(self, on):
        """LSF SERVICE clusters never auto-stop and never get a terminal-status
        cleanup, so without this the server's GPUs are held until someone runs
        `sky down` by hand."""
        inputs = _targets(on)["teardown"]["inputs"]
        assert inputs["trained"]["binding"] == "train-gold.checkpoint"
        assert inputs["vllm_cluster"]["binding"] == "vllm-server.cluster_name"

    def test_teardown_passes_a_list(self, on):
        names = _config(on, "teardown")["teardown_config"]["cluster_names"]
        assert isinstance(names, list) and len(names) == 1

    def test_the_server_carries_its_own_ceiling(self, on):
        """Teardown is gated on an artifact a CRASHED trainer never emits, and
        gbserver has no on-failure semantics. Build d77546a9 stranded 8 H100s that
        way. This is the backstop, and it must outlast the run it guards."""
        cfg = _config(on, "vllm-server")["vllm_config"]
        lifetime = int(cfg["max_lifetime_seconds"])
        assert lifetime > int(cfg["health_timeout_seconds"])
        # ~4 s/it measured on this pair, so a 2,000-step run needs ~2.2 h plus startup.
        steps = int(_config(on, "train-gold")["gold_config"]["max_steps"])
        assert lifetime >= steps * 4, "the ceiling would kill the server mid-run"


class TestItIsStage1V2PlusOnPolicy:
    """One variable. Stage 1 v2 already changed the objective; this changes where the
    sequences come from, and nothing else — otherwise an on-policy result is not
    comparable to the off-policy arm it is supposed to be judged against."""

    @pytest.mark.parametrize(
        "key",
        [
            "CE_COEF",
            "LOG_STUDENT_ENTROPY",
            "ENTROPY_GUARD_DROP_FRAC",
            "ENTROPY_GUARD_BASELINE_STEPS",
            "ENTROPY_GUARD_PATIENCE",
            "ENTROPY_GUARD_ACTION",
            "BETA",
            "TEMPERATURE",
            "GOLD_MAX_STEPS",
            "CKPT_LADDER",
            "GOLD_SAVE_STEPS",
            "GOLD_SAVE_TOTAL_LIMIT",
            "MAX_LENGTH",
            "TARGET_ROWS",
            "GOLD_LEARNING_RATE",
            "CODE_DIR",
            "CODE_EXPECT_REF",
            "NCCL_DEBUG",
            "CORPUS_DIR",
        ],
    )
    def test_it_matches_the_off_policy_arm(self, key):
        assert _params()[key] == _params(_OFFPOLICY)[key], key

    def test_the_trainer_and_the_server_read_the_same_revision(self, on):
        """On-policy is the arm where this can go wrong quietly. train-gold runs the trainer
        and vllm-server runs run_vllm_serve out of the SAME package, so two different
        checkouts would pair a patched trainer with an unpatched sampler -- and the symptom is
        not a crash, it is a floor that never applies and completions the teacher scores as
        rollouts. Both override the shared steps pin (see the off-policy arm's test for why
        that is allowed at all); this asserts they agree."""
        overrides = {}
        for name, target in _targets(on).items():
            for step in target["steps"]:
                if "code_config" in step.get("config", {}):
                    overrides[name] = step["config"]["code_config"]
        assert set(overrides) == {"train-gold", "vllm-server"}, overrides
        distinct = {tuple(sorted(c.items())) for c in overrides.values()}
        assert len(distinct) == 1, f"trainer and server disagree: {overrides}"

    def test_a_stopping_guard_and_a_fixed_ladder_cannot_both_be_asked_for(self, on):
        """Build d1acf1c0 hit this in the off-policy arm -- guard tripped at step 77 of
        2,000, and all four fixed export rungs named checkpoints the trainer never
        wrote. This recipe carries the same ladder, so it carries the same invariant."""
        gold = _config(on, "train-gold")["gold_config"]
        if float(gold["entropy_guard_drop_frac"]) <= 0:
            return
        assert gold["entropy_guard_action"] == "warn", (
            "the guard may stop at an arbitrary step while CKPT_LADDER "
            f"({_params()['CKPT_LADDER']}) demands specific ones"
        )

    def test_the_anchor_is_off_here_too(self, on):
        """The anchor's stated rationale was STRONGEST in this regime -- on-policy
        training on a student's own output is where a collapse could feed itself -- and
        it is still off, because the evidence that retired it is the only evidence there
        is. The CE sweep ran off-policy, so it says nothing about on-policy directly;
        what it says is that non-zero CE costs capability in ordered amounts and that the
        collapse it prevented damaged nothing. This arm starts where the evidence is.

        If the on-policy entropy curve argues otherwise, CE_COEF is one --param plus a
        CODE_DIR at a patched checkout -- and it must match the off-policy arm it is
        compared against, which test_it_matches_the_off_policy_arm already enforces."""
        gold = _config(on, "train-gold")["gold_config"]
        assert float(gold.get("ce_coef", 0.0)) == 0.0
        assert gold.get("log_student_entropy", False) is False
        assert float(gold.get("entropy_guard_drop_frac", 0.0)) == 0.0

    def test_the_two_code_config_overrides_are_the_steps_own_pin(self, on):
        """train-gold and vllm-server both restate code_config, and with the anchor
        retired both restate the step's PUBLIC pin rather than a /proj checkout. That is
        what makes this recipe runnable off BlueVela -- and it is the pairing that goes
        wrong quietly, since a patched trainer with an unpatched sampler is a floor that
        never applies rather than a crash."""
        assert _params()["CODE_DIR"] == ""
        for name in ("train-gold", "vllm-server"):
            assert _config(on, name)["code_config"]["code_dir"] == ""

    def test_the_effective_batch_is_unchanged_by_the_server_node(self):
        """The server has its own allocation, so GOLD_NUM_NODES still counts trainers
        only and the product must still be 96 — otherwise the on-policy arm is also a
        different optimization."""
        p = _params()
        product = (
            p["GOLD_PER_DEVICE_TRAIN_BATCH_SIZE"]
            * p["GOLD_GRADIENT_ACCUMULATION_STEPS"]
            * p["GOLD_NUM_NODES"]
            * p["GOLD_NUM_GPUS"]
        )
        assert product == 96


# ─── The generation smoke test ─────────────────────────────────────────────────


class TestGenSmoke:
    """The shared VALUES are covered by parity with the off-policy arm above. What is
    NOT shared is the rendered text, and this target is where it broke: gen-smoke
    assembles one argument per rung through a Jinja loop, and this file previously had
    no gen-smoke coverage at all, which is how build bb779f1f's failure shipped in two
    recipes at once."""

    def test_it_runs_the_gen_smoke_step(self, on):
        step = _targets(on)["gen-smoke"]["steps"][0]
        assert step["step_uri"] == "space://steps/distill/gen-smoke"

    def test_the_interpreter_actually_receives_every_rung(self, tmp_path):
        """Executes the assembly instead of parsing it. `bash -n` cannot see this
        failure: a <% %> block tag left a blank line after a `\\`, which ENDS the
        command, and bash then ran the next rung as a program name -- exit 127 with
        `500:/proj/.../export-500: No such file or directory`. This renders the
        recipe's own config through the published step template and swaps only the
        interpreter."""
        rendered = _render(tmp_path, INCLUDE_SFT=False, WORKDIR_ROOT=str(tmp_path))
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
            _config(rendered, "gen-smoke"),
            gen_smoke_config={"python": str(stub)},
        )

        result = subprocess.run(
            ["bash", "-c", script], cwd=step_dir, text=True, capture_output=True
        )

        assert result.returncode == 0, result.stderr
        argv = json.loads(argv_log.read_text(encoding="utf-8"))
        rungs = _params()["CKPT_LADDER"].split(",")
        assert argv[0] == "./src/gen_smoke.py", f"interpreter got {argv}"
        assert len(argv) == 5 + len(rungs), f"interpreter got {argv}"
        assert argv[4] == "true", "the on-policy arm gates on its final rung"
        for rung, got in zip(rungs, argv[5:]):
            step, _, path = got.partition(":")
            assert step == rung, f"expected rung {rung}, got {got}"
            assert path.endswith(f"/export-{rung}"), got

    def test_no_rung_is_appended_through_a_line_continuation(self):
        """The shape that broke, guarded at the source."""
        lines = (_RECIPE / "build.yaml").read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines[:-1]):
            if line.rstrip().endswith("\\") and lines[i + 1].lstrip().startswith("<%"):
                raise AssertionError(
                    f"{_RECIPE.name}/build.yaml:{i + 1}: continuation followed by a "
                    f"block tag renders to a blank line and ends the command:\n"
                    f"  {line}\n  {lines[i + 1]}"
                )


def test_the_student_default_cannot_silently_train_the_wrong_model():
    """On-policy continues from a good off-policy checkpoint. A plausible-looking
    default — the base retagged student, say — would run, report a loss, and answer a
    question nobody asked. An invalid one fails at align in minutes."""
    student = _params()["STUDENT_MODEL"]
    assert not student.startswith("/"), student
    assert "SET-ME" in student.upper()
    assert student != _params(_OFFPOLICY)["STUDENT_MODEL"]


# ─── Reusing an existing corpus ────────────────────────────────────────────────


_PIN = "/proj/granite-build/g4os/distill/distill-350m-s1v2-ce040/c20ed3c0/corpus"


def _corpus_inputs(rendered):
    out = {}
    for name, target in _targets(rendered).items():
        spec = (target.get("inputs") or {}).get("corpus")
        if spec is not None:
            out[name] = spec
    return out


@pytest.fixture(name="pinned")
def fixture_pinned(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False, CORPUS_DIR=_PIN)


class TestTheCorpusPin:
    """The off-policy arm's tests cover what the pin MEANS. What this file has to
    cover is that this recipe carries the same wiring -- it has its own copy of all
    four consumers, and an on-policy arm that rebuilt the corpus while the off-policy
    arm reused one would not be comparable to it."""

    def test_the_corpus_is_built_in_the_build_by_default(self, on):
        assert "sources" in _targets(on)
        assert "corpus" in _targets(on)
        for name, spec in _corpus_inputs(on).items():
            assert spec.get("binding") == "corpus.corpus", name

    def test_a_pin_removes_the_targets_that_would_rebuild_it(self, pinned):
        assert "sources" not in _targets(pinned)
        assert "corpus" not in _targets(pinned)
        assert "corpus-pin-check" in _targets(pinned)

    def test_a_pin_is_read_directly_and_gated_on_the_check(self, pinned):
        consumers = _corpus_inputs(pinned)
        assert consumers
        for name, spec in consumers.items():
            assert "binding" not in spec, name
            assert spec["uri"] == f"env://{_PIN}/train.jsonl", name
            bindings = {
                s.get("binding") for s in _targets(pinned)[name]["inputs"].values()
            }
            assert "corpus-pin-check.pin_check" in bindings, name

    def test_a_pin_is_never_re_registered_as_an_output(self, pinned):
        for name, target in _targets(pinned).items():
            for out_name, spec in (target.get("outputs") or {}).items():
                assert _PIN not in str(spec.get("uri", "")), f"{name}.{out_name}"

    def test_the_transfer_evals_read_the_pinned_eval_split(self, pinned):
        evals = [n for n in _targets(pinned) if n.startswith("eval-transfer-")]
        assert evals
        for name in evals:
            assert _config(pinned, name)["eval_config"]["corpus"] == (
                f"{_PIN}/eval.jsonl"
            )

    def test_the_server_is_not_gated_on_the_corpus(self, pinned):
        """vllm-server serves the student and never reads the corpus. If pinning had
        given it a corpus input it would also have acquired the gate's ordering edge,
        and the server would wait on a check it has nothing to do with."""
        assert "corpus" not in (_targets(pinned)["vllm-server"].get("inputs") or {})

    def test_the_pin_check_matches_the_off_policy_arm(self, pinned, tmp_path):
        """Rather than duplicate the off-policy arm's behavioural tests of the check,
        assert this arm runs the same step with the same configuration. Those tests
        then cover this recipe too, and a change applied to one arm cannot silently
        miss the other."""
        other = yaml.safe_load(
            apply_parameters(
                (_OFFPOLICY / "build.yaml").read_text(encoding="utf-8"),
                [],
                _params(_OFFPOLICY, INCLUDE_SFT=False, CORPUS_DIR=_PIN),
                str(tmp_path),
            )
        )
        mine = _targets(pinned)["corpus-pin-check"]["steps"][0]
        theirs = other["granite.build"]["targets"]["corpus-pin-check"]["steps"][0]
        assert mine["step_uri"] == theirs["step_uri"]
        assert mine["config"]["pin_check_config"].keys() == (
            theirs["config"]["pin_check_config"].keys()
        )
        assert (
            mine["config"]["pin_check_config"]["corpus_dir"]
            == theirs["config"]["pin_check_config"]["corpus_dir"]
        )
        # The tokenizer is the step's `tokenizer` input, not config.
        assert (
            _targets(pinned)["corpus-pin-check"]["inputs"]["tokenizer"]
            == other["granite.build"]["targets"]["corpus-pin-check"]["inputs"][
                "tokenizer"
            ]
        )
