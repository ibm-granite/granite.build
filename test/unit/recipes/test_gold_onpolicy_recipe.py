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

"""Unit tests for the gold-onpolicy-smoke recipe.

THE RECIPE HAS NOT RUN. These tests pin the graph wiring, which is where this shape
differs from every other recipe here and where the failures are silent: a target
that never dispatches because a binding was dereferenced with the wrong accessor, a
server that never gets torn down, a URL mangled by the wrong URI scheme. None of
those produce an error — they produce a build that sits, or two allocations held
until someone notices.

What no test can settle is the open question the recipe exists for: whether the
trainer's NCCL weight-sync group can span two LSF allocations. That needs BlueVela.
"""

import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-gold-distillation"
    / "lsf"
    / "gold-onpolicy-smoke"
)

_GOLD_STEP = (
    pathlib.Path(__file__).resolve().parents[3]
    / "configurations"
    / "assets"
    / "environments"
    / "skypilot"
    / "steps"
    / "distill"
    / "gold"
)

# apply_parameters' delimiters: variable_start_string="$${", variable_end_string="}".
_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")


@pytest.fixture
def params():
    return get_params_from_file(str(_RECIPE / "parameters.yaml"))


@pytest.fixture
def rendered(tmp_path, params):
    """apply_parameters writes a side-effect file into the folder it is handed, so
    it gets tmp_path rather than the recipe dir."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))


@pytest.fixture
def targets(rendered):
    return rendered["granite.build"]["targets"]


def _gold(targets):
    return targets["train"]["steps"][0]["config"]["gold_config"]


# ─── Parameter hygiene, as for every recipe here ───────────────────────────────


def test_parameter_sets_match_exactly(params):
    """Rendering catches markers with no value, by raising under StrictUndefined.
    An unused declaration is silent and reads as a knob that does something."""
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    used = set(_MARKER.findall(text))

    assert used - set(params) == set(), "markers with no declared parameter"
    assert set(params) - used == set(), "parameters declared but never used"


def test_no_single_dollar_substitution_markers():
    """`${VAR}` survives rendering verbatim and reaches the cluster as a literal.
    Note this recipe legitimately contains `{{ }}` — gbserver's own Jinja, for the
    binding dereferences — which is why the check targets only `${`."""
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")

    assert "${" not in text.replace("$${", "")


# ─── The graph ─────────────────────────────────────────────────────────────────


def test_three_targets_in_the_expected_roles(targets):
    assert set(targets) == {"vllm-server", "train", "teardown"}
    assert (
        targets["vllm-server"]["steps"][0]["step_uri"]
        == "space://steps/distill/vllm-server"
    )
    assert targets["train"]["steps"][0]["step_uri"] == "space://steps/distill/gold"
    assert (
        targets["teardown"]["steps"][0]["step_uri"] == "space://steps/skypilot-teardown"
    )


def test_the_server_is_a_starting_target(targets):
    """A target with no input bindings dispatches immediately; one with them waits.
    The server must have none, or nothing ever starts. Its one input, the model it
    serves, is a plain env:// uri, which resolves at once."""
    inputs = targets["vllm-server"].get("inputs", {})
    assert all("binding" not in spec for spec in inputs.values()), inputs


def test_service_outputs_use_the_mem_store(targets):
    """mem:// and never env://. env:// runs the value through filesystem-path
    normalisation and mangles http://host:8001 into /http:/host:8001 — the bug
    recipes/granite4-350m/lsf/bcb-server still demonstrates. mem:// keys the URI to
    the producer's verbatim binding["state"].

    Also: no `type`. Nothing is transferred, so a lineage artifact type would claim
    a data flow that does not exist.
    """
    outputs = targets["vllm-server"]["outputs"]
    assert set(outputs) == {"vllm_url", "cluster_name"}
    for name, spec in outputs.items():
        assert spec["uri"].startswith("mem://"), name
        assert "type" not in spec, name


def test_the_trainer_waits_on_the_server_url(targets):
    """This binding is both the ordering primitive and the health gate: the server
    publishes it only after /health answers, and a target does not dispatch until
    every input binding resolves."""
    assert targets["train"]["inputs"]["vllm"]["binding"] == "vllm-server.vllm_url"


def test_the_url_is_dereferenced_as_state_not_path(targets):
    """The single most likely silent break. mem:// bindings carry `state`;
    `.binding.path` would render empty or wrong, and the trainer would launch
    pointed at nothing. Filesystem stores are the ones that use `.path`.

    The URL reaches the gold step as its declared `vllm` input, which the step
    itself dereferences, so the recipe must not also copy it into a config key —
    and the published step must read it as `state`.
    """
    assert "vllm_server_url" not in _gold(targets)

    step_yaml = (_GOLD_STEP / "step.yaml").read_text(encoding="utf-8")
    assert "bindings.vllm.binding.state" in step_yaml
    assert "bindings.vllm.binding.path" not in step_yaml


def test_teardown_gates_on_the_checkpoint_and_the_cluster(targets):
    """`gate` has no special meaning — it is a conventional name for a binding
    consumed only for ordering. The trainer emits checkpoint as its LAST log line,
    which is what puts teardown after training rather than beside it."""
    inputs = targets["teardown"]["inputs"]

    assert inputs["gate"]["binding"] == "train.checkpoint"
    assert inputs["vllm_cluster"]["binding"] == "vllm-server.cluster_name"


def test_teardown_actually_downs_the_server_cluster(targets):
    """Not optional on LSF: the provisioner refuses idle_minutes_to_autostop for
    SSH/HPC clouds, so a SERVICE cluster never autostops and never gets a
    terminal-status cleanup. Without this the allocation is held indefinitely."""
    config = targets["teardown"]["steps"][0]["config"]["teardown_config"]

    assert config["cluster_names"] == ["{{ bindings.vllm_cluster.binding.state }}"]


def test_no_target_is_orphaned(targets):
    """Every target is either a starting target or reachable from one. An orphan
    would sit forever with no error."""
    produced = {
        f"{name}.{out}"
        for name, target in targets.items()
        for out in target.get("outputs", {})
    }
    for name, target in targets.items():
        for input_name, spec in target.get("inputs", {}).items():
            if "binding" in spec:
                assert spec["binding"] in produced, f"{name}.{input_name} is dangling"


# ─── On-policy correctness ─────────────────────────────────────────────────────


def test_the_server_serves_the_student_not_the_teacher(targets, params):
    """The on-policy contract, and a mistake that RUNS: the student generates and
    the teacher scores those generations. Serving the teacher gives a different
    algorithm that completes and reports a loss."""
    served = targets["vllm-server"]["inputs"]["model"]["uri"]

    assert served == "env://" + params["STUDENT_MODEL"]
    assert served == targets["train"]["inputs"]["student"]["uri"]
    assert served != "env://" + params["TEACHER_MODEL"]
    assert (
        "model_path" not in targets["vllm-server"]["steps"][0]["config"]["vllm_config"]
    )


def test_the_run_is_genuinely_on_policy(targets):
    """lmbda > 0 is what makes the student generate at all. The renderer refuses a
    server URL at lmbda 0 — the server would be allocated and never used — so this
    also pins the recipe on the right side of that check."""
    gold = _gold(targets)

    assert gold["lmbda"] > 0
    assert gold["vllm_num_servers"] >= 1, "0 would omit the on-policy config block"
    assert gold["vllm_mode"] == "server"


def test_weights_are_synced_every_step(targets):
    """On-policy means generating from the CURRENT student. At 2 max_steps a sync
    frequency above 1 would mean the weight-sync path — the thing this recipe
    exists to test — never runs at all."""
    gold = _gold(targets)

    assert gold["vllm_sync_frequency"] == 1
    assert gold["max_steps"] >= 2, "a single step would never exercise a sync"


def test_the_server_window_is_not_smaller_than_the_trainer_context(targets):
    """max_model_len below the trainer's max_length means the server refuses prompts
    the trainer happily builds — mid-run, after both allocations are held."""
    server = targets["vllm-server"]["steps"][0]["config"]["vllm_config"]
    gold = _gold(targets)

    assert server["max_model_len"] >= gold["max_length"]


def test_all_trainer_nodes_train(targets):
    """With an external server no node is carved out of this allocation. A recipe
    that also set a nonzero in-allocation split would silently lose a node."""
    compute = targets["train"]["steps"][0]["config"]["compute_config"]

    assert compute["num_nodes"] >= 1
    # The two targets are separately allocated, so the server's node count is its
    # own and must not be folded into the trainer's.
    assert (
        targets["vllm-server"]["steps"][0]["config"]["compute_config"]["num_nodes"] == 1
    )


def test_both_targets_share_one_trainer_checkout_and_image(targets, params):
    """A server built from different code than the trainer expects is a protocol
    mismatch surfacing as a connection or tensor-shape error mid-run.

    Neither target's build.yaml overrides code_config, so both resolve to their
    step-template's default -- the same pinned public-repo clone -- and this
    recipe-level render cannot see that default to compare it directly. What it CAN
    see, and what this asserts, is the one thing the recipe itself controls: both
    targets run the same image.
    """
    server_step = targets["vllm-server"]["steps"][0]["config"]
    train_step = targets["train"]["steps"][0]["config"]

    assert (
        server_step["launcher_config"]["image_id"]
        == train_step["launcher_config"]["image_id"]
    )


# ─── Lineage and types ─────────────────────────────────────────────────────────


def test_three_lineage_inputs_on_the_training_target(targets):
    """Lineage is built from a target's input artifacts. The vllm binding is NOT one
    of them — it carries no artifact and has no type — so the three model/dataset
    inputs must be declared alongside it rather than displaced by it."""
    inputs = targets["train"]["inputs"]

    assert {"teacher", "student", "corpus"} <= set(inputs)
    assert inputs["teacher"]["type"] == "model"
    assert inputs["student"]["type"] == "model"
    assert inputs["corpus"]["type"] == "dataset"
    for name in ("teacher", "student", "corpus"):
        assert inputs[name]["uri"].startswith("env:///"), name
    assert "type" not in inputs["vllm"]


def test_inputs_are_the_only_route_to_the_trainer(targets, params):
    """The gold step reads student, teacher and corpus from its declared inputs.
    A config key carrying the same path again could only disagree with them."""
    inputs = targets["train"]["inputs"]
    gold = _gold(targets)

    assert inputs["student"]["uri"] == "env://" + params["STUDENT_MODEL"]
    assert inputs["teacher"]["uri"] == "env://" + params["TEACHER_MODEL"]
    assert inputs["corpus"]["uri"] == "env://" + params["TRAINING_DATASET"]
    for retired in ("model_name_or_path", "teacher_model_name_or_path", "dataset_name"):
        assert retired not in gold, retired


def test_response_template_transports_its_newline_as_an_escape(targets):
    """The trailing newline crosses the wire as a literal two-character escape.

    A real newline here does not survive: gbserver's config fill runs every string
    through Jinja and strips one trailing newline, so the step would receive a
    template with no line boundary and mask loss from the wrong token, silently.
    distill-gold's renderer decodes the escape in the container. See
    ../gold-sweep-100/parameters.yaml for the mechanism and test_distill_gold.py
    for the decode."""
    template = _gold(targets)["response_template"]

    assert template == "<|im_start|>assistant\\n"
    assert not template.endswith(
        "\n"
    ), "a real newline here is stripped by the config fill before the step sees it"


def test_the_server_cannot_outlive_a_failed_trainer(targets):
    """The server must bound its own allocation, because teardown cannot.

    `teardown` is gated on `train.checkpoint`, which a crashed trainer never
    emits, and gbserver's schema has no on-failure semantics — so on any trainer
    failure the vLLM node is stranded. LSF SERVICE clusters never autostop
    either. Build d77546a9 held 8 H100s that way until they were downed by hand.

    A cap on the SERVICE is the only backstop available, so this recipe sets one.
    It must comfortably exceed the health timeout, or a slow model load would trip
    the cap before the server ever became useful.
    """
    vllm = targets["vllm-server"]["steps"][0]["config"]["vllm_config"]

    cap = vllm["max_lifetime_seconds"]
    assert isinstance(cap, int) and cap > 0, f"no lifetime cap on the server: {cap!r}"
    assert cap > vllm["health_timeout_seconds"], (
        "the cap must outlast the health timeout, or a slow load reaps the server "
        "before it is ever healthy"
    )


def test_numeric_parameters_keep_their_types(targets):
    gold = _gold(targets)
    server = targets["vllm-server"]["steps"][0]["config"]["vllm_config"]

    for key in ("max_steps", "max_length", "save_steps", "vllm_sync_frequency"):
        assert isinstance(gold[key], int), key
    assert isinstance(gold["lmbda"], float)
    assert isinstance(gold["use_liger_fused_jsd"], bool)
    for key in ("port", "max_model_len", "health_timeout_seconds"):
        assert isinstance(server[key], int), key


def test_nccl_diagnostics_are_on(targets):
    """The one recipe where this is not optional: a cross-allocation group that
    cannot form HANGS rather than erroring, so INIT,NET plus a bounded timeout is
    what turns two held allocations into a diagnostic."""
    gold = _gold(targets)

    assert gold["nccl_debug"] == "INFO"
    assert "INIT" in gold["nccl_debug_subsys"] and "NET" in gold["nccl_debug_subsys"]
    assert gold["nccl_enable_monitoring"] is True
    assert 0 < gold["nccl_timeout_ms"] <= 900000, "a hang must report in minutes"
