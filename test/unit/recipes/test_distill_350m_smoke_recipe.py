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

"""Unit tests for the distill-smoke recipe (granite4-350m).

This is distill-pipeline-smoke's graph retargeted to a new pair, and retargeting a
distillation recipe is exactly where the silent failures live. Three of them are
specific to THIS pair and none of them would fail loudly on a cluster:

* **require_chatml and the chat template must agree.** False with a ChatML template
  aligns cleanly and then has the teacher score a prompt format it has never seen.
* **The response template is a fact about the chat template**, not a style. The
  reference family's ``<|im_start|>assistant\\n`` matches nothing against a granite
  4.x template, which leaves every label at ignore_index — a run that trains on
  nothing while producing a loss curve and checkpoints.
* **The student must be the SFT checkpoint**, because its recorded 27-benchmark row
  is this work's control. A base student here would leave the comparison with no
  control at all.

So the assertions below are mostly about agreement between values that a human
would have to keep in step by hand.
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
    / "granite4-350m"
    / "lsf"
    / "distill-smoke"
)

_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")

_TARGETS_WITHOUT_SFT = [
    "align",
    "corpus",
    "train-gold",
    "export",
    "eval-transfer-baseline",
    "eval-transfer",
    "eval-bfcl",
]

# The granite 4.x assistant turn header, in full. Nothing follows it in the
# template — which is why, unlike the ChatML family, there is no trailing newline
# to carry as an escape.
_GRANITE_MARKER = "<|start_of_role|>assistant<|end_of_role|>"


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


@pytest.fixture(name="off")
def fixture_off(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False)


@pytest.fixture(name="on")
def fixture_on(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=True)


# ─── The parameter surface ─────────────────────────────────────────────────────


def test_every_marker_has_a_parameter():
    """apply_parameters uses StrictUndefined, so a marker with no parameter fails
    the render rather than rendering empty."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert set(_MARKER.findall(contents)) - set(_params()) == set()


def test_every_parameter_is_referenced():
    """A parameter nothing reads is a value somebody will change expecting an
    effect."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    unread = set(_params()) - set(_MARKER.findall(contents)) - {"INCLUDE_SFT"}
    assert unread == set()


def test_the_recipe_renders_and_declares_the_smoke_graph(off):
    assert list(_targets(off)) == _TARGETS_WITHOUT_SFT


# ─── The pair ──────────────────────────────────────────────────────────────────


class TestThePair:
    def test_the_teacher_is_a_pinned_copy_not_the_model_directory(self, off):
        """granite-4.1-3b's own directory declares tokenizer_class GPT2Tokenizer over
        a trained Sequence[Split,ByteLevel] pre_tokenizer, so AutoTokenizer
        mis-segments it. align pins the STUDENT only; the teacher needs its own
        pinned copy, and pointing this at the raw model directory reintroduces the
        exact bug align exists to remove."""
        teacher = _params()["TEACHER_MODEL_URI"]
        assert "granite-4.1-3b" in teacher
        assert teacher.endswith(
            "-pinned"
        ), "TEACHER_MODEL_URI must be the pinned-tokenizer copy, not the model dir"

    def test_the_student_is_the_sft_checkpoint_not_the_base_model(self, off):
        """The recorded after-SFT eval row is this work's control. Initialising from
        base, or from a different epoch, silently replaces a measured control with an
        unmeasured one."""
        student = _params()["STUDENT_MODEL_URI"]
        assert student.endswith("/epoch_hf_2"), student
        assert "-base" not in student

    def test_the_teacher_and_student_reach_every_step_that_needs_them(self, off):
        """One teacher, named in align, in the trainer and in both evals. A pipeline
        that aligns against one teacher and scores against another reports a
        divergence from a model it never trained towards.

        Every consumer reads the RESOLVED BINDING PATH now, not the raw
        TEACHER_MODEL_URI parameter — that's the whole point of routing it through
        an input binding rather than a bare path, so the same assertion holds
        regardless of which uri scheme TEACHER_MODEL_URI names.
        """
        align_teacher = _config(off, "align")["align_config"]["teacher_model"]
        assert align_teacher == "{{ bindings.teacher_model.binding.path }}"
        gold = _config(off, "train-gold")["gold_config"]
        assert (
            gold["teacher_model_name_or_path"]
            == "{{ bindings.teacher_model.binding.path }}"
        )
        for target in ("eval-transfer", "eval-transfer-baseline"):
            assert (
                _config(off, target)["eval_config"]["teacher_model"]
                == "{{ bindings.teacher_model.binding.path }}"
            )

    def test_lineage_records_the_pair_as_input_artifacts(self, off):
        """Lineage is built from a target's input artifacts, not from step config: a
        path handed only to align_config would run and record nothing."""
        align_inputs = _targets(off)["align"]["inputs"]
        assert set(align_inputs) == {
            "teacher_model",
            "student_model",
            "chat_template",
        }
        for name, spec in align_inputs.items():
            assert spec["type"] == ("dataset" if name == "chat_template" else "model")
            # The URI's SCHEME is a parameter (TEACHER_MODEL_URI / STUDENT_MODEL_URI /
            # CHAT_TEMPLATE_URI), not hardcoded — this recipe's default happens to be
            # env://, but the point of the *_URI convention is that hf:// or s3://
            # work identically. See test_source_uris_can_be_overridden_with_a_different_scheme.
            assert spec["uri"].startswith("env://")

    def test_source_uris_can_be_overridden_with_a_different_scheme(self, tmp_path):
        """The *_URI parameters carry a full uri, not a bare path, so a recipe user
        can point align at a Hub model or an S3 object instead of /proj without any
        change to build.yaml — only --param."""
        rendered = _render(
            tmp_path,
            TEACHER_MODEL_URI="hf:///ibm-granite/granite-4.1-3b",
            DATASET_URI="s3://my-bucket/corpora/smoke.jsonl",
        )
        align_inputs = _targets(rendered)["align"]["inputs"]
        assert (
            align_inputs["teacher_model"]["uri"] == "hf:///ibm-granite/granite-4.1-3b"
        )
        corpus_inputs = _targets(rendered)["corpus"]["inputs"]
        assert (
            corpus_inputs["source_dataset"]["uri"]
            == "s3://my-bucket/corpora/smoke.jsonl"
        )


# ─── The markup family: the coupled invariant ──────────────────────────────────


class TestMarkupFamily:
    def test_chatml_is_not_required_of_this_pair(self, off):
        """No granite 4.x model carries <|im_start|>, so align's stage 1 of 4 aborts
        on a granite-4.1 teacher under the upstream default."""
        assert _config(off, "align")["align_config"]["require_chatml"] is False

    def test_not_requiring_chatml_forces_a_granite_native_template(self, off):
        """THE coupled invariant. require_chatml false with a ChatML template is the
        one combination that fails silently: alignment succeeds and the teacher then
        scores a format it has never seen. The step cannot catch it — the template is
        a path, and its contents are never compared against the vocabulary.

        align_config.chat_template is now the RESOLVED BINDING PATH (see
        TestThePair.test_the_teacher_and_student_reach_every_step_that_needs_them for
        why), so the actual template URI is asserted from CHAT_TEMPLATE_URI instead.
        """
        align = _config(off, "align")["align_config"]
        template_uri = _params()["CHAT_TEMPLATE_URI"]
        if align["require_chatml"] is False:
            assert "chatml" not in template_uri.lower()
            assert "role" in template_uri

    def test_the_chat_template_is_absolute(self, off):
        """run-align.sh only prefixes the steps' code_dir for a RELATIVE value. A
        relative granite-native path would resolve inside the pinned upstream
        checkout, which does not contain one.

        CHAT_TEMPLATE_URI carries the env:// scheme prefix; the path after it must
        still be absolute for the same reason the old bare CHAT_TEMPLATE had to be.
        """
        template_uri = _params()["CHAT_TEMPLATE_URI"]
        assert template_uri.startswith("env:///")

    def test_the_response_template_is_the_granite_role_marker(self, off):
        """A fact about the installed chat template, not a style. The reference
        family's marker matches nothing here, and render_gold_config.py's own default
        is the ChatML one — so every target must state this explicitly or train on
        nothing while reporting a loss."""
        assert (
            _config(off, "train-gold")["gold_config"]["response_template"]
            == _GRANITE_MARKER
        )

    def test_the_marker_needs_no_escape_and_no_strip_protection(self, off):
        """The ChatML recipes carry a doubled backslash because their marker ends in
        a line boundary and gbserver's Jinja fill strips exactly one real trailing
        newline per value (build d8470f14). granite 4.x puts nothing after
        <|end_of_role|>, so there is nothing to protect — and asserting that keeps
        someone from reintroducing the escape as cargo cult."""
        template = _config(off, "train-gold")["gold_config"]["response_template"]
        assert "\\" not in template  # nothing for printf %b to reinterpret
        assert not template.endswith(("\n", " "))  # nothing for the fill to strip

    def test_both_trainers_mask_on_the_same_boundary(self, on):
        """If the SFT arm is ever switched on as a control, it has to differ from the
        treatment in the OBJECTIVE only. A different response template is a different
        labelled span."""
        assert (
            _config(on, "train-sft")["sft_config"]["response_template"]
            == _config(on, "train-gold")["gold_config"]["response_template"]
        )


# ─── The objective ─────────────────────────────────────────────────────────────


class TestObjective:
    def test_it_is_off_policy_and_allocates_no_server(self, off):
        """lmbda 0 means the student never generates. render_gold_config.py refuses a
        server URL at lmbda 0, and a non-zero vllm_num_servers here would carve a
        node out of the allocation to serve a model nothing asks for."""
        gold = _config(off, "train-gold")["gold_config"]
        assert float(gold["lmbda"]) == 0.0
        assert int(gold["vllm_num_servers"]) == 0

    def test_beta_is_a_genuine_mixture(self, off):
        """The trainer short-circuits 0.0 to forward KL and 1.0 to reverse KL, so
        0.5 is the only one of the three that mixes. Reverse KL's mode-seeking half
        is what a student with an eighth of the teacher's hidden size needs."""
        assert float(_config(off, "train-gold")["gold_config"]["beta"]) == 0.5

    def test_the_fused_jsd_kernel_stays_off(self, off):
        """The teacher's logits_scaling is 10.0, which overflows the fused bf16 JSD
        kernel and yields NaN loss."""
        assert _config(off, "train-gold")["gold_config"]["use_liger_fused_jsd"] is False

    def test_the_learning_rate_suits_a_converged_student(self, off):
        """The student is a two-epoch SFT checkpoint, not a base model. The reference
        recipes' 1e-5 peak was tuned for a base student; at that rate a converged
        model is moved off its optimum before the soft labels do any work — and its
        optimum is the baseline row this work is measured against."""
        gold = _config(off, "train-gold")["gold_config"]
        peak, floor = float(gold["learning_rate"]), float(gold["min_lr"])
        assert peak <= 5e-6, "too hot for a converged SFT checkpoint"
        assert 0 < floor < peak


# ─── Agreement between steps ───────────────────────────────────────────────────


class TestAgreement:
    def test_one_length_budget_for_the_whole_pipeline(self, off):
        """Four steps measure lengths in tokens and have to agree about the budget."""
        budget = _params()["MAX_LENGTH"]
        assert _config(off, "corpus")["corpus_config"]["max_length"] == budget
        assert _config(off, "train-gold")["gold_config"]["max_length"] == budget
        for target in ("eval-transfer", "eval-transfer-baseline"):
            assert _config(off, target)["eval_config"]["max_length"] == budget

    def test_documents_are_kept_because_this_template_renders_them(self, off):
        """The step defaults to dropping rows carrying documents, on the grounds that
        granite's chat template does not render them. The granite 4.x template this
        pipeline installs DOES, via documents_system_message_prefix — so the default
        would silently discard grounded rows."""
        assert _config(off, "corpus")["corpus_config"]["documents_policy"] == "keep"

    def test_the_sft_arm_is_off_by_default(self, off):
        """The student already IS an SFT checkpoint and its eval row is the control.
        Re-running SFT would replace a measured control with an unmeasured one. The
        switch is kept rather than deleted so a same-corpus control remains one
        --param away."""
        assert _params()["INCLUDE_SFT"] is False
        assert "train-sft" not in _targets(off)

    def test_gold_saves_a_checkpoint_the_export_can_select(self, off):
        gold = _config(off, "train-gold")["gold_config"]
        assert gold["max_steps"] > 0
        assert gold["save_steps"] <= gold["max_steps"]

    def test_every_composed_path_lands_under_one_per_build_directory(self, off):
        """BUILD_SUBDIR is a correctness setting: a colliding artifact URI makes the
        target report SUCCESS with an EMPTY output list (build b5f030cd)."""
        assert _params()["BUILD_SUBDIR"] == "{{ run_metadata.build_id }}"
        root = f"{_params()['WORKDIR_ROOT']}/{_params()['RUN_NAME']}"
        rendered = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
        composed = re.findall(r'"(\$\$\{WORKDIR_ROOT\}[^"]*)"', rendered)
        assert composed, "no composed paths found; has the recipe changed shape?"
        for path in composed:
            assert path.startswith("$${WORKDIR_ROOT}/$${RUN_NAME}/$${BUILD_SUBDIR}/")
        assert root  # the interpolated form is what lands on disk


class TestThePinnedUpstreamCheckout:
    """Every distillation step reads its Python from a checkout THIS project controls.

    It used to be a shared, continuously-advancing tree, and that broke mid-build: the
    shared tree is advanced by its upstream author, and every time it moved, all six
    ported steps' pins stopped matching and the affected step exited 1. So the pin now
    names a small public repo (gb-steps-distillation) that only advances when this
    project deliberately bumps it.

    That lives in the STEP templates, identically across all six -- the source contract
    asserts those blocks are identical to each other, not that they hold any particular
    value, so a uniform change keeps it green. These tests assert the recipes do NOT
    re-specify it, because a per-recipe override is how five steps end up pointed at a
    tree their own pin rejects.
    """

    def test_no_target_overrides_the_shared_code_config(self, off):
        """The pin belongs to the step, once, for all six. A recipe-level override was
        the first attempt here and it left `corpus` reading the shared tree while
        `align` read the patched one -- which is precisely how build 1820703f failed."""
        for name in _targets(off):
            assert "code_config" not in _config(
                off, name
            ), f"{name} overrides code_config; the pin belongs in the step template"
