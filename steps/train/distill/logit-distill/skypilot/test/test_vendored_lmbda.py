"""The vendored trainer's lmbda schedule, generation floor and stop_at_step.

These pin the two local patches folded into steps/train/distill/src (see VENDORED.md there):
the restored linear lmbda schedule, and the on-policy study's ramp window, generation floor
(min_completion_length) and stop_at_step.

The trainer module imports torch, trl and transformers at module scope, none of which the
granite.build venv carries, so it cannot be imported here. Instead each method under test is
lifted out of the vendored source with ``ast`` and executed against minimal stubs (``.args``,
``.state``). That runs the real code -- the source text the image bakes in -- without the
heavy stack, and the structural checks below cover the call sites a stub cannot reach.
"""

import ast
import types
from pathlib import Path

import pytest

_PKG = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "gb_steps_post_training"
    / "distillation"
)
_TRAINER = _PKG / "custom_gold_trainer.py"
_CONFIG = _PKG / "custom_gold_config.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _method_node(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {cls.name}.{name} not found")


def _compile(node: ast.stmt, namespace: dict) -> dict:
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(_TRAINER), "exec"), namespace)  # noqa: S102
    return namespace


def _trainer_method(name: str):
    cls = _class(_tree(_TRAINER), "CustomGOLDTrainer")
    return _compile(_method_node(cls, name), {})[name]


def _stub(*, max_steps: int, global_step: int, **args):
    defaults = {"lmbda_schedule": "linear", "lmbda_init": 0.0, "lmbda": 1.0}
    defaults.update(args)
    return types.SimpleNamespace(
        args=types.SimpleNamespace(**defaults),
        state=types.SimpleNamespace(max_steps=max_steps, global_step=global_step),
        lmbda=defaults["lmbda"],
    )


def _lmbda(**kwargs) -> float:
    return _trainer_method("_get_current_lmbda")(_stub(**kwargs))


class TestLinearLmbdaSchedule:
    """At a5d59bc4 the linear branch sat after _draw_on_policy's return: unreachable."""

    @pytest.mark.parametrize("step, expected", [(0, 0.0), (50, 0.5), (100, 1.0)])
    def test_linear_ramps_over_max_steps(self, step, expected):
        assert _lmbda(max_steps=100, global_step=step) == pytest.approx(expected)

    def test_linear_never_returns_none(self):
        # None is what made draw_on_policy raise TypeError on `None <= 0.0` at step 1.
        assert _lmbda(max_steps=100, global_step=1) is not None

    def test_constant_is_unchanged(self):
        assert _lmbda(
            max_steps=100, global_step=50, lmbda_schedule="constant", lmbda=0.3
        ) == pytest.approx(0.3)

    def test_the_upstream_verification_points(self):
        # The patch header's own check: 0.25 -> 0.5 over 2,500 steps.
        got = [
            _lmbda(max_steps=2500, global_step=s, lmbda_init=0.25, lmbda=0.5)
            for s in (0, 625, 1250, 1875, 2500)
        ]
        assert got == pytest.approx([0.25, 0.3125, 0.375, 0.4375, 0.5])

    def test_draw_on_policy_no_longer_carries_the_dead_branch(self):
        cls = _class(_tree(_TRAINER), "CustomGOLDTrainer")
        src = ast.unparse(_method_node(cls, "_draw_on_policy"))
        assert "lmbda_schedule" not in src


class TestLmbdaRampWindow:
    @pytest.mark.parametrize(
        "step, expected",
        [(0, 0.0), (40, 0.0), (50, 0.5), (60, 1.0), (100, 1.0)],
    )
    def test_ramp_start_40_length_20(self, step, expected):
        got = _lmbda(
            max_steps=100,
            global_step=step,
            lmbda_ramp_start_step=40,
            lmbda_ramp_steps=20,
        )
        assert got == pytest.approx(expected)

    def test_a_resumed_run_ramps_over_its_own_steps(self):
        # A run resumed at 2,000 of an 8,150-step schedule ramps over its own steps.
        got = [
            _lmbda(
                max_steps=8150,
                global_step=s,
                lmbda_ramp_start_step=2000,
                lmbda_ramp_steps=2500,
            )
            for s in (2000, 3250, 4500)
        ]
        assert got == pytest.approx([0.0, 0.5, 1.0])

    def test_defaults_reproduce_the_plain_linear_schedule(self):
        assert _lmbda(
            max_steps=100, global_step=50, lmbda_ramp_start_step=0, lmbda_ramp_steps=0
        ) == pytest.approx(0.5)


class TestStopAtStep:
    def _callback(self):
        tree = _tree(_TRAINER)
        ns = {"TrainerCallback": object}
        return _compile(_class(tree, "_StopAtStepCallback"), ns)["_StopAtStepCallback"]

    @staticmethod
    def _control():
        return types.SimpleNamespace(should_save=False, should_training_stop=False)

    def test_stops_and_saves_at_the_step(self):
        cb = self._callback()(30)
        control = cb.on_step_end(
            None, types.SimpleNamespace(global_step=30), self._control()
        )
        assert control.should_training_stop and control.should_save

    def test_runs_on_before_the_step(self):
        cb = self._callback()(30)
        control = cb.on_step_end(
            None, types.SimpleNamespace(global_step=29), self._control()
        )
        assert not control.should_training_stop and not control.should_save

    def test_the_trainer_registers_it_when_set(self):
        cls = _class(_tree(_TRAINER), "CustomGOLDTrainer")
        init = ast.unparse(_method_node(cls, "__init__"))
        assert "_StopAtStepCallback" in init and "stop_at_step" in init


class TestGenerationFloor:
    def test_the_floor_is_sent_as_min_tokens(self):
        fn = _trainer_method("_min_tokens_kwargs")
        stub = types.SimpleNamespace(
            args=types.SimpleNamespace(min_completion_length=16)
        )
        assert fn(stub) == {"generation_kwargs": {"min_tokens": 16}}

    def test_no_floor_sends_nothing(self):
        fn = _trainer_method("_min_tokens_kwargs")
        stub = types.SimpleNamespace(
            args=types.SimpleNamespace(min_completion_length=0)
        )
        assert fn(stub) == {}

    def test_every_vllm_server_request_carries_the_floor(self):
        """Both server-mode request sites: the overlap submit and the direct generate."""
        tree = _tree(_TRAINER)
        sites = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = ast.unparse(node.func)
            direct = func == "self.vllm_client.generate"
            submitted = func.endswith(".submit") and any(
                ast.unparse(a) == "self.vllm_client.generate" for a in node.args
            )
            if direct or submitted:
                sites.append(node)
        assert len(sites) >= 2
        for call in sites:
            spread = [ast.unparse(k.value) for k in call.keywords if k.arg is None]
            assert "self._min_tokens_kwargs()" in spread, ast.unparse(call)[:200]

    def test_colocate_sampling_params_carry_min_tokens(self):
        calls = [
            n
            for n in ast.walk(_tree(_TRAINER))
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "SamplingParams"
        ]
        assert calls
        for call in calls:
            assert "min_tokens" in {k.arg for k in call.keywords}


class TestConfigFields:
    """CustomGOLDConfig imports trl, so its fields are read from the source."""

    def _fields(self) -> dict:
        cls = _class(_tree(_CONFIG), "CustomGOLDConfig")
        out = {}
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                default = None
                if isinstance(node.value, ast.Call):
                    for kw in node.value.keywords:
                        if kw.arg == "default":
                            default = ast.literal_eval(kw.value)
                out[node.target.id] = default
        return out

    @pytest.mark.parametrize(
        "name",
        [
            "lmbda_ramp_start_step",
            "lmbda_ramp_steps",
            "min_completion_length",
            "stop_at_step",
        ],
    )
    def test_new_field_defaults_to_off(self, name):
        fields = self._fields()
        assert name in fields
        assert fields[name] == 0

    def test_ce_anchor_and_entropy_guard_fields_are_gone(self):
        """Removed in #664 (see VENDORED.md); a re-import must not bring them back."""
        fields = self._fields()
        for name in (
            "ce_coef",
            "log_student_entropy",
            "entropy_guard_drop_frac",
            "entropy_guard_baseline_steps",
            "entropy_guard_patience",
            "entropy_guard_action",
        ):
            assert name not in fields

    def test_ce_anchor_and_entropy_guard_logic_is_gone(self):
        trainer = _TRAINER.read_text()
        for needle in (
            "ce_coef",
            "log_student_entropy",
            "entropy_guard",
            "_maybe_act_on_entropy_collapse",
            "_accumulate_collapse_diagnostics",
        ):
            assert needle not in trainer

    def test_post_init_rejects_negative_values(self):
        cls = _class(_tree(_CONFIG), "CustomGOLDConfig")
        post = ast.unparse(_method_node(cls, "__post_init__"))
        for name in (
            "lmbda_ramp_start_step",
            "lmbda_ramp_steps",
            "min_completion_length",
            "stop_at_step",
        ):
            assert name in post
