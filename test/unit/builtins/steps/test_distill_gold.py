"""Unit tests for the distill-gold step asset's response-template transport.

The trainer masks loss from the first token AFTER ``response_template``, so the
template's trailing newline is part of the mask boundary rather than cosmetic.
Getting it from a recipe to the trainer crosses a layer that silently eats it:
``fill_objtemplate`` (targetsteprun.py) runs every config string through Jinja,
and gbserver's SandboxedEnvironments are built without ``keep_trailing_newline``,
so Jinja's default strips exactly one trailing newline from each VALUE.

That is not hypothetical. Build d8470f14 (gold-sweep-smoke on BlueVela, 2026-09-14)
rendered ``"<|im_start|>assistant\\n"`` correctly into build.yaml, dispatched
``--response-template '<|im_start|>assistant'``, trained two steps against the
wrong span, and reported success — the exact silent failure the recipe exists to
gate. test_gold_sweep_recipe.py passed throughout, because it asserts on the
build.yaml render, one layer ABOVE where the newline dies.

The fix moves the newline out of the transport: recipes carry a literal backslash-n,
which has no trailing whitespace for anything to strip, and the step's renderer
decodes the escape exactly once at the far end. These tests cover both halves.
"""

import subprocess
import sys
from pathlib import Path

import yaml

from gbserver.utils.template import fill_objtemplate, fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
GOLD_STEP_DIR = (
    REPO_ROOT / "configurations/assets/environments/skypilot/steps/distill/distill-gold"
)
RENDERER = GOLD_STEP_DIR / "src/render_gold_config.py"

# What a recipe transports: a literal backslash followed by 'n', 22 chars, no
# trailing whitespace. This is the form that has to reach the step intact.
ESCAPED = "<|im_start|>assistant\\n"
# What the trainer must end up with.
DECODED = "<|im_start|>assistant\n"


def _render_config(tmp_path, response_template):
    """Run the step's renderer the way the launch script does, and load its output."""
    out = tmp_path / "gold.yaml"
    subprocess.run(
        [
            sys.executable,
            str(RENDERER),
            "--output",
            str(out),
            "--total-nodes",
            "1",
            "--model-name-or-path",
            "/proj/student",
            "--teacher-model-name-or-path",
            "/proj/teacher",
            "--dataset-name",
            "/proj/train.jsonl",
            "--response-template",
            response_template,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def test_renderer_decodes_the_response_template_escape(tmp_path):
    """A literal backslash-n on the command line becomes a real newline in the config.

    This is the decode half of the fix. The launch script cannot pass a real
    newline (the value arrives already stripped), so the escape is the only thing
    that can carry a line boundary across the config fill.
    """
    config = _render_config(tmp_path, ESCAPED)

    assert config["response_template"] == DECODED


def test_renderer_leaves_an_unescaped_template_alone(tmp_path):
    """Templates without an escape are untouched — gold-smoke deliberately has no
    trailing newline, and decoding must not invent one."""
    config = _render_config(tmp_path, "<|im_start|>assistant")

    assert config["response_template"] == "<|im_start|>assistant"


def test_the_escaped_template_survives_the_config_fill():
    """The transport half: the escaped form passes gbserver's config fill unharmed.

    fill_objtemplate is what silently ate the newline in build d8470f14. Asserting
    the escaped form survives it is what makes the escape convention load-bearing
    rather than incidental — and asserting the real newline does NOT survive is
    what documents why the convention is needed at all.
    """
    filled = fill_objtemplate({"response_template": ESCAPED}, {}, strict=False)
    assert filled["response_template"] == ESCAPED

    stripped = fill_objtemplate({"response_template": DECODED}, {}, strict=False)
    assert stripped["response_template"] == "<|im_start|>assistant", (
        "a real trailing newline is expected to be stripped here; if this now "
        "survives, the transport was fixed and the escape convention can be revisited"
    )


# ─── use_vllm: the on-policy server switch ─────────────────────────────────────

GOLD_STEP_YAML = GOLD_STEP_DIR / "step.yaml"


def _render_run(gold_overrides=None):
    """The step's launcher `run` script rendered against its own defaults."""
    cfg = yaml.safe_load(GOLD_STEP_YAML.read_text(encoding="utf-8"))
    gold = {**cfg["config"]["gold_config"], **(gold_overrides or {})}
    run = cfg["environment_configs"]["Skypilot"]["launchers"]["gold"]["config"]["run"]
    return fill_template(run, {"config": {"gold_config": gold}}, strict=False)


def test_on_policy_run_turns_use_vllm_on():
    """A configured vLLM server must actually receive the generation traffic.

    gold.py's --use_vllm defaults to False, and custom_gold_trainer.py:2003
    branches on it: False sends generation down the local ZeRO-3 path
    (generate_on_policy_outputs) instead of _generate_on_policy_outputs_vllm.
    Build d77546a9 allocated a whole node for vLLM, brought it up, health-checked
    it, plumbed its address in — and then generated locally anyway, because the
    step only ever passed the flag in the NEGATIVE case. The server logged not one
    request, and the local path died in transformers 5.8.0 on a shape mismatch.

    So the flag has to be stated in both directions, driven by the same
    vllm_num_servers the rest of the on-policy config keys are driven by.
    """
    run = _render_run({"vllm_num_servers": 1})

    assert "--use_vllm=True" in run
    assert "--use_vllm=False" not in run


def test_off_policy_run_turns_use_vllm_off():
    """The existing off-policy behaviour, unchanged: no server, no vLLM path."""
    run = _render_run({"vllm_num_servers": 0})

    assert "--use_vllm=False" in run
    assert "--use_vllm=True" not in run


def test_launch_command_has_no_comment_inside_a_continuation():
    """A `#` comment between backslash-continued lines silently eats the rest.

    Line continuations join physical lines into ONE logical line, so a comment
    placed mid-command comments out every argument that follows it — including
    ones on later continuation lines. The rendered text still CONTAINS those
    arguments, so a test that greps the text passes while the shell ignores them.
    This asserts the shape instead: no continued line may start a comment.
    """
    run = _render_run({"vllm_num_servers": 1})

    lines = run.splitlines()
    offenders = [
        (i + 1, line.strip())
        for i, line in enumerate(lines)
        if line.strip().startswith("#")
        and i > 0
        and lines[i - 1].rstrip().endswith("\\")
    ]

    assert not offenders, f"comment inside a line continuation: {offenders}"
