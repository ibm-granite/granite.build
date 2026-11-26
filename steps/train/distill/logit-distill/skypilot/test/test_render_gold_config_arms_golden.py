# SPDX-License-Identifier: Apache-2.0

"""Every loss arm renders byte-identically to ced53644, except `uld`, which changes as stated.

At ced53644 the renderer dropped last_message_only off-policy (_ONLINE_ONLY), so the uld
arm could not get the last_message_only=true its alignment needs. The change gives the uld
arm that requirement and emits the key off-policy for that arm alone. This pins the
"alone": for every arm, on- and off-policy, with last_message_only false and true, the
renderer's exit code and output bytes are compared against the ced53644 renderer's,
recorded in render_gold_config_golden_ced53644.json. For uld the expected difference is
spelled out: false now fails to render, and true off-policy gains exactly one trailing
`last_message_only: true` line.

The fixture was written by running this module against the ced53644 renderer:

    git show ced53644:steps/train/distill/logit-distill/skypilot/src/render_gold_config.py \\
        > /tmp/render_ced53644.py
    python test/test_render_gold_config_arms_golden.py --write /tmp/render_ced53644.py \\
        test/render_gold_config_golden_ced53644.json
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RENDER = _HERE.parent / "src" / "render_gold_config.py"
_GOLDEN = _HERE / "render_gold_config_golden_ced53644.json"

_REQUIRED = [
    "--model-name-or-path",
    "/proj/kd/student_overlays/granite-4.1-3b-base-hub",
    "--teacher-model-name-or-path",
    "/proj/kd/teacher_overlays/granite-4.2-30b",
    "--dataset-name",
    "/proj/kd/data/subsampled_0.4_shuffled_nothink.jsonl",
]

# What each arm needs to render at all; requirements it does not meet are covered by the
# last_message_only axis and by the policy axis (an arm that is on-policy only fails
# off-policy, at ced53644 and now).
_ARM_ARGS = {
    "": [],
    "jsd": [],
    "kl_interpolation": ["--beta", "0.5"],
    "liger_fused_jsd": ["--use-liger-fused-jsd", "true"],
    "liger_fused_kl_interpolation": ["--use-liger-fused-jsd", "true", "--beta", "0.5"],
    "distillm2": [],
    "distillm2_like": [],
    "reversed_distillm2_like": [],
    "sampled_opd": ["--use-sampled-opd-loss", "true"],
    "uld": [],
    "ce": [],
    "adaptive_kld": [],  # refused outright
}
_POLICY_ARGS = {
    "off": (1, ["--lmbda", "0.0"]),
    "on": (2, ["--vllm-num-servers", "1", "--lmbda", "0.5"]),
    # sampled_opd's lmbda; also a second on-policy shape for every other arm.
    "on-lmbda1": (2, ["--vllm-num-servers", "1", "--lmbda", "1.0"]),
}
_LMO = ("false", "true")


def _cases():
    for arm, arm_args in _ARM_ARGS.items():
        for policy, (nodes, policy_args) in _POLICY_ARGS.items():
            for lmo in _LMO:
                case_id = (
                    f"arm={arm or '<none>'} policy={policy} last_message_only={lmo}"
                )
                extra = ["--loss-arm", arm, *arm_args, *policy_args]
                extra += ["--last-message-only", lmo]
                yield case_id, nodes, extra


def _run(renderer, nodes, extra):
    """Render one case; return (exit code, output text or None)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "gold_config.yaml"
        cmd = [sys.executable, str(renderer), "--output", str(out)]
        cmd += ["--total-nodes", str(nodes), *_REQUIRED, *extra]
        rc = subprocess.run(cmd, capture_output=True, text=True, check=False).returncode
        return rc, (out.read_text(encoding="utf-8") if rc == 0 else None)


def _render_all(renderer):
    return {
        case_id: dict(zip(("rc", "yaml"), _run(renderer, nodes, extra)))
        for case_id, nodes, extra in _cases()
    }


def test_every_arm_but_uld_is_byte_identical_and_uld_changes_as_stated():
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    current = _render_all(_RENDER)
    assert sorted(current) == sorted(
        golden
    ), "the case list moved; regenerate the fixture"
    # Sanity: the matrix is not vacuous. Every arm renders somewhere (but adaptive_kld,
    # which is refused), and both policies render for the arms that allow both.
    rendered_arms = {c.split()[0] for c, g in golden.items() if g["rc"] == 0}
    assert rendered_arms == {f"arm={a or '<none>'}" for a in _ARM_ARGS} - {
        "arm=adaptive_kld"
    }
    for case_id, want in golden.items():
        got = current[case_id]
        if not case_id.startswith("arm=uld "):
            assert got == want, case_id
            continue
        if case_id.endswith("last_message_only=false"):
            # Rendered at ced53644, and trained on nothing; refused now.
            assert want["rc"] == 0 and got == {"rc": 2, "yaml": None}, case_id
        elif "policy=off " in case_id:
            # The one off-policy addition, appended after the arm's own switch.
            assert want["rc"] == 0, case_id
            assert got["rc"] == 0, case_id
            assert got["yaml"] == want["yaml"] + "last_message_only: true\n", case_id
        else:
            # On-policy the key was already emitted, for every arm.
            assert got == want, case_id
            assert "last_message_only: true\n" in got["yaml"], case_id


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--write":
        sys.exit(
            "usage: test_render_gold_config_arms_golden.py --write RENDERER OUT.json"
        )
    Path(sys.argv[3]).write_text(
        json.dumps(_render_all(Path(sys.argv[2])), indent=1, sort_keys=True) + "\n",
        encoding="utf-8",
    )
