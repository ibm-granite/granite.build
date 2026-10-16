"""The trainer_state.json fixture shared by the resume tests.

test_step_template.py builds whole checkpoints around it (weights, ZeRO shards) to
run the resume block; test_align_resume_state.py uses it alone to run the rewrite.
One writer keeps the two in step with the fields the step reads.
"""

import json


def write_trainer_state(ckpt, step, save_steps, logging_steps):
    """Write ckpt/trainer_state.json as HF would at global_step `step` on the grid
    (save_steps, logging_steps); ckpt must exist. Returns the file's path."""
    path = ckpt / "trainer_state.json"
    state = dict(global_step=step, save_steps=save_steps, logging_steps=logging_steps)
    path.write_text(json.dumps(state))
    return path
