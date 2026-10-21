"""Decide whether a training step draws from the teacher, identically on every rank.

THIS IS PART OF THE COLLECTIVE SCHEDULE, NOT A SAMPLING DETAIL. Both call sites in
custom_gold_trainer.training_step guard a `gather_object` on the default process group -- directly on
the overlap path, and inside `_generate_on_policy_outputs_vllm` on the standard one. A collective must
be entered by every rank or by none, so a rank that disagrees about this draw does not get a slightly
different batch: it corrupts the process group.

WHAT IT USED TO BE, AND WHAT THAT COST. The guard was `random.random() <= current_lmbda`, reading
Python's GLOBAL random module, whose state is per PROCESS. It agrees across ranks only because
Trainer.__init__ calls set_seed(args.seed) on all of them and nothing afterwards consumes the module
unevenly -- an invariant nobody declared and nothing enforced. A direct measurement (sweep arm 2, lmbda 0.3,
24 steps) is what breaking it looks like: one rank entered gather_object alone and unpickled whatever
bytes the next collective had left in the buffer --

    _pickle.UnpicklingError: invalid load key, '\xe7'.

-- while the other three blocked in the following collective for the full 1,800,000 ms NCCL watchdog
and SIGABRTed. 2,081 s of a two-node allocation, ZERO optimizer steps logged.

THE EXPOSURE IS EXACTLY THE DELIVERABLE'S CASE, which is why it survived this long. lmbda 1.0 never
reaches the guard (a `current_lmbda >= 1.0` branch decides without the RNG) and lmbda 0.0 never takes
it, so the gen-cost probes and sweep arm 1 were structurally immune and both finished; arm 2 is the
only arm that consults the RNG and the only one that died. No commit between the last good on-policy
run (1148715, 10 steps at lmbda 0.3) and the crash touched this path -- the trigger is exposure, not a
regression. The deliverable runs 4,224 steps at lmbda 0.3.

THE DRAW IS NOW STATELESS IN THE STEP, which buys three things a shared generator would not:
  - it cannot drift, because nothing carries over from the previous draw;
  - it needs no broadcast, because every rank computes the same number from the same two integers;
  - it RESUMES correctly for free. Nothing in this trainer checkpoints RNG state, and on the
    `preemptable` queue a restart is the normal case, not the exception -- a stateful draw would
    silently take a different path through the run after every preemption.

WHY THIS IS ITS OWN MODULE rather than a method body. The property that matters is testable without a
GPU, a model, or an allocation, but importing custom_gold_trainer pulls in vllm, confirmed directly, on
`No module named 'vllm_ascend'`). A decision function that cannot be exercised cheaply is a decision
function that ships untested, and this one had already cost a run.
"""
from __future__ import annotations

import os
import random

import torch
import torch.distributed as dist

DRAW_SALT = "gold-on-policy-draw"


def verify_enabled() -> bool:
    """The agreement probe is on unless explicitly disabled.

    It costs one integer per step and converts a 30-minute silent watchdog hang into a named error.
    The probe is itself unconditional across ranks, so it cannot become the thing that desynchronises
    them.
    """
    return os.environ.get("GOLD_VERIFY_DRAW_AGREEMENT", "1").lower() not in ("0", "false", "no")


def draw_on_policy(current_lmbda: float, seed: int, global_step: int, device=None) -> bool:
    """True if this step should draw from the teacher. Same answer on every rank, by construction.

    `device` is only used by the agreement probe; pass the accelerator's device so the probe rides
    the same backend as the training collectives it is protecting.
    """
    if current_lmbda <= 0.0:
        return False
    if current_lmbda >= 1.0:
        return True

    step = int(global_step or 0)
    decision = random.Random(f"{int(seed)}:{step}:{DRAW_SALT}").random() <= current_lmbda

    if verify_enabled() and dist.is_available() and dist.is_initialized():
        probe = torch.tensor([1 if decision else 0], dtype=torch.int64, device=device)
        lo, hi = probe.clone(), probe.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if int(lo.item()) != int(hi.item()):
            raise RuntimeError(
                f"on-policy draw diverged across ranks at global_step {step} (this rank chose "
                f"{decision}, lmbda {current_lmbda}). The next collective would have been entered by "
                f"a subset of ranks; the observable symptom is an UnpicklingError on one rank and a "
                f"1800 s NCCL watchdog abort on the others, which is how a direct measurement lost a two-node "
                f"allocation with zero logged steps. The draw is derived from (seed, global_step) "
                f"precisely so this cannot happen, so reaching this line means seed or global_step "
                f"itself differs across ranks."
            )
    return decision
