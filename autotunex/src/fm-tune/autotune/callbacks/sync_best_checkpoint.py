# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""Make every rank agree with rank 0 on the best checkpoint after each save.

HF's ``Trainer._save_checkpoint`` updates ``state.best_model_checkpoint`` on
**each rank independently**, gated on ``os.path.exists(best_checkpoint_dir)`` as
that rank sees it, while checkpoint rotation (``save_total_limit``) runs on rank
0 only and protects only rank 0's best. When a rank's view disagrees -- its
exists-check misses the directory rank 0 just wrote, or its metric differs -- it
silently keeps an older "best", rank 0's rotation later deletes that directory,
and the run dies at the very end, inside ``_load_best_model``, after all the
compute is spent::

    ValueError: Can't find a valid checkpoint at .../checkpoint-74

Seen on a 2-rank KTO + ZeRO-3 final training (rank 0: best=checkpoint-111, rank
1: checkpoint-74, eval_loss 0.32270 vs 0.32357); the identical job passed on two
other runs. Rank 0 decides which directories survive rotation, so its view is the
one that matches the disk: broadcast it after every save. A no-op without an
initialized multi-rank process group, so it is safe in every driver.
"""

from transformers import TrainerCallback


class SyncBestCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
            return control
        payload = [(state.best_model_checkpoint, state.best_metric, state.best_global_step)]
        dist.broadcast_object_list(payload, src=0)
        state.best_model_checkpoint, state.best_metric, state.best_global_step = payload[0]
        return control
