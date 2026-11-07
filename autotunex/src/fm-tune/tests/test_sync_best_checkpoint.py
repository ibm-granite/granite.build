"""Pins autotune/callbacks/sync_best_checkpoint.py on a real 2-rank (gloo, CPU) group."""

import os
import socket

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import TrainerControl, TrainerState

from autotune.callbacks.sync_best_checkpoint import SyncBestCheckpointCallback


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    try:
        state = TrainerState()
        # rank 1 missed rank 0's newer best (the observed failure)
        state.best_model_checkpoint = "/ckpt/checkpoint-111" if rank == 0 else "/ckpt/checkpoint-74"
        state.best_metric = 0.32270 if rank == 0 else 0.32357
        state.best_global_step = 111 if rank == 0 else 74
        SyncBestCheckpointCallback().on_save(None, state, TrainerControl())
        out[rank] = (state.best_model_checkpoint, state.best_metric, state.best_global_step)
    finally:
        dist.destroy_process_group()


def test_every_rank_adopts_rank0_best():
    with mp.Manager() as m:
        out = m.dict()
        mp.spawn(_worker, args=(_free_port(), out), nprocs=2, join=True)
        assert out[0] == out[1] == ("/ckpt/checkpoint-111", 0.32270, 111)


def test_noop_without_process_group():
    assert not dist.is_initialized()
    state = TrainerState()
    state.best_model_checkpoint = "/ckpt/checkpoint-74"
    SyncBestCheckpointCallback().on_save(None, state, TrainerControl())
    assert state.best_model_checkpoint == "/ckpt/checkpoint-74"


@pytest.mark.parametrize("driver", ["hf_ds", "hf_fsdp", "trl_ds", "trl_fsdp"])
def test_multi_gpu_drivers_register_it(driver):
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[1] / f"autotune/trainers/driver_multi_{driver}.py").read_text()
    assert "trainer.add_callback(SyncBestCheckpointCallback())" in src
