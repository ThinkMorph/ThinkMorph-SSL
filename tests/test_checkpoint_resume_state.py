"""Verify DCP populates dict-valued state_dict entries on load only when a template is provided."""

import os
import random

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp


@pytest.fixture(scope="module", autouse=True)
def _dist_env():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29677")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _make_saved_state():
    return {
        "extra_state": {
            "lr_scheduler_state": {
                "last_epoch": 8000,
                "base_lrs": [2e-5],
                "_last_lr": [2e-5],
            },
            "rng": {
                "cpu": torch.get_rng_state().clone(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "total_tokens": 123_456_789,
            "accumulated_grad_steps": 7,
            "current_stage_idx": 1,
        },
        "dataloader_state": {"cur_idx": 42},
    }


def test_empty_placeholder_drops_extra_state(tmp_path):
    """Current code path: load with {} placeholders -> nothing restored."""
    ckpt_dir = tmp_path / "ckpt_empty"
    dcp.save(_make_saved_state(), checkpoint_id=str(ckpt_dir))

    loaded = {"extra_state": {}, "dataloader_state": {}}
    dcp.load(loaded, checkpoint_id=str(ckpt_dir))

    assert loaded["extra_state"] == {}, (
        f"Expected empty dict to stay empty (documents the bug), "
        f"got: {loaded['extra_state']}"
    )
    assert loaded["dataloader_state"] == {}


def test_template_placeholder_restores_extra_state(tmp_path):
    """Fix path: load with live template -> values come back from ckpt."""
    ckpt_dir = tmp_path / "ckpt_template"
    saved = _make_saved_state()
    dcp.save(saved, checkpoint_id=str(ckpt_dir))

    loaded = {
        "extra_state": {
            "lr_scheduler_state": {"last_epoch": 0, "base_lrs": [0.0], "_last_lr": [0.0]},
            "rng": {
                "cpu": torch.zeros_like(saved["extra_state"]["rng"]["cpu"]),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "total_tokens": 0,
            "accumulated_grad_steps": 0,
            "current_stage_idx": 0,
        },
        "dataloader_state": {"cur_idx": 0},
    }
    dcp.load(loaded, checkpoint_id=str(ckpt_dir))

    assert loaded["extra_state"]["total_tokens"] == 123_456_789
    assert loaded["extra_state"]["accumulated_grad_steps"] == 7
    assert loaded["extra_state"]["current_stage_idx"] == 1
    assert loaded["extra_state"]["lr_scheduler_state"]["last_epoch"] == 8000
    assert loaded["extra_state"]["lr_scheduler_state"]["base_lrs"] == [2e-5]
    assert loaded["dataloader_state"]["cur_idx"] == 42


def test_real_scheduler_roundtrip(tmp_path):
    """Warmup scheduler at step 8000 should remain at step 8000 after DCP roundtrip."""
    from torch.optim.lr_scheduler import LambdaLR

    def _make_scheduler():
        optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=2e-5)
        return LambdaLR(optim, lr_lambda=lambda step: min(step / 1000.0, 1.0))

    saved_sched = _make_scheduler()
    for _ in range(8000):
        saved_sched.step()
    saved_lr = saved_sched.get_last_lr()
    assert saved_lr == [2e-5], f"sanity: warmup done by step 8000, got {saved_lr}"

    ckpt_dir = tmp_path / "ckpt_sched"
    saved = {"extra_state": {"lr_scheduler_state": saved_sched.state_dict()}}
    dcp.save(saved, checkpoint_id=str(ckpt_dir))

    fresh_sched = _make_scheduler()
    assert fresh_sched.get_last_lr() == [0.0], "fresh scheduler should be at step 0"

    loaded = {"extra_state": {"lr_scheduler_state": fresh_sched.state_dict()}}
    dcp.load(loaded, checkpoint_id=str(ckpt_dir))
    fresh_sched.load_state_dict(loaded["extra_state"]["lr_scheduler_state"])

    assert fresh_sched.get_last_lr() == [2e-5], (
        f"After DCP roundtrip + load_state_dict, scheduler should be back at peak lr; "
        f"got {fresh_sched.get_last_lr()}"
    )
