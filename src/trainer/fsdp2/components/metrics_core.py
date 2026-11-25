from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from utils import TrainUtilities


def calculate_training_metrics(
    *,
    flops_tensor: torch.Tensor,
    sp_size: int,
    promised_flops: float,
    device: torch.device,
    seq_len: List[float],
    total_tokens: int,
    delta_time: float,
    world_size: int,
) -> Tuple[Dict[str, float | int | str], int]:
    metrics: Dict[str, float | int | str] = {}

    mfu = flops_tensor.item() / sp_size / promised_flops
    mfu = torch.tensor(mfu, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(mfu, op=dist.ReduceOp.AVG)
    mfu_value = mfu.item()

    seq_len_tensor = torch.tensor(seq_len, device=device, dtype=torch.float32)
    total_seq_len = seq_len_tensor.sum() / sp_size
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_seq_len, op=dist.ReduceOp.SUM)

    global_seq_len_avg = seq_len_tensor.sum()
    global_seq_len_min = seq_len_tensor.sum()
    global_seq_len_max = seq_len_tensor.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_seq_len_avg, op=dist.ReduceOp.AVG)
        dist.all_reduce(global_seq_len_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(global_seq_len_max, op=dist.ReduceOp.MAX)

    metrics["perf/global_seq_len_avg"] = global_seq_len_avg.item()
    metrics["perf/global_seq_len_min"] = global_seq_len_min.item()
    metrics["perf/global_seq_len_max"] = global_seq_len_max.item()
    metrics["train/mfu"] = round(mfu_value, 2)

    total_tokens += int(total_seq_len.item())
    tokens_per_second = total_seq_len.item() / delta_time
    tokens_per_gpu = tokens_per_second / world_size

    metrics["train/total_tokens"] = TrainUtilities.format_tokens(total_tokens)
    metrics["train/tokens_per_second"] = round(tokens_per_second)
    metrics["train/tokens_per_gpu"] = round(tokens_per_gpu)

    return metrics, total_tokens
