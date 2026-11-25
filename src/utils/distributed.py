import os
import warnings
from typing import Optional

import torch
import torch.distributed as dist


def install_mixed_pg_default_device_warning_filter() -> None:
    """Hide PyTorch's benign mixed-backend default-device warnings.

    This repo intentionally initializes the default process group as
    ``cpu:gloo,cuda:nccl`` so DCP async checkpointing has a CPU backend while
    training collectives still use CUDA/NCCL. Some PyTorch 2.8 state-dict helper
    paths do not expose a device argument and still call the deprecated internal
    ``_get_pg_default_device`` helper. Those warnings are expected for this
    configuration and are not actionable in user code.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"`_get_pg_default_device` will be deprecated.*",
        category=UserWarning,
        module=r"torch\.distributed\.distributed_c10d",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"Multiple backends are registered with this ProcessGroup.*",
        category=UserWarning,
        module=r"torch\.distributed\.distributed_c10d",
    )


def _local_cuda_barrier_device_id() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return None
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        local_rank = 0
    return local_rank % device_count


def dist_barrier() -> None:
    """Run a distributed barrier without asking PyTorch to infer a default device.

    With a mixed default process group such as ``cpu:gloo,cuda:nccl``, a bare
    ``dist.barrier()`` makes PyTorch call its deprecated default-device
    inference path. Supplying the local CUDA device avoids that warning while
    preserving CPU-only fallback behavior.
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    device_id = _local_cuda_barrier_device_id()
    if device_id is None:
        dist.barrier()
    else:
        dist.barrier(device_ids=[device_id])


def object_collective_device() -> torch.device:
    """Explicit device for small Python-object collectives.

    The default process group is initialized with a CPU backend in this repo, so
    CPU is the right device for serialized metadata and avoids PyTorch's
    deprecated default-device inference path.
    """
    return torch.device("cpu")
