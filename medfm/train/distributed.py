"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist
from typing import Optional


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> None:
    """Initialize distributed training."""
    if init_method is None:
        init_method = "env://"

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    """Get the number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get the rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary values across all processes."""
    if not dist.is_initialized():
        return input_dict

    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict

    keys = sorted(input_dict.keys())
    values = [input_dict[k] for k in keys]

    tensor = torch.tensor(values, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor)

    if average:
        tensor /= world_size

    reduced_dict = {k: tensor[i].item() for i, k in enumerate(keys)}
    return reduced_dict
