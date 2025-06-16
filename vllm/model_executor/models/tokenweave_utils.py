import os
import json
import torch
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union
from vllm.model_executor.layers.layernorm import RMSNorm

@lru_cache(maxsize=None)
def load_config(config_path="tokenweave_configs/llama_config_8.json"):
    """
    TokenWeave Config Loader function: Load the JSON configuration file, 
    convert its string keys to integers, and return as a dictionary.
    The function uses caching to avoid redundant file reads for the same config path.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.normpath(os.path.join(base_dir, "..", "..", config_path))
    with open(full_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def fused_allreduce_layernorm(
    layernorm: RMSNorm,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    symm_mem_hdl: Any,
    num_tokens_per_rank: int,
    rank: int,
    world_size: int,
    MAX_CTAS: int,
    offset_symm_mem: Optional[int] = 0,
) -> None:
    """
    Applies fused residual addition + RMSNorm + AllReduce

    Args:
        layernorm (RMSNorm): Custom RMSNorm implementation with fused execution support.
        hidden_states (torch.Tensor): Full hidden state tensor.
                                      Shape: [total_tokens, hidden_dim]
        residual (torch.Tensor): Residual tensor to be added before normalization.
                                 Shape: [total_tokens, hidden_dim]
        symm_mem_hdl (Any): Symmetric memory handle for distributed synchronization.
        num_tokens_per_rank (int): Number of tokens assigned to each rank.
        rank (int): This process's rank.
        world_size (int): Total number of distributed processes.
        MAX_CTAS (int): Max concurrency (used to tune GPU kernel launches).

    Returns:
        None
    """

    # ============================
    # Safety checks and assertions
    # ============================

    # Ensure tensors are properly shaped
    assert hidden_states.shape == residual.shape, \
        "hidden_states and residual must have the same shape"

    # Confirm the total number of tokens is divisible by world size
    total_tokens = hidden_states.shape[0]
    expected_tokens = num_tokens_per_rank * world_size
    assert total_tokens == expected_tokens, (
        f"Mismatch in token count: expected {expected_tokens}, got {total_tokens}"
    )

    # Ensure rank is within valid bounds
    assert 0 <= rank < world_size, f"Invalid rank {rank}, must be in [0, {world_size - 1}]"

    # ============================
    # Compute rank-specific slice
    # ============================

    start_idx = rank * num_tokens_per_rank
    end_idx = (rank + 1) * num_tokens_per_rank

    # ============================
    # Apply fused RMSNorm + AllReduce
    # ============================

    layernorm(
        hidden_states[start_idx:end_idx],                # This rank's portion of hidden_states
        residual[start_idx:end_idx],                     # This rank's corresponding residuals
        MAX_CTAS=min(MAX_CTAS, num_tokens_per_rank),     # Limit concurrency to token count
        fused_ar=True,                                   # Enable fused AllReduce
        symm_mem_hdl=symm_mem_hdl,                       # Distributed communication handle
        rank=rank,                                       # This rank's ID
        world_size=world_size,                           # Total number of participating ranks
        offset=offset_symm_mem + start_idx * hidden_states.shape[1] * hidden_states.element_size(),  # Memory offset in bytes
    )