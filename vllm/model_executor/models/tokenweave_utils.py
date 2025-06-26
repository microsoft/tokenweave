import os
import json
import torch
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union, Callable
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.distributed.triton_comm.triton_comm import (
    multimem_all_reduce, multimem_reduce_scatter, multimem_all_gather, multimem_all_gather_async)

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
    offset_symm_mem: int = 0,
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


def tokenweave_with_fuse_only(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    symm_mem_hdl: Any,
    layer_id: int,
    rank: int = 0,
    world_size: int = 1,
    next_layer_norm: RMSNorm = None,
    actual_tokens: int = None,
    nearest_multiple_of_world_size: int = None,
    MAX_CTAS_ATTN: int = 16,
    MAX_CTAS_MLP: int = 16,
    mlp_fn: Callable = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass with fused all-reduce + RMSNorm + residual add only (no TokenWeave overlap).

    Args:
        positions (torch.Tensor): Positional encoding indices.
        hidden_states (torch.Tensor): Input hidden states.
        residual (torch.Tensor): Residual.
        symm_mem_hdl (Any): Symmetric memory handle for all-reduce.
        layer_id (int): Current layer index.
        rank (int): Local process rank.
        world_size (int): Total number of distributed processes.
        next_layer_norm (RMSNorm): LayerNorm for the next block.
        actual_tokens (int): Number of valid tokens.
        nearest_multiple_of_world_size (int): Padding length (multiple of world_size).
        MAX_CTAS_ATTN (int): Max CTAs for attention norm kernel.
        MAX_CTAS_MLP (int): Max CTAs for MLP norm kernel.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated hidden_states and residual.
    """
    assert mlp_fn is not None, "mlp_fn must be provided"
    num_tokens_per_rank = nearest_multiple_of_world_size // world_size

    if residual is None:
        residual = torch.empty_like(hidden_states)
    if layer_id == 0:
        self.input_layernorm(hidden_states, out=residual)

    # === Self-Attention (non-overlapping) ===
    self.self_attn(
        positions=positions,
        hidden_states=hidden_states[:actual_tokens]
    )

    # === Post-Attention Norm (fused residual add + RMSNorm + allreduce) ===
    fused_allreduce_layernorm(
        layernorm=self.post_attention_layernorm,
        hidden_states=hidden_states,
        residual=residual,
        symm_mem_hdl=symm_mem_hdl,
        num_tokens_per_rank=num_tokens_per_rank,
        rank=rank,
        world_size=world_size,
        MAX_CTAS=MAX_CTAS_ATTN,
    )

    # === MLP ===
    mlp_fn(hidden_states[:actual_tokens])

    # === Final Norm ((fused residual add + RMSNorm + allreduce)) ===
    fused_allreduce_layernorm(
        layernorm=next_layer_norm,
        hidden_states=hidden_states,
        residual=residual,
        symm_mem_hdl=symm_mem_hdl,
        num_tokens_per_rank=num_tokens_per_rank,
        rank=rank,
        world_size=world_size,
        MAX_CTAS=MAX_CTAS_MLP,
    )

    return hidden_states, residual


def tokenweave_overlap(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    symm_mem_hdl: Any,
    layer_id: int,
    end_layer: int = None,
    rank: int = 0,
    world_size: int = 1,
    current_stream: torch.cuda.Stream = None,
    copy_stream: torch.cuda.Stream = None,
    next_layer_norm: RMSNorm = None,
    split_size: int = None,
    actual_tokens: int = None,
    num_tokens_padded: int = None,
    MAX_CTAS_ATTN: int = 16,
    MAX_CTAS_MLP: int = 16,
    mlp_fn: Callable = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs forward pass of a transformer block using TokenWeave overlap strategy.
    Processes two token splits (interleaved) across GPUs with communication-compute overlap.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated hidden_states and residual tensors.
    """
    assert mlp_fn is not None, "mlp_fn must be provided"
    num_bytes_per_token = hidden_states.shape[1] * hidden_states.element_size()
    # Self Attention
    offset_second = split_size * hidden_states.shape[1] * hidden_states.element_size()
    if residual is None:
        residual = torch.empty_like(hidden_states)
    # Split hidden states and residuals
    hidden_states_1, hidden_states_2 = hidden_states[:split_size], hidden_states[split_size:]
    residual_1, residual_2 = residual[:split_size], residual[split_size:]
    blpr_1 = split_size // world_size
    blpr_2 = hidden_states_2.shape[0] // world_size

    # === LayerNorm & Comm for First Layer ===
    if layer_id == 0:
        hidden_states_1 = self.input_layernorm(hidden_states_1, out=residual_1)
        multimem_reduce_scatter(
            hidden_states_2,
            symm_mem_hdl,
            offset_second,
            MAX_CTAS=8
        )
        self.input_layernorm(
            hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], 
            out=residual_2[rank * blpr_2: (rank + 1) * blpr_2])
        symm_mem_hdl.barrier(channel=7)
        multimem_all_gather_async(
            hidden_states_2,
            symm_mem_hdl,
            offset_second,
            blpr_2 * num_bytes_per_token,
            current_stream,
        )
        symm_mem_hdl.barrier(channel=9)
    else:
        # === Fused all reduce + Pre-Attn Norm + residual add on split-1 ===
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(current_stream)
            fused_allreduce_layernorm(
                layernorm=self.input_layernorm,
                hidden_states=hidden_states_2,
                residual=residual_2,
                symm_mem_hdl=symm_mem_hdl,
                num_tokens_per_rank=blpr_2,
                rank=rank,
                world_size=world_size,
                MAX_CTAS=MAX_CTAS_ATTN,
                offset_symm_mem=offset_second
            )
    # === Self-Attn on split-0 ===
    with torch.cuda.stream(current_stream):
        hidden_states_1 = self.self_attn(
            positions=positions[:split_size],
            hidden_states=hidden_states_1,
            split_id=0,
            split_size=split_size,
            num_actual_tokens=split_size,
        )
        current_stream.wait_stream(copy_stream)

    # === Fused all reduce + Post-Attn Norm + residual add on split-0 ===
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(current_stream)
        fused_allreduce_layernorm(
            layernorm=self.post_attention_layernorm,
            hidden_states=hidden_states_1,
            residual=residual_1,
            symm_mem_hdl=symm_mem_hdl,
            num_tokens_per_rank=blpr_1,
            rank=rank,
            world_size=world_size,
            MAX_CTAS=MAX_CTAS_ATTN,
            offset_symm_mem=0
        )
    
    # === Self-Attn on split-1 ===
    with torch.cuda.stream(current_stream):
        hidden_states_2 = self.self_attn(
            positions=positions[split_size:],
            hidden_states=hidden_states_2,
            split_id=1,
            split_size=split_size,
            num_actual_tokens=actual_tokens - split_size,
        )
        current_stream.wait_stream(copy_stream)
    
    # === Fused all reduce + Post-Attn Norm + residual add on split-1 ===
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(current_stream)
        fused_allreduce_layernorm(
            layernorm=self.post_attention_layernorm,
            hidden_states=hidden_states_2,
            residual=residual_2,
            symm_mem_hdl=symm_mem_hdl,
            num_tokens_per_rank=blpr_2,
            rank=rank,
            world_size=world_size,
            MAX_CTAS=MAX_CTAS_MLP,
            offset_symm_mem=offset_second
        )

    # === MLP on split-0 ===
    with torch.cuda.stream(current_stream):
        hidden_states_1 = mlp_fn(hidden_states_1)
        current_stream.wait_stream(copy_stream)

    # === Fused all reduce + Post-MLP Norm + residual add on split-0 ===
    with torch.cuda.stream(copy_stream):
        copy_stream.wait_stream(current_stream)
        fused_allreduce_layernorm(
            layernorm=next_layer_norm,
            hidden_states=hidden_states_1,
            residual=residual_1,
            symm_mem_hdl=symm_mem_hdl,
            num_tokens_per_rank=blpr_1,
            rank=rank,
            world_size=world_size,
            MAX_CTAS=MAX_CTAS_MLP,
            offset_symm_mem=0
        )
    
    # === MLP on split-1 ===
    with torch.cuda.stream(current_stream):
        hidden_states_2 = mlp_fn(hidden_states_2)
        current_stream.wait_stream(copy_stream)    

    # === Fused all reduce + Post-MLP Norm + residual add on split-1 (only on last layer) ===
    if layer_id == end_layer - 1:
        fused_allreduce_layernorm(
            layernorm=next_layer_norm,
            hidden_states=hidden_states_2,
            residual=residual_2,
            symm_mem_hdl=symm_mem_hdl,
            num_tokens_per_rank=blpr_2,
            rank=rank,
            world_size=world_size,
            MAX_CTAS=16 if actual_tokens < 16384 else 32,
            offset_symm_mem=offset_second
        )
    return hidden_states, residual