# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
from typing import Iterable, Optional, Set, Tuple, Union, Any
import os
import torch
import torch.distributed._symmetric_memory as symm_mem
from torch import nn
from transformers import MixtralConfig

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size, pytorch_all_reduce, get_device_group, get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed.triton_comm.triton_comm import (
    multimem_all_reduce, multimem_reduce_scatter, multimem_all_gather, multimem_all_gather_async)
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

from .tokenweave_utils import (load_config, fused_allreduce_layernorm)

class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self,
                 num_experts: int,
                 top_k: int,
                 hidden_size: int,
                 intermediate_size: int,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 tp_size: Optional[int] = None,
                 dp_size: Optional[int] = None,
                 prefix: str = ""):
        super().__init__()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.

        self.gate = ReplicatedLinear(hidden_size,
                                     num_experts,
                                     bias=False,
                                     params_dtype=params_dtype,
                                     quant_config=None,
                                     prefix=f"{prefix}.gate")

        self.experts = FusedMoE(num_experts=num_experts,
                                top_k=top_k,
                                hidden_size=hidden_size,
                                intermediate_size=intermediate_size,
                                params_dtype=params_dtype,
                                reduce_results=True,
                                renormalize=True,
                                quant_config=quant_config,
                                tp_size=tp_size,
                                dp_size=dp_size,
                                prefix=f"{prefix}.experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        self.experts(hidden_states, router_logits, is_tokenweave=True)
        return hidden_states.view(orig_shape)


class MixtralAttention(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MixtralConfig has an optional head_dim argument
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        split_id: Optional[int] = None,
        chunk_size: Optional[int] = None,
        num_actual_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the attention layer with optional TokenWeave mode.
        """
        # ----------------------------------------
        # Arguments:
        # - positions: Tensor containing positional indices for rotary embeddings.
        #              Shape: [num_tokens]
        #
        # - hidden_states: Input tensor containing embeddings to be processed by the attention mechanism.
        #                  Shape: [num_tokens, hidden_dim]
        #
        # - split_id: Optional identifier (int): 0 or 1 — 0 for the first split batch, 1 for the second.
        #             Relevant only in TokenWeave mode.
        #
        # - chunk_size: Optional identifier (int): Number of tokens in the first split batch.
        #             Relevant only in TokenWeave mode.
        #
        # - num_actual_tokens: The number of tokens used to exclude padding or non-real tokens in TokenWeave mode.
        #
        # Returns:
        # - Updated hidden_states tensor after attention and output projection.
        # ----------------------------------------
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if split_id is not None and chunk_size is not None:
            # TokenWeave Mode
            assert num_actual_tokens is not None
            self.rotary_emb(positions, q[:num_actual_tokens], k[:num_actual_tokens])
            attn_output = self.attn(q, k, v, split_id, chunk_size)
        else:
            # Default Mode
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v)
        # inplace + no all reduce
        self.o_proj(attn_output, hidden_states,
                                is_tokenweave=True)
        return hidden_states


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn")
        self.block_sparse_moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe")
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward_with_fuse_only(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        symm_mem_hdl: Any,
        layer_id: int,
        rank: int = 0,
        world_size: int = 1,
        next_layer_norm: Optional[RMSNorm] = None,
        actual_tokens: Optional[int] = None,
        nearest_multiple_of_world_size: Optional[int] = None,
        MAX_CTAS_ATTN: int = 16,
        MAX_CTAS_MLP: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with fused all-reduce + RMSNorm + residual add only (no TokenWeave overlap).

        Args:
            positions (torch.Tensor): Positional encoding indices.
            hidden_states (torch.Tensor): Input hidden states.
            residual (Optional[torch.Tensor]): Optional residual storage.
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
        assert actual_tokens is not None, "actual_tokens must be provided"
        assert nearest_multiple_of_world_size is not None, "nearest_multiple_of_world_size must be set"
        assert next_layer_norm is not None, "next_layer_norm must be provided"

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

        # === block_sparse_moe ===
        self.block_sparse_moe(hidden_states[:actual_tokens])

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


    def forward_tokenweave(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        symm_mem_hdl: Any,
        layer_id: int,
        end_layer: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        current_stream: Optional[torch.cuda.Stream] = None,
        copy_stream: Optional[torch.cuda.Stream] = None,
        next_layer_norm: Optional[RMSNorm] = None,
        chunk_size: Optional[int] = None,
        actual_tokens: Optional[int] = None,
        nearest_multiple_of_256: Optional[int] = None,
        MAX_CTAS_ATTN: int = 16,
        MAX_CTAS_MLP: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass of a transformer block using TokenWeave overlap strategy.
        Processes two token chunks (interleaved) across GPUs with communication-compute overlap.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated hidden_states and residual tensors.
        """
        assert chunk_size is not None and actual_tokens is not None, "chunk_size and actual_tokens are required"
        assert current_stream is not None and copy_stream is not None, "CUDA streams must be provided"
        assert next_layer_norm is not None, "next_layer_norm must be provided"
        num_bytes_per_token = hidden_states.shape[1] * hidden_states.element_size()
        # Self Attention
        offset_second = chunk_size * hidden_states.shape[1] * hidden_states.element_size()
        if residual is None:
            residual = torch.empty_like(hidden_states)
        # Split hidden states and residuals
        hidden_states_1, hidden_states_2 = hidden_states[:chunk_size], hidden_states[chunk_size:]
        residual_1, residual_2 = residual[:chunk_size], residual[chunk_size:]
        blpr_1 = chunk_size // world_size
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
            # # === Fused all reduce + Pre-Attn Norm + residual add on split-1 ===
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
                positions=positions[:chunk_size],
                hidden_states=hidden_states_1,
                split_id=0,
                chunk_size=chunk_size,
                num_actual_tokens=chunk_size,
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
                positions=positions[chunk_size:],
                hidden_states=hidden_states_2,
                split_id=1,
                chunk_size=chunk_size,
                num_actual_tokens=actual_tokens - chunk_size,
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

        # === block_sparse_moe on split-0 ===
        with torch.cuda.stream(current_stream):
            hidden_states_1 = self.block_sparse_moe(hidden_states_1)
            current_stream.wait_stream(copy_stream)

        # === Fused all reduce + Post-block_sparse_moe Norm + residual add on split-0 ===
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
                MAX_CTAS=MAX_CTAS_ATTN,
                offset_symm_mem=0
            )
        
        # === block_sparse_moe on split-1 ===
        with torch.cuda.stream(current_stream):
            hidden_states_2 = self.block_sparse_moe(hidden_states_2)
            current_stream.wait_stream(copy_stream)    

        # === Fused all reduce + Post-block_sparse_moe Norm + residual add on split-1 (only on last layer) ===
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


@support_torch_compile
class MixtralModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        ## --------- TokenWeave: pq_overlap_fused --------- #
        CHUNK_SIZE = vllm_config.scheduler_config.max_num_batched_tokens + 512
        self.staging_buffer = symm_mem.empty((CHUNK_SIZE, config.hidden_size),
                                          dtype=vllm_config.model_config.dtype,
                                          device="cuda")
        self.symm_mem_hdl = symm_mem.rendezvous(self.staging_buffer, get_device_group())
        self.current_stream = torch.cuda.current_stream()
        self.copy_stream = torch.cuda.Stream(priority=-1)
        self.buff = None

        world_size = get_tensor_model_parallel_world_size()
        self.config_data = load_config(f"tokenweave_configs/mixtral_config_{world_size}.json")
        self.MAX_CTAS_ATTN = 16
        self.MAX_CTAS_MLP = 16
        self.CHUNK_OFFSET = 0

        ## --------- TokenWeave: pq_overlap_fused --------- ##

        self.config = config
        self.quant_config = quant_config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MixtralDecoderLayer(
                config, cache_config, quant_config=quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(
        self, 
        input_ids: torch.Tensor, 
        output_buffer: torch.Tensor, 
        is_tokenweave: Optional[bool] = False, 
        chunk_size: Optional[int] = None) -> torch.Tensor:
        return self.embed_tokens(
            input_ids, 
            output_parallel=output_buffer, 
            use_pytorch_all_reduce=False, 
            is_overlap=is_tokenweave, 
            symm_mem_hdl=self.symm_mem_hdl, 
            chunk_size=chunk_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        rank, world_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        num_tokens = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        is_tokenweave = num_tokens >= 1024 # spliting requires at least 1024 tokens
        tokenweave_chunk_size = None
        if is_tokenweave:
            # Load the tokenweave config based on the number of tokens
            closest_len = min(self.config_data.keys(), key=lambda k: abs(k - num_tokens))
            tokenweave_config = self.config_data[closest_len]
            self.MAX_CTAS_ATTN = tokenweave_config["attention_ctas"]
            self.MAX_CTAS_MLP = tokenweave_config["mlp_ctas"]
            self.CHUNK_OFFSET = tokenweave_config["chunk_offset"]
            tokenweave_chunk_size = (((num_tokens + 255) & ~255) // 2 + self.CHUNK_OFFSET) if is_tokenweave else None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                self.buff = self.staging_buffer[:inputs_embeds.shape[0]]
                hidden_states = inputs_embeds
            else:
                self.buff = self.staging_buffer[:input_ids.shape[0]]
                hidden_states = self.get_input_embeddings(input_ids, self.buff, is_tokenweave, tokenweave_chunk_size)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        if not is_tokenweave: # with fuse only
            nearest_multiple_of_world_size = (num_tokens + world_size - 1) // world_size * world_size
            hidden_states = self.staging_buffer[:nearest_multiple_of_world_size]
            for layer_id in range(self.start_layer, self.end_layer):
                layer = self.layers[layer_id]
                next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
                hidden_states, residual = layer.forward_with_fuse_only(
                    positions, 
                    hidden_states, 
                    residual, 
                    self.symm_mem_hdl, 
                    layer_id,
                    # end_layer is not used in with fuse only flow
                    rank,
                    world_size,
                    # current_stream is not used in with fuse only flow
                    # copy_stream is not used in with fuse only flow
                    next_layer_norm,
                    # tokenweave_chunk_size is not used in with fuse only flow
                    num_tokens,
                    nearest_multiple_of_world_size,
                    self.MAX_CTAS_ATTN,
                    self.MAX_CTAS_MLP,
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })
            return hidden_states[:num_tokens]
        # TokenWeave
        nearest_multiple_of_256 = (num_tokens + 255) & ~255
        hidden_states = self.staging_buffer[:nearest_multiple_of_256]
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_id]
            next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
            hidden_states, residual = layer.forward_tokenweave(positions, 
                                            hidden_states, 
                                            residual, 
                                            self.symm_mem_hdl, 
                                            layer_id, 
                                            self.end_layer, 
                                            rank, 
                                            world_size,
                                            self.current_stream,
                                            self.copy_stream,
                                            next_layer_norm,
                                            tokenweave_chunk_size,
                                            num_tokens,
                                            nearest_multiple_of_256,
                                            self.MAX_CTAS_ATTN,
                                            self.MAX_CTAS_MLP,
                                            )
        return hidden_states[:num_tokens]
    
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if ((name.endswith(".bias") or name.endswith("_bias"))
                        and name not in params_dict):
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MixtralForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = MixtralModel(vllm_config=vllm_config,
                                  prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["rotary_emb.inv_freq"])
        return loader.load_weights(weights)
