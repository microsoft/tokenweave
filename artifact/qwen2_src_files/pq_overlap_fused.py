# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, Optional, Set, Tuple, Union, Any

import torch
from torch import nn
import os
import torch.distributed._symmetric_memory as symm_mem
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank, get_device_group,
                              get_tensor_model_parallel_world_size,
                              pytorch_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, PoolerOutput
from vllm.distributed.triton_comm.triton_comm import (
    multimem_all_reduce, multimem_reduce_scatter, multimem_all_gather, multimem_all_gather_async)
from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, WeightsMapper,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

logger = init_logger(__name__)
import json
from functools import lru_cache

@lru_cache(maxsize=None)
def load_config(config_path="tokenweave_configs/qwen2_config_8.json"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.normpath(os.path.join(base_dir, "..", "..", config_path))
    with open(full_path, "r") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward_default(self, hidden_states):
        x, _ = self.gate_up_proj(hidden_states)
        x = self.act_fn(x)
        self.down_proj(x, hidden_states, is_tokenweave=True)
        return hidden_states

    def forward(self, hidden_states):
        x, _ = self.gate_up_proj(hidden_states)
        x = self.act_fn(x)
        self.down_proj(x, hidden_states, is_tokenweave=True)
        return hidden_states


class Qwen2Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[Tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
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
            base=self.rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)

    def forward_default(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        self.o_proj(attn_output, hidden_states,
                                is_tokenweave=True)
        return hidden_states
    
    def forward_split1(
        self,
        positions_1: torch.Tensor,
        hidden_states_1: torch.Tensor,
    ):
        qkv1, _ = self.qkv_proj(hidden_states_1)
        q1, k1, v1 = qkv1.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q1, k1 = self.rotary_emb(positions_1, q1, k1)
        attn_output = self.attn(q1, k1, v1)
        self.o_proj(attn_output, hidden_states_1,
                                is_tokenweave=True)
        return hidden_states_1

    def forward_split2(
        self,
        positions_2: torch.Tensor,
        hidden_states_2: torch.Tensor,
    ):
        qkv2, _ = self.qkv_proj(hidden_states_2)
        q2, k2, v2 = qkv2.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q2, k2 = self.rotary_emb(positions_2, q2, k2)
        attn_output = self.attn(q2, k2, v2)
        self.o_proj(attn_output, hidden_states_2,
                                is_tokenweave=True)
        return hidden_states_2  
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        split_id: Optional[int] = None,
        split_size: Optional[int] = None,
        num_actual_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        self.rotary_emb(positions, q[:num_actual_tokens], k[:num_actual_tokens])
        attn_output = self.attn(q, k, v, split_id, split_size)
        self.o_proj(attn_output, hidden_states,
                                is_tokenweave=True)
        return hidden_states

class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward_default(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        symm_mem_hdl: Any,
        layer_id: int,
        rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
        next_layer_norm: Optional[RMSNorm] = None,
        actual_tokens: Optional[int] = None,
        num_tokens_padded: Optional[int] = None,
        MAX_CTAS_ATTN: Optional[int] = 16,
        MAX_CTAS_MLP: Optional[int] = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens_per_rank = num_tokens_padded // world_size
        # Self Attention
        if residual is None:
            residual = torch.empty_like(hidden_states)
        if layer_id == 0: # First layer
            self.input_layernorm(hidden_states, out=residual)

        self.self_attn.forward_default(positions=positions,
                                       hidden_states=hidden_states[:actual_tokens])
        # Fused_RS_LN_AG
        self.post_attention_layernorm(
            hidden_states[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank], 
            residual[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank],
            MAX_CTAS=min(MAX_CTAS_ATTN, num_tokens_per_rank),
            fused_ar=True,
            symm_mem_hdl=symm_mem_hdl,
            rank=rank,
            world_size=world_size,
            offset=rank * num_tokens_per_rank * hidden_states.shape[1] * hidden_states.element_size(),
        )
        self.mlp.forward_default(hidden_states[:actual_tokens])
        next_layer_norm(
            hidden_states[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank], 
            residual[rank * num_tokens_per_rank: (rank + 1) * num_tokens_per_rank],
            MAX_CTAS=min(MAX_CTAS_MLP, num_tokens_per_rank),
            fused_ar=True,
            symm_mem_hdl=symm_mem_hdl,
            rank=rank,
            world_size=world_size,
            offset=rank * num_tokens_per_rank * hidden_states.shape[1] * hidden_states.element_size(),
        )

        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        symm_mem_hdl: Any,
        layer_id: int,
        end_layer: Optional[int] = None,
        rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
        current_stream: Optional[torch.cuda.Stream] = None,
        copy_stream: Optional[torch.cuda.Stream] = None,
        next_layer_norm: Optional[RMSNorm] = None,
        split_size: Optional[int] = None,
        actual_tokens: Optional[int] = None,
        num_tokens_padded: Optional[int] = None,
        MAX_CTAS_ATTN: Optional[int] = 16,
        MAX_CTAS_MLP: Optional[int] = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        offset_second = split_size * hidden_states.shape[1] * hidden_states.element_size()
        if residual is None:
            residual = torch.empty_like(hidden_states)
        hidden_states_1, hidden_states_2 = hidden_states[:split_size], hidden_states[split_size:]
        residual_1, residual_2 = residual[:split_size], residual[split_size:]
        blpr_1, blpr_2 = split_size // world_size, hidden_states_2.shape[0] // world_size # bl_per_rank

        # Attention Block
        if layer_id == 0: # First layer
            hidden_states_1 = self.input_layernorm(hidden_states_1, out=residual_1) 
            multimem_reduce_scatter(
                hidden_states_2,
                symm_mem_hdl,
                offset_second,
                MAX_CTAS=8
            )
            self.input_layernorm(hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], out=residual_2[rank * blpr_2: (rank + 1) * blpr_2])
            symm_mem_hdl.barrier(channel=7)
            multimem_all_gather_async(
                hidden_states_2,
                symm_mem_hdl,
                offset_second,
                blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(), # nbytes_per_rank
                current_stream,
            )
            symm_mem_hdl.barrier(channel=9)
        else:
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_stream(current_stream)
                self.input_layernorm(
                    hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], 
                    residual_2[rank * blpr_2: (rank + 1) * blpr_2],
                    MAX_CTAS=MAX_CTAS_ATTN,
                    fused_ar=True,
                    symm_mem_hdl=symm_mem_hdl,
                    rank=rank,
                    world_size=world_size,
                    offset=offset_second +  rank * blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(),
                )
        with torch.cuda.stream(current_stream):
            hidden_states_1 = self.self_attn(
                positions=positions[:split_size],
                hidden_states=hidden_states_1,
                split_id=0,
                split_size=split_size,
                num_actual_tokens=split_size,
            )
            current_stream.wait_stream(copy_stream)

        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(current_stream)
            self.post_attention_layernorm(
                hidden_states_1[rank * blpr_1: (rank + 1) * blpr_1], 
                residual_1[rank * blpr_1: (rank + 1) * blpr_1],
                MAX_CTAS=MAX_CTAS_ATTN,
                fused_ar=True,
                symm_mem_hdl=symm_mem_hdl,
                rank=rank,
                world_size=world_size,
                offset=rank * blpr_1 * hidden_states_1.shape[1] * hidden_states_1.element_size(),
            )
        with torch.cuda.stream(current_stream):
            hidden_states_2 = self.self_attn(
                positions=positions[split_size:],
                hidden_states=hidden_states_2,
                split_id=1,
                split_size=split_size,
                num_actual_tokens=actual_tokens - split_size,
            )
            current_stream.wait_stream(copy_stream)
        
        # MLP Block
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(current_stream)
            self.post_attention_layernorm(
                hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], 
                residual_2[rank * blpr_2: (rank + 1) * blpr_2],
                MAX_CTAS=MAX_CTAS_MLP,
                fused_ar=True,
                symm_mem_hdl=symm_mem_hdl,
                rank=rank,
                world_size=world_size,
                offset=offset_second +  rank * blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(),
            )
        
        with torch.cuda.stream(current_stream):
            hidden_states_1 = self.mlp(hidden_states_1)
            current_stream.wait_stream(copy_stream)

        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(current_stream)
            next_layer_norm(
                hidden_states_1[rank * blpr_1: (rank + 1) * blpr_1], 
                residual_1[rank * blpr_1: (rank + 1) * blpr_1],
                MAX_CTAS=MAX_CTAS_MLP,
                fused_ar=True,
                symm_mem_hdl=symm_mem_hdl,
                rank=rank,
                world_size=world_size,
                offset=rank * blpr_1 * hidden_states_1.shape[1] * hidden_states_1.element_size(),
            )
        with torch.cuda.stream(current_stream):
            hidden_states_2 = self.mlp(hidden_states_2)
            current_stream.wait_stream(copy_stream)        
        if layer_id == end_layer - 1:
            next_layer_norm(
                hidden_states_2[rank * blpr_2: (rank + 1) * blpr_2], 
                residual_2[rank * blpr_2: (rank + 1) * blpr_2],
                MAX_CTAS=16 if actual_tokens < 16384 else 32,
                fused_ar=True,
                symm_mem_hdl=symm_mem_hdl,
                rank=rank,
                world_size=world_size,
                offset=offset_second +  rank * blpr_2 * hidden_states_2.shape[1] * hidden_states_2.element_size(),
            )
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen2Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                             "supported. This model uses sliding window "
                             "but `max_window_layers` = {} is less than "
                             "`num_hidden_layers` = {}. Please open an issue "
                             "to discuss this feature.".format(
                                 config.max_window_layers,
                                 config.num_hidden_layers,
                             ))

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
        self.config_data = load_config(f"tokenweave_configs/qwen2_config_{world_size}.json")
        self.MAX_CTAS_ATTN = 16
        self.MAX_CTAS_MLP = 16
        self.SPLIT_OFFSET = 0

        ## --------- TokenWeave: pq_overlap_fused --------- ##
        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor, output_buffer: torch.Tensor, is_tokenweave: Optional[bool] = False, split_size: Optional[int] = None) -> torch.Tensor:
        return self.embed_tokens(input_ids, output_parallel=output_buffer, use_pytorch_all_reduce=False, is_overlap=is_tokenweave, symm_mem_hdl=self.symm_mem_hdl, split_size=split_size)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        rank, world_size = get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
        num_tokens = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        is_tokenweave = num_tokens >= 1024 
        tokenweave_split_size = None
        if is_tokenweave:
            # Load the tokenweave config based on the number of tokens
            closest_len = min(self.config_data.keys(), key=lambda k: abs(k - num_tokens))
            tokenweave_config = self.config_data[closest_len]
            self.MAX_CTAS_ATTN = tokenweave_config["attention_ctas"]
            self.MAX_CTAS_MLP = tokenweave_config["mlp_ctas"]
            self.SPLIT_OFFSET = tokenweave_config["split_offset"]
            tokenweave_split_size = (((num_tokens + 255) & ~255) // 2 + self.SPLIT_OFFSET) if is_tokenweave else None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                self.buff = self.staging_buffer[:inputs_embeds.shape[0]]
                hidden_states = inputs_embeds
            else:
                self.buff = self.staging_buffer[:input_ids.shape[0]]
                hidden_states = self.get_input_embeddings(input_ids, self.buff, is_tokenweave, tokenweave_split_size)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        if not is_tokenweave: # default
            num_tokens_padded = (num_tokens + world_size - 1) // world_size * world_size
            hidden_states = self.staging_buffer[:num_tokens_padded]
            for layer_id in range(self.start_layer, self.end_layer):
                layer = self.layers[layer_id]
                next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
                hidden_states, residual = layer.forward_default(
                    positions, 
                    hidden_states, 
                    residual, 
                    self.symm_mem_hdl, 
                    layer_id,
                    # end_layer is not used in default flow
                    rank,
                    world_size,
                    # current_stream is not used in default flow
                    # copy_stream is not used in default flow
                    next_layer_norm,
                    # tokenweave_split_size is not used in default flow
                    num_tokens,
                    num_tokens_padded,
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
        num_tokens_padded = (num_tokens + 255) & ~255
        hidden_states = self.staging_buffer[:num_tokens_padded]
        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_id]
            next_layer_norm = self.layers[layer_id + 1].input_layernorm if layer_id < self.end_layer - 1 else self.norm
            hidden_states, residual = layer(positions, 
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
                                            tokenweave_split_size,
                                            num_tokens,
                                            num_tokens_padded,
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
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
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
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
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
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


class Qwen2EmbeddingModel(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.": ""})

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # TODO: Replace this model class with as_embedding_model(
        # Qwen2ForCausalLM) after changing the default pooling method
        if pooler_config.pooling_type is None:
            logger.warning(
                "This embedding model will default to last-token pooling in "
                "an upcoming version. To avoid breaking changes, you should "
                "pass `--override-pooler-config '{\"pooling_type\": \"MEAN\"}'`"
                " explicitly.")

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.MEAN,
            normalize=True,
            softmax=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        weights = ((name, data) for name, data in weights
                   if not name.startswith("lm_head."))
        self.model.load_weights(weights)
