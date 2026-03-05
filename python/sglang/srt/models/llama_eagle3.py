"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from sglang.srt.utils import add_prefix

# Adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py
"""Inference-only LLaMA-EAGLE model compatible with HuggingFace weights."""

import copy
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

logger = logging.getLogger(__name__)

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP


class LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix)

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )

        if config.model_type == "llama4_text":
            inter_size = config.intermediate_size_mlp
        else:
            inter_size = config.intermediate_size

        self.mlp = LlamaMLP(
            config.hidden_size, inter_size, config.hidden_act, quant_config, prefix
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.is_mrope_enabled = (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and "mrope_section" in config.rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            config.rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        self.midlayer = LlamaDecoderLayer(config, 0, quant_config, prefix)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Multi-expert state (populated by register_experts)
        self._has_experts = False
        self._num_experts = 0
        self._profile_names: List[str] = []
        self._default_profile_idx: int = 0
        self.fc_experts: Optional[nn.ModuleList] = None
        self.midlayer_experts: Optional[nn.ModuleList] = None
        self.norm_experts: Optional[nn.ModuleList] = None

    def register_experts(
        self,
        profile_names: List[str],
        default_profile_idx: int,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str = "",
    ):
        """Create N expert copies of fc, midlayer, norm for MoE-style parallel routing.

        Each midlayer expert uses a different layer_id (0, 1, 2, ...)
        so they have independent KV caches.

        The default expert (at default_profile_idx) shares weights with
        self.fc / self.midlayer / self.norm (layer_id=0).
        Other experts are new instances with different layer_ids.
        """
        n = len(profile_names)
        if n <= 1:
            return

        self._has_experts = True
        self._num_experts = n
        self._profile_names = profile_names
        self._default_profile_idx = default_profile_idx

        fc_list: List[nn.Module] = []
        mid_list: List[nn.Module] = []
        norm_list: List[nn.Module] = []

        for i in range(n):
            if i == default_profile_idx:
                # Reuse the existing modules (layer_id=0)
                fc_list.append(self.fc)
                mid_list.append(self.midlayer)
                norm_list.append(self.norm)
            else:
                # Create new instances with unique layer_id
                new_fc = torch.nn.Linear(
                    self.hidden_size_in * 3,
                    config.hidden_size,
                    bias=getattr(config, "bias", False),
                ).to(device=next(self.fc.parameters()).device,
                     dtype=next(self.fc.parameters()).dtype)

                new_mid = LlamaDecoderLayer(
                    config, layer_id=i, quant_config=quant_config, prefix=prefix
                ).to(device=next(self.midlayer.parameters()).device,
                     dtype=next(self.midlayer.parameters()).dtype)

                new_norm = RMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                ).to(device=next(self.norm.parameters()).device,
                     dtype=next(self.norm.parameters()).dtype)

                fc_list.append(new_fc)
                mid_list.append(new_mid)
                norm_list.append(new_norm)

        self.fc_experts = nn.ModuleList(fc_list)
        self.midlayer_experts = nn.ModuleList(mid_list)
        self.norm_experts = nn.ModuleList(norm_list)

        logger.info(
            f"Registered {n} MoE experts for profiles: {profile_names}, "
            f"default_idx={default_profile_idx}"
        )

    def _get_profile_ids(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        """Extract per-token profile IDs from forward_batch.

        In EAGLE3 draft decode, each token in the batch corresponds to a request.
        The profile is determined by the request's custom_params['eagle_draft_profile'].

        Returns:
            A tensor of shape [num_tokens] with integer profile indices,
            or None if multi-expert is not active.
        """
        if not self._has_experts:
            return None

        custom_params_list = None
        if (
            forward_batch.sampling_info is not None
            and forward_batch.sampling_info.custom_params is not None
        ):
            custom_params_list = forward_batch.sampling_info.custom_params

        num_tokens = forward_batch.input_ids.shape[0]

        if custom_params_list is None:
            # All tokens use default profile
            return torch.full(
                (num_tokens,), self._default_profile_idx,
                dtype=torch.long, device=forward_batch.input_ids.device
            )

        # Build per-request profile index
        # In draft decode, batch_size = num_reqs, and num_tokens = batch_size * topk
        # custom_params_list has one entry per request
        num_reqs = len(custom_params_list)
        profile_name_to_idx = {name: i for i, name in enumerate(self._profile_names)}

        req_profile_ids = []
        for cp in custom_params_list:
            if isinstance(cp, dict):
                p = cp.get("eagle_draft_profile")
                if p is not None and p in profile_name_to_idx:
                    req_profile_ids.append(profile_name_to_idx[p])
                else:
                    req_profile_ids.append(self._default_profile_idx)
            else:
                req_profile_ids.append(self._default_profile_idx)

        req_profile_tensor = torch.tensor(
            req_profile_ids, dtype=torch.long,
            device=forward_batch.input_ids.device
        )

        # Expand from per-request to per-token
        # In draft decode, tokens are organized as: [req0_tok0, req0_tok1, ..., req1_tok0, ...]
        # with topk tokens per request. The pattern depends on how tokens are laid out.
        # In EAGLE3 draft, topk tokens per request are interleaved:
        # after select_top_k_tokens, shape is [batch_size * topk].
        # The repeat_interleave in eagle_worker maps positions as:
        #   positions = seq_lens.repeat_interleave(topk)
        # So tokens are grouped by request.
        if num_tokens == num_reqs:
            # 1:1 mapping (e.g. draft extend or prefill)
            return req_profile_tensor
        elif num_tokens > num_reqs and num_tokens % num_reqs == 0:
            # topk tokens per request
            topk = num_tokens // num_reqs
            return req_profile_tensor.repeat_interleave(topk)
        else:
            # Fallback: use default for all
            return torch.full(
                (num_tokens,), self._default_profile_idx,
                dtype=torch.long, device=forward_batch.input_ids.device
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = forward_batch.spec_info.hidden_states

        # idle batch
        if hidden_states.shape[0] == 0:
            if hidden_states.shape[-1] != embeds.shape[-1]:
                hidden_states = self.fc(hidden_states)
            return hidden_states, [hidden_states]

        # ----- Multi-expert MoE-style parallel forward -----
        if self._has_experts:
            return self._forward_moe(
                input_ids, positions, forward_batch, embeds, hidden_states
            )

        # ----- Single-profile (original) forward -----
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]

    def _forward_moe(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """MoE-style forward: each expert processes ALL tokens, then we
        select per-token outputs based on profile routing.

        Each expert's midlayer has a unique layer_id, so KV caches are independent.
        The "wasted" compute (processing tokens belonging to other profiles) is
        acceptable because EAGLE3 has only 1 decoder layer.
        """
        profile_ids = self._get_profile_ids(forward_batch)  # [num_tokens]
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device

        # If all tokens belong to one profile, fast-path: just use that expert
        if profile_ids is not None:
            unique_profiles = profile_ids.unique()
            if len(unique_profiles) == 1:
                idx = unique_profiles.item()
                return self._forward_single_expert(
                    idx, positions, embeds, hidden_states, forward_batch
                )

        # Need fc first (since it may change hidden_size)
        need_fc = hidden_states.shape[-1] != embeds.shape[-1]

        # Allocate output buffers
        # We'll fill them from each expert
        hidden_dim = self.config.hidden_size
        hs_logits_out = torch.empty(
            num_tokens, hidden_dim, device=device, dtype=embeds.dtype
        )
        hs_aux_out = torch.empty(
            num_tokens, hidden_dim, device=device, dtype=embeds.dtype
        )

        # Process each expert on ALL tokens, but only keep the relevant outputs
        for expert_idx in range(self._num_experts):
            # Check if any token needs this expert
            mask = profile_ids == expert_idx
            if not mask.any():
                continue

            # FC
            if need_fc:
                hs_expert = self.fc_experts[expert_idx](hidden_states)
            else:
                hs_expert = hidden_states

            # Midlayer (attention + MLP) — processes ALL tokens
            # Each expert has its own layer_id, so KV cache is independent
            residual = None
            hs_expert, residual = self.midlayer_experts[expert_idx](
                positions,
                embeds,
                hs_expert,
                forward_batch,
                residual,
            )

            # Norm
            hs_to_logits, hs_to_aux = self.norm_experts[expert_idx](
                hs_expert, residual
            )

            # Only keep outputs for tokens belonging to this expert
            hs_logits_out[mask] = hs_to_logits[mask]
            hs_aux_out[mask] = hs_to_aux[mask]

        return hs_logits_out, [hs_aux_out]

    def _forward_single_expert(
        self,
        expert_idx: int,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Fast path: all tokens use the same expert."""
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc_experts[expert_idx](hidden_states)

        residual = None
        hidden_states, residual = self.midlayer_experts[expert_idx](
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hs_to_logits, hs_to_aux = self.norm_experts[expert_idx](
            hidden_states, residual
        )

        return hs_to_logits, [hs_to_aux]


class LlamaForCausalLMEagle3(LlamaForCausalLM):
    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        # EAGLE3 draft model has exactly 1 hidden layer (midlayer).
        # Multi-expert KV cache layers are extended dynamically after pool creation.
        if self.config.num_hidden_layers != 1:
            raise ValueError(
                f"EAGLE3 requires exactly 1 hidden layer, got {self.config.num_hidden_layers}"
            )

        self.model = LlamaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = (
            config_.draft_vocab_size
        )  # draft logits processor has it's own vocab size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

        # Multi-expert state
        self._has_experts = False
        self._num_experts = 0
        self._profile_names: List[str] = []
        self._default_profile_idx: int = 0
        self.lm_head_experts: Optional[nn.ModuleList] = None

    # ------------------------------------------------------------------ #
    #  Multi-expert registration
    # ------------------------------------------------------------------ #

    def register_experts(
        self,
        profile_names: List[str],
        default_profile_idx: int,
    ):
        """Register multiple lm_head experts for MoE-style routing.

        Also delegates to self.model.register_experts() for fc/midlayer/norm.
        The lm_head experts are created here since lm_head is on the CausalLM level.

        The default expert reuses self.lm_head.
        """
        n = len(profile_names)
        if n <= 1:
            return

        self._has_experts = True
        self._num_experts = n
        self._profile_names = profile_names
        self._default_profile_idx = default_profile_idx

        # Create lm_head experts
        lm_head_list: List[nn.Module] = []
        for i in range(n):
            if i == default_profile_idx:
                lm_head_list.append(self.lm_head)
            else:
                if self.config.tie_word_embeddings:
                    lm_head_list.append(self.model.embed_tokens)
                else:
                    new_lm_head = ParallelLMHead(
                        self.config.draft_vocab_size,
                        self.config.hidden_size,
                        quant_config=self.quant_config,
                    ).to(device=next(self.lm_head.parameters()).device,
                         dtype=next(self.lm_head.parameters()).dtype)
                    lm_head_list.append(new_lm_head)

        self.lm_head_experts = nn.ModuleList(lm_head_list)

        # Register experts in the inner model (fc, midlayer, norm)
        self.model.register_experts(
            profile_names=profile_names,
            default_profile_idx=default_profile_idx,
            config=self.config,
            quant_config=self.quant_config,
        )

        logger.info(
            f"LlamaForCausalLMEagle3 registered {n} experts: {profile_names}"
        )

    def _extract_own_profile_weights(self) -> Dict[str, torch.Tensor]:
        """Extract (clone) the current model weights that are profile-specific."""
        weights: Dict[str, torch.Tensor] = {}
        for name, param in self.model.fc.named_parameters():
            weights[f"fc.{name}"] = param.data.clone()
        for name, param in self.model.midlayer.named_parameters():
            weights[f"midlayer.{name}"] = param.data.clone()
        for name, param in self.model.norm.named_parameters():
            weights[f"norm.{name}"] = param.data.clone()
        for name, param in self.lm_head.named_parameters():
            weights[f"lm_head.{name}"] = param.data.clone()
        return weights

    def load_expert_weights(
        self,
        expert_idx: int,
        weights: Dict[str, torch.Tensor],
    ):
        """Load weights into a specific expert (fc, midlayer, norm, lm_head).

        Args:
            expert_idx: which expert to load into
            weights: dict of param_name -> tensor (keys like fc.weight, midlayer.*, etc.)
        """
        if not self._has_experts:
            raise RuntimeError("Experts not registered yet")

        # Load fc expert weights
        for name, param in self.model.fc_experts[expert_idx].named_parameters():
            key = f"fc.{name}"
            if key in weights:
                param.data.copy_(weights[key])

        # Load midlayer expert weights (need to handle different layer_id)
        for name, param in self.model.midlayer_experts[expert_idx].named_parameters():
            key = f"midlayer.{name}"
            if key in weights:
                param.data.copy_(weights[key])

        # Load norm expert weights
        for name, param in self.model.norm_experts[expert_idx].named_parameters():
            key = f"norm.{name}"
            if key in weights:
                param.data.copy_(weights[key])

        # Load lm_head expert weights
        if self.lm_head_experts is not None:
            for name, param in self.lm_head_experts[expert_idx].named_parameters():
                key = f"lm_head.{name}"
                if key in weights:
                    param.data.copy_(weights[key])

    # ------------------------------------------------------------------ #
    #  Override forward for MoE routing
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional["PPProxyTensors"] = None,
    ):
        if not self._has_experts:
            # Original single-profile forward
            return super().forward(
                input_ids,
                positions,
                forward_batch,
                input_embeds,
                get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )

        # MoE-style forward with multiple experts
        # self.model.forward() handles fc/midlayer/norm routing internally
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        if self.pp_group.is_last_rank:
            if not get_embedding:
                # MoE logits: route to different lm_heads per token
                return self._moe_logits(
                    input_ids, hidden_states, forward_batch, aux_hidden_states
                )
            else:
                return self.pooler(hidden_states, forward_batch)
        else:
            return hidden_states

    def _moe_logits(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        aux_hidden_states: Optional[List[torch.Tensor]],
    ) -> LogitsProcessorOutput:
        """Compute logits using per-token expert lm_heads.

        If all tokens share the same profile, use the fast path.
        Otherwise, compute logits for each expert and scatter.
        """
        profile_ids = self.model._get_profile_ids(forward_batch)

        if profile_ids is not None:
            unique_profiles = profile_ids.unique()
            if len(unique_profiles) == 1:
                # Fast path: single expert
                idx = unique_profiles.item()
                return self.logits_processor(
                    input_ids, hidden_states, self.lm_head_experts[idx],
                    forward_batch, aux_hidden_states,
                )

        # Mixed profiles: compute logits per expert and merge
        # This is trickier because logits_processor returns a LogitsProcessorOutput
        # We need to run the full pipeline for all tokens, then overwrite per-expert
        #
        # Strategy: run logits_processor with default lm_head first to get the output
        # structure, then overwrite next_token_logits per-expert.
        result = self.logits_processor(
            input_ids, hidden_states, self.lm_head_experts[self._default_profile_idx],
            forward_batch, aux_hidden_states,
        )

        # Now overwrite logits for non-default experts
        for expert_idx in range(self._num_experts):
            if expert_idx == self._default_profile_idx:
                continue
            mask = profile_ids == expert_idx
            if not mask.any():
                continue

            # Compute logits for this expert's tokens
            expert_result = self.logits_processor(
                input_ids, hidden_states, self.lm_head_experts[expert_idx],
                forward_batch, aux_hidden_states,
            )
            result.next_token_logits[mask] = expert_result.next_token_logits[mask]

        return result

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue

            if "t2d" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id


EntryClass = [LlamaForCausalLMEagle3]
