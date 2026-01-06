import torch
from torch import nn
from typing import Optional, Tuple
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.model_executor.models.hunyuan_v1 import (HunYuanAttention,
                                                    HunYuanModel,
                                                    HunyuanV1ModelBase)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.logits_processor import LogitsProcessor

class HpuHunYuanAttention(HunYuanAttention):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_states: Optional[tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        ori_k = k
        if self.use_qk_norm:
            q_by_head = self.query_layernorm(
                q.view(-1, self.num_heads, self.head_dim).contiguous())
            k_by_head = self.key_layernorm(
                k.view(-1, self.num_kv_heads, self.head_dim).contiguous())

            q = q_by_head.reshape(q.shape)
            k = k_by_head.reshape(k.shape)
        attn_output = self.attn(q, k, v)
        # For o_proj
        output, _ = self.o_proj(attn_output)
        return output, (ori_k, v)

class HpuHunYuanModel(HunYuanModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        for layer in self.layers:
            if isinstance(layer.self_attn, HunYuanAttention) and not isinstance(layer.self_attn, HpuHunYuanAttention):
                layer.self_attn.__class__ = HpuHunYuanAttention

class HpuHunyuanV1ModelBase(HunyuanV1ModelBase):
    packed_modules_mapping = HunyuanV1ModelBase.packed_modules_mapping
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = HpuHunYuanModel(vllm_config=vllm_config, prefix="model")
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()
            self.logits_processor = None 

class HpuHunYuanDenseV1Base(HpuHunyuanV1ModelBase):
    pass

class HpuHunYuanDenseV1ForCausalLM(HpuHunYuanDenseV1Base):
    pass