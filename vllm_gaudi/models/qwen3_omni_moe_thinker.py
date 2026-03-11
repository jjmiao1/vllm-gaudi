from functools import partial
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3Omni_VisionTransformer,
    Qwen3_VisionMLP,
    Qwen3_VisionPatchEmbed,
    Qwen3_VisionPatchMerger,
    SinusoidsPositionEmbedding,
    _get_feat_extract_output_lengths,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm_gaudi.models.qwen2_5_vl import HPUQwen2_5_VisionAttention
from vllm_gaudi.models.qwen3_moe import upgrade_qwen3_moe_blocks_inplace

logger = init_logger(__name__)

class HpuQwen3_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = HPUQwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        seqlens: Optional[list[int]] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            attn_mask=attn_mask,
        )

        x = x + self.mlp(self.norm2(x))
        return x

class HpuQwen3Omni_VisionTransformer(Qwen3Omni_VisionTransformer):
    def __init__(
        self,
        vision_config,
        norm_eps: float = 1e-6,
        quant_config: Optional["QuantizationConfig"] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(vision_config, norm_eps, quant_config, prefix)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.blocks = nn.ModuleList([
            HpuQwen3_VisionBlock(
                dim=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_dim=vision_config.intermediate_size,
                act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                norm_layer=norm_layer,
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
            ) for layer_idx in range(vision_config.depth)
        ])

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        if self.apply_vit_abs_pos_embed:
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds
            
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                dim=0,
                dtype=grid_thw.dtype
                if torch.jit.is_tracing() else torch.int32,
            )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        hidden_states = hidden_states.unsqueeze(1)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(hidden_states.device)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(hidden_states.device)
        
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        hidden_states_list = []
        deepstack_visual_indexes = self.deepstack_visual_indexes

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
            )
            
            if (deepstack_visual_indexes is not None
                    and layer_num in deepstack_visual_indexes):
                hidden_states_list.append(hidden_states)

        hidden_states = self.merger(hidden_states)

        if deepstack_visual_indexes is not None:
            processed_hidden_states_list = [hidden_states]
            for idx, hidden_x in enumerate(hidden_states_list):
                hidden_x = self.merger_list[idx](hidden_x)
                processed_hidden_states_list.append(hidden_x)
            
            hidden_states = torch.cat(
                processed_hidden_states_list,
                dim=1)

        return hidden_states


class Qwen3Omni_VisionTransformerStaticShape(HpuQwen3Omni_VisionTransformer):
    """Static-shape wrapper for HPU graph capture."""

    def pad_multimodal_data(self, pixel_values, vision_buckets, constant_value=0):
        desired_number_of_pixels = vision_buckets.get_multimodal_bucket(
            pixel_values.shape[0]
        )
        padding_len = desired_number_of_pixels - pixel_values.shape[0]
        if padding_len <= 0:
            return pixel_values

        logger.debug(
            "Padding current number pixel %d to %d",
            pixel_values.shape[0],
            desired_number_of_pixels,
        )
        pixel_values = F.pad(
            pixel_values, (0, 0, 0, padding_len), "constant", constant_value
        )
        return pixel_values

    def pre_attn(self, x, grid_thw, vision_buckets):
        hidden_states = x.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(hidden_states)

        if self.apply_vit_abs_pos_embed:
            pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
            hidden_states = hidden_states + pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(device=self.device)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(device=self.device)
        attention_mask = torch.ones(hidden_states.size(0), 1).to(
            device=self.device
        )

        hidden_states = self.pad_multimodal_data(
            hidden_states, vision_buckets, 0
        )
        rotary_pos_emb_cos = self.pad_multimodal_data(
            rotary_pos_emb_cos, vision_buckets, 0
        )
        rotary_pos_emb_sin = self.pad_multimodal_data(
            rotary_pos_emb_sin, vision_buckets, 0
        )
        attention_mask = self.pad_multimodal_data(
            attention_mask, vision_buckets, 0
        )

        return hidden_states, rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb_cos: Optional[torch.Tensor] = None,
        rotary_pos_emb_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = x.unsqueeze(1)
        rotary_pos_emb_cos = rotary_pos_emb_cos.to(hidden_states.device)
        rotary_pos_emb_sin = rotary_pos_emb_sin.to(hidden_states.device)
        hidden_states_list = []
        deepstack_visual_indexes = self.deepstack_visual_indexes

        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                attn_mask=attn_mask,
                cu_seqlens=None,
                max_seqlen=None,
            )
            if (
                deepstack_visual_indexes is not None
                and layer_num in deepstack_visual_indexes
            ):
                hidden_states_list.append(hidden_states)

        hidden_states = self.merger(hidden_states)

        # Processing deepstack
        if deepstack_visual_indexes is not None:
            processed_hidden_states_list = [hidden_states]
            for idx, x in enumerate(hidden_states_list):
                x = self.merger_list[idx](x)
                processed_hidden_states_list.append(x)
            hidden_states = torch.cat(processed_hidden_states_list, dim=1)

        return hidden_states

    def get_image_embeds(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        vision_buckets,
    ) -> torch.Tensor:
        import habana_frameworks.torch as htorch
        import habana_frameworks.torch.core as htcore

        offset = 0
        results = []

        for img_idx in range(grid_thw.shape[0]):
            img_shape = grid_thw[img_idx, :].unsqueeze(0).clone()
            grid_t = grid_thw[img_idx, 0]
            img_shape[0, 0] = 1
            curr_img_size = img_shape.prod()

            for _ in torch.arange(0, grid_t):
                pixel_values_curr_img = pixel_values[
                    offset : offset + curr_img_size, :
                ]
                offset += curr_img_size

                (pixel_values_curr_img_padded,
                    rot_pos_emb_cos,
                    rot_pos_emb_sin,
                    attention_mask,) = self.pre_attn(
                    pixel_values_curr_img, img_shape, vision_buckets)

                fullatt_block_attn_mask = (
                    attention_mask.squeeze(1).unsqueeze(0) * attention_mask
                )

                extra_forward_kwargs = {}
                if htorch.utils.internal.is_lazy():
                    padded_len = pixel_values_curr_img_padded.shape[0]
                    use_graph = vision_buckets.use_graph(padded_len)
                    extra_forward_kwargs.update(
                        {"bypass_hpu_graphs": not use_graph}
                    )

                htcore.mark_step()
                hidden_states = self.forward(
                    pixel_values_curr_img_padded,
                    rotary_pos_emb_cos=rot_pos_emb_cos,
                    rotary_pos_emb_sin=rot_pos_emb_sin,
                    attn_mask=fullatt_block_attn_mask,
                    **extra_forward_kwargs,
                )
                htcore.mark_step()

                post_embed_size = curr_img_size // self.spatial_merge_unit
                results.append(hidden_states[:post_embed_size, :].clone())

        return torch.cat(results)



class HpuQwen3OmniMoeThinkerForConditionalGeneration(
    Qwen3OmniMoeThinkerForConditionalGeneration,
):
    """HPU-enabled Qwen3-Omni-Moe thinker."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        thinker_config = vllm_config.model_config.hf_config.thinker_config
        quant_config = vllm_config.quant_config

        upgraded_count = upgrade_qwen3_moe_blocks_inplace(self.language_model)
        logger.info(f"Successfully upgraded {upgraded_count} Qwen3 MoE layers for HPU 3D shape support.")

        # self.audio_tower = Qwen3OmniMoeAudioEncoderStaticShape(
        #     thinker_config.audio_config
        # )

        self.visual = Qwen3Omni_VisionTransformerStaticShape(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "visual"),
        )

    def _process_image_input(self, image_input):
        if image_input["type"] == "image_embeds":
            return image_input["image_embeds"].type(self.visual.dtype)

        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values = image_input["pixel_values"].type(self.visual.dtype)
        image_embeds = self.visual.get_image_embeds(
            pixel_values,
            grid_thw=grid_thw,
            vision_buckets=self.vision_bucket_manager,
        )
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return image_embeds.split(sizes.tolist())

    def _process_video_input(
        self, video_input, video_hashes=None, cached_video_embeds=None
    ):
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype
        )
        video_embeds = self.visual.get_image_embeds(
            pixel_values_videos,
            grid_thw=grid_thw,
            vision_buckets=self.vision_bucket_manager,
        )
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return video_embeds.split(sizes.tolist())

    def _process_audio_input(
        self, audio_input, audio_hashes=None, cached_audio_features=None
    ):
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]

        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)

        if not isinstance(audio_feature_lengths, torch.Tensor):
            audio_feature_lengths = torch.cat(audio_feature_lengths)
        if audio_feature_lengths.ndim == 2:
            audio_feature_lengths = audio_feature_lengths.reshape(-1)

        audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths
        )

        audio_features = self.audio_tower.get_audio_embeddings(
            input_features.to(self.audio_tower.dtype),
            audio_feature_lengths=audio_feature_lengths,
            aftercnn_lens=audio_output_lengths,
            audio_buckets=self.vision_bucket_manager,
        )
        return audio_features.split(audio_output_lengths.tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        """Convert index-style multimodal positions to a boolean mask."""
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        is_multimodal_aligned = torch.zeros_like(input_ids, dtype=torch.bool)
        is_multimodal_aligned[is_multimodal.long()] = True

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal_aligned,
            handle_oov_mm_token=handle_oov_mm_token,
        )
