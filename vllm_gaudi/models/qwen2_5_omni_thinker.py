import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerForConditionalGeneration)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_gaudi.models.qwen2_5_vl import Qwen2_5_VisionTransformerStaticShape

class HpuQwen2_5OmniThinkerForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        
        thinker_config = vllm_config.model_config.hf_config.thinker_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        

        if multimodal_config.get_limit_per_prompt("image") or \
           multimodal_config.get_limit_per_prompt("video"):
            
            self.visual = Qwen2_5_VisionTransformerStaticShape(
                vision_config=thinker_config.vision_config,
                norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
                multimodal_config=multimodal_config,
            )
        else:
            self.visual = None

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
    
    def _process_video_input(self, video_input):
        if video_input["type"] == "video_embeds":
            return video_input["video_embeds"].type(self.visual.dtype)

        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2

        pixel_values_videos = video_input["pixel_values_videos"].type(
            self.visual.dtype)

        video_embeds = self.visual.get_image_embeds(
            pixel_values_videos,
            grid_thw=grid_thw,
            vision_buckets=self.vision_bucket_manager,
        )
        merge_size = self.visual.spatial_merge_size
        sizes = grid_thw.prod(-1) // merge_size // merge_size
        return video_embeds.split(sizes.tolist())