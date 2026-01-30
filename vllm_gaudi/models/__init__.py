from vllm.model_executor.models.registry import ModelRegistry


def register_model():
    from vllm_gaudi.models.gemma3_mm import HpuGemma3ForConditionalGeneration  # noqa: F401

    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",  # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3_mm:HpuGemma3ForConditionalGeneration")

    from vllm_gaudi.models.qwen2_5_vl import HpuQwen2_5_VLForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen2_5_VLForConditionalGeneration",
                                 "vllm_gaudi.models.qwen2_5_vl:HpuQwen2_5_VLForConditionalGeneration")

    from vllm_gaudi.models.qwen3_vl import HpuQwen3_VLForConditionalGeneration  # noqa: F401
    ModelRegistry.register_model("Qwen3VLForConditionalGeneration",
                                 "vllm_gaudi.models.qwen3_vl:HpuQwen3_VLForConditionalGeneration")
    
    from vllm_gaudi.models.qwen2_5_omni_thinker import HpuQwen2_5OmniThinkerForConditionalGeneration # noqa: F401
    ModelRegistry.register_model("Qwen2_5OmniModel",
                                "vllm_gaudi.models.qwen2_5_omni_thinker:HpuQwen2_5OmniThinkerForConditionalGeneration")

    ModelRegistry.register_model("Qwen2_5OmniForConditionalGeneration",
                                "vllm_gaudi.models.qwen2_5_omni_thinker:HpuQwen2_5OmniThinkerForConditionalGeneration")
