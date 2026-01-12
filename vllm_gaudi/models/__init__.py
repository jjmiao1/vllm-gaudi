from vllm.model_executor.models.registry import ModelRegistry


def register_model():
    from vllm_gaudi.models.gemma3_mm import HpuGemma3ForConditionalGeneration  # noqa: F401
    from vllm_gaudi.models.hunyuan_v1 import HpuHunYuanDenseV1ForCausalLM
    ModelRegistry.register_model(
        "Gemma3ForConditionalGeneration",  # Original architecture identifier in vLLM
        "vllm_gaudi.models.gemma3_mm:HpuGemma3ForConditionalGeneration")
    ModelRegistry.register_model(
         "HunYuanDenseV1ForCausalLM",
         "vllm_gaudi.models.hunyuan_v1:HpuHunYuanDenseV1ForCausalLM")
    ModelRegistry.register_model(
         "HunYuanMoEV1ForCausalLM",
         "vllm_gaudi.models.hunyuan_v1:HpuHunYuanMoEV1ForCausalLM")