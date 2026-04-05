from .configuration_internvlu_generation_decoder import InternVLUGenerationDecoderConfig
from .modeling_internvlu_generation_decoder import InternVLUGenerationDecoder
from .pipeline_internvlu_generation_decoder import InternVLUDiffusionPipeline

from transformers import AutoConfig, AutoModel, PreTrainedModel

AutoConfig.register("internvlu_generation_decoder", InternVLUGenerationDecoderConfig)
AutoModel.register(InternVLUGenerationDecoderConfig, InternVLUGenerationDecoder)

__all__ = [
    "InternVLUGenerationDecoderConfig",
    "InternVLUGenerationDecoder",
    "InternVLUDiffusionPipeline",
    "PreTrainedModel",
]
