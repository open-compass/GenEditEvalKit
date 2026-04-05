from .configuration_internvlu_chat import InternVLUChatConfig
from .modeling_internvlu_chat import InternVLUChatModel

from transformers import AutoConfig, AutoModel, PreTrainedModel

AutoConfig.register("internvlu_chat", InternVLUChatConfig)
AutoModel.register(InternVLUChatConfig, InternVLUChatModel)

__all__ = ["InternVLUChatConfig", "InternVLUChatModel", "PreTrainedModel"]
