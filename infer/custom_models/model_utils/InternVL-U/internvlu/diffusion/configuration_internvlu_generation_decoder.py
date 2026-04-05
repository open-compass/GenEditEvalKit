# --------------------------------------------------------
# InternVL-U
# Copyright (c) 2026 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import json
import copy
from typing import Optional, Union, List
from transformers.configuration_utils import PretrainedConfig


class InternVLUGenerationDecoderConfig(PretrainedConfig):
    model_type = "internvlu_generation_decoder"
    is_composition = False

    def __init__(
        self,
        decoder_projector_type: str = "mlp2x_gelu",
        vae_projector_type: str = "mlp2x_gelu",
        visual_input_hidden_size: int = 4096,
        input_hidden_size: int = 1536,
        vae_input_hidden_size: int = 64,
        output_hidden_size: int = 2304,
        decoder_config: Optional[Union[str, dict]] = None,
        padding_encoder_hidden_states=True,
        pad_to_fix_length: bool = True,
        sigmas_as_weight: bool = False,  # Used in Flux
        discrete_timestep=False,
        weighting_scheme: Optional[str] = (
            "logit_normal"  # ["sigma_sqrt", "logit_normal", "mode", "cosmap", "null"]
        ),
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
        all_cfg_scale: float = 0.0,
        part_cfg_scale: float = 0.0,
        mask_weight_type: Optional[str] = None,  # ['log', 'exp']
        max_sequence_length: int = 512,
        gen_image_height: int = 512,
        gen_image_width: int = 512,
        video_position_scale_factor: float = 1.0,
        vlm_cond_position_scale_factor: float = 1.0,
        vae_downsample_factor: int = 32,
        region_weighting: bool = False,
        flow_shift: float = 3.0,
        vlm_select_layer: Optional[List[int]] = [-1],
        qk_norm: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.decoder_projector_type = decoder_projector_type
        self.vae_projector_type = vae_projector_type

        self.visual_input_hidden_size = visual_input_hidden_size
        self.input_hidden_size = input_hidden_size
        self.vae_input_hidden_size = vae_input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.decoder_config = decoder_config

        self.padding_encoder_hidden_states = padding_encoder_hidden_states
        self.pad_to_fix_length = pad_to_fix_length

        self.sigmas_as_weight = sigmas_as_weight
        self.discrete_timestep = discrete_timestep
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale

        self.all_cfg_scale = all_cfg_scale
        self.part_cfg_scale = part_cfg_scale

        self.mask_weight_type = mask_weight_type
        self.gen_image_height = gen_image_height
        self.gen_image_width = gen_image_width
        self.video_position_scale_factor = video_position_scale_factor
        self.vlm_cond_position_scale_factor = vlm_cond_position_scale_factor

        self.vae_downsample_factor = vae_downsample_factor
        self.region_weighting = region_weighting

        self.vlm_select_layer = vlm_select_layer

        self.max_sequence_length = max_sequence_length

        self.qk_norm = qk_norm

        self.flow_shift = flow_shift

        if isinstance(decoder_config, str):
            with open(decoder_config, "r") as f:
                self.decoder_config = json.load(f)
        else:
            self.decoder_config = decoder_config

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """

        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        output["is_composition"] = self.__class__.is_composition

        return output
