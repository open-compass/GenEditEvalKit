# --------------------------------------------------------
# InternVL-U
# Copyright (c) 2026 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Union, List

import torch

from torch import nn
from transformers.modeling_utils import PreTrainedModel

from .configuration_internvlu_generation_decoder import InternVLUGenerationDecoderConfig
from .internvlu_transformer import InternVLUTransformer2DModel


class InternVLUGenerationDecoder(PreTrainedModel):
    """Transformer-based generation decoder used by the InternVLU diffusion stack."""

    config_class = InternVLUGenerationDecoderConfig
    base_model_prefix = ""

    def __init__(self, config: InternVLUGenerationDecoderConfig):
        super().__init__(config)

        self.config = config
        config.decoder_config["video_position_scale_factor"] = (
            config.video_position_scale_factor
        )
        config.decoder_config["vlm_cond_position_scale_factor"] = (
            config.vlm_cond_position_scale_factor
        )
        self.decoder = InternVLUTransformer2DModel.from_config(config.decoder_config)

        if self.config.padding_encoder_hidden_states:
            self.encoder_padding_token = nn.Parameter(
                torch.zeros(self.config.input_hidden_size)
            )

        self.do_denormalize = True

        self._init_decoder_projector()
        self._init_visual_embeds_projector()

    def _init_decoder_projector(self):
        """Initialize the decoder_projector for multi model input."""
        if self.config.decoder_projector_type == "mlp2x_gelu":
            self.decoder_projector = nn.Sequential(
                nn.Linear(
                    self.config.input_hidden_size,
                    self.config.output_hidden_size * 3,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.config.output_hidden_size * 3, self.config.output_hidden_size
                ),
            )
        elif self.config.decoder_projector_type == "identity":
            self.decoder_projector = nn.Identity()
        else:
            raise ValueError(
                f"Unknown decoder_projector_type: {self.config.decoder_projector_type}"
            )

    def _init_visual_embeds_projector(self):
        """Initialize the visual_embeds_projector for visual encoder input."""
        self.visual_embeds_projector = nn.Identity()
        return

    @property
    def device(self):
        return self.decoder.device

    def prepare_forward_input(
        self,
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_image_token_mask: Union[torch.Tensor, List[torch.Tensor]] = None,
        **kwargs,
    ):
        return self._prepare_forward_input_default(
            encoder_hidden_states, encoder_image_token_mask, **kwargs
        )

    def _prepare_forward_input_default(
        self,
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_image_token_mask: Union[torch.Tensor, List[torch.Tensor]] = None,
        **kwargs,
    ):
        """Project encoder hidden states and build attention masks (no visual embeds)."""
        assert encoder_hidden_states is not None

        if isinstance(encoder_hidden_states, list):
            assert (
                self.config.padding_encoder_hidden_states
            ), "encoder_hidden_states should be a list only when padding is enabled."
            # if padding is enabled, we need to pad the encoder_hidden_states
            # if self.config.padding_encoder_hidden_states:
            import os

            pad_to_fix_length = self.config.pad_to_fix_length

            if pad_to_fix_length:
                max_length = self.config.max_sequence_length
            else:
                max_length = max([x.shape[0] for x in encoder_hidden_states])
                max_length = max(
                    max_length, self.config.max_sequence_length
                )  # at least max_sequence_length

            attention_masks = torch.zeros(
                len(encoder_hidden_states), max_length, device=self.device
            ).bool()
            encoder_hidden_states_padded = self.encoder_padding_token[
                None, None
            ].repeat(len(encoder_hidden_states), max_length, 1)
            if encoder_image_token_mask is not None:
                encoder_image_token_mask_padded = torch.zeros(
                    len(encoder_hidden_states), max_length, device=self.device
                ).bool()
            for idx, encoder_hidden_state in enumerate(encoder_hidden_states):
                encoder_seq_len = min(encoder_hidden_state.shape[0], max_length)
                encoder_hidden_states_padded[idx, :encoder_seq_len] = (
                    encoder_hidden_state[:encoder_seq_len]
                )
                attention_masks[idx, :encoder_seq_len] = True
                if encoder_image_token_mask is not None:
                    encoder_image_token_mask_padded[idx, :encoder_seq_len] = (
                        encoder_image_token_mask[idx][:encoder_seq_len]
                    )
            if encoder_image_token_mask is not None:
                encoder_image_token_mask = encoder_image_token_mask_padded

            encoder_hidden_states = encoder_hidden_states_padded

        prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
        if prefix_prompt_embeds is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states, prefix_prompt_embeds], dim=1
            )
            # attention mask
            extended_attention_masks = torch.ones(
                prefix_prompt_embeds.shape[0],
                prefix_prompt_embeds.shape[1],
                device=self.device,
            ).bool()
            attention_masks = torch.concat(
                [attention_masks, extended_attention_masks], dim=1
            )

        encoder_hidden_states = self.decoder_projector(encoder_hidden_states)
        return encoder_hidden_states, attention_masks, encoder_image_token_mask

    def get_sigmas(self, timesteps, n_dim=4):
        """Convert scheduler timesteps to sigma values with broadcast to `n_dim`."""
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
