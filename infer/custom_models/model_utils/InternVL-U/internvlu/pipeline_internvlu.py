# --------------------------------------------------------
# InternVL-U
# Copyright (c) 2026 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional, Union, List
from dataclasses import dataclass

import numpy as np
import torch

from PIL import Image
from transformers.generation.utils import GenerateOutput
from transformers.utils import logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import AutoencoderKLQwenImage
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import BaseOutput

from .diffusion import InternVLUGenerationDecoder
from .diffusion import InternVLUDiffusionPipeline
from .vlm import InternVLUChatModel
from .vlm.constants import (
    IMG_CONTEXT_TOKEN,
    IMG_START_TOKEN,
    IMG_END_TOKEN,
    IMG_UNCOND_TOKEN,
    SPECIAL_TOKEN_LIST,
)
from .processing_internvlu import InternVLUProcessor

logger = logging.get_logger(__name__)

GENERATION_KWARGS = {
    "max_new_tokens",
    "min_new_tokens",
    "do_sample",
    "temperature",
    "top_p",
    "num_beams",
    "return_dict_in_generate",
    "output_hidden_states",
}

DIFFUSION_KWARGS = {
    "num_inference_steps",
    "num_images_per_prompt",
    "all_cfg_scale",
    "part_cfg_scale",
    "height",
    "width",
    "generator",
}


@dataclass
class InternVLUPipelineOutput(BaseOutput):
    """
    Output class for InternVLU pipelines.

    Args:
        images (`List[Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    generate_output: Union[torch.LongTensor, GenerateOutput] = None
    images: Union[List[Image.Image], np.ndarray] = None


class InternVLUPipeline(DiffusionPipeline):
    """End-to-end InternVLU pipeline for text and image generation."""

    model_cpu_offload_seq = "vlm->generation_decoder->vae"

    def __init__(
        self,
        vlm: InternVLUChatModel,
        generation_decoder: InternVLUGenerationDecoder,
        vae: AutoencoderKLQwenImage,
        scheduler: DPMSolverMultistepScheduler,
        processor: InternVLUProcessor,
    ):
        super().__init__()

        self.register_modules(
            vlm=vlm,
            generation_decoder=generation_decoder,
            vae=vae,
            scheduler=scheduler,
            processor=processor,
        )
        self.tokenizer = self.processor.tokenizer
        self.processor.template_name = self.vlm.config.template
        self.image_pipeline = InternVLUDiffusionPipeline(
            vae=vae,
            generation_decoder=generation_decoder,
            scheduler=scheduler,
        )

        self._init_special_tokens()

    def _split_kwargs(self, kwargs):
        """Split keyword arguments into generation, diffusion, and unused buckets."""
        generation_kwargs = {}
        diffusion_kwargs = {}
        unused_kwargs = {}

        for k, v in kwargs.items():
            if k in GENERATION_KWARGS:
                generation_kwargs[k] = v
            elif k in DIFFUSION_KWARGS:
                diffusion_kwargs[k] = v
            else:
                unused_kwargs[k] = v

        return generation_kwargs, diffusion_kwargs, unused_kwargs

    def _init_special_tokens(self):
        """Cache special token IDs on the VLM for faster generation."""
        special_token_id_list = []
        for token in SPECIAL_TOKEN_LIST:
            special_token_id_list.append(self.tokenizer.convert_tokens_to_ids(token))
        self.vlm.special_token_id_list = special_token_id_list
        self.vlm.img_context_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_CONTEXT_TOKEN
        )
        self.vlm.img_start_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_START_TOKEN
        )
        self.vlm.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.vlm.img_uncond_token_id = self.tokenizer.convert_tokens_to_ids(
            IMG_UNCOND_TOKEN
        )
        self.vlm.im_start_token_id = self.tokenizer.convert_tokens_to_ids(
            "<|im_start|>"
        )
        self.vlm.im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _prepare_hidden_state_mask(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_flags: Optional[torch.LongTensor] = None,
        padding_type="pad",
    ) -> torch.FloatTensor:
        """
        Build the hidden-state mask used for image generation conditioning.

        The mask selects token positions from the second `<|im_start|>` token up to each image start token.

        Args:
            input_ids (`torch.LongTensor`): Input token IDs of shape `(B, N)` where `B` is batch size and
                `N` is sequence length.
            attention_mask (`torch.Tensor`, *optional*):
                - For `padding_type="pad"`: standard attention mask of shape `(B, N)`.
                - For `padding_type="pack"`: BOS token positions of shape `(1, B')`.
            generation_flags (`torch.LongTensor`, *optional*): Flags indicating which image start tokens
                should be generated. Shape `(K_total,)`.
            padding_type (`str`, *optional*, defaults to `"pad"`): Padding strategy, `"pad"` or `"pack"`.

        Returns:
            `torch.FloatTensor`: Hidden-state mask of shape `(K, B, N)` where `K` is the number of images
            to generate. `True` positions indicate selected hidden states.
        """

        B, N = input_ids.shape

        img_start_positions = (input_ids == self.vlm.img_start_token_id).nonzero()
        gen_img_start_positions = img_start_positions[generation_flags.bool()]  # [K,2]
        state_positions_row = torch.arange(B, device=input_ids.device)[None]  # [1,B]
        state_positions_col = torch.arange(N, device=input_ids.device)[None]  # [1,N]
        state_mask_row = state_positions_row == gen_img_start_positions[:, :1]  # [K, B]
        state_mask_col = state_positions_col <= gen_img_start_positions[:, 1:]  # [K, N]

        if padding_type == "pad":
            im_start_positions = (input_ids == self.vlm.im_start_token_id).nonzero()
            # find the second <|im_start|> token in each sample
            im_state_mask_row = state_positions_row == im_start_positions[:, :1]
            im_start_second_idxs = (im_state_mask_row.cumsum(dim=0) == 2).nonzero(
                as_tuple=True
            )[0]
            im_start_second_positions = im_start_positions[
                im_start_second_idxs
            ]  # [K',2]

            # find the second <|im_start|> token for each img_start_token in each sample
            # NOTE here we assume the second <|im_start|> token always exists for each sample !!!
            gen_img_start_positions_global = (
                gen_img_start_positions[:, 0] * N + gen_img_start_positions[:, 1]
            )
            bos_positions_global = gen_img_start_positions[:, 0] * N
            im_start_second_positions_global = (
                im_start_second_positions[:, 0] * N + im_start_second_positions[:, 1]
            )
            im_start_second_positions_mask = (
                im_start_second_positions_global[None, :]
                <= gen_img_start_positions_global[:, None]
            ) & (
                im_start_second_positions_global[None, :]
                >= bos_positions_global[:, None]
            )  # [K,K']
            im_start_second_positions_idxs = (
                im_start_second_positions_mask.int().argmax(dim=1)
            )  # [K]
            im_start_second_positions_to_gen_img_start = im_start_second_positions[
                im_start_second_positions_idxs
            ]  # [K, 2]

            state_mask_col = state_mask_col & (
                im_start_second_positions_to_gen_img_start[:, 1:] <= state_positions_col
            )

        if padding_type == "pack":
            bos_token_ids = attention_mask  # [1, B']
            boi_token_ids = gen_img_start_positions[:, 1:]  # [K, 1]
            image_in_seq_idxs = (boi_token_ids >= bos_token_ids).sum(dim=-1) - 1  # [K]
            image_seq_positions = bos_token_ids[0, image_in_seq_idxs][:, None]
            state_mask_col = state_mask_col & (
                image_seq_positions <= state_positions_col
            )

            im_start_positions_global = (
                input_ids == self.vlm.im_start_token_id
            ).nonzero()[:, 1]
            gen_img_start_positions_global = gen_img_start_positions[:, 1]
            im_start_positions_mask = (
                im_start_positions_global[None, :]
                <= gen_img_start_positions_global[:, None]
            ) & (
                im_start_positions_global[None, :] >= image_seq_positions
            )  # [K,K']
            # find the second <|im_start|> token for each img_start_token in each sample
            # NOTE here we assume the second <|im_start|> token always exists for each sample !!!
            im_start_second_idxs = (im_start_positions_mask.cumsum(dim=1) == 2).nonzero(
                as_tuple=True
            )[1]
            im_start_second_positions_to_gen_img_start = im_start_positions_global[
                im_start_second_idxs
            ][:, None]
            if (
                im_start_second_positions_to_gen_img_start.shape[0]
                != state_mask_col.shape[0]
            ):
                print(
                    f"Warning: im_start_second_positions_to_gen_img_start shape"
                    f"{im_start_second_positions_to_gen_img_start.shape} does not match state_mask_col shape "
                    f"{state_mask_col.shape}. This may cause issues in the state mask calculation.",
                    force=True,
                )
            state_mask_col = state_mask_col & (
                im_start_second_positions_to_gen_img_start <= state_positions_col
            )

        state_mask = (
            (state_mask_row[..., None] & state_mask_col[:, None])
        ).bool()  # [K,B,N]

        if padding_type == "pad":
            state_mask = (state_mask & attention_mask[None]).bool()

        return state_mask

    def _prepare_image_hidden_state_mask(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_flags: Optional[torch.LongTensor] = None,
        padding_type="pad",
    ):
        """
        Build the image-conditioning mask and optional relative indices.

        This identifies which prior images in the sequence can be used as conditioning
        for each image generation step.

        Args:
            input_ids (`torch.LongTensor`): Input token IDs of shape `(B, N)`.
            attention_mask (`torch.Tensor`, *optional*):
                - For `padding_type="pad"`: standard attention mask of shape `(B, N)`.
                - For `padding_type="pack"`: BOS token positions of shape `(1, B')`.
            generation_flags (`torch.LongTensor`, *optional*): Flags indicating which image start tokens
                should be generated. Shape `(K_total,)`.
            padding_type (`str`, *optional*, defaults to `"pad"`): Padding strategy, `"pad"` or `"pack"`.

        Returns:
            `Tuple[torch.BoolTensor, torch.BoolTensor, Optional[List[torch.LongTensor]]]`:
            - `gen_img_cond_image_mask`: Mask of shape `(K, K_total)` selecting conditioning images.
            - `input_img_cond_image_mask`: Mask of shape `(K, K_real)` selecting input images.
            - `image_rel_idxs`: Optional list of relative indices for channel-add conditioning.

        Notes:
            Only images that appear before the target image in the same sequence are eligible
            as conditioning inputs.
        """

        gen_img_cond_image_mask = None
        image_rel_idxs = None
        img_start_positions = (input_ids == self.vlm.img_start_token_id).nonzero()
        gen_img_start_positions = img_start_positions[generation_flags.bool()]

        if padding_type == "pad":
            img_start_positions_real = img_start_positions[~generation_flags.bool()]
            input_img_cond_image_mask = (
                gen_img_start_positions[:, 0][:, None]
                == img_start_positions_real[:, 0][None, :]
            ) & (
                gen_img_start_positions[:, 1][:, None]
                > img_start_positions_real[:, 1][None, :]
            )
            gen_img_cond_image_mask = (
                gen_img_start_positions[:, 0][:, None]
                == img_start_positions[:, 0][None, :]
            ) & (
                gen_img_start_positions[:, 1][:, None]
                > img_start_positions[:, 1][None, :]
            )

        else:
            bos_token_ids = attention_mask  # [1, B']
            boi_token_ids = gen_img_start_positions[:, 1:]  # [K, 1]
            image_in_seq_idxs = (boi_token_ids >= bos_token_ids).sum(dim=-1) - 1  # [K]
            gen_img_bos_positions = bos_token_ids[0, image_in_seq_idxs][:, None]
            gen_img_cond_image_mask = (
                (
                    gen_img_start_positions[:, 0][:, None]
                    == img_start_positions[:, 0][None, :]
                )
                & (
                    gen_img_start_positions[:, 1][:, None]
                    > img_start_positions[:, 1][None, :]
                )
                & (gen_img_bos_positions < img_start_positions[:, 1][None, :])
            )

        return gen_img_cond_image_mask, input_img_cond_image_mask, image_rel_idxs

    def _prepare_diffusion_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.Tensor,
        pixel_values_gen: torch.Tensor,
        generation_flags: torch.Tensor,
        image_grid_thw_gen: list,
        vlm_hidden_states: torch.Tensor,
    ):
        """Prepare decoder inputs and conditioning tensors for diffusion generation."""
        selected = input_ids == self.vlm.img_context_token_id
        B, L = input_ids.shape
        vlm_hidden_states = [
            vlm_hidden_states[i].view(B, L, -1)
            for i in self.generation_decoder.config.vlm_select_layer
        ]
        vlm_hidden_states = torch.cat(vlm_hidden_states, dim=-1)  # B, N, C * num_layers
        state_mask = self._prepare_hidden_state_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_flags=generation_flags,
            padding_type="pad",
        )

        vlm_hidden_states = [vlm_hidden_states[s] for s in state_mask]
        vlm_image_token_mask = [
            selected[s] for s in state_mask
        ]  # if pixel_values is not None else None

        image_hidden_states = None
        image_rel_idxs = None
        pixel_values_cond = None
        image_fhw_cond = None
        image_grid_thw_gen_cond = None
        if pixel_values is not None and (selected.sum() != 0):
            gen_img_cond_image_mask, input_img_cond_image_mask, image_rel_idxs = (
                self._prepare_image_hidden_state_mask(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_flags=generation_flags,
                    padding_type="pad",
                )
            )
            image_fhw = (
                1,
                int(
                    pixel_values.shape[2]
                    // self.vlm.config.vision_config.patch_size
                    * self.vlm.config.downsample_ratio
                ),
                int(
                    pixel_values.shape[3]
                    // self.vlm.config.vision_config.patch_size
                    * self.vlm.config.downsample_ratio
                ),
            )

            image_fhw = torch.tensor(
                [image_fhw] * gen_img_cond_image_mask.shape[1],
                device=pixel_values.device,
            )
            image_fhw_cond = [image_fhw[m] for m in gen_img_cond_image_mask]

            if image_grid_thw_gen is not None:
                image_grid_thw_gen_cond = [
                    image_grid_thw_gen[m] for m in gen_img_cond_image_mask
                ]
            pixel_values_cond = [pixel_values_gen[m] for m in input_img_cond_image_mask]

        else:
            image_fhw_cond = [
                torch.zeros(
                    [0, 3], device=vlm_hidden_states[0].device, dtype=torch.long
                )
            ] * B

        if image_grid_thw_gen is not None:
            image_grid_thw_gen = image_grid_thw_gen[generation_flags.bool()]
        generation_inputs = {
            "encoder_hidden_states": vlm_hidden_states,
            "encoder_image_token_mask": vlm_image_token_mask,
            "image_fhw_cond": image_fhw_cond,
            "image_hidden_states": image_hidden_states,
            "image_rel_idxs": image_rel_idxs,
            "conditional_image": pixel_values_cond,
            "image_grid_thw_gen": image_grid_thw_gen,
            "image_grid_thw_gen_cond": image_grid_thw_gen_cond,
        }
        return generation_inputs

    def _generate(self, input_ids, attention_mask, pixel_values=None, **kwargs):
        """Generate text-only outputs using the VLM."""
        generation_returns = self.vlm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            eos_token_id=[self.tokenizer.eos_token_id, self.vlm.img_start_token_id],
            **kwargs,
        )
        return InternVLUPipelineOutput(generate_output=generation_returns)

    def _generate_image(
        self,
        input_ids,
        attention_mask,
        generation_flags,
        pixel_values=None,
        pixel_values_gen=None,
        image_grid_thw_gen=None,
        all_cfg_scale=4.5,
        part_cfg_scale=2.0,
        num_inference_steps=20,
        num_images_per_prompt=1,
        height=None,
        width=None,
        generator=None,
    ):
        """Generate images conditioned on VLM hidden states and optional inputs."""

        outputs = self.vlm.generate_hidden_states(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )
        vlm_hidden_states = outputs.hidden_states

        diffusion_inputs = self._prepare_diffusion_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_values_gen=pixel_values_gen,
            image_grid_thw_gen=image_grid_thw_gen,
            generation_flags=generation_flags,
            vlm_hidden_states=vlm_hidden_states,
        )
        output = self.image_pipeline(
            **diffusion_inputs,
            all_cfg_scale=all_cfg_scale,
            part_cfg_scale=part_cfg_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            use_resolution_binning=False,
            height=height,
            width=width,
            generator=generator,
        ).images

        output = ((127.5 * output + 128.0) / 255).clamp(0, 1)
        output_images = [
            Image.fromarray(
                (img.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            for img in output
        ]

        return InternVLUPipelineOutput(images=output_images)

    def _generate_image_with_cot(
        self,
        input_ids,
        attention_mask,
        generation_flags,
        pixel_values=None,
        pixel_values_gen=None,
        image_grid_thw_gen=None,
        all_cfg_scale=3.5,
        part_cfg_scale=1.5,
        num_inference_steps=20,
        num_images_per_prompt=1,
        max_new_tokens=200,
        height=None,
        width=None,
        generator=None,
        **kwargs,
    ):
        """Generate images after a text chain-of-thought expansion stage."""
        B, L = input_ids.shape
        assert B % 3 == 0
        input_ids_text = input_ids[: B // 3]
        attention_mask_text = attention_mask[: B // 3]
        if pixel_values is not None:
            pixel_values_text = pixel_values[: pixel_values.shape[0] // 2]
        else:
            pixel_values_text = None
        outputs_text_ids = self._generate(
            input_ids=input_ids_text,
            attention_mask=attention_mask_text,
            pixel_values=pixel_values_text,
            max_new_tokens=max_new_tokens,
            **kwargs,
        ).generate_output
        device = input_ids.device

        # concat input_ids & attention_masks
        new_input_ids_text = torch.cat(
            [input_ids_text, outputs_text_ids], dim=1
        )  # [B // 3, T_in + T_out]

        eos_mask = (outputs_text_ids == self.vlm.im_end_token_id).long()
        img_start_token_mask = (outputs_text_ids == self.vlm.img_start_token_id).long()
        default_pos = outputs_text_ids.shape[1] - 1

        has_eos = eos_mask.any(dim=1)  # [B]
        has_img_start = img_start_token_mask.any(dim=1)  # [B]

        eos_pos = torch.where(
            has_eos, eos_mask.float().argmax(dim=1), default_pos
        )  # [B]

        # img_start position
        img_start_pos = torch.where(
            has_img_start, img_start_token_mask.argmax(dim=1), default_pos
        )  # [B]
        eos_pos = torch.min(img_start_pos, eos_pos)
        eos_pos = eos_pos + input_ids_text.shape[1] - 1

        input_ids_cot = (
            torch.ones(
                [B, eos_pos.max()], dtype=new_input_ids_text.dtype, device=device
            )
            * self.vlm.im_end_token_id
        )
        attention_mask_cot = torch.zeros(
            [B, eos_pos.max()], dtype=attention_mask_text.dtype, device=device
        )

        for batch_idx in range(B // 3):
            sample_eos_pos = eos_pos[batch_idx]
            input_ids_cot[batch_idx, -sample_eos_pos:] = new_input_ids_text[
                batch_idx, :sample_eos_pos
            ]
            attention_mask_cot[batch_idx, -sample_eos_pos:] = 1
        input_ids_cot[B // 3 :, -L:] = input_ids[B // 3 :]
        attention_mask_cot[B // 3 :, -L:] = attention_mask[B // 3 :]

        img_start_token = (
            torch.ones([B, 1], dtype=input_ids_cot.dtype, device=device)
            * self.vlm.img_start_token_id
        )
        img_start_token_attention_mask = torch.ones(
            [B, 1], dtype=attention_mask_cot.dtype, device=device
        )

        input_ids_cot = torch.cat([input_ids_cot, img_start_token], dim=1)
        attention_mask_cot = torch.cat(
            [attention_mask_cot, img_start_token_attention_mask], dim=1
        )

        output_images = self._generate_image(
            input_ids=input_ids_cot,
            attention_mask=attention_mask_cot,
            pixel_values=pixel_values,
            pixel_values_gen=pixel_values_gen,
            generation_flags=generation_flags,
            image_grid_thw_gen=image_grid_thw_gen,
            all_cfg_scale=all_cfg_scale,
            part_cfg_scale=part_cfg_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            generator=generator,
        ).images
        return InternVLUPipelineOutput(
            generate_output=outputs_text_ids, images=output_images
        )

    def __call__(self, prompt, image=None, generation_mode="text", **kwargs):
        """Run the pipeline in text, image, or text+image generation modes.

        Args:
            prompt (`str` or `List[str]`): Input text prompt(s).
            image (`PIL.Image.Image` or `List[PIL.Image.Image]`, *optional*):
                Optional image(s) for multimodal generation.
            generation_mode (`str`, *optional*, defaults to `"text"`):
                One of `"text"`, `"image"`, or `"text_image"`.
            kwargs: Additional keyword arguments forwarded to the processor and generation methods.

        Returns:
            `InternVLUPipelineOutput`: Generated text and/or image outputs depending on mode.
        """
        system_prompt = kwargs.pop("system_prompt", None)
        call_kwargs = dict(kwargs)

        if generation_mode != "text":
            call_kwargs.setdefault(
                "height",
                self.generation_decoder.config.gen_image_height,
            )
            call_kwargs.setdefault(
                "width",
                self.generation_decoder.config.gen_image_width,
            )
            height, width = call_kwargs["height"], call_kwargs["width"]
            stride = self.processor.image_gen_processor.merge_size * self.processor.image_gen_processor.patch_size
            gen_height, gen_width =  height // stride * stride, width // stride * stride
        else:
            height, width = None, None
            gen_height, gen_width = None, None

        generation_kwargs, diffusion_kwargs, unused = self._split_kwargs(call_kwargs)
        if len(unused) > 0:
            logger.warning(f"Unused kwargs: {unused}")

        inputs = self.processor(
            prompt=prompt,
            image=image,
            generation_mode=generation_mode,
            padding=True,
            return_tensors="pt",
            height=gen_height,
            width=gen_width,
            system_prompt=system_prompt,
        )
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = inputs[k].to(self.vlm.device)
        if generation_mode == "text":
            generation_returns = self._generate(**inputs, **generation_kwargs)
            if len(diffusion_kwargs) > 0:
                logger.warning(
                    f"Diffusion kwargs: {diffusion_kwargs}, but generation mode is {generation_mode}"
                )
        elif generation_mode == "image":
            generation_returns = self._generate_image(**inputs, **diffusion_kwargs)
            if len(generation_kwargs) > 0:
                logger.warning(
                    f"Text Generation kwargs: {generation_kwargs}, but generation mode is {generation_mode}"
                )
        elif generation_mode == "text_image":
            generation_returns = self._generate_image_with_cot(
                **inputs, **generation_kwargs, **diffusion_kwargs
            )
        else:
            raise NotImplementedError(f"Unknown generation_mode: {generation_mode}")
    
        images: Union[List[Image.Image], np.ndarray] = getattr(generation_returns, "images", None)

        if images is not None:
            if isinstance(images, list):
                generation_returns.images = [
                    img.resize((width, height), Image.Resampling.LANCZOS)
                    for img in images
                ]
            elif isinstance(images, np.ndarray):
                if images.ndim == 3:
                    generation_returns.images = np.array(
                        Image.fromarray(images).resize((width, height), Image.Resampling.LANCZOS)
                    )
                elif images.ndim == 4:
                    generation_returns.images = np.stack([
                        np.array(
                            Image.fromarray(img).resize((width, height), Image.Resampling.LANCZOS)
                        )
                        for img in images
                    ])
                else:
                    raise ValueError(f"Unsupported images ndim: {images.ndim}")
        return generation_returns

