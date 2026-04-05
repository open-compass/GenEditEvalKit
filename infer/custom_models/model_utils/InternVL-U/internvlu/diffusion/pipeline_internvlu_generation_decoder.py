# --------------------------------------------------------
# InternVL-U
# Modifications Copyright (c) 2026 OpenGVLab
# This file includes code from PixArt-Sigma and HuggingFace,
# licensed under the Apache License, Version 2.0.
# --------------------------------------------------------
# Copyright 2025 PixArt-Sigma Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import inspect

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import numpy as np
import torch

from einops import rearrange
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PixArtImageProcessor
from diffusers.models import AutoencoderDC
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from diffusers.pipelines.pipeline_utils import StableDiffusionMixin
from diffusers.utils import BaseOutput

from .modeling_internvlu_generation_decoder import InternVLUGenerationDecoder

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


ASPECT_RATIO_4096_BIN = {
    "0.25": [2048.0, 8192.0],
    "0.26": [2048.0, 7936.0],
    "0.27": [2048.0, 7680.0],
    "0.28": [2048.0, 7424.0],
    "0.32": [2304.0, 7168.0],
    "0.33": [2304.0, 6912.0],
    "0.35": [2304.0, 6656.0],
    "0.4": [2560.0, 6400.0],
    "0.42": [2560.0, 6144.0],
    "0.48": [2816.0, 5888.0],
    "0.5": [2816.0, 5632.0],
    "0.52": [2816.0, 5376.0],
    "0.57": [3072.0, 5376.0],
    "0.6": [3072.0, 5120.0],
    "0.68": [3328.0, 4864.0],
    "0.72": [3328.0, 4608.0],
    "0.78": [3584.0, 4608.0],
    "0.82": [3584.0, 4352.0],
    "0.88": [3840.0, 4352.0],
    "0.94": [3840.0, 4096.0],
    "1.0": [4096.0, 4096.0],
    "1.07": [4096.0, 3840.0],
    "1.13": [4352.0, 3840.0],
    "1.21": [4352.0, 3584.0],
    "1.29": [4608.0, 3584.0],
    "1.38": [4608.0, 3328.0],
    "1.46": [4864.0, 3328.0],
    "1.67": [5120.0, 3072.0],
    "1.75": [5376.0, 3072.0],
    "2.0": [5632.0, 2816.0],
    "2.09": [5888.0, 2816.0],
    "2.4": [6144.0, 2560.0],
    "2.5": [6400.0, 2560.0],
    "2.89": [6656.0, 2304.0],
    "3.0": [6912.0, 2304.0],
    "3.11": [7168.0, 2304.0],
    "3.62": [7424.0, 2048.0],
    "3.75": [7680.0, 2048.0],
    "3.88": [7936.0, 2048.0],
    "4.0": [8192.0, 2048.0],
}


@dataclass
class InternVLUDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    """Retrieve latent samples from an encoder output object.

    Args:
        encoder_output (`torch.Tensor` or object): Output that may expose `latent_dist` or `latents`.
        generator (`torch.Generator`, *optional*): RNG to use when sampling.
        sample_mode (`str`, *optional*, defaults to `"sample"`): `"sample"` to sample, `"argmax"` to use mode.

    Returns:
        `torch.Tensor`: Latent tensor extracted from the encoder output.
    """
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class InternVLUDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    """Diffusion pipeline wrapper around the InternVLU generation decoder."""

    def __init__(
        self,
        vae: AutoencoderDC,
        generation_decoder: InternVLUGenerationDecoder,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae, generation_decoder=generation_decoder, scheduler=scheduler
        )
        self.scheduler.config["flow_shift"] = self.generation_decoder.config.flow_shift

        self.transformer = self.generation_decoder.decoder

        if hasattr(self, "vae") and self.vae is not None:
            if hasattr(self.vae.config, "encoder_block_out_channels"):
                self.vae_scale_factor = 2 ** (
                    len(self.vae.config.encoder_block_out_channels) - 1
                )
            elif hasattr(self.vae.config, "temperal_downsample"):
                self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
            else:
                self.vae_scale_factor = 32

            self.latent_channels = (
                self.vae.config.z_dim if getattr(self, "vae", None) else 16
            )
        self.image_processor = PixArtImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def do_classifier_free_guidance(self):
        return True

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def pixels_to_latents(self, x, max_bs=32):
        x = x.to(dtype=self.dtype)
        x = x.unsqueeze(2)

        max_bs = math.floor(max_bs * 3 * 512**2 / x[0].numel())
        num_iter = math.ceil(x.shape[0] / max_bs)
        z_list = []
        for i in range(num_iter):
            x_bs = x[i * max_bs : (i + 1) * max_bs]
            image_latents = retrieve_latents(
                self.vae.encode(x_bs), generator=None, sample_mode="argmax"
            )
            z_list.append(image_latents)
        image_latents = torch.cat(z_list, dim=0)
        # image_latents = retrieve_latents(self.vae.encode(x), generator=None, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        z = (image_latents - latents_mean) / latents_std
        z = z.squeeze(2)
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z, max_bs=8):
        latents = z.to(self.vae.dtype).unsqueeze(2)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        max_bs = math.floor(max_bs * 16 * 64**2 / latents[0].numel())
        num_iter = math.ceil(latents.shape[0] / max_bs)
        x_list = []
        for i in range(num_iter):
            latents_bs = latents[i * max_bs : (i + 1) * max_bs]
            x_rec = self.vae.decode(latents_bs, return_dict=False)[0][:, :, 0]
            x_list.append(x_rec)
        x_rec = torch.cat(x_list)
        return x_rec

    @torch.no_grad()
    def __call__(
        self,
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_image_token_mask: Union[torch.Tensor, List[torch.Tensor]] = None,
        image_fhw_cond: Union[torch.Tensor, List[torch.Tensor]] = None,
        image_hidden_states: Union[torch.Tensor, List[torch.Tensor]] = None,
        image_rel_idxs: Union[torch.Tensor, List[torch.Tensor]] = None,
        conditional_image: Union[torch.Tensor, List[torch.Tensor]] = None,
        num_images_per_prompt: Optional[int] = 1,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        all_cfg_scale: float = 4.5,
        part_cfg_scale: float = 2.0,
        height: int = 256,
        width: int = 256,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        use_resolution_binning: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        conditional_input: Optional[List[torch.Tensor]] = None,
        timestep_trunc=930,
        timestep_shift=3.0,
        image_grid_thw_gen: torch.Tensor = None,
        image_grid_thw_gen_cond: torch.Tensor = None,
    ) -> Union[InternVLUDiffusionPipelineOutput, Tuple]:
        """
        Generate images from VLM conditioning and optional image guidance.

        Args:
            encoder_hidden_states (`torch.Tensor` or `List[torch.Tensor]`):
                Conditioning hidden states from the VLM.
            encoder_image_token_mask (`torch.Tensor` or `List[torch.Tensor]`, *optional*):
                Mask indicating image token positions in the encoder states.
            image_fhw_cond (`torch.Tensor` or `List[torch.Tensor]`, *optional*):
                Frame/height/width metadata for conditioning images.
            image_hidden_states (`torch.Tensor` or `List[torch.Tensor]`, *optional*):
                Optional image hidden states used for conditioning.
            image_rel_idxs (`torch.Tensor` or `List[torch.Tensor]`, *optional*):
                Relative indices for channel-add conditioning.
            conditional_image (`torch.Tensor` or `List[torch.Tensor]`, *optional*):
                Conditional images or latents for image-based guidance.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                Number of images to generate per prompt.
            num_inference_steps (`int`, *optional*, defaults to 20):
                Number of denoising steps.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process.
            all_cfg_scale (`float`, *optional*, defaults to 4.5):
                Classifier-free guidance scale for full conditioning.
            part_cfg_scale (`float`, *optional*, defaults to 2.0):
                Classifier-free guidance scale for partial conditioning.
            height (`int`, *optional*, defaults to 256):
                Height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256):
                Width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Eta (η) for DDIM-like schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random generator(s) for deterministic output.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents. If not provided, latents are sampled.
            output_type (`str`, *optional*, defaults to `"pil"`):
                Output format: `"pil"` or `"np"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a pipeline output dataclass or a tuple.
            use_resolution_binning (`bool`, *optional*, defaults to `True`):
                Map requested size to the closest aspect ratio bin before decoding.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Extra kwargs passed to attention processors.
            callback_on_step_end (`Callable`, *optional*):
                Callback invoked at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*):
                Tensor names to pass to the callback.
            conditional_input (`List[torch.Tensor]`, *optional*):
                Optional conditional inputs for the decoder.
            timestep_trunc (`int`, *optional*, defaults to 930):
                Maximum timestep used when truncating the schedule.
            timestep_shift (`float`, *optional*, defaults to 3.0):
                Shift value applied to timesteps.
            image_grid_thw_gen (`torch.Tensor`, *optional*):
                Generation grid metadata for image tokens.
            image_grid_thw_gen_cond (`torch.Tensor`, *optional*):
                Conditioning grid metadata for image tokens.

        Returns:
            `InternVLUDiffusionPipelineOutput` or `tuple`: Generated images.
        """

        # 0. perpare inputs

        assert (
            len(encoder_hidden_states) % 3 == 0
        ), "encoder_hidden_states should be mod by 3, as it is split into three parts for dual cfg guidance."

        conditional_input = None

        image_hidden_states = None
        if conditional_image is not None:
            split_sizes = None
            if isinstance(conditional_image, list):
                split_sizes = [len(c) for c in conditional_image]
                conditional_image = torch.cat(conditional_image)
            if conditional_image.shape[0] > 0:
                image_hidden_states = self.pixels_to_latents(conditional_image)
                if split_sizes is not None:
                    image_hidden_states = list(
                        torch.split(image_hidden_states, split_sizes)
                    )
                conditional_input = image_hidden_states
                image_hidden_states = None

        encoder_hidden_states, attention_masks, encoder_image_token_mask = (
            self.generation_decoder.prepare_forward_input(
                encoder_hidden_states,
                encoder_image_token_mask=encoder_image_token_mask,
            )
        )

        prompt_embeds = encoder_hidden_states
        prompt_attention_mask = attention_masks

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        def _duplicate(embeds: torch.Tensor, attention_mask: torch.Tensor):
            bs_embed, seq_len, _ = embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            embeds = embeds.repeat(1, num_images_per_prompt, 1)
            embeds = embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            attention_mask = attention_mask.view(bs_embed, -1)
            attention_mask = attention_mask.repeat_interleave(num_images_per_prompt, 0)
            # attention_mask = attention_mask.repeat(num_images_per_prompt, 1)

            return embeds, attention_mask

        prompt_embeds, prompt_attention_mask = _duplicate(
            prompt_embeds, prompt_attention_mask
        )

        if conditional_input is not None:
            conditional_input_new = []
            for c in conditional_input:
                conditional_input_new.extend([c] * num_images_per_prompt)
            conditional_input = conditional_input_new
        if encoder_image_token_mask is not None:
            encoder_image_token_mask = encoder_image_token_mask.repeat_interleave(
                num_images_per_prompt, 0
            )
        if image_fhw_cond is not None:
            image_fhw_cond_new = []
            for c in image_fhw_cond:
                image_fhw_cond_new.extend([c] * num_images_per_prompt)
            image_fhw_cond = image_fhw_cond_new
        if image_grid_thw_gen is not None:
            image_grid_thw_gen = image_grid_thw_gen.repeat_interleave(
                num_images_per_prompt, 0
            )
        if image_grid_thw_gen_cond is not None:
            image_grid_thw_gen_cond_new = []
            for c in image_grid_thw_gen_cond:
                image_grid_thw_gen_cond_new.extend([c] * num_images_per_prompt)
            image_grid_thw_gen_cond = image_grid_thw_gen_cond_new

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_4096_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 16:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            else:
                raise ValueError("Invalid sample size")
            height, width = self.image_processor.classify_height_width_bin(
                height, width, ratios=aspect_ratio_bin
            )

        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default height and width to transformer
        batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        lora_scale = (
            self.attention_kwargs.get("scale", None)
            if self.attention_kwargs is not None
            else None
        )

        # print(self.scheduler, self.scheduler.config)
        # 4. Prepare timesteps
        if timestep_shift is not None:
            self.scheduler.config.flow_shift = timestep_shift
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels // (
            self.transformer.config.patch_size**2
        )
        if image_grid_thw_gen is not None:
            height = max(image_grid_thw_gen[:, 1]) * self.vae_scale_factor
            width = max(image_grid_thw_gen[:, 2]) * self.vae_scale_factor
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt // 3,
            latent_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        transformer_dtype = self.transformer.dtype
        self.set_progress_bar_config(disable=True)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = torch.cat([latents] * 3)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                timestep = timestep * getattr(
                    self.transformer.config, "timestep_scale", 1.0
                )

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input.to(dtype=transformer_dtype),
                    encoder_hidden_states=prompt_embeds.to(dtype=transformer_dtype),
                    encoder_attention_mask=prompt_attention_mask,
                    encoder_image_token_mask=encoder_image_token_mask,
                    image_fhw_cond=image_fhw_cond,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                    conditional_input=conditional_input,
                    image_grid_thw_gen=image_grid_thw_gen,
                    image_grid_thw_gen_cond=image_grid_thw_gen_cond,
                )[0]
                noise_pred = noise_pred.float()

                noise_pred_cond, noise_pred_part_cond, noise_pred_uncond = (
                    noise_pred.chunk(3)
                )

                if timestep_trunc > 0 and t > timestep_trunc:
                    part_diff_norm = torch.norm(
                        noise_pred_part_cond - noise_pred_uncond, dim=1, keepdim=True
                    )
                    diff_norm = torch.norm(
                        noise_pred_cond - noise_pred_part_cond, dim=1, keepdim=True
                    )
                    noise_pred = (
                        noise_pred_uncond
                        + part_cfg_scale
                        * (noise_pred_part_cond - noise_pred_uncond)
                        / self.process_diff_norm(part_diff_norm, k=0.4)
                        + all_cfg_scale
                        * (noise_pred_cond - noise_pred_part_cond)
                        / self.process_diff_norm(diff_norm, k=0.4)
                    )
                else:
                    noise_pred = (
                        noise_pred_uncond
                        + part_cfg_scale * (noise_pred_part_cond - noise_pred_uncond)
                        + all_cfg_scale * (noise_pred_cond - noise_pred_part_cond)
                    )

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            image = self.latents_to_pixels(latents.to(self.generation_decoder.dtype))

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return InternVLUDiffusionPipelineOutput(images=image)

    @staticmethod
    def process_diff_norm(diff_norm, k):
        pow_result = torch.pow(diff_norm, k)

        result = torch.where(
            diff_norm > 1.0,
            pow_result,
            torch.where(diff_norm < 1.0, torch.ones_like(diff_norm), diff_norm),
        )
        return result
