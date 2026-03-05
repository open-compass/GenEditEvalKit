# -*- coding: utf-8 -*-
"""
Image-to-image generator (supports DDP)
"""
import torch
import math
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk


@torch.no_grad()
def generate_i2i(
    model,
    prompt: torch.LongTensor,
    *,
    seq_len: int = 1024,
    newline_every: int = 16,
    timesteps: int = 18,
    mask_token_id: int = 126336,
    newline_id: int = 126084,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    cfg_img: float = 0.0,
    uncon_text: torch.LongTensor,
    uncon_image: torch.LongTensor,
    code_start: Optional[int] = None,
    codebook_size: int = 8192,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    text_vocab_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.LongTensor:
    """
    Image-to-image MaskGit generation (supports CFG for text and image)
    
    Args:
        model: Model
        prompt: Prompt tensor
        seq_len: Sequence length
        newline_every: Newline interval per row
        timesteps: Number of timesteps
        mask_token_id: Mask token id
        newline_id: Newline token id
        temperature: Temperature
        cfg_scale: Text CFG scale
        cfg_img: Image CFG scale
        code_start: Prediction image token satrt index
        uncon_text: Unconditional text input
        uncon_image: Unconditional image input
        codebook_size: Codebook size
        noise_schedule: Noise schedule function
        text_vocab_size: Text vocabulary size
        generator: Random number generator
    
    Returns:
        Final VQ codes (1, seq_len)
    """
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    B, P = prompt.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"
    
    x = prompt

    vq_mask = x == mask_token_id
    unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
    vq_len = unknown_cnt

    # Infer text vocabulary size
    if text_vocab_size is None:
        vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
        text_vocab_size = vocab_total - codebook_size
    vocab_offset = text_vocab_size

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # Calculate number of tokens to keep (continue masking) this round
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)

        # Forward pass (with/without CFG)
        if cfg_scale > 0 or cfg_img > 0:
            # CFG text
            uncond_text = torch.cat((uncon_text.to(x.device), x[:, code_start-2:]), dim=1)
            uncond_text_vq_mask = torch.cat((torch.zeros((1, uncon_text.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start-2:]), dim=1)
            # CFG image
            uncond_img = torch.cat((uncon_image.to(x.device), x[:, code_start-2:]), dim=1)
            uncond_img_vq_mask = torch.cat((torch.zeros((1, uncon_image.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start-2:]), dim=1)

            cond_logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            uncond_logits_text = model(uncond_text, infer=True).logits[:, uncond_text_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            uncond_logits_img = model(uncond_img, infer=True).logits[:, uncond_img_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
            logits = cond_logits + cfg_scale * (cond_logits - uncond_logits_text) + cfg_img * (cond_logits - uncond_logits_img)
        else:
            logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

        sampled = gumbel_max_sample(logits, temperature, generator=generator)
        sampled_full = sampled + vocab_offset
        probs = torch.softmax(logits, dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
        x.view(-1)[flat_idx] = sampled_full.view(-1)

        conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
        conf_map.view(-1)[flat_idx] = conf.view(-1)

        mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
        x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

    # Remove newline tokens
    vq_ids = x[0, code_start:-2]
    vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
    return vq_ids
