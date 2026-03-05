# -*- coding: utf-8 -*-
"""
Text understanding generator
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from utils.generation_utils import add_gumbel_noise, get_num_transfer_tokens


@torch.no_grad()
def generate_text_understanding(
    model,
    prompt, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336, 
    code_start: Optional[int] = None,
):
    """
    Text understanding generation function
    
    Args:
        model: Mask predictor
        prompt: Input prompt tensor (1, L)
        steps: Sampling steps, less than or equal to gen_length
        gen_length: Generated answer length
        block_length: Block length, less than or equal to gen_length
        temperature: Categorical distribution sampling temperature
        cfg_scale: Unsupervised classifier-free guidance scale
        remasking: Remasking strategy 'low_confidence' or 'random'
        mask_id: The token id of [MASK] is 126336
        code_start: Prediction text token satrt index
    """
    device = next(model.parameters()).device

    x = prompt

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, code_start + num_block * block_length: code_start + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_, infer=True).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, infer=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, code_start + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
        
        # early stop
        if (x==126081).sum().item() > 0 and num_blocks > 0:
            return x[:,: input_p_len + prompt.shape[0] + 3 + rows + (num_block + 1) * block_length]
    
    return x
