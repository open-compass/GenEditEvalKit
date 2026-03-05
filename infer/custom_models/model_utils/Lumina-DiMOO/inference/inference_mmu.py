# -*- coding: utf-8 -*-
"""
Text understanding inference script
"""
import os
import json
import argparse
from PIL import Image
import torch
import time
from transformers import AutoConfig, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.image_utils import preprocess_image, encode_img_with_breaks, calculate_vq_params, generate_crop_size_list, var_center_crop, add_break_line
from generators.text_understanding_generator import generate_text_understanding
from utils.prompt_utils import generate_multimodal_understanding_prompt


def main():
    parser = argparse.ArgumentParser(description="Text understanding inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--steps", type=int, default=128, help="Generation steps")
    parser.add_argument("--gen_length", type=int, default=1024, help="Generation length")
    parser.add_argument("--block_length", type=int, default=256, help="Block length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="outputs_text_understanding", help="Output directory")
    
    args = parser.parse_args()

    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
    )
    
    # Load VQ-VAE
    from diffusers import VQModel
    vqvae = VQModel.from_pretrained(args.vae_ckpt, subfolder="vqvae").to(device)

    # Calculate VQ parameters
    vae_scale = 2 ** (len(vqvae.config.block_out_channels) - 1)

    # Get prompt and image
    question = args.prompt
    image_path = args.image_path
    print(f"Processing image: {image_path}")
    print(f"Question: {question}")

    # Generate prompt using utility function
    input_prompt = generate_multimodal_understanding_prompt(question)
    input_ids = tokenizer(input_prompt)['input_ids']
    
    # Preprocess image
    img = Image.open(image_path)
    crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
    image = var_center_crop(img, crop_size_list=crop_size_list)
    image_width, image_height = image.size
    
    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
        image_height, image_width, vae_scale
    )
    
    # Encode image
    input_img_token = encode_img_with_breaks(image, vqvae=vqvae)

    # Build input image token
    img_token = add_break_line(input_img_token, token_grid_height, token_grid_width, new_number = NEW_LINE)
    input_img_token = img_token

    # Build input sequence
    input_token = input_ids[:-1] + input_img_token + input_ids[-1:]

    # Prediction text token start index
    code_start = len(input_token) + 1 

    # Build text mask predition sequence
    input_token = input_token + [BOA] + args.gen_length*[MASK] + [EOA]
    input_ids = torch.tensor(input_token, device=device).unsqueeze(0)
    
    # Generate text
    start_time = time.time()
    out_new = generate_text_understanding(
        model, input_ids,
        steps=args.steps, 
        gen_length=args.gen_length, 
        block_length=args.block_length, 
        temperature=args.temperature, 
        cfg_scale=args.cfg_scale, 
        remasking='low_confidence',
        code_start=code_start
    )

    text_new = tokenizer.batch_decode(
        out_new[:, code_start : -1], 
        skip_special_tokens=True
    )[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[✓] (Time {elapsed_time:.2f}s)")
    
    print(f"Generated text: {text_new}")


if __name__ == '__main__':
    main()
