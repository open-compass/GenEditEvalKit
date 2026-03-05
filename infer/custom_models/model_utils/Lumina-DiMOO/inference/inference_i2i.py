# -*- coding: utf-8 -*-
"""
Image-to-image inference script
"""
import os
import json
import argparse
import time
from PIL import Image
import torch
from transformers import AutoConfig, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import preprocess_image, decode_vq_to_image, calculate_vq_params, generate_crop_size_list, var_center_crop, add_break_line, encode_img_with_breaks
from generators.image_to_image_generator import generate_i2i
from utils.prompt_utils import generate_image_to_image_prompt, create_prompt_templates


def main():
    parser = argparse.ArgumentParser(description="Image-to-image inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--ref_image_path", type=str, default=None, help="Input image path for image-reference style transfer")
    parser.add_argument("--edit_type", type=str, default="canny_pred", help="Edit type")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=2.5, help="CFG scale")
    parser.add_argument("--cfg_img", type=float, default=4.0, help="Image CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--output_dir", type=str, default="results_image_to_image", help="Output directory")
    
    args = parser.parse_args()
    
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image

    # Set Random seed
    if args.seed != 0:
        setup_seed(args.seed)
    
    # Create Output directory
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
    
    # Get prompt templates
    templates = create_prompt_templates()
    
    # Get prompt and image
    prompt_text = args.prompt
    edit_type = args.edit_type

    # Generate prompts using utility function
    # Support edit type: dense prediction, controllable generation, subject driven, multi-view generation, editing, reference style transfer
    input_prompt, uncon_text, system_prompt = generate_image_to_image_prompt(
        prompt_text, edit_type, templates
    )
    
    # Handle special cases for different edit types
    if "image_ref_transfer" in edit_type:
        input_img, input_ref = args.image_path, args.ref_image_path
        img = Image.open(input_img).convert("RGB")
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        img = var_center_crop(img, crop_size_list=crop_size_list)
        img_token_input = encode_img_with_breaks(img, vqvae)
        input_image = input_ref
    else:
        input_image = args.image_path
    
    prompt_ids = tokenizer(input_prompt)["input_ids"]
    uncon_text_ids = tokenizer(uncon_text)["input_ids"]
    
    # Read and preprocess image
    img = Image.open(input_image).convert("RGB")
    crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
    img = var_center_crop(img, crop_size_list=crop_size_list)
    
    image_width, image_height = img.size
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(image_height, image_width, vae_scale)
    
    # Encode image to tokens
    input_img_token = encode_img_with_breaks(img, vqvae)
    
    # Conditional/unconditional input concatenation
    if "image_ref_transfer" in edit_type:
        con_input = prompt_ids[:-1] + img_token_input + input_img_token + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + img_token_input + input_img_token + uncon_text_ids[-1:]
    else:
        con_input = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
    uncon_input_image = prompt_ids
    
    # Build image mask predition
    img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
    img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

    # Prediction image token satrt index
    code_start = len(con_input) + 2 

    con_input = torch.tensor(con_input + img_pred_token, device=device).unsqueeze(0)
    uncon_input_text = torch.tensor(uncon_input_text, device=device).unsqueeze(0)
    uncon_input_image = torch.tensor(uncon_input_image, device=device).unsqueeze(0)

    # Generate
    start_time = time.time()
    vq_tokens = generate_i2i(
        model,
        con_input,
        seq_len=seq_len,
        newline_every=newline_every,
        timesteps=args.timesteps,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        cfg_img=args.cfg_img,
        uncon_text=uncon_input_text,
        uncon_image=uncon_input_image,
        code_start=code_start
    )
    
    # Generate filename
    words = (prompt_text or "").split()
    filename_words = words[:10] if len(words) > 10 else words
    filename = "_".join(filename_words)
    filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    filename = f"{filename}_{image_height}x{image_width}_t{args.timesteps}_cfg{args.cfg_scale}_cfgimg{args.cfg_img}_{edit_type}.png"
    
    save_path = os.path.join(args.output_dir, filename)
    
    # Decode and save
    out_img = decode_vq_to_image(
        vq_tokens, save_path, 
        vae_ckpt=args.vae_ckpt, 
        image_height=image_height, 
        image_width=image_width, 
        vqvae=vqvae
    )
    
    # Create side-by-side image (original + generated)
    w1, h1 = img.size
    w2, h2 = out_img.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
    canvas.paste(img, (0, 0))
    canvas.paste(out_img, (w1, 0))
    concat_path = save_path.replace(".png", "_concat.png")
    canvas.save(concat_path)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"[✓] Saved {concat_path} (Time {elapsed_time:.2f}s)")
    
   

if __name__ == '__main__':
    main()
