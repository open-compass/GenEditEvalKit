# Modified based on Alpha-VLLM/Lumina-DiMOO/inference/inference_i2i.py

import sys
import os
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{dir_name}/Lumina-DiMOO')

import torch
from transformers import AutoConfig, AutoTokenizer
from diffusers import VQModel
from PIL import Image

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line, encode_img_with_paint
from generators.image_generation_generator import generate_image
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates


# Special tokens
MASK = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
EOI = SPECIAL_TOKENS["eoi"]           # End of Image

class LuminaDiMOOT2I:
    def __init__(self, model_path,
                 height=1024, width=1024,
                 timesteps=64, temperature=1.0,
                 cfg_scale=4.0):
        self.model_name = model_path.split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.height = height
        self.width = width
        self.timesteps = timesteps
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.vqvae = VQModel.from_pretrained(model_path, subfolder="vqvae").to('cuda')

    def generate(self, prompt, seed=42):
        setup_seed(seed)
        # generation preparation
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(self.height, self.width)
        templates = create_prompt_templates()
        input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt, templates)
        con_prompt_token = self.tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = self.tokenizer(uncon_prompt)["input_ids"]
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
        img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]
        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device='cuda').unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device='cuda').unsqueeze(0)   
        # generation
        code_start = len(con_prompt_token) + 2
        vq_tokens = generate_image(
            self.model,
            prompt_ids,
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=self.timesteps,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            uncon_ids=uncon_ids,
            code_start=code_start
        )
        out_image = decode_vq_to_image(
            vq_tokens, save_path='', 
            vae_ckpt='', 
            image_height=self.height, 
            image_width=self.width,
            vqvae=self.vqvae
        )

        return out_image

if __name__ == '__main__':
    model = LuminaDiMOOT2I(model_path="...")  # Please specify the correct model path here if you want to run the test

    prompt = 'A plush toy resembling a white dog with large ears and a pink bow tie sits in the center of a snowy landscape. The toy wears a pink and white hat and is surrounded by small pink heart-shaped objects on thesnow, The word "Loveing" is writtenin the snow in front of the toy. The background features a vast expanse of snow with bare trees and a pale, overcast sky. The scene is serene and whimsical, with soft natural lighting and a pastel color palette.'
    image = model.generate(prompt)
    image.save("test_output/lumina_dimoo_t2i.png")