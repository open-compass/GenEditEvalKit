import sys
import os
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{dir_name}/Lumina-DiMOO')

from PIL import Image
import torch
from transformers import AutoConfig, AutoTokenizer

from .tools import split_input_list

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.generation_utils import setup_seed
from utils.image_utils import preprocess_image, decode_vq_to_image, calculate_vq_params, generate_crop_size_list, var_center_crop, add_break_line, encode_img_with_breaks
from generators.image_to_image_generator import generate_i2i
from utils.prompt_utils import generate_image_to_image_prompt, create_prompt_templates

from diffusers import VQModel

# Special tokens
MASK = SPECIAL_TOKENS["mask_token"]
NEW_LINE = SPECIAL_TOKENS["newline_token"]
BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
EOI = SPECIAL_TOKENS["eoi"]           # End of Image

class LuminaDiMOOEdit:
    def __init__(self, model_path,
                 timesteps=64, temperature=1.0,
                 cfg_scale=2.5, cfg_img=4.0):
        self.model_name = model_path.split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = LLaDAForMultiModalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map='auto') # Load model and tokenizer
        self.vqvae = VQModel.from_pretrained(model_path, subfolder="vqvae").to('cuda')  # Load VQ-VAE
        self.vae_scale = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.timesteps = timesteps
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.cfg_img = cfg_img

    def generate(self, input_list, seed=42):
        setup_seed(seed)
        input_image, prompt = split_input_list(input_list)

        # process prompt and image
        templates = create_prompt_templates()
        edit_type = 'edit' # Default to 'edit' type
        input_prompt, uncon_text, system_prompt = generate_image_to_image_prompt(
            prompt, edit_type, templates
        )
        prompt_ids = self.tokenizer(input_prompt)["input_ids"]
        uncon_text_ids = self.tokenizer(uncon_text)["input_ids"]
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        img = var_center_crop(input_image, crop_size_list=crop_size_list)
        image_width, image_height = img.size
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(image_height, image_width, self.vae_scale)
        # generation preparation
        input_img_token = encode_img_with_breaks(img, self.vqvae)
        con_input = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
        uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
        uncon_input_image = prompt_ids
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
        img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]
        code_start = len(con_input) + 2
        con_input = torch.tensor(con_input + img_pred_token, device='cuda').unsqueeze(0)
        uncon_input_text = torch.tensor(uncon_input_text, device='cuda').unsqueeze(0)
        uncon_input_image = torch.tensor(uncon_input_image, device='cuda').unsqueeze(0)
        # generate
        vq_tokens = generate_i2i(
            self.model,
            con_input,
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=self.timesteps,
            temperature=self.temperature,
            cfg_scale=self.cfg_scale,
            cfg_img=self.cfg_img,
            uncon_text=uncon_input_text,
            uncon_image=uncon_input_image,
            code_start=code_start
        )
        out_img = decode_vq_to_image(
            vq_tokens, save_path='', 
            vae_ckpt='', 
            image_height=image_height, 
            image_width=image_width, 
            vqvae=self.vqvae
        )

        return out_img

if __name__ == '__main__':
    model = LuminaDiMOOEdit(model_path="...")  # Please specify the correct model path here if you want to run the test
    image = Image.open("Lumina-DiMOO/examples/example_4.png").convert("RGB")
    prompt = "Add a beige shed with brown trim and double doors with a diamond pattern in the center-right, occupying more than a third of the image."
    input_list = [image, prompt]
    output_image = model.generate(input_list=input_list, seed=42)
    output_image.save("test_output/lumina_dimoo_edit.png")