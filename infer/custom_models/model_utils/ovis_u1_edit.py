import sys
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from .tools import split_input_list

class OvisU1Edit:
    def __init__(self, model_path,
                 steps=50, txt_cfg=6.0, img_cfg=1.5):
        self.model_name = model_path.split('/')[-1]
        self.model, loading_info = AutoModelForCausalLM.from_pretrained(model_path,
                                                torch_dtype=torch.bfloat16,
                                                output_loading_info=True,
                                                trust_remote_code=True
                                                )
        print(f'Loading info of Ovis-U1:\n{loading_info}')
        self.model = self.model.eval().to("cuda")
        self.model = self.model.to(torch.bfloat16)

        self.steps = steps
        self.txt_cfg = txt_cfg
        self.img_cfg = img_cfg

    def load_blank_image(self, width, height):
        pil_image = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')
        return pil_image

    def build_inputs(self, model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width, target_height):
        if pil_image is not None:
            target_size = (int(target_width), int(target_height))
            pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(pil_image, target_size)
            cond_img_ids[..., 0] = 1.0
            vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
            width = pil_image.width
            height = pil_image.height
            resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
            pil_image = pil_image.resize((resized_width, resized_height))
        else:
            vae_pixel_values = None
            cond_img_ids = None

        prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            prompt, 
            [pil_image], 
            generation_preface=None,
            return_labels=False,
            propagate_exception=False,
            multimodal_type='single_image',
            fix_sample_overall_length_navit=False
            )
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = torch.cat([
                pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
            ],dim=0)
        if grid_thws is not None:
            grid_thws = torch.cat([
                grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
            ],dim=0)
        return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values

    def pipe_img_edit(self, model, input_img, prompt, steps, txt_cfg, img_cfg, seed):
        text_tokenizer = model.get_text_tokenizer()
        visual_tokenizer = model.get_visual_tokenizer()
        
        width, height = input_img.size
        height, width = visual_tokenizer.smart_resize(height, width, factor=32)

        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=text_tokenizer.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True,
            height=height,
            width=width,
            num_steps=steps,
            seed=seed,
            img_cfg=img_cfg,
            txt_cfg=txt_cfg,
        )
        uncond_image = self.load_blank_image(width, height)
        uncond_prompt = "<image>\nGenerate an image."
        input_ids, pixel_values, attention_mask, grid_thws, _ = self.build_inputs(model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
        with torch.inference_mode():
            no_both_cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)

        input_img = input_img.resize((width, height))
        prompt = "<image>\n" + prompt.strip()
        with torch.inference_mode():
            input_ids, pixel_values, attention_mask, grid_thws, _ = self.build_inputs(model, text_tokenizer, visual_tokenizer, uncond_prompt, input_img, width, height)
            no_txt_cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs) 

        input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = self.build_inputs(model, text_tokenizer, visual_tokenizer, prompt, input_img, width, height)
        with torch.inference_mode():
            cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)
            cond["vae_pixel_values"] = vae_pixel_values
            images = model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
        return images
    
    def generate(self, input_list, seed=42):
        input_image, prompt = split_input_list(input_list)
        input_image = input_image.convert("RGB") # To prevent some images from not being in RGB format, causing errors
        images = self.pipe_img_edit(self.model, input_image, prompt,
                                    steps=self.steps, txt_cfg=self.txt_cfg,
                                    img_cfg=self.img_cfg, seed=seed)
        assert len(images) == 1, f"Please generate only 1 image once during evaluation. Now len(images)={len(images)}"
        image = images[0]

        return image

if __name__=="__main__":
    ovis_u1_edit = OvisU1Edit(model_path="...") # Please replace the path with your own model path if needed

    input_image = Image.open("../../../../benchmarks/RISEBench/data/logical_reasoning_images/56.png").convert("RGB")
    prompt = "A WordDescription Search game is shown in the image. \nThe task is to find the given words hidden in the grid. Rules:\n1. Words can be hidden horizontally, vertically, or diagonally.\n2. Words can read forwards or backwards.\n3. Words always follow a straight line (no zigzagging)\n\nCircle the location of \"HORSE\" with a red marker."
    input_list = [input_image, prompt]
    image = ovis_u1_edit.generate(input_list)
    image.save("test_output/output_ovis_u1_edit.png")