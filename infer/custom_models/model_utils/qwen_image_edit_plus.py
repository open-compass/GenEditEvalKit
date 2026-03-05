import os
from io import BytesIO

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

from .tools import split_input_list

class QwenImageEditPlus:
    def __init__(self, model_path: str,
                 true_cfg_scale: float = 4.0,
                 negative_prompt: str = " ",
                 num_inference_steps: int = 40,
                 guidance_scale: float = 1.0,
                 num_images_per_prompt: int = 1):
        self.model_name = model_path.split("/")[-1]
        self.pipeline = (
            QwenImageEditPlusPipeline
            .from_pretrained(model_path, torch_dtype=torch.bfloat16)
            .to("cuda")
        )
        self.pipeline.set_progress_bar_config(disable=None)

        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.num_images_per_prompt = num_images_per_prompt

    def generate(self, input_list, seed: int = 42) -> Image.Image:
        input_images, prompt = split_input_list(input_list, single_image=False)
        inputs = {
            "image": input_images,
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": self.true_cfg_scale,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "num_images_per_prompt": self.num_images_per_prompt,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]

        return output_image


if __name__ == "__main__":
    import requests

    model = QwenImageEditPlus(
        model_path="Qwen/Qwen-Image-Edit-2511",
        true_cfg_scale=4.0,
        negative_prompt=" ",
        num_inference_steps=40,
        guidance_scale=1.0,
        num_images_per_prompt=1,
    )
    
    image = Image.open("test_output/qwen_image_t2i.png").convert("RGB")
    prompt = "Please change the word '通义千问' to '书生万象', and change the word 'Qwen' to 'Intern'."
    input_list = [image, prompt]
    output_image = model.generate(input_list, seed=42)

    output_image.save("test_output/qwen_image_edit_2511.png")