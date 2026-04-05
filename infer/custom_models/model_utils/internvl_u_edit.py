import sys
import os
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{dir_name}/InternVL-U')

import torch
from PIL import Image
from internvlu import InternVLUPipeline

from .tools import split_input_list

class InternVLUEdit:
    def __init__(self, model_path):
        self.pipeline = InternVLUPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    def generate(self, input_list, generation_mode="text_image", seed=42):
        input_image, prompt = split_input_list(input_list)
        with torch.no_grad():
            image = self.pipeline(
                prompt=prompt,
                image=input_image,
                generation_mode=generation_mode,
                height=input_image.size[1],
                width=input_image.size[0],
                generator=torch.Generator(device=self.pipeline.device).manual_seed(seed)
            ).images[0]
        return image

if __name__ == "__main__":
    internvl_u_edit = InternVLUEdit(model_path="../checkpoints/InternVL-U") # Please replace the path with your own model path if needed
    input_image = Image.open("test_output/output_internvlu_t2i.png").convert("RGB")
    prompt = "Change the plant to a rose with throny stem and red petals."
    input_list = [input_image, prompt]

    output_image = internvl_u_edit.generate(input_list, generation_mode="image")
    os.makedirs("test_output", exist_ok=True)
    output_image.save("test_output/output_internvlu_edit.png")