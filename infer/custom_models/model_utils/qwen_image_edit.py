import os
from PIL import Image
import torch
dir_name = os.path.dirname(os.path.abspath(__file__))

from diffusers import QwenImageEditPipeline

from .tools import split_input_list

class QwenImageEdit:
    def __init__(self, model_path,
                 true_cfg_scale=4.0,
                 negative_prompt=" ",
                 num_inference_steps=50):
        self.model_name = model_path.split("/")[-1]
        self.pipeline = QwenImageEditPipeline.from_pretrained(model_path).to(torch.bfloat16).to("cuda")
        self.pipeline.set_progress_bar_config(disable=None)
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps

    def generate(self, input_list, seed=42):
        input_image, prompt = split_input_list(input_list)
        inputs = {
            "image": input_image,
            "prompt": prompt,
            "generator": torch.manual_seed(seed),
            "true_cfg_scale": self.true_cfg_scale,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": self.num_inference_steps,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]

        return output_image
        
        
if __name__ == "__main__":
    model = QwenImageEdit(model_path="...")  # Please change to your model path
    
    image = Image.open("test_output/qwen_image_t2i.png").convert("RGB")
    prompt = "Please change the word '通义千问' to '书生万象', and change the word 'Qwen' to 'Intern'."
    input_list = [image, prompt]
    output_image = model.generate(input_list, seed=42)

    output_image.save("test_output/qwen_image_edit.png")