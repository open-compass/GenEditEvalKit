import os
dir_name = os.path.dirname(os.path.abspath(__file__))

from diffusers import QwenImagePipeline
import torch


class QwenImageT2I:
    positive_magic = {
        "en": "Ultra HD, 4K, cinematic composition.", # for english prompt,
        "zh": "超清，4K，电影级构图" # for chinese prompt,
    }
    negative_prompt = " "
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }

    def __init__(self, model_path,
                 aspect_ratio="16:9",
                 num_inference_steps=50,
                 true_cfg_scale=4.0):
        self.model_name = model_path.split('/')[-1]
        self.pipe = QwenImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(torch.device("cuda:0")) # Only use the 0th card visible to this process (restricted by CUDA_VISIBLE_DEVICES) to avoid falling to physical card 0 when using the "cuda" alias

        self.aspect_ratio = aspect_ratio
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale

    def generate(self, prompt, seed=42):
        width, height = self.aspect_ratios[self.aspect_ratio]
        image = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=self.num_inference_steps,
            true_cfg_scale=self.true_cfg_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed)
        ).images[0]
        return image

if __name__=='__main__':
    prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
    negative_prompt = " " # using an empty string if you do not have specific concept to remove

    qwen_image = QwenImageT2I(model_path="...") # Please replace the path to the real model path if necessary.
    image = qwen_image.generate(prompt=prompt, seed=42)
    image.save('test_output/qwen_image_t2i.png')