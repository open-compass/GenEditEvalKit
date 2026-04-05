import sys
import os
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{dir_name}/InternVL-U')

import torch
from internvlu import InternVLUPipeline

class InternVLUT2I:
    def __init__(self, model_path):
        self.pipeline = InternVLUPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to("cuda")

    def generate(self, prompt, generation_mode="text_image", seed=42):
        with torch.no_grad():
            output = self.pipeline(
                prompt=prompt,
                generation_mode=generation_mode,
                generator=torch.Generator(device="cuda").manual_seed(seed)
            )
            # output_text = output.generate_output[0]
            image = output.images[0]
        return image

if __name__ == "__main__":
    internvlu_t2i = InternVLUT2I(model_path="../checkpoints/InternVL-U")  # Please replace the path with your own model path if needed

    prompt = "生成一张展现希望正在生长、充满力量感的画面。"
    image = internvlu_t2i.generate(prompt)
    os.makedirs("test_output", exist_ok=True)
    image.save("test_output/output_internvlu_t2i.png")