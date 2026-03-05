import os
import io
import base64
from PIL import Image
from openai import OpenAI

from .tools import split_input_list
# from tools import split_input_list

class OpenAIAPIEdit:
    def __init__(self, model_path, api_key=os.getenv('API_KEY'), base_url=os.getenv('BASE_URL')):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_path

    def pil_image_to_bytes(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return buffered.getvalue()

    def generate(self, input_list, seed=None, **kwargs):
        # OpenAI's image generation does not support setting seed
        input_images, prompt = split_input_list(input_list, single_image=False)
        input_images_bytes = [self.pil_image_to_bytes(img) for img in input_images]
        image = self.client.images.edit(
            model=self.model_name,
            image=input_images_bytes,
            prompt=prompt,
            **kwargs
        )
        image = image.data[0]
        image_base64 = base64.b64decode(image.b64_json)
        image = Image.open(io.BytesIO(image_base64)).convert("RGB")
        return image

if __name__ == "__main__":
    model = OpenAIAPIEdit(model_path="gpt-image-1")
    input_list = [
        "Please change the background from sunset to daytime.",
        Image.open("test_output/openaiapi_t2i.png").convert("RGB")
    ]
    image = model.generate(input_list)
    image.save("test_output/openaiapi_edit.png")