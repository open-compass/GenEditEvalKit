import os
import io
import base64
from PIL import Image
from openai import OpenAI

class OpenAIAPIT2I:
    def __init__(self, model_path, api_key=os.getenv('API_KEY'), base_url=os.getenv('BASE_URL')):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_path

    def generate(self, prompt, seed=None, **kwargs):
        # OpenAI's image generation does not support setting seed
        image = self.client.images.generate(
            model=self.model_name,
            prompt=prompt,
            **kwargs
        )
        image = image.data[0]
        image_base64 = base64.b64decode(image.b64_json)
        image = Image.open(io.BytesIO(image_base64)).convert("RGB")
        return image

if __name__ == "__main__":
    model = OpenAIAPIT2I(model_path="gpt-image-1")
    image = model.generate(prompt="A futuristic cityscape at sunset", seed=123)
    image.save("test_output/openaiapi_t2i.png")