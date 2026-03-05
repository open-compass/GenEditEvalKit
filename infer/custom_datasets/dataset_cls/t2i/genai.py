import os
import json

MODEL_NAME='<MODEL_NAME>'

class GenAI:
    def __init__(self):
        self.dataset_name = 'genai'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/genai/eval_prompts/genai1600/genai_image.json'
        self.num_samples = 1600
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = json.load(open(self.ann_path))
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data.values():
            processed_data.append({
                'prompt': item['prompt'],
                'seed': 0, 
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', f"{int(item['id']):09d}.jpg")
            })

        return processed_data