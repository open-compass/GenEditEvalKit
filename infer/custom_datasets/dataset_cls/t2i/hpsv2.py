import os
import json

MODEL_NAME='<MODEL_NAME>'

class HPSv2:
    def __init__(self):
        self.dataset_name = 'hpsv2'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/HPSv2/prompts_extracted.json'
        self.num_samples = 3200
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = json.load(open(self.ann_path))
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data:
            processed_data.append({
                'prompt': item['prompt'],
                'seed': 0,
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', f"{item['style']}", f"{item['idx']:05d}.jpg")
            })

        return processed_data