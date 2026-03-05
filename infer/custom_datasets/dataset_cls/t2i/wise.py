import os
import json
import glob

MODEL_NAME='<MODEL_NAME>'

class WISE:
    def __init__(self):
        self.dataset_name = 'wise'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/wise/data'
        self.num_samples = 1000
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        json_paths = list(glob.glob(f"{self.ann_path}/*.json"))
        raw_data = []
        for json_path in json_paths:
            with open(json_path, "r") as f:
                raw_data += json.load(f)
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data:
            processed_data.append({
                'prompt': item['Prompt'],
                'seed': 0,
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', f"{item['prompt_id']}.png")
            })

        return processed_data