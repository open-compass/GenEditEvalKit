import os
import json
from PIL import Image

MODEL_NAME = '<MODEL_NAME>'

class RISE:
    def __init__(self):
        self.dataset_name = 'rise'
        self.dataset_task = 'edit'
        self.ann_path = 'benchmarks/RISEBench/data/datav2_total_w_subtask.json'
        self.num_samples = 360
        self.image_dir = 'benchmarks/RISEBench/data'
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = json.load(open(self.ann_path))
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data:
            processed_data.append({
                'input_list': [Image.open(os.path.join(self.image_dir, item['image'])), item['instruction']], 
                'seed': 0, 
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'outputs', MODEL_NAME, 'images', "{}/{}.png".format(item['category'], item['index']))
            })

        return processed_data