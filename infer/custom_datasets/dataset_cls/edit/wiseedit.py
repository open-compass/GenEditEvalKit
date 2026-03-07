import os
import json
import glob
from PIL import Image

MODEL_NAME = '<MODEL_NAME>'

class WiseEdit:
    def __init__(self):
        self.dataset_name = 'wiseedit'
        self.dataset_task = 'edit'
        self.ann_path = 'benchmarks/WiseEdit/WiseEdit-Benchmark'
        self.num_samples = 1220
        self.image_dir = 'benchmarks/WiseEdit/WiseEdit-Benchmark'
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = []
        for path in glob.glob(os.path.join(self.ann_path, "**/ins.json"), recursive=True):
            subset_name = path.split('/')[-2]
            data_subset = json.load(open(path))
            raw_data.extend([{**item, 'subset': subset_name} for item in data_subset])
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data:
            processed_data.append({
                'input_list': [Image.open(os.path.join(self.image_dir, img_path)) for img_path in item['input']] + [item['prompt']],
                'seed': 0,
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, MODEL_NAME, item['subset'], 'en', f"{item['id']}.png"),
            })
            processed_data.append({
                'input_list': [Image.open(os.path.join(self.image_dir, img_path)) for img_path in item['input']] + [item['promptcn']],
                'seed': 0,
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, MODEL_NAME, item['subset'], 'cn', f"{item['id']}.png"),
            })

        return processed_data