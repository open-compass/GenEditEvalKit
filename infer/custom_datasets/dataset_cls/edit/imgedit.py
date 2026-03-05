import os
import json
from PIL import Image

MODEL_NAME = '<MODEL_NAME>'

class ImgEdit:
    def __init__(self):
        self.dataset_name = 'imgedit'
        self.dataset_task = 'edit'
        self.ann_path = 'benchmarks/imgedit/eval_prompts/basic_edit.json'
        self.num_samples = 737
        self.image_dir = 'benchmarks/imgedit/imgedit_asset/Benchmark/singleturn'
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = list(json.load(open(self.ann_path)).items())
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for key, metadata in raw_data:
            processed_data.append({
                'input_list': [Image.open(os.path.join(self.image_dir, metadata['id'])), metadata["prompt"]],
                'seed': 0,
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', f"{key}.png")
            })

        return processed_data