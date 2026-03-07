import os
import json
import glob
from PIL import Image

MODEL_NAME = '<MODEL_NAME>'

class KRISBench:
    def __init__(self):
        self.dataset_name = 'krisbench'
        self.dataset_task = 'edit'
        self.ann_path = 'benchmarks/Kris_Bench/KRIS_Bench'
        self.num_samples = 1267
        self.image_dir = 'benchmarks/Kris_Bench/KRIS_Bench'
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = []
        for path in glob.glob(os.path.join(self.ann_path, "*/annotation.json")):
            category = os.path.basename(os.path.dirname(path))
            data_category = json.load(open(path))
            for key, value in data_category.items():
                item = value.copy()
                item['category'] = category
                item['id'] = key
                raw_data.append(item)
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data:
            if isinstance(item['ori_img'], str):
                processed_data.append({
                    'input_list': [Image.open(os.path.join(self.image_dir, item['category'], item['ori_img'])), item['ins_en']],
                    'seed': 0,
                    'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', MODEL_NAME, f"{item['category']}", f"{item['id']}.jpg")
                })
            elif isinstance(item['ori_img'], list):
                processed_data.append({
                    'input_list': [Image.open(os.path.join(self.image_dir, item['category'], img_path)) for img_path in item['ori_img']] + [item['ins_en']],
                    'seed': 0,
                    'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', MODEL_NAME, f"{item['category']}", f"{item['id']}.jpg")
                })
            else:
                raise ValueError(f"ori_img must be either a string or a list of strings. Got: {item['ori_img']}")

        return processed_data