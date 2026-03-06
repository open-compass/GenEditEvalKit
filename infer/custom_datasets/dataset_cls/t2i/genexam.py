import os
import json

MODEL_NAME='<MODEL_NAME>'

class GenExam:
    def __init__(self):
        self.dataset_name = 'genexam'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/GenExam/data/annotations/All_Subjects.jsonl'
        self.num_samples = 1000
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        with open(self.ann_path, 'r') as f:
            raw_data = [json.loads(line) for line in f]
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data:
            processed_data.append({
                'prompt': item['prompt'],
                'seed': 0,
                'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', f"{item['id']}.png")
            })

        return processed_data