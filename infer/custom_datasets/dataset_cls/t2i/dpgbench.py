import os
import json

MODEL_NAME='<MODEL_NAME>'

class DPGBench:
    def __init__(self):
        self.dataset_name = 'dpgbench'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/dpgbench/eval_prompts/dpgbench_prompts.json'
        self.num_samples = 1065
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = json.load(open(self.ann_path, 'r'))
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for filename, prompt in raw_data.items():
            for it in range(4):
                processed_data.append({
                    'prompt': prompt,
                    'seed': it,
                    'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', filename.replace(".txt", ""), f"{it}.png")
                })

        return processed_data