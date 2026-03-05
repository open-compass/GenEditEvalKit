import os
import json

MODEL_NAME='<MODEL_NAME>'

class GenEval:
    def __init__(self):
        self.dataset_name = 'geneval'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/geneval/prompts/evaluation_metadata.jsonl'
        self.num_samples = 553
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        with open(self.ann_path, 'r') as f:
            raw_data = [json.loads(line) for line in f]
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for idx, item in enumerate(raw_data):
            for it in range(4):
                processed_data.append({
                    'prompt': item['prompt'],
                    'seed': it,
                    'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images', f"{idx:0>5}", "samples", f"{it:05}.png")
                })

        return processed_data