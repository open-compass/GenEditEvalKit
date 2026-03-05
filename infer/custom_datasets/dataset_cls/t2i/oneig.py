import os
import json

MODEL_NAME='<MODEL_NAME>'

class OneIG:
    class_item = {
        "Anime_Stylization" : "anime",
        "Portrait" : "human",
        "General_Object" : "object",
        "Text_Rendering" : "text",
        "Knowledge_Reasoning" : "reasoning",
        "Multilingualism" : "multilingualism"
    }

    def __init__(self):
        self.dataset_name = 'oneig'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/OneIG-Benchmark/OneIG-Bench/OneIG-Bench.json'
        self.num_samples = 1120
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = json.load(open(self.ann_path))
        assert len(raw_data) == self.num_samples, f"Expected {self.num_samples} samples, but got {len(raw_data)}"

        # Process data into required format
        processed_data = []
        for item in raw_data:
            for it in range(4):
                processed_data.append({
                    'prompt': item['prompt_en'],
                    'seed': it, 
                    'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, 'images_before_combination', self.class_item[item['category']], MODEL_NAME, f"{item['id']}_{it}.png") # Save as png first, then convert to webp later
                })

        return processed_data