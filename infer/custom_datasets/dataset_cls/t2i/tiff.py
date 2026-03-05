import os
import json
import pandas as pd
import glob

MODEL_NAME='<MODEL_NAME>'

class TIIFBench:
    def __init__(self):
        self.dataset_name = 'tiff'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/TIIF-Bench/data/testmini_prompts'
        self.num_samples = 554
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = {}
        for path in glob.glob(os.path.join(self.ann_path, "*.jsonl")):
            # import pdb;pdb.set_trace()
            jsonl_name =  os.path.basename(path).split('.')[0]

            with open(path, 'r') as f:
                prompts_data = [json.loads(line) for line in f]

            prompts_list = []
            
            for idx, item in enumerate(prompts_data):
                item_id = idx
                long_description = item['long_description']
                short_description = item['short_description']
                prompts_list.append(
                    {
                        'index': idx,
                        'prompt': long_description,
                        'desc_type': 'long_description'
                    }
                )
                prompts_list.append(
                    {
                        'index': idx,
                        'prompt': short_description,
                        'desc_type': 'short_description'
                    }
                )
            
            raw_data[jsonl_name] = prompts_list
        len_raw_data = sum([len(values) for key, values in raw_data.items()])
        assert len_raw_data == self.num_samples, f"Expected {self.num_samples} samples, but got {len_raw_data}"


        # Process data into required format
        processed_data = []
        for sub_dataset_name, prompt_dicts in raw_data.items():
            for prompt_dict in prompt_dicts:
                processed_data.append({
                    'prompt': prompt_dict['prompt'],
                    'seed': 0,
                    'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, sub_dataset_name, prompt_dict['desc_type'], f"{str(int(prompt_dict['index']))}.png")
                })
        return processed_data