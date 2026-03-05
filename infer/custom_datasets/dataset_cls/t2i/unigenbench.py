import os
import json
import pandas as pd
import glob

MODEL_NAME='<MODEL_NAME>'

class UniGeBench:
    def __init__(self):
        self.dataset_name = 'unigenbench'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/UniGenBench/data'
        self.num_samples = 1200
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = {}
        for path in glob.glob(os.path.join(self.ann_path, "*.csv")):
            # import pdb;pdb.set_trace()
            csv_name =  os.path.basename(path).split('.')[0]
            prompts_data = pd.read_csv(path)
            prompts_list = []
            # for _, row in prompts_data.iterrows():
            #     prompts_list.append({'index': row['index'], 'prompt': row['prompt_en']})
            prompts_list = [
                {'index': row['index'],
                'prompt': row.get('prompt_en') or row.get('prompt_zh') or ''}
                for _, row in prompts_data.iterrows()
            ]
            raw_data[csv_name] = prompts_list
        len_raw_data = sum([len(values) for key, values in raw_data.items()])
        assert len_raw_data == self.num_samples, f"Expected {self.num_samples} samples, but got {len_raw_data}"


        # Process data into required format
        processed_data = []
        for sub_dataset_name, prompt_dicts in raw_data.items():
            for prompt_dict in prompt_dicts:
                for it in range(4):
                    processed_data.append({
                        'prompt': prompt_dict['prompt'],
                        'seed': it+3407,
                        'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, sub_dataset_name, 'samples', f"{str(int(prompt_dict['index']))}_{it}.png")
                    })



        return processed_data