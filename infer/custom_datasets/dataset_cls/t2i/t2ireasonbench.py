import os
import json

MODEL_NAME='<MODEL_NAME>'

class T2IReasonBench:
    def __init__(self):
        self.dataset_name = 't2ireasonbench'
        self.dataset_task = 't2i'
        self.ann_path = 'benchmarks/T2I-ReasonBench/prompts'
        self.num_samples = 800
        self.data = self.__load_data()

    def __load_data(self):
        # Load raw data
        raw_data = {}
        data_names = ['entity_reasoning', 'idiom_interpretation', 'scientific_reasoning', 'textual_image_design']
        for data_name in data_names:
            prompts_data = json.loads(open(os.path.join(self.ann_path, f"{data_name}.json"), 'r').read())
            prompts_list = []
            for item in prompts_data:
                prompts_list.append({'id': item['id'], 'prompt': item['prompt']})
            raw_data[data_name] = prompts_list
        len_raw_data = sum([len(values) for key, values in raw_data.items()])
        assert len_raw_data == self.num_samples, f"Expected {self.num_samples} samples, but got {len_raw_data}"

        # Process data into required format
        processed_data = []
        for sub_dataset_name, prompt_dicts in raw_data.items():
            for prompt_dict in prompt_dicts:
                processed_data.append({
                    'prompt': prompt_dict['prompt'],
                    'seed': 0,
                    'image_save_path': os.path.join(MODEL_NAME, self.dataset_name, sub_dataset_name, f"{prompt_dict['id']:04d}.png")
                })

        return processed_data