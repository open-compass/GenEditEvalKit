import pickle
from typing import List, Tuple
import random
from accelerate import init_empty_weights
import torch
import os
import numpy as np
import torch.nn as nn
import math
from transformers import AutoTokenizer, AutoConfig
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import LLaDAForMultiModalGeneration
from xllmx.data.item_processor import ItemProcessorBase
from xllmx.solvers.finetune import FinetuneSolverBase



def add_break_line(sequence, H, W, new_number=0):
    result = []
    for i in range(H):
        start = i * W
        end = start + W
        row = sequence[start:end]
        result.extend(row + [new_number])
    return result

def mask_codes(codes, sch="cosine", mask = False, editing = False):
    r = random.uniform(0, 1)
    if len(codes) <= 5 and mask == False:
        mask_ratio=1.0
    elif sch=="cosine":
        mask_ratio = math.cos(r * math.pi / 2) # cosine scheduler
    elif sch=="linear":
        if r < 0.05:
            r = r + 0.05
        mask_ratio = r
    else:
        print("Not Implement")
    num_to_mask = int(len(codes) * mask_ratio)
    if num_to_mask < 1:
        num_to_mask = 1      
    indices_to_mask = random.sample(range(len(codes)), num_to_mask)
    masked_codes = codes[:]
    labels = [-100] * len(codes)
    for index in indices_to_mask:
        labels[index] = codes[index]
        masked_codes[index] = 126336 # <|mdm_mask|> --> for mask token
    return masked_codes, labels

def load_image_tokens(image_path):
    with open(image_path, "rb") as f:
        data_pkl = pickle.load(f)
    assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
    height, width = data_pkl["width"] // 16, data_pkl["height"] // 16
    # add breakline for image
    tokens = add_break_line(data_pkl["input_ids"], height, width, new_number=126084)
    return tokens

class ItemProcessor(ItemProcessorBase):
    def __init__(self, tokenizer, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
        # Understanding Data
        if data_item["user_image"] != "" and data_item["answer_image"] == "":
            """
                Final instruction template: 
                <system> system prompt </system> <user> user prompt <IMAGE> image tokens</IMAGE> </user> 
            """
            instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item["user_prompt"] + "</user>"
            instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
            
            image_tokens = load_image_tokens(data_item["user_image"])
            # 126349 --> <Image>, 126350 --> </Image>
            instruction_token = instruction_token[:-1] + [126349] + image_tokens + [126350] + instruction_token[-1:]
            instruction_label = [-100] * len(instruction_token)

            """
                Final answer template: 
                <answer> answer token </answer> <padding> <padding> <padding> <padding> .....
            """
            answer = data_item["answer_text"] + "</answer>"
            answer_token = self.tokenizer(answer, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
            answer_token, answer_label = mask_codes(answer_token)
            padding_len = 1024 - len(answer_token)
            # 126339 for padding
            padding_token = [126339] * padding_len
            padding_token, padding_label = mask_codes(padding_token, mask=True)
            # 126354 --> <answer>
            all_token = instruction_token + [126354] + answer_token + padding_token
            all_label = instruction_label + [-100] + answer_label + padding_label
        
        # Text-to-Image Data
        elif data_item["user_image"] == "" and data_item["answer_image"] != "":
            # CFG --> drop 10% text prompt
            if np.random.rand() < 0.1:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
            else:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item["user_prompt"] + "</user>"
            instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
            instruction_label = [-100] * len(instruction_token)
            
            with open(data_item["answer_image"], "rb") as f:
                data_pkl = pickle.load(f)
            image_tokens = data_pkl["input_ids"]
            assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
            image_height, image_width = data_pkl["width"]//16, data_pkl["height"]//16
            image_masked_codes, image_labels = mask_codes(image_tokens)
            image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
            image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)
            all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
            all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

        # Image-to-Image Data
        elif data_item["user_image"] != "" and data_item["answer_image"] != "":
            if np.random.rand() < 0.1:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + "<uncondition>" + "</user>"
                instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                instruction_label = [-100] * len(instruction_token)
            else:
                instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item["user_prompt"] + "</user>"
                instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                with open(data_item["user_image"], "rb") as f:
                    data_pkl = pickle.load(f)
                image_tokens = data_pkl["input_ids"]
                assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
                image_height, image_width = data_pkl["width"]//16, data_pkl["height"]//16
                image_tokens = add_break_line(image_tokens, image_height, image_width, new_number=126084)
                instruction_token = instruction_token[:-1] + [126349] + image_tokens + [126350] + instruction_token[-1:]
                instruction_label = [-100] * len(instruction_token)
            with open(data_item["answer_image"], "rb") as f:
                data_pkl = pickle.load(f)
            image_tokens = data_pkl["input_ids"]
            assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
            image_height, image_width = data_pkl["width"]//16, data_pkl["height"]//16
            image_masked_codes, image_labels = mask_codes(image_tokens)
            image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
            image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)
            all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
            all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]
        return all_token, all_label

    def predict_item_token_length(self, data_item: dict) -> int:
        # breakpoint()
        if "token" in data_item:
            return len(data_item["token"])
        elif "len" in data_item:
            return data_item["len"]
        else:
            raise ValueError()


class Solver(FinetuneSolverBase):
    @classmethod
    def get_args_parser(cls):
        parser = super().get_args_parser()
        # task-specific parameters
        parser.add_argument("--max_seq_len", default=1024, type=int, help="max token length")
        parser.add_argument("--dropout", type=float, default=0.05)
        return parser

    def _model_func(
        self,
        init_from: str,
    ) -> (LLaDAForMultiModalGeneration, None):
        # Final SFT
        tokenizer = AutoTokenizer.from_pretrained(init_from, trust_remote_code=True)
        model = LLaDAForMultiModalGeneration.from_pretrained(init_from, torch_dtype=torch.bfloat16, device_map="cpu")
        model.model.set_activation_checkpointing("whole_layer")
        return model, tokenizer

    def _item_processor_func(self, tokenizer=None, max_len=None) -> ItemProcessorBase:
        return ItemProcessor(tokenizer, max_len)

    def _make_and_save_starting_point(self, save_path: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.args.init_from, trust_remote_code=True)
        base_config = AutoConfig.from_pretrained(self.args.init_from)
        model = LLaDAForMultiModalGeneration(base_config)
        model.resize_token_embeddings(len(tokenizer))
        model.model.transformer.ff_out = torch.nn.Linear(4096, len(tokenizer), bias=False) # model dim --> 4096

if __name__ == "__main__":
    args = Solver.get_args_parser().parse_args()
    solver = Solver(args)
    solver.run()
