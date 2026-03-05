import os
import sys
sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

from argparse import ArgumentParser
import json
import math
import pickle
from PIL import Image
import random
import pandas as pd

from data.item_processor import DimooItemProcessor
from data.item_processor import var_center_crop, generate_crop_size_list


class ItemProcessor(DimooItemProcessor):
    """Task-aware pre-tokenization adapter built on top of `DimooItemProcessor`.

    This subclass converts diverse raw dataset items into the standard format
    expected by the base processor and downstream training code. It handles
    multiple task types and image preparation strategies:

    - "t2i": Text-to-image generation with a single target image.
    - "edit": Image editing with an original and an edited image.
    - "mmu_single_image": Multimodal understanding with one image.
    - "mmu_multi_image": Multimodal understanding with multiple images.
    """
    def __init__(
        self,
        tokenizer=".Alpha-VLLM/Lumina-DiMOO",
        vq_ckpt_path="Alpha-VLLM/Lumina-DiMOO",
        target_size=512,
    ):
        super().__init__(tokenizer, vq_ckpt_path, target_size)
        print(self.crop_size_list)

    def process_item(self, raw_item, task_type=None):
        """Convert a raw item dict into tokenized image records.

        Args:
            raw_item: A dict describing the sample. Recognized keys include
                - "image_path": str | list[str] | PIL.Image depending on task
                - "edit_path": str, required when task_type == "edit"
                - one of {"prompt", "caption", "instruction"} for text
            task_type: One of {"t2i", "edit", "mmu_single_image",
                "mmu_multi_image"}. Controls image loading/cropping logic.

        Returns:
            The output of `DimooItemProcessor.process_item`, which contains
            image token ids and metadata. For multi-image cases, a list of
            such dicts is returned; otherwise a single dict.
        """

        # Add custom codes here to convert raw_item to the standard format
        # The standard format contains the "conversations" and "image" keys

        # ********* <start>  Add your custom codes here *******
        if "image_path" in raw_item and task_type == "t2i":
            image = Image.open(raw_item["image_path"])
            img_path = raw_item["image_path"]
        elif "image_path" in raw_item and task_type == "edit":
            img_ori, img_edit = raw_item['image_path'], raw_item['edit_path']
            img_ori = Image.open(img_ori)
            crop_size_list = generate_crop_size_list((512 //32) ** 2, 32)
            img_ori = var_center_crop(img_ori, crop_size_list=crop_size_list)
            img_edit = Image.open(rimg_edit)
            crop_size_list = generate_crop_size_list((512 //32) ** 2, 32)
            img_edit = var_center_crop(img_edit, crop_size_list=crop_size_list)
            img_path = [img_ori, img_edit]
        elif "image_path" in raw_item and task_type == "mmu_single_image":
            image = Image.open(os.path.realpath(raw_item['image']))
            area = image.size[0] * image.size[1]
            if area < 512*512:
                crop_size_list = generate_crop_size_list((512 //32) ** 2, 32)
                image = var_center_crop(image, crop_size_list=crop_size_list)
            elif area >= 2048*2048:
                raise ValueError(f"Image size is too large {raw_item}")
            elif area > 1024*1024:
                crop_size_list = generate_crop_size_list((1024 //32) ** 2, 32)
                image = var_center_crop(image, crop_size_list=crop_size_list)
            else:
                crop_size_list = [(image.size[0]//16*16, image.size[1]//16*16)]
                image = var_center_crop(image, crop_size_list=crop_size_list)
            img_path = image
        elif "image_path" in raw_item and task_type == "mmu_multi_image":
            img_list = []
            for img in raw_item['image_path']:
                image = Image.open(rimg)
                crop_size_list = generate_crop_size_list((512 //32) ** 2, 32)
                image = var_center_crop(image, crop_size_list=crop_size_list)
                img_list.append(image)
            img_path = img_list
        else:
            raise ValueError(f"No image_path found in {raw_item}")

        if "prompt" in raw_item:
            caption = raw_item["prompt"]
        elif "caption" in raw_item:
            caption = raw_item["caption"]
        elif "instruction" in raw_item:
            caption = raw_item['instruction']
        else:
            raise ValueError(f"No prompt/caption/instruction found in {raw_item}")
       
            
        return super(ItemProcessor, self).process_item(caption, img_path)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--splits",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--in_filename",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
    )
    parser.add_argument("--target_size", type=int, default=512)
    args = parser.parse_args()

    item_processor = ItemProcessor(target_size=args.target_size, tokenizer="Alpha-VLLM/Lumina-DiMOO")
    
    if args.in_filename.endswith("jsonl"):
        ori_contents = []
        with open(args.in_filename, 'r') as file:
            for line in file.readlines():
                dic = json.loads(line, strict=False)
                ori_contents.append(dic)
    elif args.in_filename.endswith("json"):
        with open(args.in_filename, 'r') as json_file:
            f = json_file.read()
            ori_contents = json.loads(f)
    else:
        raise ValueError("Input file must be either .json or .jsonl format")


    num = len(ori_contents)

    splits = args.splits
    rank = args.rank
    output_dir = args.out_dir
    save_dir = os.path.join(output_dir, "files")
    os.makedirs(save_dir, exist_ok=True)

    num_per_rank = math.ceil(num / splits)

    try:
        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "r") as f:
            start_idx = int(f.read()) + 1
        print(f"resume from {start_idx}")
    except:
        start_idx = num_per_rank * rank
        print(f"start from {start_idx}")

    end_idx = min(num_per_rank * (rank + 1), len(ori_contents))
    for i in range(start_idx, end_idx):
        if i % 10 == 0:
            print(f"{i}/{end_idx}")
        record = None
        pkl_path = os.path.join(save_dir, f"{i}.pkl")
        try:
            new_item  = item_processor.process_item(ori_contents[i], task_type=args.type)
            if args.type == "t2i":
                with open(pkl_path, "wb") as f:
                    pickle.dump(new_item, f)
                if "prompt" in ori_contents[i]:
                    prompt = ori_contents[i]['prompt']
                record = {
                    "system_prompt": "Generate an image accroding to the text prompt.",
                    "user_prompt": prompt,
                    "user_image": "",
                    "answer_text": "",
                    "answer_image": pkl_path,
                    "answer_thinking": "",
                    "id": i,
                    "len": len(new_item["input_ids"])
                }
            elif "edit" in args.type:
                pkl_path0 = os.path.join(save_dir, f"{i}_0.pkl") ## image_ori token
                pkl_path1 = os.path.join(save_dir, f"{i}_1.pkl") ## image_edit token
                with open(pkl_path0, "wb") as f:
                    pickle.dump(new_item[0], f)
                with open(pkl_path1, "wb") as f:
                    pickle.dump(new_item[1], f)
                record = {
                    "system_prompt": "Generate an image applying the following editing instruction based on the original image.",
                    "user_prompt": ori_contents[i]['prompt'],
                    "user_image": pkl_path0,
                    "answer_text": "",
                    "answer_image": pkl_path1,
                    "answer_thinking": "",
                    "id": i,
                    "len": len(new_item[0]["input_ids"])
                }
            elif args.type == "mmu":
                if isinstance(new_item, list):
                    pkl_path = []
                    for idx, item in enumerate(new_item):
                        path = os.path.join(save_dir, f"{i}_{idx}.pkl")
                        with open(path, "wb") as f:
                            pickle.dump(item, f)
                        pkl_path.append(path)
                else:
                    with open(pkl_path, "wb") as f:
                        pickle.dump(new_item, f)
                for k in ori_contents[i]['conversations']:
                    if k["from"] == "human":
                        user_prompt = k['value'].replace("<image>\n","")
                    elif k["from"] == "gpt":
                        answer_text = k['value']
                    else:
                        print(f"Error key in in conversations: {k}")
                record = {
                    "system_prompt": "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer.",
                    "user_prompt": user_prompt,
                    "user_image": pkl_path,
                    "answer_text": answer_text,
                    "answer_image": "",
                    "answer_thinking": "",
                    "id": i,
                    "len": len(new_item["input_ids"])
                }
            else:
                raise ValueError(f"Error task_type")


        except Exception as e:
            from traceback import format_exc

            print(f"item {i} error: \n{ori_contents[i]}")
            print(format_exc())

        if record is not None:
            with open(os.path.join(output_dir, f"{rank}-of-{splits}-record.jsonl"), "a", encoding="utf-8") as f:
                record_str = json.dumps(record, ensure_ascii=False) + "\n"
                f.write(record_str)

        with open(os.path.join(output_dir, f"{rank}-of-{splits}-progress.txt"), "w", encoding="utf-8") as f:
            if i == end_idx - 1:
                f.write("finished")
            else:
                f.write(f"{i}")
