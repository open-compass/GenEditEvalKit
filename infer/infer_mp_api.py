# Supporting infer with OpenAI's API.

import multiprocessing as mp
import json
import os
from PIL import Image
from tqdm import tqdm
import argparse

from custom_datasets.load_dataset import load_dataset
from custom_models.model_utils.openai_api_edit import OpenAIAPIEdit
from custom_models.model_utils.openai_api_t2i import OpenAIAPIT2I

MODEL_NAME = '<MODEL_NAME>'


def create_blank_image(width, height):
    blank_image = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')
    return blank_image

def parse_model_kwargs(model_kwargs):
    # Parse model kwargs from string to dictionary. Currently only supports int and str types.
    if model_kwargs is None or model_kwargs=="":
        return {}
    else:
        kwargs = {}
        for pair in model_kwargs.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Try to convert value to float or int if possible, otherwise keep as string
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass # keep as string
                kwargs[key] = value
        return kwargs

def inference_single_worker(args, worker_id, items_chunk):
    if args.dataset_task == 't2i':
        model = OpenAIAPIT2I(args.model_name, **args.custom_model_kwargs)
    elif args.dataset_task == 'edit':
        model = OpenAIAPIEdit(args.model_name, **args.custom_model_kwargs)
    
    pbar = tqdm(items_chunk, desc=f"Worker {worker_id}", position=worker_id)
    for item in pbar:
        image_save_path = os.path.join(args.image_save_dir, item['image_save_path'])
        image_save_path = image_save_path.replace(MODEL_NAME, args.model_name)
        if os.path.exists(image_save_path):
            pbar.set_postfix(status="skipped")
            continue

        output_image = None
        if args.dataset_task == 't2i':
            try:
                output_image = model.generate(item['prompt'], seed=args.seed+item['seed'])
            except Exception as e:
                print("\n", f"T2I error: {e}", f"Prompt: {item['prompt']}", f"Save Path: {image_save_path}", "\n", sep="\n")
                if args.generate_blank_image_on_error:
                    w = getattr(model, 'width', 256)
                    h = getattr(model, 'height', 256)
                    output_image = create_blank_image(w, h)
        elif args.dataset_task == 'edit':
            try:
                output_image = model.generate(item['input_list'], seed=args.seed+item['seed'])
            except Exception as e:
                print("\n", f"Editing error: {e}", "Editing Prompt: {}".format('\n'.join([input_data for input_data in item['input_list'] if isinstance(input_data, str)])), f"Save Path: {image_save_path}", "\n", sep="\n")
                if args.generate_blank_image_on_error:
                    for input_data in item['input_list']: # find first image to get size
                        if isinstance(input_data, Image.Image):
                            width, height = input_data.size
                            break
                    output_image = create_blank_image(width, height)

        if output_image: # Pass the samples where generation failed and no blank image is to be created
            os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
            output_image.save(image_save_path)
            pbar.set_postfix(status="done")

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--image_save_dir', type=str, default='output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--generate_blank_image_on_error', action='store_true')
    parser.add_argument('--custom_model_kwargs', type=str, default=None, help='(optional) Extra parameters to pass to the model, e.g., "top_p=0.8,temperature=0.6"')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(), help='Number of worker processes to use for inference. Default is the number of CPU cores.')
    args = parser.parse_args()

    # Parse user custom parameters
    args.custom_model_kwargs = parse_model_kwargs(args.custom_model_kwargs)

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    args.dataset_task = dataset.dataset_task
    chunks = [[] for _ in range(args.num_workers)]
    for idx, item in enumerate(dataset.data):
        chunks[idx % args.num_workers].append(item)

    # Run inference including model loading and inference
    processes = []
    for worker_id in range(args.num_workers):
        if not chunks[worker_id]:
            continue
        p = mp.Process(target=inference_single_worker, args=(args, worker_id, chunks[worker_id]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("All workers have completed inference.")

if __name__ == "__main__":
    inference()