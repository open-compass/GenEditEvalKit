import multiprocessing as mp
import json
import os
from PIL import Image
from tqdm import tqdm
import argparse

from custom_datasets.load_dataset import load_dataset
from custom_models.load_model import load_model

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

def get_available_device_ids(): # Two methods to get device ids
    env = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env and env.strip():
        ids = [d.strip() for d in env.split(',') if d.strip()]
    else:
        import torch
        n = torch.cuda.device_count()

        ids = [str(i) for i in range(n)]
    return ids

def inference_single_gpu(args, gpu_id, items_chunk):
    model = load_model(args.model_name, args.dataset_task, gpu_id, args.custom_model_kwargs)
    pbar = tqdm(items_chunk, desc=f"GPU {gpu_id}", position=gpu_id)
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
                    output_image = create_blank_image(model.width, model.height)
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
    args = parser.parse_args()

    # Parse user custom parameters
    args.custom_model_kwargs = parse_model_kwargs(args.custom_model_kwargs)

    # Get gpu ids
    device_ids = get_available_device_ids()
    if not device_ids:
        raise RuntimeError("No GPUs available. Please ensure CUDA is available.")
    num_gpus = len(device_ids)

    # Load dataset
    dataset = load_dataset(args.dataset_name)
    args.dataset_task = dataset.dataset_task
    chunks = [[] for _ in range(num_gpus)]
    for idx, item in enumerate(dataset.data):
        chunks[idx % num_gpus].append(item)

    # Run inference including model loading and inference
    processes = []
    for rank, device_id in enumerate(device_ids):
        if not chunks[rank]:
            continue
        p = mp.Process(target=inference_single_gpu, args=(args, int(device_id), chunks[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("All GPUs have completed inference.")

if __name__ == "__main__":
    inference()