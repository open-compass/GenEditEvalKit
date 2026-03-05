# -*- coding: utf-8 -*-
"""
Text-to-image inference script (DDP version)
"""
import os
import json
import argparse
import time
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import SPECIAL_TOKENS
from model import LLaDAForMultiModalGeneration
from utils.image_utils import decode_vq_to_image, calculate_vq_params, add_break_line
from generators.image_generation_generator import generate_image
from utils.generation_utils import setup_seed
from utils.prompt_utils import generate_text_to_image_prompt, create_prompt_templates


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Text-to-image inference (DDP version)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Fine-tuned checkpoint path")
    parser.add_argument("--prompt_path", type=str, required=True, help="Prompt file path(.json/.jsonl/.txt)")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--timesteps", type=int, default=64, help="Number of timesteps")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_ckpt", type=str, default="./vae_ckpt", help="VAE checkpoint path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="results_text_to_image_ddp", help="Output directory")
    parser.add_argument("--output_json", type=str, default="results_text_to_image_ddp/results.json", help="Output JSON file")
    parser.add_argument("--use-cache", action='store_true', help="Enable caching for faster inference")
    parser.add_argument("--cache_ratio", type=float, default=0.9, help="Ratio of reused tokens, in (0,1); the higher the faster")
    parser.add_argument("--warmup_ratio", type=float, default=0.3, help="Warmup ratio for caching, in [0,1); the lower the faster")
    parser.add_argument("--refresh_interval", type=int, default=5, help="Refresh all cache every `refresh_interval` steps, in (1, timesteps-int(warmup_ratio*timesteps)-1]; the higher the faster")
    
    args = parser.parse_args()
    
    # Special tokens
    MASK = SPECIAL_TOKENS["mask_token"]
    NEW_LINE = SPECIAL_TOKENS["newline_token"]
    BOA = SPECIAL_TOKENS["answer_start"]  # Begin of Answer
    EOA = SPECIAL_TOKENS["answer_end"]    # End of Answer
    BOI = SPECIAL_TOKENS["boi"]           # Begin of Image
    EOI = SPECIAL_TOKENS["eoi"]           # End of Image

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Set random seed
    if int(args.seed) != 0:
        setup_seed(args.seed + local_rank)
    
    # Only rank0 creates directory
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = LLaDAForMultiModalGeneration.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
    )
    
    # Wrap with DDP
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Read prompts
    prompts = []
    if args.prompt_path.endswith(".jsonl"):
        with open(args.prompt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    prompt_text = obj.get("prompt", "").strip()
                    if prompt_text:
                        prompts.append(prompt_text)
    elif args.prompt_path.endswith(".json"):
        with open(args.prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
    elif args.prompt_path.endswith(".txt"):
        with open(args.prompt_path, "r", encoding="utf-8") as f:
            raw = f.read()
        prompts = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        raise ValueError("Unsupported file format, please use .json/.jsonl/.txt")
    
    # Create dataset and data loader
    dataset = PromptDataset(prompts)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    
    # Calculate VQ parameters
    seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(args.height, args.width)
    
    if is_main_process():
        print(f"Generate image size: {args.height}x{args.width}")
        print(f"Calculated VQ sequence length: {seq_len}")
        print(f"Tokens per line (newline_every): {newline_every}")
    
    # Get prompt templates
    templates = create_prompt_templates()
    
    # rank-specific jsonl, avoid write conflicts
    rank = dist.get_rank()
    per_rank_jsonl = os.path.splitext(args.output_json)[0] + f".rank{rank}.jsonl"
    if os.path.exists(per_rank_jsonl):
        os.remove(per_rank_jsonl)
    
    time_list = []
    
    # Main loop (each rank processes its own subset)
    for i, prompt_text in enumerate(dataloader):
        prompt_text = prompt_text[0]  # Unpack from batch
        
        if is_main_process():
            print(f"Processing prompt {i+1}/{len(prompts)}: {prompt_text}")
        
        # Generate prompts using utility function
        input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt_text, templates)
        
        # build initial sequence
        con_prompt_token = tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = tokenizer(uncon_prompt)["input_ids"]
        
        # build image mask predition
        img_mask_token = add_break_line([MASK] * seq_len, token_grid_height, token_grid_width, new_number = NEW_LINE)
        img_pred_token = [BOA] + [BOI] + img_mask_token + [EOI] + [EOA]

        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=device).unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device=device).unsqueeze(0)

        # image satrt index
        code_start = len(con_prompt_token) + 2 
        
        # Generate VQ tokens
        start_time = time.time()
        vq_tokens = generate_image(
            model,
            prompt_ids,
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=args.timesteps,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            uncon_ids=uncon_ids,
            code_start=code_start,
            use_cache=args.use_cache,
            cache_ratio=args.cache_ratio,
            refresh_interval=args.refresh_interval,
            warmup_ratio=args.warmup_ratio
        )
        
        # Generate filename
        words = prompt_text.split()
        filename_words = words[:10] if len(words) > 10 else words
        filename = "_".join(filename_words)
        filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
        filename = f"{i}_{filename}_{args.height}x{args.width}_t{args.timesteps}_cfg{args.cfg_scale}_seed{args.seed}_rank{rank}.png"
        save_path = os.path.join(args.output_dir, filename)
        
        # Decode VQ codes to PNG and save
        decode_vq_to_image(
            vq_tokens, save_path, 
            vae_ckpt=args.vae_ckpt, 
            image_height=args.height, 
            image_width=args.width
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        time_list.append(elapsed_time)
        
        if is_main_process():
            print(f"Time: {elapsed_time:.2f}s")
            print("-" * 50)
        
        # Write each rank's own JSONL
        result = {
            "image_path": filename,
            "prompt": prompt_text,
            "elapsed_time": elapsed_time,
            "rank": rank
        }
        with open(per_rank_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    barrier()
    
    # Merge JSONL (only on rank0)
    if is_main_process():
        merged = []
        base, ext = os.path.splitext(args.output_json)
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        
        # Collect all per-rank files
        world = dist.get_world_size()
        for r in range(world):
            fpath = os.path.splitext(args.output_json)[0] + f".rank{r}.jsonl"
            if not os.path.exists(fpath):
                continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))
        
        # Save merged json (list)
        with open(args.output_json, "w", encoding="utf-8", buffering=1) as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        
        print(f"Average sampling time: {sum(time_list)/len(prompts):.2f}s")
        print(f"[✓] Merged JSON saved to {args.output_json} (items={len(merged)})")
    
    barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
