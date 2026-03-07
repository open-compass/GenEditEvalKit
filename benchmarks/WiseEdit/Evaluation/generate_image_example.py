# How to generate images on WiseEdit (we use flux2 dev as the example):
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Awareness/Awareness_1/Awareness_1.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Awareness/Awareness_1/Awareness_1.csv --eng 1
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Awareness/Awareness_2/Awareness_2.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Awareness/Awareness_2/Awareness_2.csv --eng 1

# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Interpretation/Interpretation_1/Interpretation_1.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Interpretation/Interpretation_1/Interpretation_1.csv --eng 1

# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Imagination/Imagination_1/Imagination_1.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Imagination/Imagination_1/Imagination_1.csv --eng 1
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Imagination/Imagination_2/Imagination_2.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Imagination/Imagination_2/Imagination_2.csv --eng 1
# ...
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Imagination/Imagination_5/Imagination_5.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit/Imagination/Imagination_5/Imagination_5.csv --eng 1

# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit-Complex/WiseEdit_Complex_2/WiseEdit_Complex_2.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit-Complex/WiseEdit_Complex_2/WiseEdit_Complex_2.csv --eng 1
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit-Complex/WiseEdit_Complex_3/WiseEdit_Complex_3.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit-Complex/WiseEdit_Complex_3/WiseEdit_Complex_3.csv --eng 1
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit-Complex/WiseEdit_Complex_4/WiseEdit_Complex_4.csv --eng 0
# python generate_image_example.py --input_path /path/to/WiseEdit-Benchmark/WiseEdit-Complex/WiseEdit_Complex_4/WiseEdit_Complex_4.csv --eng 1

import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
import pandas as pd
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input_path", 
        type=str, 
        default='/path/to/WiseEdit-Benchmark/WiseEdit/Awareness/Awareness_1/Awareness_1.csv',
        help="Input CSV Path"
    )

    parser.add_argument(
        "--eng", 
        type=int, 
        default=1, 
        help="English or Chinese Instruction for image editing."
    )

    parser.add_argument(
        "--output_path", 
        type=str, 
        default='/path/to/result_images_root/Flux2Dev', 
        help="Path to save the output image."
    )
    args = parser.parse_args()

    input_path = args.input_path
    print('current sub-task: ', input_path)
    df = pd.read_csv(input_path,encoding="utf-8-sig") 
    input_num = int(input_path.split('.')[0].split('_')[-1])
    output_dir = input_path.split('.')[0].split('/')[-1]

    repo_id = "Path_to_black-forest-labs/FLUX.2-dev"
    device = "cuda:0"
    torch_dtype = torch.bfloat16

    pipe = Flux2Pipeline.from_pretrained(
        repo_id, torch_dtype=torch_dtype
    )
    pipe.enable_model_cpu_offload()

    
    if args.eng>0:
        lang = 'en'
    else:
        lang = 'cn'
    output_path = os.path.join(args.output_path, output_dir, lang)
    os.makedirs(output_path, exist_ok=True)
    for i in range(len(df)):
        curdata = df.iloc[i]
        allimgs = []
        idx = curdata['idx']
        print('current test case index:', idx, '/', len(df))

        tgt_path = os.path.join(output_path, f'{idx}.png')
        if os.path.exists(tgt_path):
            continue

        prompt, promptcn = curdata['prompt'], curdata['promptcn']
        for j in range(input_num):
            curimgpath = curdata[f'input_{j+1}']

            allimgs.append(load_image(curimgpath))
        
        if args.eng>0:
            input_instruction = prompt
        else:
            input_instruction = promptcn

        prompt = input_instruction
        images = allimgs
        try:
            image = pipe(
                prompt=prompt,
                image=allimgs,
                generator=torch.Generator(device=device).manual_seed(42),
                num_inference_steps=50,
                guidance_scale=4,
            ).images[0]

            image.save(tgt_path)
            print('save image in:', tgt_path)
        except Exception as e:
            continue


