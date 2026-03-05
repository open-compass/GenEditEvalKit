import base64
from PIL import Image
import io
import os
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_prompts(prompts_json_path):
    with open(prompts_json_path, 'r') as f:
        return json.load(f)


def image_to_base64(image_path, target_size=None, dst_path=None):
    """
    将图片转换为base64编码
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸 (width, height)，如果提供则会resize图片
    
    Returns:
        base64编码的字符串，失败时返回None
    """
    try:
        with Image.open(image_path) as image:
            # 如果指定了目标尺寸，则resize图片
            if target_size:
                w_target, h_target = target_size
                w_ori, h_ori = image.size
                if (w_target / h_target) != (w_ori / h_ori):
                    image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            if dst_path is not None:
                image.save(dst_path)
            
            # 将图片转换为字节流
            buffer = io.BytesIO()
            # 保持原始格式，如果是PNG则保存为PNG，否则保存为JPEG
            format = image.format if image.format in ['PNG', 'JPEG', 'JPG'] else 'JPEG'
            image.save(buffer, format=format)
            
            # 转换为base64
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_image_size(image_path):
    """
    获取图片尺寸
    
    Args:
        image_path: 图片路径
    
    Returns:
        (width, height) 元组，失败时返回None
    """
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception as e:
        print(f"Error getting image size for {image_path}: {e}")
        return None

@retry(wait=wait_exponential(multiplier=1, min=2, max=2), stop=stop_after_attempt(100))
def call_gpt(original_image_path, result_image_path, edit_prompt, edit_type, prompts, api_key, base_url):

    try:
        # 获取原始图片的尺寸
        original_size = get_image_size(original_image_path)
        if original_size is None:
            print("Failed to get original image size")
            return None
        
        original_image_path_new = result_image_path.replace("/images/", "/images_ori/")
        os.makedirs(os.path.dirname(original_image_path_new), exist_ok=True)
        result_image_path_resize = result_image_path.replace("/images/", "/images_resize/")
        os.makedirs(os.path.dirname(result_image_path_resize), exist_ok=True)

        # 转换原始图片为base64
        original_image_base64 = image_to_base64(original_image_path, dst_path=original_image_path_new)
        
        # 将结果图片resize到原始图片尺寸后转换为base64
        result_image_base64 = image_to_base64(result_image_path, target_size=original_size, dst_path=result_image_path_resize)

        if not original_image_base64 or not result_image_base64:
            return {"error": "Image conversion failed"}

        client = OpenAI(
            api_key=api_key,
            # base_url=base_url  
        )

        prompt = prompts[edit_type]
        full_prompt = prompt.replace('<edit_prompt>', edit_prompt)

        response = client.chat.completions.create(
            model="gpt-4.1",
            stream=False,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}}
                ]
            }]
        )

        result_json_path = os.path.join(os.path.dirname(os.path.dirname(result_image_path_resize)), "log_debug.jsonl")

        data = {
            "original_image_path": original_image_path,
            "result_image_path": result_image_path,
            "original_image_path_new": original_image_path_new,
            "result_image_path_resize": result_image_path_resize,
            "text": full_prompt,
            "response": response.choices[0].message.content
        }

        with open(result_json_path, 'a') as wf:
            wf.write(json.dumps(data) + '\n')

        return response
    except Exception as e:
        print(f"Error in calling GPT API: {e}")
        import traceback
        traceback.print_exc()
        raise

def process_single_item(key, item, result_img_folder, origin_img_root, prompts, api_key, base_url):
    result_img_name = f"{key}.png"
    result_img_path = os.path.join(result_img_folder, result_img_name)
    origin_img_path = os.path.join(origin_img_root, item['id'])
    edit_prompt = item['prompt']
    edit_type = item['edit_type']

    response = call_gpt(origin_img_path, result_img_path, edit_prompt, edit_type, prompts, api_key, base_url)
    return key, response.choices[0].message.content

def process_json(edit_json, result_img_folder, origin_img_root, num_threads, prompts, result_json, api_key, base_url):
    with open(edit_json, 'r') as f:
        edit_infos = json.load(f)

    results = {}
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_key = {
            executor.submit(process_single_item, key, item, result_img_folder, origin_img_root, prompts, api_key, base_url): key
            for key, item in edit_infos.items()
        }

        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Processing edits"):
            key = future_to_key[future]
            try:
                k, result = future.result()
                results[k] = result
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                results[key] = {"error": str(e)}

    # Save results to the specified output JSON file
    with open(result_json, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder with subfolders of edited images")
    parser.add_argument('--edit_json', type=str, required=True, help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    parser.add_argument('--prompts_json', type=str, required=True, help="JSON file containing prompts") 
    parser.add_argument('--result_json', type=str, required=True, help="Path to output JSON file")  
    parser.add_argument('--api_key', type=str, required=True, help="API key for authentication")  # Add API key argument
    parser.add_argument('--base_url', type=str, default="https://api.openai.com/v1/chat/completions", help="Base URL for the API")  # Add base_url argument
    
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)  

    process_json(args.edit_json, args.result_img_folder, args.origin_img_root, args.num_processes, prompts, args.result_json, args.api_key, args.base_url)

if __name__ == "__main__":
    main()
