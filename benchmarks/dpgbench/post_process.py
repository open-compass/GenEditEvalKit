import os
import argparse
from PIL import Image
import shutil
# --- 固定配置 ---

# 每个子文件夹中子图的数量和名称。
# 0.png, 1.png, 2.png, 3.png
SUB_IMAGE_NAMES = ["0.png", "1.png", "2.png", "3.png"]

# 拼接的布局: 2行 x 2列 (从左到右，从上到下)
LAYOUT = (2, 2) 

# --- 核心图像处理函数 ---

def stitch_images_in_folder(input_folder, sub_image_paths):
    """
    将一个文件夹内的四张子图按照 2x2 顺序拼接成一张大图。
    
    Args:
        input_folder (str): 包含子图的文件夹路径 (例如: 'dpgbench/images/0')
        sub_image_paths (list): 四张子图的完整路径列表。
        
    Returns:
        Image or None: 拼接好的 PIL Image 对象，失败则返回 None。
    """
    try:
        # 1. 打开并加载所有子图
        # .convert("RGB") 确保所有图像模式一致，避免拼接错误
        images = [Image.open(path).convert("RGB") for path in sub_image_paths]
    except FileNotFoundError as e:
        print(f"    ❌ 错误: 缺少子图文件: {e}")
        return None
    except Exception as e:
        print(f"    ❌ 错误: 加载图像时发生异常: {e}")
        return None

    # 检查图像数量是否正确
    if len(images) != len(SUB_IMAGE_NAMES):
        print(f"    ⚠️ 警告: 找到的图像数量 ({len(images)}) 与预期 ({len(SUB_IMAGE_NAMES)}) 不符，跳过。")
        return None

    # 假设所有子图尺寸相同，获取单张子图的尺寸
    try:
        width, height = images[0].size
    except IndexError:
        return None # 无法获取尺寸，已在上面捕获

    rows, cols = LAYOUT # 2行, 2列

    # 计算最终大图的尺寸
    stitched_width = cols * width
    stitched_height = rows * height
    
    # 2. 创建一张空白的新图片用于拼接
    stitched_image = Image.new('RGB', (stitched_width, stitched_height))

    # 3. 按照从左到右、从上到下的顺序进行粘贴 (2x2)
    # 顺序: 0.png -> (0, 0); 1.png -> (w, 0); 2.png -> (0, h); 3.png -> (w, h)
    coordinates = [
        (0, 0),         # 对应 0.png (左上角)
        (width, 0),     # 对应 1.png (右上角)
        (0, height),    # 对应 2.png (左下角)
        (width, height) # 对应 3.png (右下角)
    ]

    for img, (x, y) in zip(images, coordinates):
        stitched_image.paste(img, (x, y))

    return stitched_image

def process_all_directories(base_directory):
    """
    遍历基目录下的所有子文件夹，进行图像拼接并保存。
    """
    base_directory = os.path.abspath(base_directory) # 转换为绝对路径
    
    if not os.path.isdir(base_directory):
        print(f"❌ 错误: 基目录不存在: {base_directory}")
        return

    print(f"🚀 开始处理目录: {base_directory}")
    # 记录待删除的文件夹，防止在遍历时直接删除影响 listdir
    folders_to_delete = [] 
    processed_count = 0
    # os.listdir 获取基目录下所有文件和文件夹名称
    for item_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, item_name)
        
        if os.path.isdir(folder_path) and not item_name.startswith('.'):
            # 跳过我们自己创建的新的拼接图文件（它们不是文件夹）
            if item_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                 continue 
                 
            print(f"--- 处理文件夹: {item_name} ---")
            
            sub_image_paths = [os.path.join(folder_path, name) for name in SUB_IMAGE_NAMES]
            stitched_image = stitch_images_in_folder(folder_path, sub_image_paths)
            
            if stitched_image:
                output_filename = f"{item_name}.png"
                output_path = os.path.join(base_directory, output_filename)
                
                # 4. 保存拼接后的大图
                try:
                    stitched_image.save(output_path)
                    print(f"    ✨ 成功保存拼接图: {output_path}")
                    
                    # 5. 图片保存成功后，标记该子文件夹待删除
                    folders_to_delete.append(folder_path)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 错误: 保存图像时发生异常，跳过删除 {item_name}: {e}")
            else:
                 print(f"    ⚠️ 警告: 文件夹 {item_name} 图像拼接失败或缺少文件，跳过。")
    
    print("\n--- 阶段二: 清理子文件夹 ---")
    
    # 统一删除已标记的文件夹
    deleted_count = 0
    for folder_path in folders_to_delete:
        try:
            shutil.rmtree(folder_path)
            print(f"    🗑️ 成功删除目录: {folder_path}")
            deleted_count += 1
        except Exception as e:
            print(f"    ❌ 错误: 删除目录 {folder_path} 失败: {e}")


    print("---")
    print(f"🎉 脚本执行完毕。成功处理 {processed_count} 个文件夹，删除了 {deleted_count} 个子目录。")

# --- 主程序入口 (处理命令行参数) ---
if __name__ == "__main__":
    # 检查 PIL 库是否已导入，给出友好提示
    # try:
    #     Image.VERSION
    # except NameError:
    #     print("🚨 致命错误: 未安装 Pillow 库。请运行 'pip install Pillow'。")
    #     exit(1)

    # 创建解析器
    parser = argparse.ArgumentParser(
        description="遍历指定基目录下的所有子文件夹，将每个子文件夹内的 4 张图片 (0.png, 1.png, 2.png, 3.png) 拼接成一张 2x2 的大图，并以文件夹名命名保存到基目录中。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 添加 BASE_DIR 参数
    parser.add_argument(
        '--base_dir', 
        type=str, 
        help=(
            "包含所有编号子文件夹 (如 0, 1, ..., partiprompts308) 的父目录路径。\n"
            "例如: /path/to/lumina-dimoo/dpgbench/images"
        )
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用主函数
    process_all_directories(args.base_dir)