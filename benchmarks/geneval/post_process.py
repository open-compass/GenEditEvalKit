import os
import argparse

# --- 配置参数 ---

# 源 metadata.jsonl 文件的路径 (保持不变，或也可以通过参数传递，这里先固定)
SOURCE_METADATA_PATH = 'benchmarks/geneval/prompts/evaluation_metadata.jsonl'

# 期望处理的目录总数 (553 行对应 00000 到 00552)
TOTAL_DIRECTORIES = 553 

# --- 脚本逻辑 ---

def create_indexed_metadata_files(base_directory, source_metadata_file, total_count):
    """
    根据源文件的行索引，为 BASE_DIR 下的编号目录创建对应的 metadata.jsonl 文件。
    
    Args:
        base_directory (str): 包含 00000, 00001... 文件夹的根目录。
        source_metadata_file (str): 源 evaluation_metadata.jsonl 文件的路径。
        total_count (int): 需要处理的目录总数 (也是源文件的总行数)。
    """
    
    print(f"🚀 阶段一: 读取源文件 '{source_metadata_file}' 的内容...")
    
    # 1. 读取源文件的所有行
    try:
        with open(source_metadata_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 错误: 源文件不存在: {source_metadata_file}")
        return
    except Exception as e:
        print(f"❌ 错误: 读取源文件时发生异常: {e}")
        return
    
    # 检查行数是否匹配预期
    if len(lines) != total_count:
        print(f"⚠️ 警告: 源文件行数 ({len(lines)}) 与预期 ({total_count}) 不匹配。")
    
    print(f"✨ 成功读取 {len(lines)} 行数据。")
    print("---")
    print(f"🚀 阶段二: 开始在 '{base_directory}' 中根据索引创建目标 metadata.jsonl 文件...")

    # 2. 遍历索引并创建目标文件
    lines_written_count = 0
    for i in range(total_count):
        if i >= len(lines):
            print(f"🛑 停止: 源文件行数不足，已处理到索引 {i-1}。")
            break

        # 目录名是 i 的零填充五位数 (e.g., 0 -> '00000', 552 -> '00552')
        dir_name = f"{i:05d}" 
        
        # 目标目录的完整路径
        target_dir = os.path.join(base_directory, dir_name)
        
        # 目标 metadata.jsonl 的完整路径
        target_metadata_path = os.path.join(target_dir, 'metadata.jsonl')
        
        # 对应的源文件内容 (第 i+1 行, 因为 lines 是零索引)
        content_to_write = lines[i]
        
        # 检查目标目录是否存在
        if not os.path.isdir(target_dir):
            # 这是一个预期中的目录，如果不存在，则打印警告并跳过
            print(f"⚠️ 目标目录不存在，跳过: {target_dir}")
            continue

        try:
            # 写入内容到目标文件 ('w' 模式会创建或覆盖文件)
            with open(target_metadata_path, 'w', encoding='utf-8') as outfile:
                outfile.write(content_to_write)
            
            lines_written_count += 1
            # 为了简洁，只在特定索引打印成功信息
            if i % 50 == 0 or i == total_count - 1:
                 print(f"-> 写入: {target_metadata_path} (对应源文件第 {i+1} 行)")
                 
        except Exception as e:
            print(f"❌ 错误: 写入文件 {target_metadata_path} 时发生异常: {e}")

    print("---")
    print(f"🎉 脚本执行完毕。成功在 {lines_written_count} 个目录下创建了 metadata.jsonl 文件。")

# --- 主程序入口 (处理命令行参数) ---
if __name__ == "__main__":
    # 创建解析器
    parser = argparse.ArgumentParser(
        description="根据索引映射，为指定基目录下的编号文件夹创建 metadata.jsonl 文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 添加 BASE_DIR 参数
    parser.add_argument(
        '--base_dir', 
        type=str, 
        help=(
            "包含 '00000', '00001', ... 文件夹的根目录路径。\n"
            "例如: /path/to/output/geneval_long/images"
        )
    )
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用主函数
    create_indexed_metadata_files(args.base_dir, SOURCE_METADATA_PATH, TOTAL_DIRECTORIES)