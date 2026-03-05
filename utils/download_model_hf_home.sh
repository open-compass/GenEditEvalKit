#!/usr/bin/env bash

# ==========================
# 环境配置
# ==========================
export HF_TOKEN="your_hf_token"
export HF_ENDPOINT="https://hf-mirror.com"      # 镜像站（可换官方 https://huggingface.co）
export HF_HOME="your_local_path_to_hf_cache"  # 本地缓存路径
export TRANSFORMERS_CACHE="$HF_HOME/hub"
# export HF_HUB_ENABLE_HF_TRANSFER=1              # 启用更快的传输协议（可选）
mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"

# ==========================
# 模型列表（可扩展）
# ==========================
MODELS=(
  "your/model"
)

# ==========================
# 下载逻辑
# ==========================
for i in {1..5}; do # 循环多遍，防止中途报错
echo ">>> 第 ${i} 次尝试下载模型..."
for model in "${MODELS[@]}"; do
  echo ">>> 开始下载模型：${model}"

  hf download "${model}" \
      --repo-type model \
      --token $HF_TOKEN

  echo ">>> 模型 ${model} 下载完成 ✅"
done
done

echo ">>> 所有模型下载完成"
