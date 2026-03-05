#!/bin/bash
set -x

# Change cuda version from 12.1 to 11.8
source ../../utils/use_cuda.sh 11.8
# Conda initialization (for subsequent environment switching)
source $(conda info --base)/etc/profile.d/conda.sh

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;; # mode (EN/ZH)
    --image_dir)
      IMAGE_DIR="$2"; shift 2 ;; # image_root_dir
    --model_names)
      IFS=',' read -r -a MODEL_NAMES <<< "$2"; shift 2 ;; # model list
    --eval_env1)
      EVAL_ENV_1="$2"; shift 2 ;; # env 1
    --eval_env2)
      EVAL_ENV_2="$2"; shift 2 ;; # env 2
    *)
      echo "Unknown arg: $1" ;;
  esac
done

# start_time
start_time=$(date +%s)

# image grid
IMAGE_GRID=(2)

# ----------------------- #
# Overall, the main input parameters are MODE, IMAGE_DIR, MODEL_NAMES, and IMAGE_GRID.
# For our model, the mode should be set to English.
# The image_dirname is the path where the output images are saved.
# The model_names is the list of model names. The input image path should be image_dirname/{model_names}, so this parameter should not be set arbitrarily.
# The image_grid is the grid size of the images, which should be fine to keep the default parameter.
# -----------------------
echo "Running all evaluation scripts"

# pip install transformers==4.50.0
# conda activate $EVAL_ENV_1
# echo "Current Env: $(conda info --envs | grep '*')"

# Alignment Score

echo "It's alignment time."

$CONDA_BASE/envs/${EVAL_ENV_1}/bin/python -m scripts.alignment.alignment_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \
  --class_items "anime" "human" "object" \

# In ZH mode, the class_items list can be extended to include "multilingualism".

# Text Score

echo "It's text time."

$CONDA_BASE/envs/${EVAL_ENV_1}/bin/python -m scripts.text.text_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR/text" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \

# Diversity Score

echo "It's diversity time."

$CONDA_BASE/envs/${EVAL_ENV_1}/bin/python -m scripts.diversity.diversity_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \
  --class_items "anime" "human" "object" "text" "reasoning" \

# Style Score

echo "It's style time."

$CONDA_BASE/envs/${EVAL_ENV_1}/bin/python -m scripts.style.style_score \
  --mode "$MODE" \
  --image_dirname "$IMAGE_DIR/anime" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \

# Reasoning Score

echo "It's reasoning time."

# # pip install transformers==4.46.1
# conda activate $EVAL_ENV_2
# echo "Current Env: $(conda info --envs | grep '*')"


$CONDA_BASE/envs/${EVAL_ENV_2}/bin/python -m scripts.reasoning.reasoning_score \
  --mode "$MODE" \
  --image_dirname "${IMAGE_DIR}/reasoning" \
  --model_names "${MODEL_NAMES[@]}" \
  --image_grid "${IMAGE_GRID[@]}" \


rm -rf tmp_*
# end_time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "✅ All evaluations finished in $duration seconds."