GPU_NUM=8 

MODEL_PATH=black-forest-labs/FLUX.1-dev
OUTPUT_DIR='flux_output'

mkdir -p ${OUTPUT_DIR}

torchrun --nproc_per_node=$GPU_NUM --master_port 19000 \
    flux_en_multi_node_inference.py \
    --output_dir $OUTPUT_DIR \
    --prompt_dir "data/test_prompts_en.csv" \
    --model_path ${MODEL_PATH}
