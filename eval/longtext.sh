MODEL_NAME=$1
EVAL_ENV=$2

current_dir=$(pwd)
cd benchmarks/LongText/textbench

echo "Start evaluating Chinese"
SAMPLE_FOLDER="${current_dir}/output/${MODEL_NAME}/longtext/text_prompts_zh"
MODE=zh # en or zh
OUTPUT_DIR=eval_results

"$CONDA_BASE/envs/${EVAL_ENV}/bin/python" -m torch.distributed.run --nnodes=1 --node-rank=0 --nproc_per_node=$CUDA_VISIBLE_COUNT \
    evaluate_text_reward.py \
    --sample_dir $SAMPLE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --mode $MODE

cat $OUTPUT_DIR/results_chunk*.jsonl > $OUTPUT_DIR/results_${MODE}.jsonl
rm $OUTPUT_DIR/results_chunk*.jsonl

python3 summary_scores.py $OUTPUT_DIR/results_${MODE}.jsonl --mode $MODE



echo "Start evaluating English"
SAMPLE_FOLDER="${current_dir}/output/${MODEL_NAME}/longtext/text_prompts"
MODE=en # en or zh
OUTPUT_DIR=eval_results

"$CONDA_BASE/envs/${EVAL_ENV}/bin/python" -m torch.distributed.run --nnodes=1 --node-rank=0 --nproc_per_node=$CUDA_VISIBLE_COUNT \
    evaluate_text_reward.py \
    --sample_dir $SAMPLE_FOLDER \
    --output_dir $OUTPUT_DIR \
    --mode $MODE

cat $OUTPUT_DIR/results_chunk*.jsonl > $OUTPUT_DIR/results_${MODE}.jsonl
rm $OUTPUT_DIR/results_chunk*.jsonl

python3 summary_scores.py $OUTPUT_DIR/results_${MODE}.jsonl --mode $MODE