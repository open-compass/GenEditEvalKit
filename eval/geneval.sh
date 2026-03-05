set -x

MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}/geneval"
python -u benchmarks/geneval/post_process.py --base_dir "${RESULT_DIR}/images" || exit

touch "${RESULT_DIR}/results.jsonl"
geneval_dir="benchmarks/geneval"
echo "geneval directory: $geneval_dir"
cd $geneval_dir || exit

NODE_COUNT=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
PROC_PER_NODE=$CUDA_VISIBLE_COUNT
MASTER_PORT=32427

"$CONDA_BASE/envs/${EVAL_ENV}/bin/python" -m torch.distributed.run \
  --nnodes=${NODE_COUNT} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${PROC_PER_NODE} \
  evaluation/evaluate_images.py \
    --imagedir "${RESULT_DIR}/images" \
    --outfile "${RESULT_DIR}/results.jsonl" \
    --model-path "../../eval_models" || exit


$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u evaluation/merge_result_files.py "${RESULT_DIR}/results.jsonl" || exit
$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u evaluation/summary_scores.py "${RESULT_DIR}/results.jsonl" 