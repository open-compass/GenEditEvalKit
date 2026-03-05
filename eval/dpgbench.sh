# set -x

MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}/dpgbench"

python -u benchmarks/dpgbench/post_process.py --base_dir "${RESULT_DIR}/images" || exit

dpgbench_dir="benchmarks/dpgbench/"
echo "dpgbench directory: $dpgbench_dir"
cd $dpgbench_dir || exit

export MPLUG_LOCAL_PATH="../../eval_models/mplug"
export PYTHONNOUSERSITE=1

"${CONDA_BASE}/envs/${EVAL_ENV}/bin/python" -m accelerate.commands.launch --num_machines 1 --num_processes $CUDA_VISIBLE_COUNT \
    --multi_gpu \
    --mixed_precision "fp16" \
    step2_compute_dpg_bench.py \
    --image_root_path ${RESULT_DIR}/images \
    --pic_num 4 \
    --res_path ${RESULT_DIR}/dpgbench.txt \
    --vqa_model mplug \
    --mplug_local_path ${MPLUG_LOCAL_PATH} \
    --csv eval_prompts/dpgbench.csv
cat ${RESULT_DIR}/dpgbench.txt