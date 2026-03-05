MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}/wise"

wise_dir="benchmarks/wise/"
echo "wise directory: $wise_dir"
cd $wise_dir || exit

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step2_gpt_eval.py \
    --json_path data/cultural_common_sense.json \
    --output_dir ${RESULT_DIR}/Results/cultural_common_sense \
    --image_dir ${RESULT_DIR}/images \
    --api_key ${API_KEY} \
    --model "gpt-4o-2024-05-13" \
    --result_full ${RESULT_DIR}/Results/cultural_common_sense_full_results.json \
    --result_scores ${RESULT_DIR}/Results/cultural_common_sense_scores_results.jsonl \
    --max_workers 96

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step2_gpt_eval.py \
    --json_path data/spatio-temporal_reasoning.json \
    --output_dir ${RESULT_DIR}/Results/spatio-temporal_reasoning \
    --image_dir ${RESULT_DIR}/images \
    --api_key ${API_KEY} \
    --model "gpt-4o-2024-05-13" \
    --result_full ${RESULT_DIR}/Results/spatio-temporal_reasoning_results.json \
    --result_scores ${RESULT_DIR}/Results/spatio-temporal_reasoning_results.jsonl \
    --max_workers 96

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step2_gpt_eval.py \
    --json_path data/natural_science.json \
    --output_dir ${RESULT_DIR}/Results/natural_science \
    --image_dir ${RESULT_DIR}/images \
    --api_key ${API_KEY} \
    --model "gpt-4o-2024-05-13" \
    --result_full ${RESULT_DIR}/Results/natural_science_full_results.json \
    --result_scores ${RESULT_DIR}/Results/natural_science_scores_results.jsonl \
    --max_workers 96

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step3_wise_cal.py \
    "${RESULT_DIR}/Results/cultural_common_sense_scores_results.jsonl" \
    "${RESULT_DIR}/Results/natural_science_scores_results.jsonl" \
    "${RESULT_DIR}/Results/spatio-temporal_reasoning_results.jsonl" \
    --category all