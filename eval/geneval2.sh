# Local Variables
MODEL_NAME=$1
EVAL_ENV=$2

# Environment Variables
RESULT_DIR="$PWD/output/${MODEL_NAME}"

# Change Working Directory
geneval2_dir="$PWD/benchmarks/GenEval2"
cd $geneval2_dir || exit

# Execute Evaluation Commands
${CONDA_BASE}/envs/${EVAL_ENV}/bin/python build_json.py \
    --image_dir ${RESULT_DIR}/geneval2
${CONDA_BASE}/envs/${EVAL_ENV}/bin/python evaluation.py \
    --benchmark_data geneval2_data.jsonl \
    --image_filepath_data ${RESULT_DIR}/geneval2/image_paths.json \
    --method soft_tifa_gm \
    --output_file ${RESULT_DIR}/geneval2/score_lists.json
${CONDA_BASE}/envs/${EVAL_ENV}/bin/python soft_tifa_analysis.py \
    --benchmark_data geneval2_data.jsonl \
    --score_data ${RESULT_DIR}/geneval2/score_lists.json