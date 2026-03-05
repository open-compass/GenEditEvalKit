MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

unigenbench_dir="benchmarks/UniGenBench/"
echo "unigenbench directory: $unigenbench_dir"
cd $unigenbench_dir || exit

DATA_PATH_EN="${RESULT_DIR}/unigenbench/test_prompts_en/samples"
DATA_PATH_ZH="${RESULT_DIR}/unigenbench/test_prompts_zh/samples"

echo "Start evaluating unigenbench-en"
CSV_FILE="data/test_prompts_en.csv" # English test prompt file
# English Evaluation
$CONDA_BASE/envs/${EVAL_ENV}/bin/python eval/gemini_en_eval.py \
  --data_path "$DATA_PATH_EN" \
  --api_key "$API_KEY" \
  --base_url "$BASE_URL" \
  --csv_file "$CSV_FILE"