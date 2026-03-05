MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

tiff_dir="benchmarks/TIIF-Bench"
echo "tiff directory: $tiff_dir"
cd $tiff_dir || exit

echo "Set variables"
JSONL_DIR=data/testmini_eval_prompts
IMAGE_DIR=${RESULT_DIR}/tiff
OUTPUT_DIR=eval-results
MODEL="gpt-5"

echo "Run evaluation"
$CONDA_BASE/envs/${EVAL_ENV}/bin/python eval/eval_with_vlm.py \
    --jsonl_dir $JSONL_DIR \
    --image_dir $IMAGE_DIR \
    --eval_model $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --api_key $API_KEY \
    --base_url $BASE_URL \
    --model "$MODEL"

echo "Start Summarizing results"
$CONDA_BASE/envs/${EVAL_ENV}/bin/python eval/summary_results.py --input_dir $OUTPUT_DIR --output_excel_name result_summary_${MODEL_NAME}.xlsx
$CONDA_BASE/envs/${EVAL_ENV}/bin/python eval/summary_dimension_results.py --input_excel $OUTPUT_DIR/result_summary_${MODEL_NAME}.xlsx --output_txt $OUTPUT_DIR/result_summary_dimension_${MODEL_NAME}.txt