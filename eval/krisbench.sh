set -x

MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

krisbench_dir="benchmarks/Kris_Bench"
echo "krisbench directory: $krisbench_dir"
cd $krisbench_dir || exit

export OPENAI_API_KEY=$API_KEY
# If there are previous results, remove them
rm -rf results/*

cp -r ${RESULT_DIR}/krisbench/images/${MODEL_NAME} results/ # Copy the results to the `results` directory for evaluation

$CONDA_BASE/envs/${EVAL_ENV}/bin/python metrics_common.py --models ${MODEL_NAME}
$CONDA_BASE/envs/${EVAL_ENV}/bin/python metrics_knowledge.py --models ${MODEL_NAME}
$CONDA_BASE/envs/${EVAL_ENV}/bin/python metrics_multi_element_robust.py --models ${MODEL_NAME} ## Use the modified version to support evaluation of models which cannot support multi-image input
$CONDA_BASE/envs/${EVAL_ENV}/bin/python metrics_temporal_prediction_robust.py --models ${MODEL_NAME} ## Use the modified version to support evaluation of models which cannot support multi-image input
$CONDA_BASE/envs/${EVAL_ENV}/bin/python metrics_view_change.py --models ${MODEL_NAME}
$CONDA_BASE/envs/${EVAL_ENV}/bin/python summarize.py --results_dir results/${MODEL_NAME}

# Copy the generated results to the output directory
cp -r results/${MODEL_NAME}/results.json ${RESULT_DIR}/krisbench/
for metrics_file in results/${MODEL_NAME}/*/metrics.json; do
    category_name=$(basename "$(dirname "$metrics_file")")
    cp "$metrics_file" "${RESULT_DIR}/krisbench/images/${MODEL_NAME}/${category_name}/metrics.json"
done

cat ${RESULT_DIR}/krisbench/results.json