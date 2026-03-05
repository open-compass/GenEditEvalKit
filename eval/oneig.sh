MODEL_NAME=$1
EVAL_ENV=$2

# Warning message
echo "############################################################"
echo ""
echo "⚠️⚠️⚠️ Before running all evaluation scripts, please make sure the models required for evaluation are downloaded and placed correctly. If you don't know what models are needed and where to place them, please refer to https://github.com/OneIG-Bench/OneIG-Benchmark?tab=readme-ov-file#get-started or the README.md in this repository."
echo ""
echo "############################################################"

# Split EVAL_ENV into two parts, EVAL_ENV1 and EVAL_ENV2
{
IFS=' ' read -r -a EVAL_ENVS <<< "$EVAL_ENV"
EVAL_ENV1="${EVAL_ENVS[0]:-}"
EVAL_ENV2="${EVAL_ENVS[1]:-}"
}

RESULT_DIR="$PWD/output/${MODEL_NAME}"
echo "HF_HOME:${HF_HOME}"

oneig_dir="benchmarks/OneIG-Benchmark"
echo "oneig directory: $oneig_dir"
cd $oneig_dir || exit

# If results directory exists, remove it
rm -rf results

# Since inference saves each webp's 4 sub-images as png, we need to recombine them into webp before evaluation
python combine_subimages_to_webp.py --subimage_dir "${RESULT_DIR}/oneig/images_before_combination"

bash run_overall.sh --mode EN --image_dir "${RESULT_DIR}/oneig/images" --model_names "${MODEL_NAME}" --eval_env1 "${EVAL_ENV1}" --eval_env2 "${EVAL_ENV2}"

# Copy the generated results to the output directory
cp -r results "${RESULT_DIR}/oneig/"

# Look through all results
for result_file in results/*; do
    cat $result_file
done