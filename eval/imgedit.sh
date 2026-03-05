set -x

MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}/imgedit"
sudo chmod -R 777 ${RESULT_DIR}
cp benchmarks/imgedit/eval_prompts/imgedit_bench.json ${RESULT_DIR}
touch "${RESULT_DIR}/result.json"

imgedit_dir="benchmarks/imgedit"
unset LD_LIBRARY_PATH
imgedit_dir="benchmarks/imgedit/"
echo "imgedit directory: $imgedit_dir"
cd $imgedit_dir || exit

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step2_basic_bench.py \
   --result_img_folder ${RESULT_DIR}/images \
   --result_json ${RESULT_DIR}/imgedit_bench.json \
   --edit_json eval_prompts/basic_edit.json \
   --prompts_json eval_prompts/prompts.json \
   --origin_img_root ./imgedit_asset/Benchmark/singleturn \
   --api_key ${API_KEY} 

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step3_get_avgscore.py \
   --input ${RESULT_DIR}/imgedit_bench.json \
   --meta_json eval_prompts/basic_edit.json \
   --output_json ${RESULT_DIR}/result.json

cat ${RESULT_DIR}/result.json

