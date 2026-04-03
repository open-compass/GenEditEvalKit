set -x

MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}/gedit"
sudo chmod -R 777 ${RESULT_DIR}
cp benchmarks/gedit/gedit_edit.json ${RESULT_DIR}
touch "${RESULT_DIR}/result.txt"

touch "benchmarks/gedit/secret_t2.env"
printf '%s\n' "${API_KEY}" > benchmarks/gedit/secret_t2.env
echo "Update API_KEY config for gedit evaluating successfully."

gedit_dir="benchmarks/gedit"
unset LD_LIBRARY_PATH
gedit_dir="benchmarks/gedit/"
echo "gedit directory: $gedit_dir"
cd $gedit_dir || exit


$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step2_gedit_bench.py \
   --model_name $MODEL_NAME \
   --save_path ${RESULT_DIR}/images \
   --backbone gpt4o \
   --source_path gedit_asset \
   --language en

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step3_calculate_statistics.py \
   --model_name $MODEL_NAME \
   --save_path ${RESULT_DIR}/images \
   --backbone gpt4o \
   --language en > ${RESULT_DIR}/result.txt

cat ${RESULT_DIR}/result.txt

