MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

unset LD_LIBRARY_PATH
rise_dir="benchmarks/RISEBench/"
echo "RISEBench directory: $rise_dir"
cd $rise_dir || exit

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u gpt_eval.py \
    --data data/datav2_total_w_subtask.json \
    --output ${RESULT_DIR}/rise/outputs/${MODEL_NAME}

cat ${RESULT_DIR}/rise/outputs/${MODEL_NAME}/${MODEL_NAME}_judge.csv