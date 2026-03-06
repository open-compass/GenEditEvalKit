MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

genexam_dir="benchmarks/GenExam/"
echo "GenExam directory: $genexam_dir"
cd $genexam_dir || exit

$CONDA_BASE/envs/${EVAL_ENV}/bin/python run_eval.py --data_dir ./data/ --img_save_dir ${RESULT_DIR}/genexam/images --eval_save_dir ${RESULT_DIR}/genexam/eval_results
$CONDA_BASE/envs/${EVAL_ENV}/bin/python cal_score.py --eval_results_dir ${RESULT_DIR}/genexam/eval_results > ${RESULT_DIR}/genexam/final_scores.log 2>&1
cat ${RESULT_DIR}/genexam/final_scores.log