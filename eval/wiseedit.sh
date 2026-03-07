MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

wiseedit_dir="benchmarks/WiseEdit/"
echo "wiseedit directory: $wiseedit_dir"
cd $wiseedit_dir || exit

$CONDA_BASE/envs/${EVAL_ENV}/bin/python run_eval.py \
  --name ${MODEL_NAME} \
  --dataset_dir ./WiseEdit-Benchmark \
  --result_img_root ${RESULT_DIR}/wiseedit \
  --score_output_root ${RESULT_DIR}/wiseedit \
  --num_workers 128 # number of threads used for evaluation

$CONDA_BASE/envs/${EVAL_ENV}/bin/python statistic_single.py \
  --dataset_dir ./WiseEdit-Benchmark \
  --score_root ${RESULT_DIR}/wiseedit \
  --name ${MODEL_NAME} \
  --statistic_output_dir ${RESULT_DIR}/wiseedit 