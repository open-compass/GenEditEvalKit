MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

t2ireasonbench_dir="benchmarks/T2I-ReasonBench"
echo "t2ireasonbench directory: $t2ireasonbench_dir"
cd $t2ireasonbench_dir || exit

bash eval_all.sh $RESULT_DIR/t2ireasonbench $MODEL_NAME $EVAL_ENV