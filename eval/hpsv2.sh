MODEL_NAME=$1
EVAL_ENV=$2

export HPS_ROOT=$HF_HOME
RESULT_DIR="$PWD/output/${MODEL_NAME}"

hpsv2_dir="benchmarks/HPSv2"
echo "hpsv2 directory: $hpsv2_dir"
cd $hpsv2_dir || exit

echo "Start evaluating HPSv2..."
$CONDA_BASE/envs/${EVAL_ENV}/bin/python -c "
import hpsv2
hpsv2.evaluate('${RESULT_DIR}/hpsv2/images', hps_version='v2.0')
"
echo "Start evaluating HPSv2.1"
$CONDA_BASE/envs/${EVAL_ENV}/bin/python -c "
import hpsv2
hpsv2.evaluate('${RESULT_DIR}/hpsv2/images', hps_version='v2.1')
"