# Load parameters
MODEL_NAME=$1
EVAL_ENV=$2

# Set environment variables
RESULT_DIR="$PWD/output/${MODEL_NAME}"
VISION_TOWER="$PWD/eval_models/clip-vit-large-patch14-336"
T5_PATH="$PWD/eval_models/clip-flant5-xxl"
source utils/use_cuda.sh 11.8

# Change working directory
genai_dir="$PWD/benchmarks/genai"
cd $genai_dir || exit

# Execute evaluation command
META_DIR="eval_prompts/genai1600"
IMAGE_DIR="${RESULT_DIR}/genai/images"
VISION_TOWER=${VISION_TOWER} ${CONDA_BASE}/envs/${EVAL_ENV}/bin/python -m step2_run_model \
    --model_path ${T5_PATH} \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR} > ${RESULT_DIR}/genai/results.txt
cat ${RESULT_DIR}/genai/results.txt