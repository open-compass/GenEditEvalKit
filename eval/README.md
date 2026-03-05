# Guidelines for Writing Evaluation Scripts

A typical evaluation script consists of four parts:
1. Read parameters, including the name of the model to be evaluated and the environment used for evaluation.
2. Set environment variables, for example the path where images are stored, the path for model weights used during evaluation, and activate Conda or similar environments.
3. Change the working directory to the root directory of the evaluation repository.
4. Execute the evaluation command.

For example, the following script (genai.sh):
```bash
# Read parameters
MODEL_NAME=$1
EVAL_ENV=$2

# Set environment variables
RESULT_DIR="$PWD/output/${MODEL_NAME}"
VISION_TOWER="$PWD/eval_models/clip-vit-large-patch14-336"
T5_PATH="$PWD/eval_models/clip-flant5-xxl"
source utils/use_cuda.sh 11.8

# Change working directory
genai_dir="$PWD/benchmarks/genai"
echo "genai directory: $genai_dir"
cd $genai_dir || exit

# Execute the evaluation command
META_DIR="eval_prompts/genai1600"
IMAGE_DIR="${RESULT_DIR}/genai/images"
VISION_TOWER=${VISION_TOWER} $CONDA_BASE/envs/yph-genai/bin/python -m step2_run_model \
    --model_path ${T5_PATH} \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR} > ${RESULT_DIR}/genai/results.txt
cat ${RESULT_DIR}/genai/results.txt
```

You may of course modify the structure of the evaluation script as needed, provided the changes are reasonable.