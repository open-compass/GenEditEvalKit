############################################
# Minimum Required Configurations (For Quick Start)
############################################
# 1. List of Benchmarks to Evaluate (Multiple entries allowed; command input also supported)
DS_NAMES=('')

# 2. List of Models to Evaluate (Multiple entries allowed; command input also supported)
MODEL_NAMES=('')

# 3. Whether to Perform Inference (Set to false if inference results are already generated in the output folder)
ENABLE_INFER=false

# 4. Whether to Perform Evaluation
ENABLE_EVAL=false

# 5. API Used for Evaluation (No modification needed if API is not used for evaluation)
API_KEY='your_api_key_here' # Replace with your actual API key
BASE_URL="your_api_base_url_here" # Replace with your actual API base URL, such as "https://api.openai.com/v1"

# 6. Environment Variables and Cache Paths. If you want to use the default paths, you can annotate the variables below.
CONDA_BASE="YOUR_CONDA_PATH_HERE" # Replace with your actual conda installation path, e.g., "/home/username/miniconda3"
CUDA_BASE="YOUR_CUDA_PATH_HERE" # Replace with your actual CUDA installation path, e.g., "/usr/local/cuda"
TORCH_HOME="YOUR_TORCH_CACHE_PATH_HERE" # Replace with your desired torch cache path, e.g., "/home/username/.cache/torch"
HOME="YOUR_HOME_PATH_HERE" # Path for ~, recommended to set to the root directory of GMEvalKit_dev to control some cache paths
HF_HOME="YOUR_HF_CACHE_PATH_HERE" # Replace with your desired Hugging Face cache path, e.g., "/home/username/.cache/huggingface"
HF_ENDPOINT="https://huggingface.co" # You can change to mirror endpoint such as https://hf-mirror.com if needed
TRANSFORMERS_CACHE="$HF_HOME/hub"

# 7. Offline Mode Switch (If the GPU used cannot connect to the internet to download models, set these two environment variables to 1 and manually download the required models to the above cache paths; otherwise, set to 0)
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# 8. # Environment Names Used During Inference and Evaluation (If there are multiple environments, it is recommended to separate them with spaces in the string)
declare -gA INFER_ENV_MAP
INFER_ENV_MAP=(
  ['gpt-image-1.5']=''
  ['gemini-3-pro-image-preview']=''
  ['bagel']=''
  ['lumina-dimoo']=''
  ['ovis-u1']=''
  ['qwen-image']=''
  ['qwen-image-2512']=''
  ['qwen-image-edit']=''
  ['qwen-image-edit-2509']=''
  ['qwen-image-edit-2511']=''
  # You can add more models here as needed
)
declare -gA EVAL_ENV_MAP
EVAL_ENV_MAP=(
  # T2I
  ['dpgbench']=''
  ['genai']=''
  ['geneval']=''
  ['geneval2']=''
  ['genexam']=''
  ['hpsv2']=''
  ['longtext']=''
  ['oneig']='' # It needs two environments, please use space to separate them, e.g., 'oneig-env-1 oneig-env-2'
  ['t2ireasonbench']=''
  ['tiff']=''
  ['unigenbench']=''
  ['wise']=''

  # Editing
  ['imgedit']=''
  ['rise']=''
)

# Number of GPUs Used During Benchmark Evaluation (The default configuration is based on H200 with 140 GB memory. Please adjust according to your actual GPU memory size and the number of GPUs available. For example, if you are using A100 with 80GB GPUs, it is recommended to set t2ireasonbench from 2 to 4 GPUs to ensure Qwen2.5-VL-72B can run successfully.)
## key: Benchmark Name
## al: Number of GPUs Required for Evaluation Phase of the Benchmark, 0 indicates no GPU needed; these benchmarks usually use APIs for evaluation
declare -gA EVAL_GPU_MAP
EVAL_GPU_MAP=(
  ['dpgbench']=1
  ['geneval']=1
  ['geneval2']=1
  ['imgedit']=0
  ['wise']=0
  ['genai']=1
  ['hpsv2']=1
  ['oneig']=1
  ['t2ireasonbench']=2
  ['genexam']=0
  ['rise']=0
  ['longtext']=1
  ['tiff']=0
  ['unigenbench']=0
)

## 10. Whether to Run Evaluations of Multiple Model-Benchmark Combinations in Parallel (Recommended to enable if sufficient GPU memory is available to speed up evaluation)
PARALLEL=false

## 11. Whether to Use Blank Images as Substitutes When Errors Occur
GENERATE_BLANK_IMAGE_ON_ERROR=false
