#!/bin/bash
# set -x

# ========= Auto Cleanup ==========
function setup_cleanup() {
    trap 'echo "Stopping all background jobs..."; jobs -p | xargs -r kill' SIGINT SIGTERM
}

# ========= Set CUDA Visible Devices =========
function set_visible_gpus() {
    local threshold_mib=${1:-1024}  # Can pass in threshold in MiB, default is 1024. The GPUs with more used memory than this will be hidden.
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Read GPU index and used memory (MiB), filter out GPUs with used memory <= threshold_mib
        local ids
        ids=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
              awk -F',' -v th="$threshold_mib" '{
                    gsub(/ /,"",$1); gsub(/ /,"",$2);
                    if ($2+0 <= th) print $1
              }' | paste -sd, -)
        if [ -n "$ids" ]; then
            export CUDA_VISIBLE_DEVICES="$ids"
            # Count number of CUDA devices
            local cuda_count
            cuda_count=$(echo "$ids" | awk -F',' '{print NF}')
            export CUDA_VISIBLE_COUNT="$cuda_count"
            echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (Memory Usage <= ${threshold_mib}MiB)"
            echo "Set CUDA_VISIBLE_COUNT=$CUDA_VISIBLE_COUNT"
            return 0
        else
            echo "GPU with memory usage less than or equal to ${threshold_mib}MiB not found."
        fi
    fi
}

# ========= Configuration ==========
function setup_config() {
    source config.sh # Load manually set parameters in config.sh

    WORK_DIR=$(pwd)
    LOG_DIR="$WORK_DIR/log"

    export API_KEY="${API_KEY}"
    export BASE_URL="${BASE_URL}"
    export CONDA_BASE="${CONDA_BASE}"
    export CUDA_BASE="${CUDA_BASE}"
    export TORCH_HOME="${TORCH_HOME}"
    export HF_HOME="${HF_HOME}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE}"
}

# ========= Parse Command Line Arguments, Override Defaults in config.sh if Set ==========
function parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --ds_names)
                IFS=';' read -r -a DS_NAMES <<< "$2"
                shift 2
                ;;
            --model_names)
                IFS=';' read -r -a MODEL_NAMES <<< "$2"
                shift 2
                ;;
            --custom_model_kwargses)
                custom_model_kwargses="$2"
                shift 2
                ;;
            --use_api)
                USE_API="$2"
                shift 2
                ;;
            --num_workers)
                NUM_WORKERS="$2"
                shift 2
                ;;
            --infer)
                ENABLE_INFER="$2"
                shift 2
                ;;
            --eval)
                ENABLE_EVAL="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: bash eval.sh [options]"
                echo
                echo "Common Options:"
                echo "  --ds_names \"wiseedit;gedit\"        Override DS_NAMES in config.sh"
                echo "  --model_names \"internvl-o;qwen-image\" Override MODEL_NAMES in config.sh"
                echo "  --custom_model_kwargses [optional] \"p1=v1,p2=v2;p3=v3\" Pass custom parameters for different models"
                echo "  --use_api true|false               Whether to use API for inference (default: false, which uses GPU)"
                echo "  --num_workers N                    Number of workers for API inference (default: Your CPU core count, only effective if --use_api is true)"
                echo "  --infer true|false            Override ENABLE_INFER"
                echo "  --eval  true|false            Override ENABLE_EVAL"
                echo "  --parallel     true|false            Override PARALLEL"
                echo
                echo "If no arguments are passed, the default configuration in config.sh is used."
                exit 0
                ;;
            *)
                echo "Unknown argument: $1"
                echo "Use --help to see available options"
                exit 1
                ;;
        esac
    done

    parse_custom_model_kwargses
}

# ========= Parse Custom Model Kwargs ==========
function parse_custom_model_kwargses() {
    # If custom_model_kwargses is not passed, return an empty list equal to the number of models
    if [ -z "$custom_model_kwargses" ]; then
        CUSTOM_MODEL_KWARGSES=()
        for _ in "${MODEL_NAMES[@]}"; do
            CUSTOM_MODEL_KWARGSES+=("")
        done
        return 0
    fi

    # Parse CUSTOM_MODEL_KWARGSES
    IFS=';' read -r -a CUSTOM_MODEL_KWARGSES <<< "$custom_model_kwargses"
    if [ "${#MODEL_NAMES[@]}" -ne "${#CUSTOM_MODEL_KWARGSES[@]}" ]; then
        echo "Command Error: Number of models (${#MODEL_NAMES[@]}) does not match number of custom parameter sets (${#CUSTOM_MODEL_KWARGSES[@]})" >&2
        exit 1
    fi
}

# ========= Log Management ==========
function setup_log_file() {
    local model_name=$1
    local ds_name=$2
    local step=$3
    TIMESTAMP=$(date +%Y%m%d%H%M%S)
    LOG_FILE="${LOG_DIR}/${model_name}/${ds_name}_${step}_${TIMESTAMP}.log"
    rm -rf "${LOG_DIR}/${model_name}/${ds_name}_${step}_"*.log # Delete old log files
    mkdir -p "$(dirname "${LOG_FILE}")"
    echo "${LOG_FILE}"
}

# ========= Run Inference Task ==========
function run_inference_task() {
    local model_name=$1
    local ds_name=$2
    local custom_model_kwargs=$3
    local use_api=${4:-false}
    if [ "$use_api" = true ]; then
        local num_workers=$5
    fi
    local log_file=$(setup_log_file "${model_name}" "${ds_name}" "inference")

    echo "Starting Inference Model: ${model_name} Dataset: ${ds_name}" | tee -a "${log_file}"

    if [ "$use_api" = true ]; then
        echo "Using API for inference with ${num_workers} workers" | tee -a "${log_file}"
        local -a infer_cmd=( # Assemble the command yourself, so you can better control the generate_blank_image_on_error parameter
            "${CONDA_BASE}/envs/${INFER_ENV_MAP[$model_name]}/bin/python"
            "infer/infer_mp_api.py"
            "--model_name" "${model_name}"
            "--dataset_name" "${ds_name}"
            "--custom_model_kwargs" "${custom_model_kwargs}"
        )
        if [ -n "$num_workers" ]; then
            infer_cmd+=("--num_workers" "${num_workers}")
        fi
    else
        echo "Using GPU for inference" | tee -a "${log_file}"
        set_visible_gpus
        local -a infer_cmd=( # Assemble the command yourself, so you can better control the generate_blank_image_on_error parameter
            "${CONDA_BASE}/envs/${INFER_ENV_MAP[$model_name]}/bin/python"
            "infer/infer_mp_gpu.py"
            "--model_name" "${model_name}"
            "--dataset_name" "${ds_name}"
            "--custom_model_kwargs" "${custom_model_kwargs}"
        )
    fi

    if [ "${GENERATE_BLANK_IMAGE_ON_ERROR}" = "true" ]; then
        infer_cmd+=("--generate_blank_image_on_error")
    fi

    "${infer_cmd[@]}" 2>&1 | tee -a "${log_file}"

    check_empty_image "${model_name}" "${ds_name}" | tee -a "${log_file}" # Check for empty images

    return $?
}

# ========= Run Evaluation Task ==========
function run_eval_task() {
    local model_name=$1
    local ds_name=$2
    local eval_env=${EVAL_ENV_MAP[$ds_name]}
    local log_file=$(setup_log_file "${model_name}" "${ds_name}" "eval")
    local eval_gpu_nums=${EVAL_GPU_MAP[$ds_name]}
    local eval_cpu_nums=$((eval_gpu_nums * 2 > 1 ? eval_gpu_nums * 2 : 1))

    if [ "$eval_gpu_nums" -ne 0 ]; then
        set_visible_gpus
    fi
    echo "Starting Evaluation Model: ${model_name} Dataset: ${ds_name}" | tee -a "${log_file}"
    cd "$WORK_DIR" || exit
    bash "eval/${ds_name}.sh" "${model_name}" "${eval_env}" 2>&1 | tee -a "${log_file}"
    return $?
}

# ========= Check for Empty Images ==========
function check_empty_image() {
    local model_name=$1
    local ds_name=$2
    local results_dir="output/${model_name}/${ds_name}"

    echo "Checking for empty images: $results_dir"
    python ./utils/check_empty_image.py "$results_dir"

    if [ $? -ne 0 ]; then
        echo "❌ Empty images found, evaluation aborted. Please check if inference is correct: $results_dir"
        return 1
    else
        echo "✅ No empty images found, evaluation can proceed: $results_dir"
        return 0
    fi
}

# ========= Main Loop ==========
function main() {
    # check_proxies
    setup_cleanup
    setup_config

    parse_args "$@" # After loading the configuration in config.sh, use command line arguments for optional overrides

    for DS_NAME in "${DS_NAMES[@]}"; do
        for model_idx in "${!MODEL_NAMES[@]}"; do
            MODEL_NAME=${MODEL_NAMES[$model_idx]}
            CUSTOM_MODEL_KWARGS=${CUSTOM_MODEL_KWARGSES[$model_idx]}
            (
                for i in {1..1}; do
                    if [ "$ENABLE_INFER" = true ]; then
                        # Step 1: Inference
                        run_inference_task "${MODEL_NAME}" "${DS_NAME}" "${CUSTOM_MODEL_KWARGS}" "${USE_API}" "${NUM_WORKERS}"
                        if [ $? -eq 0 ]; then
                            echo "Inference succeeded: Model=${MODEL_NAME} Dataset=${DS_NAME}, Iteration=$i"
                        else
                            echo "Inference failed: Model=${MODEL_NAME} Dataset=${DS_NAME}, Iteration=$i"
                            break
                        fi
                    fi
                    
                    if [ "$ENABLE_EVAL" = true ]; then
                        # Step 2: Evaluation
                        run_eval_task "${MODEL_NAME}" "${DS_NAME}"
                        if [ $? -eq 0 ]; then
                            echo "Evaluation succeeded: Model=${MODEL_NAME} Dataset=${DS_NAME}, Iteration=$i"
                        else
                            echo "Evaluation failed: Model=${MODEL_NAME} Dataset=${DS_NAME}, Iteration=$i"
                        fi
                    fi
                done
            ) &
            # Serial mode ("$PARALLEL" != true): wait for the background task to complete immediately; Parallel mode ("$PARALLEL" == true): do not wait
            if [ "$PARALLEL" != true ]; then
                wait
            fi
        done
    done

    wait # wait for all tasks to finish
}

main "$@"