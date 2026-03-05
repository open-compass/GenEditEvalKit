# Converting CUDA to any version
# Usage (to switch to cuda 11.8 as an example): source utils/use_cuda.sh 11.8
# Note that after using source use_cuda.sh, it will switch to the base environment, you need to reactivate the environment or directly use the python command in the target environment folder

CUDA_HOME="$CUDA_BASE/cuda-${1:-cuda_version}"
export CUDA_HOME
export CUDA_VERSION="$(basename "$CUDA_HOME")"

# configure CUDA
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# print current cuda version
echo "Current CUDA version: $(nvcc --version | grep release) | CUDA_HOME=$CUDA_HOME"