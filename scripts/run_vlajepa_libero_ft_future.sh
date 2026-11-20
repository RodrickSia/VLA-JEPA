#!/usr/bin/env bash
set -euo pipefail
# World-model action head variant (VLA_JEPA_future + GR00T_ActionHeader_Future). Original: run_vlajepa_libero_ft.sh

REPO_ROOT=/mnt/GuestShare_NAS/WorkingSpace/Personal/nguyenmv/ActionChunking/VLA-JEPA
RUNTIME_ENV=/home/developer/miniconda3/envs/VLA_JEPA
ACCELERATE_BIN=${RUNTIME_ENV}/bin/accelerate
CONFIG_FILE=${REPO_ROOT}/scripts/configs/vlajepa_libero_ft_future.yaml
DS_CONFIG=${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2_offload_3gpu.yaml
RUN_ROOT=${REPO_ROOT}/runs
RUN_ID=${RUN_ID:-vlajepa_libero_ft_future}
RUN_DIR=${RUN_ROOT}/${RUN_ID}
LOG_DIR=${RUN_DIR}/logs
LOG_FILE=${LOG_DIR}/train.log

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_HOME="${RUNTIME_ENV}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDA_NVIDIA_SITE_PACKAGES="${CUDA_HOME}/lib/python3.10/site-packages/nvidia"
export CUDA_INCLUDE_PATHS="${CUDA_HOME}/targets/x86_64-linux/include:${CUDA_HOME}/targets/x86_64-linux/include/cccl:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_runtime/include:${CUDA_NVIDIA_SITE_PACKAGES}/cusparse/include:${CUDA_NVIDIA_SITE_PACKAGES}/cublas/include:${CUDA_NVIDIA_SITE_PACKAGES}/cusolver/include:${CUDA_NVIDIA_SITE_PACKAGES}/cufft/include:${CUDA_NVIDIA_SITE_PACKAGES}/cudnn/include:${CUDA_NVIDIA_SITE_PACKAGES}/nccl/include:${CUDA_NVIDIA_SITE_PACKAGES}/nvtx/include:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_cupti/include:${CUDA_NVIDIA_SITE_PACKAGES}/curand/include:${CUDA_NVIDIA_SITE_PACKAGES}/nvjitlink/include:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_nvrtc/include"
export CUDA_LIBRARY_PATHS="${CUDA_NVIDIA_SITE_PACKAGES}/cuda_runtime/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cusparse/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cublas/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cusolver/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cufft/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cudnn/lib:${CUDA_NVIDIA_SITE_PACKAGES}/nccl/lib:${CUDA_NVIDIA_SITE_PACKAGES}/nvtx/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_cupti/lib:${CUDA_NVIDIA_SITE_PACKAGES}/curand/lib:${CUDA_NVIDIA_SITE_PACKAGES}/nvjitlink/lib:${CUDA_NVIDIA_SITE_PACKAGES}/cuda_nvrtc/lib:${CUDA_HOME}/lib:${CUDA_HOME}/targets/x86_64-linux/lib"
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATHS}:${CUDA_HOME}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR=${REPO_ROOT}/.train_tmp
export HF_HOME=${REPO_ROOT}/.hf_cache XDG_CACHE_HOME=${REPO_ROOT}/.hf_cache/xdg TRITON_CACHE_DIR=${REPO_ROOT}/.train_tmp/triton TORCH_EXTENSIONS_DIR=${REPO_ROOT}/.train_tmp/torch_extensions
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7}   # only GPUs 5-7
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}   # /dev/shm is only 64 MB in this container; NCCL's shared-memory buffers overflowed it
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export OMP_NUM_THREADS=1
export FFMPEG_THREADS=1
mkdir -p "${TMPDIR}"

echo "VLA-JEPA training"
echo "  repo: ${REPO_ROOT}"
echo "  config: ${CONFIG_FILE}"
echo "  run_dir: ${RUN_DIR}"
echo "  log: ${LOG_FILE}"

"${ACCELERATE_BIN}" launch \
  --config_file "${DS_CONFIG}" \
  --num_processes 3 \
  "${REPO_ROOT}/starVLA/training/train_starvla.py" \
  --config_yaml "${CONFIG_FILE}" \
  --run_id "${RUN_ID}" "$@"
