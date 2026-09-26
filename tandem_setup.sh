#!/usr/bin/env bash
# Build the M2T2 grasp server inside this workspace's pixi environment: `pixi run setup`.
#
# What build_server.sh does, minus its own `pixi install`/`pixi run` calls, so it can run as a pixi task
# (tandem builds this server as a runtime and runs the task). The CUDA architecture pointnet2_ops is
# compiled for defaults to this machine's GPUs, read from nvidia-smi, instead of a fixed 8.9: kernels
# built for another architecture fail at the first grasp request.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';' -)"
fi
export TORCH_CUDA_ARCH_LIST
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

pip install --no-cache-dir ninja setuptools wheel
pip install --no-cache-dir \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple
pip install --no-cache-dir --no-build-isolation ./pointnet2_ops
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir --no-deps -e .
pip install --no-cache-dir fastapi "uvicorn[standard]" omegaconf huggingface_hub
echo "SETUP_OK"
