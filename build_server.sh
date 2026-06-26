#!/usr/bin/env bash
# Build the M2T2 grasp server environment (CUDA 12.8 / sm_120 for the RTX 5090).
set -euo pipefail
cd /home/samrat/tamp-vla/tamp-vla/M2T2

echo "=== [1/7] pixi install (CUDA 12.8 toolchain) ==="
pixi install

echo "=== [2/7] build tooling ==="
pixi run pip install --no-cache-dir ninja setuptools wheel

echo "=== [3/7] torch 2.7.1+cu128 (Blackwell-capable) ==="
pixi run pip install --no-cache-dir \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple

echo "=== [4/7] compile pointnet2_ops for sm_120 ==="
pixi run bash -c '
  set -e
  export CUDA_HOME="$CONDA_PREFIX"
  export PATH="$CONDA_PREFIX/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="12.0"
  echo "nvcc: $(command -v nvcc)"; nvcc --version | tail -2
  pip install --no-cache-dir --no-build-isolation ./pointnet2_ops
'

echo "=== [5/7] m2t2 package + requirements (torch already satisfied) ==="
pixi run pip install --no-cache-dir -r requirements.txt
pixi run pip install --no-cache-dir --no-deps -e .

echo "=== [6/7] server deps ==="
pixi run pip install --no-cache-dir fastapi "uvicorn[standard]" omegaconf huggingface_hub

echo "=== [7/7] download weights (wentao-yuan/m2t2) ==="
pixi run python -c "
from huggingface_hub import snapshot_download
import os
p = snapshot_download('wentao-yuan/m2t2', local_dir='weights')
print('weights at', p, '->', os.listdir('weights'))
"

echo "=== verify torch sees the 5090 + pointnet2_ops imports + a CUDA op ==="
pixi run bash -c '
  export TORCH_CUDA_ARCH_LIST="12.0"
  python -c "
import torch
print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda, \"arch\", torch.cuda.get_arch_list())
from pointnet2_ops import pointnet2_utils
x = torch.rand(1, 100, 3, device=\"cuda\")
idx = pointnet2_utils.furthest_point_sample(x, 16)
torch.cuda.synchronize()
print(\"pointnet2_ops FPS OK, idx\", idx.shape)
"
'
echo "BUILD_OK"
