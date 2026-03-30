#!/bin/bash
# Setup script for LTX Desktop Video Agent on NVIDIA cluster
# Creates a Python 3.12 venv with correct dependencies for CUDA 12.4
#
# Usage:
#   bash setup_agent_env.sh
#
# After setup, run the agent test:
#   source ltx-desktop/bin/activate
#   LTX_APP_DATA_DIR=/tmp/ltx-test python test_video_agent.py --director-only

set -e

VENV_DIR="ltx-desktop"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up LTX Desktop Video Agent environment ==="

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Install with: pip install uv"
    exit 1
fi

# Create venv with Python 3.12
if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

echo "Creating Python 3.12 venv..."
uv venv "$VENV_DIR" --python 3.12

# Activate
source "$VENV_DIR/bin/activate"

# Install torch 2.5.1 for CUDA 12.4 (compatible with cluster driver 550.x)
echo ""
echo "=== Installing PyTorch 2.5.1+cu124 ==="
uv pip install "torch==2.5.1+cu124" "torchaudio==2.5.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124

# Install LTX core dependencies
echo ""
echo "=== Installing LTX core dependencies ==="
uv pip install ltx-core ltx-pipelines

# Pin torch back (ltx-pipelines may upgrade it)
echo ""
echo "=== Pinning torch to 2.5.1+cu124 ==="
uv pip install "torch==2.5.1+cu124" "torchaudio==2.5.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124 --reinstall

# Pin transformers to compatible range
echo ""
echo "=== Installing compatible transformers ==="
uv pip install "transformers>=4.52,<5"

# Install remaining backend dependencies
echo ""
echo "=== Installing backend dependencies ==="
uv pip install \
    pydantic fastapi uvicorn pillow huggingface-hub starlette httpx \
    numpy diffusers peft ftfy sentencepiece imageio imageio-ffmpeg \
    opencv-python-headless python-multipart pynvml tqdm protobuf

echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'torch:        {torch.__version__}')
print(f'CUDA build:   {torch.version.cuda}')
import transformers
print(f'transformers: {transformers.__version__}')
import ltx_core, ltx_pipelines
print(f'ltx_core:     OK')
print(f'ltx_pipelines: OK')
import fastapi, pydantic
print(f'fastapi:      {fastapi.__version__}')
print(f'pydantic:     {pydantic.__version__}')
print()
print('Setup complete! Activate with: source ltx-desktop/bin/activate')
"

echo ""
echo "=== Environment ready ==="
echo ""
echo "To run the Video Agent test:"
echo "  source ltx-desktop/bin/activate"
echo "  LTX_APP_DATA_DIR=/tmp/ltx-test python test_video_agent.py --director-only"
echo ""
echo "For full GPU test (on a GPU node):"
echo "  gg  # get GPU node"
echo "  source ltx-desktop/bin/activate"
echo "  LTX_APP_DATA_DIR=/tmp/ltx-test python test_video_agent.py"
