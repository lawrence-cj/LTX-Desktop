#!/bin/bash
# Run Video Agent full pipeline test on a GPU node.
# This creates a temporary venv in /tmp to avoid inode quota issues on lustre.
#
# Usage:
#   srun -A nvr_elm_llm --partition interactive --gpus 1 --cpus-per-task 16 -t 02:00:00 bash run_agent_test.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="/tmp/ltx-agent-venv-$$"
MODEL_DIR="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23"
export UV_CACHE_DIR="/tmp/uv-cache-$$"

echo "=== Setting up environment in $VENV_DIR ==="
nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader

# Create venv in /tmp (no inode quota issues)
uv venv "$VENV_DIR" --python 3.12 2>&1 | tail -1
source "$VENV_DIR/bin/activate"

# Install torch for CUDA 12.4
echo "Installing torch..."
uv pip install "torch==2.5.1+cu124" "torchaudio==2.5.1+cu124" \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2

# Install backend deps
echo "Installing backend deps..."
uv pip install \
    pydantic fastapi uvicorn pillow huggingface-hub starlette httpx \
    numpy diffusers peft ftfy sentencepiece imageio imageio-ffmpeg \
    opencv-python-headless python-multipart pynvml tqdm protobuf \
    "transformers>=4.52,<5" einops safetensors accelerate scipy av 2>&1 | tail -2

# Install ltx packages without pulling torch 2.11
echo "Installing ltx packages..."
uv pip install ltx-core ltx-pipelines --no-deps 2>&1 | tail -2

echo ""
echo "=== Environment ready ==="
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "=== Running Video Agent Test ==="
export LTX_APP_DATA_DIR="/tmp/ltx-agent-test-$$"
export LTX_CHECKPOINT_PATH="$MODEL_DIR/ltx-2.3-22b-distilled.safetensors"
export LTX_UPSAMPLER_PATH="$MODEL_DIR/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
export LTX_TEXT_ENCODER_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/gemma3"
export USE_SAGE_ATTENTION=0

cd "$PROJECT_DIR"
python test_video_agent.py "$@"

echo ""
echo "=== Cleanup ==="
rm -rf "$VENV_DIR" "$UV_CACHE_DIR"
echo "Done."
