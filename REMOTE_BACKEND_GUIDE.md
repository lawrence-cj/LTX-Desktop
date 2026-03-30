# Remote Backend Guide: Mac Frontend + GPU Cluster Backend

## Overview

Run the LTX Desktop frontend on your local Mac, with the Python backend running on a GPU cluster node via SSH tunnel.

```
[Mac: Electron + React UI]  →  SSH Tunnel (port 8000)  →  [GPU Node: FastAPI Backend]
```

## Step 1: Start Backend on GPU Cluster

SSH into the cluster and get a GPU node:

```bash
# On cluster login node:
gg   # or: srun -A nvr_elm_llm --partition interactive --gpus 1 --cpus-per-task 16 -t 04:00:00 --pty $SHELL -l
```

Once on the GPU node, set up and start the backend:

```bash
# On GPU node:
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/junsongc/code/diffusion/LTX-Desktop

# Create venv in /tmp (avoids inode quota)
export UV_CACHE_DIR="/tmp/uv-cache-$$"
VENV_DIR="/tmp/ltx-venv-$$"
uv venv "$VENV_DIR" --python 3.12
source "$VENV_DIR/bin/activate"

# Install deps
uv pip install "torch==2.5.1+cu124" "torchaudio==2.5.1+cu124" --index-url https://download.pytorch.org/whl/cu124
uv pip install pydantic fastapi uvicorn pillow huggingface-hub starlette httpx numpy \
    diffusers peft ftfy sentencepiece imageio imageio-ffmpeg opencv-python-headless \
    python-multipart pynvml tqdm protobuf "transformers>=4.52,<5" einops safetensors \
    accelerate scipy av openai
uv pip install ltx-core ltx-pipelines --no-deps

# Set environment
export LTX_APP_DATA_DIR="/tmp/ltx-desktop-data"
export LTX_CHECKPOINT_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23/ltx-2.3-22b-distilled.safetensors"
export LTX_UPSAMPLER_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
export LTX_TEXT_ENCODER_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/gemma3"
export USE_SAGE_ATTENTION=0
export LTX_PORT=8000
export LTX_HOST=0.0.0.0

# (Optional) Enable LLM Director
source ~/.zshrc  # loads LTX_DESKTOP_GEMINI_API

mkdir -p "$LTX_APP_DATA_DIR/outputs"
echo '{"useLocalTextEncoder": true}' > "$LTX_APP_DATA_DIR/settings.json"

# Start the backend server
cd backend
PYTHONPATH=. python ltx2_server.py
```

You should see:
```
Server running on http://0.0.0.0:8000
```

**Note the GPU node hostname** (e.g., `pool0-02919`). You'll need it for the SSH tunnel.

## Step 2: SSH Tunnel from Mac

Open a new terminal on your Mac:

```bash
# Replace <gpu-node> with the actual hostname (e.g., pool0-02919)
# Replace <login-node> with your cluster login node
ssh -L 8000:<gpu-node>:8000 <login-node>

# Example:
ssh -L 8000:pool0-02919:8000 junsongc@login-node.nvidia.com
```

This forwards your Mac's `localhost:8000` to the GPU node's port 8000.

**Test the tunnel:**
```bash
curl http://localhost:8000/health
```

You should get a JSON response with GPU info.

## Step 3: Run Frontend on Mac

Clone the repo on your Mac (or sync it), then:

```bash
cd LTX-Desktop

# Install Node dependencies
pnpm install

# Start with remote backend
LTX_BACKEND_URL=http://localhost:8000 pnpm dev
```

The Electron app will open. Instead of starting its own Python backend, it connects to the remote one via the SSH tunnel.

## Step 4: Use the Video Agent

1. Click **"Video Agent"** in the sidebar
2. Enter a script in the text area
3. Choose a visual style (cinematic, anime, etc.)
4. Click **"Preview Plan"** to see scene breakdown (no GPU needed)
5. Edit prompts/camera motions if desired
6. Click **"Generate Video"** — scenes generate on the GPU cluster
7. Watch real-time progress
8. Click **"Open in Video Editor"** to edit the result

## Quick Start (One-liner)

If you already have the environment set up:

```bash
# Terminal 1 (on GPU node):
cd /lustre/fs1/.../LTX-Desktop && source /tmp/ltx-venv/bin/activate && \
LTX_APP_DATA_DIR=/tmp/ltx-data LTX_PORT=8000 \
LTX_CHECKPOINT_PATH=.../ltx-2.3-22b-distilled.safetensors \
LTX_UPSAMPLER_PATH=.../ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
LTX_TEXT_ENCODER_PATH=.../gemma3 \
USE_SAGE_ATTENTION=0 \
PYTHONPATH=backend python backend/ltx2_server.py

# Terminal 2 (on Mac - SSH tunnel):
ssh -L 8000:<gpu-node>:8000 <login-node>

# Terminal 3 (on Mac - frontend):
cd LTX-Desktop && LTX_BACKEND_URL=http://localhost:8000 pnpm dev
```

## Troubleshooting

### "Backend not reachable"
- Check SSH tunnel is running: `curl localhost:8000/health`
- Check GPU node backend is running: look for "Server running on" message
- Make sure `LTX_BACKEND_URL=http://localhost:8000` is set when starting `pnpm dev`

### "Models not downloaded"
- The backend expects model files at the `LTX_CHECKPOINT_PATH` etc.
- First run takes ~5-6 min to load the 43GB model to GPU

### "TEXT_ENCODING_NOT_CONFIGURED"
- Make sure `LTX_TEXT_ENCODER_PATH` points to the Gemma3 model directory
- Make sure `settings.json` has `{"useLocalTextEncoder": true}`

### Video files not playing in editor
- Videos are saved on the GPU node's filesystem
- The backend serves them via `/api/outputs/` endpoint
- SSH tunnel must stay active for video playback
