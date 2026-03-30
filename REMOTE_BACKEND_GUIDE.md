# Remote Backend Guide: Mac Frontend + GPU Cluster Backend

## Overview

```
[Mac: Electron + React UI]  →  SSH Tunnel (port 8000)  →  [GPU Node: FastAPI Backend]
```

## Step 1: Start Backend on GPU Cluster

```bash
# Get a GPU node
gg

# On GPU node:
cd /lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/junsongc/code/diffusion/LTX-Desktop
source ltx-desktop/bin/activate

# Set environment
export LTX_APP_DATA_DIR="/tmp/ltx-desktop-data"
export LTX_CHECKPOINT_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23/ltx-2.3-22b-distilled.safetensors"
export LTX_UPSAMPLER_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
export LTX_TEXT_ENCODER_PATH="/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/gemma3"
export USE_SAGE_ATTENTION=0
export LTX_PORT=8000
export LTX_HOST=0.0.0.0
source ~/.zshrc  # loads LTX_DESKTOP_GEMINI_API for LLM Director

mkdir -p "$LTX_APP_DATA_DIR/outputs"
echo '{"useLocalTextEncoder": true}' > "$LTX_APP_DATA_DIR/settings.json"

# Start backend
cd backend
PYTHONPATH=. python ltx2_server.py
```

Wait for: `Server running on http://0.0.0.0:8000`

Note the GPU node hostname (e.g., `pool0-02919`).

## Step 2: SSH Tunnel from Mac

```bash
# Replace <gpu-node> with actual hostname, <login-node> with cluster login
ssh -L 8000:<gpu-node>:8000 <login-node>
```

Test: `curl http://localhost:8000/health`

## Step 3: Run Frontend on Mac

```bash
cd LTX-Desktop
pnpm install
LTX_BACKEND_URL=http://localhost:8000 pnpm dev
```

## Using Video Agent

1. Click **"Video Agent"** in sidebar
2. Enter script → Choose style → **Preview Plan** (no GPU)
3. Edit prompts/cameras if needed → **Generate Video**
4. Watch progress → **Open in Video Editor**

## Troubleshooting

- **"Backend not reachable"**: Check `curl localhost:8000/health`
- **First scene slow (~5 min)**: Model loading to GPU, subsequent scenes ~40s each
- **"TEXT_ENCODING_NOT_CONFIGURED"**: Ensure `LTX_TEXT_ENCODER_PATH` and `settings.json` are set
