#!/usr/bin/env python3
"""Integration test for the Video Agent pipeline.

Usage (on a GPU node after `gg`):
    source ltx-desktop/bin/activate
    LTX_APP_DATA_DIR=/tmp/ltx-test \
    LTX_CHECKPOINT_PATH=/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23/ltx-2.3-22b-distilled.safetensors \
    LTX_UPSAMPLER_PATH=/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    USE_SAGE_ATTENTION=0 \
    python test_video_agent.py

For quick test (no GPU, just director):
    source ltx-desktop/bin/activate
    python test_video_agent.py --director-only
"""

from __future__ import annotations

import json
import os
import sys
import time

# Set defaults
MODEL_DIR = "/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/ltx23"
os.environ.setdefault("LTX_CHECKPOINT_PATH", f"{MODEL_DIR}/ltx-2.3-22b-distilled.safetensors")
os.environ.setdefault("LTX_UPSAMPLER_PATH", f"{MODEL_DIR}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
GEMMA_DIR = "/home/junsongc/junsongc/code/diffusion/Sana/output/pretrained_models/gemma3"
os.environ.setdefault("LTX_TEXT_ENCODER_PATH", GEMMA_DIR)
os.environ.setdefault("LTX_APP_DATA_DIR", "/tmp/ltx-agent-test")
os.environ.setdefault("USE_SAGE_ATTENTION", "0")

# Add backend to path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
os.chdir(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)


def test_director_only() -> None:
    """Test scene planning without GPU."""
    from api_types import AgentGenerateRequest
    from services.video_agent.director import plan_scenes

    print("=" * 60)
    print("TEST: Director Agent (scene planning)")
    print("=" * 60)

    # Test 1: Auto-segmentation
    req = AgentGenerateRequest(
        script="A robot walks through a neon-lit city at night. It encounters a stray cat under a streetlight. Together they watch the sunset from a rooftop.",
        style="cinematic",
        resolution="720p",
        duration_per_scene=5,
        total_duration=30,
    )
    plans = plan_scenes(req)
    print(f"\nAuto-segmented {len(plans)} scenes:")
    for p in plans:
        print(f"  Scene {p.scene_index}: [{p.camera_motion}] {p.duration}s - {p.description}")

    assert len(plans) == 3, f"Expected 3 scenes, got {len(plans)}"
    assert plans[0].camera_motion == "dolly_out", "First scene should be establishing shot"
    assert all(p.duration >= 2 for p in plans), "All scenes must be >= 2s"
    print("  [PASS] Auto-segmentation OK")

    # Test 2: User-provided scenes
    req2 = AgentGenerateRequest(
        script="ignored",
        style="anime",
        scenes=[
            {"description": "A hero appears", "duration": 3, "camera_motion": "dolly_in"},  # type: ignore[list-item]
            {"description": "Battle begins"},  # type: ignore[list-item]
        ],
    )
    plans2 = plan_scenes(req2)
    print(f"\nUser-provided {len(plans2)} scenes:")
    for p in plans2:
        print(f"  Scene {p.scene_index}: [{p.camera_motion}] {p.duration}s - {p.description}")
    assert len(plans2) == 2
    assert plans2[0].camera_motion == "dolly_in", "Should use user-specified camera"
    assert "anime" in plans2[0].prompt.lower(), "Should include anime style"
    print("  [PASS] User-provided scenes OK")

    # Test 3: Different styles
    for style in ["cinematic", "anime", "realistic", "fantasy", "noir", "documentary"]:
        req3 = AgentGenerateRequest(script="Test.", style=style)  # type: ignore[arg-type]
        p3 = plan_scenes(req3)
        assert len(p3) >= 1
    print("  [PASS] All styles OK")

    print("\n[PASS] All director tests passed\n")


def test_full_pipeline() -> None:
    """Test the full pipeline with actual video generation (requires GPU)."""
    print("=" * 60)
    print("TEST: Full Agent Pipeline (GPU required)")
    print("=" * 60)

    import torch
    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available\n")
        return

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Ensure output dir exists
    app_data = os.environ["LTX_APP_DATA_DIR"]
    os.makedirs(app_data + "/outputs", exist_ok=True)

    # Write settings to enable local text encoder
    import json
    settings_path = os.path.join(app_data, "settings.json")
    settings_data = {"useLocalTextEncoder": True}
    with open(settings_path, "w") as f:
        json.dump(settings_data, f)
    print(f"  Settings: useLocalTextEncoder=True -> {settings_path}")

    from starlette.testclient import TestClient

    # Import the full server - this sets up everything
    from ltx2_server import app

    client = TestClient(app)

    # Test 1: Progress endpoint
    resp = client.get("/api/agent/progress")
    print(f"\n  GET /api/agent/progress -> {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "idle"
    print("  [PASS] Progress endpoint idle")

    # Test 2: Generate a short video (2 scenes x 2s = 4s)
    request_body = {
        "script": "A beautiful sunset over the ocean with golden light reflecting on the water. Waves gently crash against a sandy beach.",
        "style": "cinematic",
        "resolution": "720p",
        "duration_per_scene": 2,
        "total_duration": 4,
        "fps": "24",
        "model": "fast",
    }

    print(f"\n  Generating {request_body['total_duration']}s video...")
    t0 = time.time()
    resp = client.post("/api/agent/generate", json=request_body, timeout=600)
    elapsed = time.time() - t0

    print(f"  Response ({elapsed:.1f}s): {resp.status_code}")
    data = resp.json()
    print(f"  Status: {data['status']}")
    print(f"  Scenes: {len(data.get('scene_plans', []))}")
    print(f"  Videos: {data.get('video_paths', [])}")
    print(f"  Final:  {data.get('final_video_path')}")

    if data["status"] == "complete" and data.get("final_video_path"):
        final_path = data["final_video_path"]
        if os.path.exists(final_path):
            size_mb = os.path.getsize(final_path) / 1e6
            print(f"  Size:   {size_mb:.1f}MB")
            print(f"\n  [PASS] Video generated: {final_path}")
        else:
            print(f"\n  [FAIL] File not found: {final_path}")
    elif data.get("error"):
        print(f"\n  [FAIL] Error: {data['error']}")
    else:
        print(f"\n  [FAIL] Unexpected status: {data['status']}")

    print()


if __name__ == "__main__":
    test_director_only()

    if "--director-only" not in sys.argv:
        try:
            test_full_pipeline()
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping GPU tests (--director-only)")
