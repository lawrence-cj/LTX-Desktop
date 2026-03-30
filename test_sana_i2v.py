#!/usr/bin/env python3
"""Standalone test for Sana Image-to-Video pipeline.

Usage:
    python test_sana_i2v.py --image /path/to/image.png --prompt "a cat running"

Outputs sana_i2v_output.mp4 in the current directory.
"""

import argparse
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SanaImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test Sana I2V pipeline")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, default="a video of the scene", help="Text prompt")
    parser.add_argument("--model", type=str, default="Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="sana_i2v_output.mp4")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    pipe = SanaImageToVideoPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=args.shift)
    pipe.text_encoder.to(torch.bfloat16)
    pipe.enable_model_cpu_offload()

    image = Image.open(args.image).convert("RGB")
    print(f"Input image: {args.image} ({image.size[0]}x{image.size[1]})")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print(f"Generating: {args.width}x{args.height}, {args.frames} frames, {args.steps} steps, cfg={args.guidance_scale}, shift={args.shift}")
    output = pipe(
        image=image,
        prompt=args.prompt,
        negative_prompt="low quality, worst quality, blurry",
        height=args.height,
        width=args.width,
        frames=args.frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        generator=generator,
    )

    frames = output.frames[0]
    export_to_video(frames, args.output, fps=16)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
