"""Sana text-to-image generation pipeline.

Two modes (auto-selected):
  1. Dedicated 2D image model (SanaPipeline) — if SANA_IMAGE_MODEL_PATH is set
  2. Video model single-frame fallback (SanaVideoPipeline, frames=1) — if only
     SANA_MODEL_PATH is available.  Lets you reuse the Sana-Video checkpoint
     (including LTX-VAE variants) for image generation without downloading
     a separate 2D image model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from PIL import Image as PILImageModule
from PIL.Image import Image as PILImage

from services.services_utils import ImagePipelineOutputLike, PILImageType, get_device_type

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _SanaImageOutput:
    images: Sequence[PILImageType]


class SanaImageGenerationPipeline:
    """Text-to-image pipeline backed by Sana.

    Configuration via environment variables:
        SANA_IMAGE_MODEL_PATH      : (optional) HuggingFace repo or local path for
                                     a dedicated 2D image model.  When set, uses
                                     diffusers.SanaPipeline.
        SANA_MODEL_PATH            : Sana-Video model path.  Used as single-frame
                                     fallback when SANA_IMAGE_MODEL_PATH is unset.
        SANA_IMAGE_GUIDANCE_SCALE  : CFG scale (default: 4.5 for image model, 6.0
                                     for video fallback)
    """

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "SanaImageGenerationPipeline":
        return SanaImageGenerationPipeline(model_path=model_path, device=device)

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self._device: str | None = None
        self._cpu_offload_active = False

        image_model_path = os.environ.get("SANA_IMAGE_MODEL_PATH", "").strip()
        video_model_path = os.environ.get("SANA_MODEL_PATH", "").strip()

        self._use_video_model = False

        if image_model_path:
            self._init_image_pipeline(image_model_path)
        elif video_model_path:
            self._init_video_pipeline_as_image(video_model_path)
        else:
            fallback = model_path or "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
            self._init_image_pipeline(fallback)

        self._guidance_scale = float(os.environ.get(
            "SANA_IMAGE_GUIDANCE_SCALE",
            "6.0" if self._use_video_model else "4.5",
        ))

        if device is not None:
            self.to(device)

    def _init_image_pipeline(self, model_path: str) -> None:
        from diffusers import SanaPipeline

        logger.info("Loading Sana 2D image pipeline from: %s", model_path)
        self.pipeline = SanaPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.pipeline.text_encoder.to(torch.bfloat16)  # type: ignore[reportUnknownMemberType]
        self._use_video_model = False

    def _init_video_pipeline_as_image(self, model_path: str) -> None:
        """Load SanaVideoPipeline and generate single-frame 'images'."""
        from diffusers import SanaVideoPipeline

        logger.info(
            "No SANA_IMAGE_MODEL_PATH set — using Sana Video model for "
            "single-frame image generation: %s",
            model_path,
        )
        self.pipeline = SanaVideoPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.pipeline.text_encoder.to(torch.bfloat16)  # type: ignore[reportUnknownMemberType]
        self._use_video_model = True

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_image_output(output: object) -> ImagePipelineOutputLike:
        images = getattr(output, "images", None)
        if not isinstance(images, Sequence):
            raise RuntimeError("Unexpected Sana pipeline output: missing images sequence")

        validated: list[PILImageType] = []
        for img in cast(Sequence[object], images):
            if not isinstance(img, PILImage):
                raise RuntimeError("Unexpected Sana pipeline output: images must be PIL.Image instances")
            validated.append(img)

        return _SanaImageOutput(images=validated)

    @staticmethod
    def _video_frames_to_images(output: object) -> ImagePipelineOutputLike:
        """Extract the first frame from SanaVideoPipeline output as a PIL image."""
        frames = getattr(output, "frames", None)
        if frames is None:
            raise RuntimeError("Unexpected Sana video pipeline output: missing frames")

        first_batch = frames[0] if isinstance(frames, (list, tuple)) else frames
        if isinstance(first_batch, (list, tuple)):
            first_frame = first_batch[0]
        elif hasattr(first_batch, "shape"):
            first_frame = first_batch[0] if first_batch.ndim == 4 else first_batch
        else:
            first_frame = first_batch

        if isinstance(first_frame, PILImage):
            return _SanaImageOutput(images=[first_frame])

        if isinstance(first_frame, np.ndarray):
            arr = first_frame
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            return _SanaImageOutput(images=[PILImageModule.fromarray(arr)])

        if isinstance(first_frame, torch.Tensor):
            t = first_frame
            if t.is_floating_point():
                t = (t * 255).clamp(0, 255).byte()
            arr_np: np.ndarray[Any, np.dtype[np.uint8]] = t.cpu().numpy()
            return _SanaImageOutput(images=[PILImageModule.fromarray(arr_np)])

        raise RuntimeError(f"Cannot convert video frame of type {type(first_frame)} to PIL image")

    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        gen_device = self._resolve_generator_device()
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        effective_guidance = guidance_scale if guidance_scale > 0 else self._guidance_scale

        pipe = cast(Any, self.pipeline)

        if self._use_video_model:
            logger.info("Sana image gen via video model (single frame): %dx%d", width, height)
            output = pipe(
                prompt=prompt,
                height=height,
                width=width,
                frames=1,
                guidance_scale=effective_guidance,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
                return_dict=True,
            )
            return self._video_frames_to_images(output)

        output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=effective_guidance,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pil",
            return_dict=True,
        )
        return self._normalize_image_output(output)

    def to(self, device: str) -> None:
        runtime_device = get_device_type(device)
        if runtime_device in ("cuda", "mps"):
            self.pipeline.enable_model_cpu_offload()  # type: ignore[reportUnknownMemberType]
            self._cpu_offload_active = True
        else:
            self._cpu_offload_active = False
            self.pipeline.to(runtime_device)  # type: ignore[reportUnknownMemberType]
        self._device = runtime_device

    def _resolve_generator_device(self) -> str:
        if self._cpu_offload_active:
            return "cuda"
        if self._device is not None:
            return self._device
        execution_device = getattr(self.pipeline, "_execution_device", None)
        return get_device_type(execution_device)
