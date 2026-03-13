"""Pure diffusers-based Sana-Video + LTX2 Refiner pipeline for LTX Desktop.

This is an alternative implementation that uses ONLY diffusers pipelines:
  - Stage 1: diffusers.SanaVideoPipeline  (Sana Video generation)
  - Stage 2: diffusers.LTX2Pipeline       (LTX2 refiner, 3 steps with distilled LoRA)

Pros:
  - Self-contained, no ltx_core dependency for Stage 2
  - Matches the proven sana_video_two_stages_diffusers.py workflow
  - clean API with from_pretrained()

Cons:
  - Requires downloading the diffusers-format LTX-2 model SEPARATELY (~40GB)
    because LTX Desktop downloads ltx_core-format checkpoint, not diffusers format
  - LTX2Pipeline uses T5 text encoder; LTX Desktop uses Gemma-3 (different embeddings)
  - Output format needs conversion to match LTX Desktop's encode_video expectations

Configuration via environment variables:
    SANA_MODEL_PATH  : HF repo id or local path for Sana Video (diffusers format)
    LTX2_MODEL_PATH  : HF repo id or local path for LTX-2 (diffusers format)
                       Default: "Lightricks/LTX-2"
    SANA_SKIP_UPSAMPLE : "true"/"false" (default: "true")
    SANA_SKIP_REFINE   : "true"/"false" (default: "false")
"""

from __future__ import annotations

import gc
import logging
import os
from collections.abc import Iterator
from typing import Final

import torch

from api_types import ImageConditioningInput
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number

logger = logging.getLogger(__name__)


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SanaLTXDiffusersPipeline:
    """Pure diffusers two-stage pipeline: SanaVideoPipeline → LTX2Pipeline refiner.

    NOTE: This pipeline requires the LTX-2 model in diffusers format, which is a
    SEPARATE download from the ltx_core-format checkpoint that LTX Desktop uses.
    Set LTX2_MODEL_PATH to the diffusers-format model path (e.g. "Lightricks/LTX-2").
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
    ) -> "SanaLTXDiffusersPipeline":
        return SanaLTXDiffusersPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
    ) -> None:
        self.device = device
        self.dtype = torch.bfloat16

        self.skip_upsample = os.environ.get("SANA_SKIP_UPSAMPLE", "true").lower() == "true"
        self.skip_refine = os.environ.get("SANA_SKIP_REFINE", "false").lower() == "true"

        sana_model_id = os.environ.get("SANA_MODEL_PATH", "Efficient-Large-Model/SANA-Video_2B_480p_diffusers")
        ltx2_model_id = os.environ.get("LTX2_MODEL_PATH", "Lightricks/LTX-2")

        from diffusers import FlowMatchEulerDiscreteScheduler, SanaVideoPipeline

        logger.info("Loading Sana pipeline from: %s", sana_model_id)
        self.sana_pipe = SanaVideoPipeline.from_pretrained(sana_model_id, torch_dtype=self.dtype)
        self.sana_pipe.text_encoder.to(self.dtype)

        self.ltx_pipe = None
        self.upsample_pipe = None

        if not self.skip_refine:
            from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

            logger.info("Loading LTX2 pipeline from: %s", ltx2_model_id)
            self.ltx_pipe = LTX2Pipeline.from_pretrained(ltx2_model_id, torch_dtype=self.dtype)

            if not self.skip_upsample:
                latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                    ltx2_model_id, subfolder="latent_upsampler", torch_dtype=self.dtype,
                )
                self.upsample_pipe = LTX2LatentUpsamplePipeline(
                    vae=self.ltx_pipe.vae, latent_upsampler=latent_upsampler,
                )

            self.ltx_pipe.load_lora_weights(
                ltx2_model_id,
                adapter_name="stage_2_distilled",
                weight_name="ltx-2-19b-distilled-lora-384.safetensors",
            )
            self.ltx_pipe.set_adapters("stage_2_distilled", 1.0)
            self.ltx_pipe.vae.enable_tiling()
            self.ltx_pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.ltx_pipe.scheduler.config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.pipelines.ltx2.export_utils import encode_video as diffusers_encode_video

        if images:
            logger.warning("Image conditioning not supported in diffusers Sana pipeline, using T2V")

        device = self.device
        dtype = self.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        full_prompt = prompt + f" motion score: {30}."

        # ── Stage 1: Sana Video ──
        self.sana_pipe.enable_model_cpu_offload()

        if self.skip_refine:
            video_frames = self.sana_pipe(
                prompt=full_prompt,
                negative_prompt="shaky, glitchy, low quality, worst quality",
                height=height, width=width, frames=num_frames,
                guidance_scale=6.0, num_inference_steps=50,
                generator=generator,
            ).frames[0]
            from diffusers.utils import export_to_video
            export_to_video(video_frames, output_path, fps=int(frame_rate))
            return

        sana_output = self.sana_pipe(
            prompt=full_prompt,
            negative_prompt="shaky, glitchy, low quality, worst quality",
            height=height, width=width, frames=num_frames,
            guidance_scale=6.0, num_inference_steps=50,
            generator=generator,
            output_type="latent", return_dict=True,
        )

        video_latent = None
        for attr in ("latents", "frames", "video_latents", "latent", "dit_latents"):
            val = getattr(sana_output, attr, None)
            if val is not None:
                video_latent = val
                break
        if video_latent is None:
            raise RuntimeError("Failed to extract latents from Sana output")
        if isinstance(video_latent, (list, tuple)):
            video_latent = video_latent[0]
        if video_latent.dim() == 4:
            video_latent = video_latent.unsqueeze(0)

        del sana_output
        self.sana_pipe.to("cpu")
        _cleanup_gpu()

        logger.info("Stage 1 done, latent: %s", video_latent.shape)

        # ── Stage 1.5: Optional Latent Upsample ──
        if self.upsample_pipe is not None and not self.skip_upsample:
            self.upsample_pipe.enable_model_cpu_offload(device=str(device))
            upscaled = self.upsample_pipe(
                latents=video_latent.to(device=device, dtype=dtype),
                latents_normalized=True,
                height=height, width=width, num_frames=num_frames,
                output_type="latent", return_dict=False,
            )[0]
            latents_mean = self.ltx_pipe.vae.latents_mean.view(1, -1, 1, 1, 1).to(upscaled.device, upscaled.dtype)
            latents_std = self.ltx_pipe.vae.latents_std.view(1, -1, 1, 1, 1).to(upscaled.device, upscaled.dtype)
            scaling_factor = self.ltx_pipe.vae.config.scaling_factor
            video_latent = (upscaled - latents_mean) * scaling_factor / latents_std
            del upscaled
            _cleanup_gpu()

        # ── Stage 2: LTX2 Refiner ──
        from diffusers.pipelines.ltx2 import LTX2Pipeline

        packed = LTX2Pipeline._pack_latents(
            video_latent.to(device=device, dtype=dtype),
            patch_size=self.ltx_pipe.transformer_spatial_patch_size,
            patch_size_t=self.ltx_pipe.transformer_temporal_patch_size,
        )

        _, _, lF, lH, lW = video_latent.shape
        pixel_h = lH * self.ltx_pipe.vae_spatial_compression_ratio
        pixel_w = lW * self.ltx_pipe.vae_spatial_compression_ratio
        pixel_t = (lF - 1) * self.ltx_pipe.vae_temporal_compression_ratio + 1

        audio_latent = self._create_audio_latent(lF, frame_rate, seed, device, dtype)

        del video_latent
        _cleanup_gpu()

        self.ltx_pipe.enable_model_cpu_offload()
        generator = torch.Generator(device=device).manual_seed(seed)

        video, audio = self.ltx_pipe(
            latents=packed,
            audio_latents=audio_latent,
            prompt=full_prompt,
            negative_prompt="shaky, glitchy, low quality, worst quality",
            height=pixel_h, width=pixel_w, num_frames=pixel_t,
            num_inference_steps=3,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            frame_rate=frame_rate,
            generator=generator,
            output_type="np", return_dict=False,
        )

        video_uint8 = torch.from_numpy((video * 255).round().astype("uint8"))
        diffusers_encode_video(
            video_uint8[0],
            fps=frame_rate,
            audio=None,
            audio_sample_rate=None,
            output_path=output_path,
        )

    def _create_audio_latent(
        self, latent_num_frames: int, frame_rate: float, seed: int,
        device: torch.device | str, dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create zero-equivalent audio latent (mean-filled so normalize → zeros)."""
        num_frames_pixel = (latent_num_frames - 1) * 8 + 1
        duration_s = num_frames_pixel / frame_rate
        audio_num_frames = round(duration_s * 16000 / 160 / 4)
        num_channels_audio, latent_mel_bins = 8, 16

        audio_latents_mean = self.ltx_pipe.audio_vae.latents_mean
        packed = (
            audio_latents_mean.unsqueeze(0).unsqueeze(0)
            .expand(1, audio_num_frames, num_channels_audio * latent_mel_bins)
            .to(dtype=dtype, device=device)
            .contiguous()
        )
        return packed.unflatten(2, (num_channels_audio, latent_mel_bins)).permute(0, 2, 1, 3).contiguous()

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        pass

    def compile_transformer(self) -> None:
        pass
