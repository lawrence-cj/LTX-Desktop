"""Sana-Video + LTX Refiner two-stage video pipeline for LTX Desktop.

Stage 1: Sana Video DiT generates latents in LTX VAE latent space (diffusers)
Stage 2: LTX spatial upsampler + LTX transformer refiner (ltx_core, 3 steps)
Stage 3: LTX VAE decode → MP4

Because Sana-Video uses the same LTX VAE (128ch, 32x spatial, 8x temporal compression),
its latent output is directly compatible with LTX's upsampler, refiner, and decoder.
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
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

logger = logging.getLogger(__name__)

SANA_DEFAULT_GUIDANCE_SCALE = 6.0
SANA_DEFAULT_NUM_STEPS = 50
SANA_DEFAULT_MOTION_SCORE = 30
SANA_MAX_FRAMES = 81


def _cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SanaLTXFastVideoPipeline:
    """Two-stage pipeline: Sana Video (Stage 1) → LTX Refiner (Stage 2).

    Implements the FastVideoPipeline Protocol so it can be injected into
    LTX Desktop's PipelinesHandler as a drop-in replacement.

    Configuration via environment variables:
        SANA_MODEL_PATH     : HuggingFace repo id or local path for Sana Video model
        SANA_ENABLE_REFINE  : "true"/"false" — run LTX refiner Stage 2 (default: true)
        SANA_ENABLE_UPSAMPLE: "true"/"false" — run LTX spatial 2x upsampler (default: false)
        SANA_GUIDANCE_SCALE : Stage 1 CFG scale (default: 6.0)
        SANA_NUM_STEPS      : Stage 1 denoising steps (default: 50)
        SANA_MOTION_SCORE   : Motion score appended to prompt (default: 30)
    """

    pipeline_kind: Final = "fast"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
    ) -> "SanaLTXFastVideoPipeline":
        return SanaLTXFastVideoPipeline(
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

        # Sana-specific LTX 2.0 model overrides (SANA_LTX_*).
        # When set, these take priority over the global paths passed from
        # the pipeline handler — which default to LTX 2.3 models.
        self.checkpoint_path = os.environ.get("SANA_LTX_CHECKPOINT_PATH", checkpoint_path)
        self.gemma_root = os.environ.get("SANA_LTX_GEMMA_ROOT") or gemma_root
        self.upsampler_path = os.environ.get("SANA_LTX_UPSAMPLER_PATH", upsampler_path)

        self.enable_refine = os.environ.get("SANA_ENABLE_REFINE", "true").lower() == "true"
        self.enable_upsample = os.environ.get("SANA_ENABLE_UPSAMPLE", "false").lower() == "true"
        self.guidance_scale = float(os.environ.get("SANA_GUIDANCE_SCALE", str(SANA_DEFAULT_GUIDANCE_SCALE)))
        self.num_steps = int(os.environ.get("SANA_NUM_STEPS", str(SANA_DEFAULT_NUM_STEPS)))
        self.motion_score = int(os.environ.get("SANA_MOTION_SCORE", str(SANA_DEFAULT_MOTION_SCORE)))

        sana_model_id = os.environ.get("SANA_MODEL_PATH", "Efficient-Large-Model/SANA-Video_2B_480p_diffusers")

        logger.info("Loading Sana Video pipeline from: %s", sana_model_id)
        from diffusers import SanaVideoPipeline

        self.sana_pipe = SanaVideoPipeline.from_pretrained(sana_model_id, torch_dtype=self.dtype)
        self.sana_pipe.text_encoder.to(self.dtype)

        if self.enable_refine:
            logger.info("LTX Refiner enabled (upsample=%s)", self.enable_upsample)
            self._init_ltx_refiner()
        else:
            logger.info("LTX Refiner disabled — Sana-only mode")
            self.model_ledger = None

    def _init_ltx_refiner(self) -> None:
        """Lazily prepare the ModelLedger config for LTX Stage 2.

        Actual model weights are loaded on-demand during generate() to save VRAM.
        We store constructor args here and build ModelLedger when needed.
        """
        self._refiner_args = {
            "checkpoint_path": self.checkpoint_path,
            "gemma_root_path": self.gemma_root,
            "upsampler_path": self.upsampler_path,
        }

    def _build_model_ledger(self) -> "ModelLedger":
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.utils import ModelLedger

        lora_path = os.environ.get(
            "SANA_LTX_DISTILLED_LORA_PATH",
            os.environ.get("LTX_DISTILLED_LORA_PATH", ""),
        )
        loras: tuple[LoraPathStrengthAndSDOps, ...] | None = None
        if lora_path and os.path.isfile(lora_path):
            loras = (LoraPathStrengthAndSDOps(lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP),)
            logger.info("Stage 2 distilled LoRA: %s", lora_path)
        else:
            logger.warning("No distilled LoRA found (SANA_LTX_DISTILLED_LORA_PATH / LTX_DISTILLED_LORA_PATH=%r)", lora_path)

        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self._refiner_args["checkpoint_path"],
            gemma_root_path=self._refiner_args["gemma_root_path"],
            spatial_upsampler_path=self._refiner_args["upsampler_path"],
            loras=loras,
            quantization=QuantizationPolicy.fp8_cast() if device_supports_fp8(self.device) else None,
        )

    # ------------------------------------------------------------------
    # Stage 1: Sana Video generation
    # ------------------------------------------------------------------

    def _run_sana_stage(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Run Sana Video DiT to produce normalized latents [1, 128, T', H', W']."""
        self.sana_pipe.enable_model_cpu_offload()

        full_prompt = prompt
        if self.motion_score > 0:
            full_prompt += f" motion score: {self.motion_score}."

        generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info("Sana Stage 1: generating %dx%d, %d frames ...", width, height, num_frames)
        sana_output = self.sana_pipe(
            prompt=full_prompt,
            negative_prompt=(
                "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, "
                "sudden disappearance, jump cuts, jerky movements, rapid shot changes, "
                "frames out of sync, inconsistent character shapes, temporal artifacts, "
                "jitter, and ghosting effects, creating a disorienting visual experience."
            ),
            height=height,
            width=width,
            frames=num_frames,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_steps,
            generator=generator,
            output_type="latent",
            return_dict=True,
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

        logger.info("Sana Stage 1 done, latent shape: %s", video_latent.shape)

        self.sana_pipe.to("cpu")
        _cleanup_gpu()

        return video_latent

    # ------------------------------------------------------------------
    # Stage 2: LTX Refiner (upsampler + 3-step denoise)
    # ------------------------------------------------------------------

    def _run_ltx_refine_stage(
        self,
        video_latent: torch.Tensor,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        """Run LTX spatial upsampler + transformer refiner + VAE decode."""
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
        from ltx_core.model.upsampler.model import upsample_video
        from ltx_core.model.video_vae import decode_video as vae_decode_video
        from ltx_core.text_encoders.gemma import encode_text
        from ltx_core.types import AudioLatentShape, VideoPixelShape
        from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            noise_audio_state,
            noise_video_state,
            simple_denoising_func,
        )
        from ltx_pipelines.utils.types import PipelineComponents

        model_ledger = self._build_model_ledger()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        components = PipelineComponents(dtype=self.dtype, device=self.device)

        # --- Text encoding for refiner ---
        logger.info("LTX Stage 2: encoding prompt ...")
        text_encoder = model_ledger.text_encoder()
        context_p = encode_text(text_encoder, prompts=[prompt])[0]
        video_context, audio_context = context_p

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        del text_encoder
        cleanup_memory()

        # --- Optional spatial upsample ---
        latent = video_latent.to(device=self.device, dtype=self.dtype)

        if self.enable_upsample:
            logger.info("LTX Stage 2: spatial upsampling 2x ...")
            video_encoder = model_ledger.video_encoder()
            upsampler = model_ledger.spatial_upsampler()
            latent = upsample_video(
                latent=latent[:1],
                video_encoder=video_encoder,
                upsampler=upsampler,
            )
            del upsampler
            cleanup_memory()
        else:
            video_encoder = model_ledger.video_encoder()

        # --- Transformer refiner (3 denoising steps) ---
        logger.info("LTX Stage 2: refiner denoising (3 steps) ...")
        transformer = model_ledger.transformer()
        stage_2_sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=self.device)

        # Derive pixel dimensions from actual latent shape, not from the
        # frontend-requested size, because Sana may produce latents at a
        # different resolution.  LTX VAE: 32x spatial, 8x temporal.
        _, _, lat_t, lat_h, lat_w = latent.shape
        VAE_SPATIAL_COMPRESSION = 32
        VAE_TEMPORAL_COMPRESSION = 8
        actual_height = lat_h * VAE_SPATIAL_COMPRESSION
        actual_width = lat_w * VAE_SPATIAL_COMPRESSION
        actual_frames = (lat_t - 1) * VAE_TEMPORAL_COMPRESSION + 1

        logger.info(
            "LTX Stage 2: latent %s → pixel %dx%d, %d frames (requested %dx%d, %d)",
            latent.shape, actual_width, actual_height, actual_frames,
            width, height, num_frames,
        )

        output_shape = VideoPixelShape(
            batch=1, frames=actual_frames, width=actual_width, height=actual_height, fps=frame_rate,
        )

        from ltx_pipelines.utils.samplers import euler_denoising_loop

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,
                ),
            )

        distilled_sigma_0 = STAGE_2_DISTILLED_SIGMA_VALUES[0]

        video_state, video_tools = noise_video_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=[],
            components=components,
            dtype=self.dtype,
            device=self.device,
            noise_scale=distilled_sigma_0,
            initial_latent=latent,
        )

        audio_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
        audio_latent = torch.zeros(
            audio_shape.to_torch_shape(), dtype=self.dtype, device=self.device,
        )
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=[],
            components=components,
            dtype=self.dtype,
            device=self.device,
            noise_scale=distilled_sigma_0,
            initial_latent=audio_latent,
        )

        video_state, audio_state = denoising_loop(
            stage_2_sigmas, video_state, audio_state, stepper,
        )
        video_state = video_tools.clear_conditioning(video_state)
        video_state = video_tools.unpatchify(video_state)
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        del transformer, video_encoder
        cleanup_memory()

        # --- VAE decode ---
        logger.info("LTX Stage 2: VAE decoding ...")
        tiling_config = default_tiling_config()
        decoded_video = vae_decode_video(
            video_state.latent, model_ledger.video_decoder(), tiling_config, generator,
        )
        decoded_audio = vae_decode_audio(
            audio_state.latent, model_ledger.audio_decoder(), model_ledger.vocoder(),
        )

        return decoded_video, decoded_audio

    # ------------------------------------------------------------------
    # Stage 2 fallback: Sana-only VAE decode (no refiner)
    # ------------------------------------------------------------------

    def _decode_sana_only(
        self,
        video_latent: torch.Tensor,
        seed: int,
    ) -> tuple[Iterator[torch.Tensor], None]:
        """Decode Sana latents directly via ltx_core VAE (no refiner)."""
        from ltx_core.model.video_vae import decode_video as vae_decode_video

        model_ledger = self._build_model_ledger()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        tiling_config = default_tiling_config()

        latent = video_latent.to(device=self.device, dtype=self.dtype)

        logger.info("Sana-only: VAE decoding (no refiner) ...")
        decoded_video = vae_decode_video(
            latent, model_ledger.video_decoder(), tiling_config, generator,
        )
        return decoded_video, None

    # ------------------------------------------------------------------
    # Public interface (FastVideoPipeline Protocol)
    # ------------------------------------------------------------------

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
        if images:
            logger.warning(
                "Sana pipeline received %d image(s) for I2V — "
                "image conditioning is not yet supported, falling back to T2V",
                len(images),
            )

        if num_frames > SANA_MAX_FRAMES:
            logger.info("Clamping num_frames from %d to %d (Sana limit)", num_frames, SANA_MAX_FRAMES)
            num_frames = SANA_MAX_FRAMES

        # Stage 1
        video_latent = self._run_sana_stage(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
        )

        # Derive actual frame count from Sana's latent temporal dimension.
        # Sana may produce a different number of latent frames than requested;
        # the chunk count for encode_video must match the real data.
        _, _, lat_t, _, _ = video_latent.shape
        VAE_TEMPORAL_COMPRESSION = 8
        actual_frames = (lat_t - 1) * VAE_TEMPORAL_COMPRESSION + 1
        if actual_frames != num_frames:
            logger.info(
                "Sana produced %d actual frames (latent T=%d) vs %d requested",
                actual_frames, lat_t, num_frames,
            )

        # Stage 2 + decode
        tiling_config = default_tiling_config()
        if self.enable_refine:
            video, audio = self._run_ltx_refine_stage(
                video_latent=video_latent,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
            )
        else:
            video, audio = self._decode_sana_only(video_latent, seed)

        chunks = video_chunks_number(actual_frames, tiling_config)
        encode_video_output(
            video=video,
            audio=audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
        )

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        try:
            self.generate(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                output_path=output_path,
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        pass
