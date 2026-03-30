"""Video Agent handler: orchestrates multi-scene video generation from a script."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from api_types import (
    AgentGenerateRequest,
    AgentGenerateResponse,
    AgentProgressResponse,
    AgentScenePlan,
    GenerateVideoRequest,
)
from handlers.base import StateHandlerBase
from handlers.video_generation_handler import VideoGenerationHandler
from services.video_agent.director import plan_scenes
from services.video_agent.llm_director import plan_scenes_with_llm

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState
    from handlers.generation_handler import GenerationHandler

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Tracks the state of an agent generation session."""

    status: str = "idle"  # idle, planning, generating, assembling, complete, error, cancelled
    current_scene: int = 0
    total_scenes: int = 0
    scene_status: str = ""
    overall_progress: int = 0
    completed_scenes: list[int] = field(default_factory=list)
    failed_scenes: list[int] = field(default_factory=list)
    scene_plans: list[AgentScenePlan] = field(default_factory=list)
    video_paths: list[str] = field(default_factory=list)
    final_video_path: str | None = None
    error: str | None = None
    cancelled: bool = False


class AgentHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        video_generation_handler: VideoGenerationHandler,
        generation_handler: GenerationHandler,
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._video_gen = video_generation_handler
        self._generation = generation_handler
        self._agent_state = AgentState()
        self._agent_lock = threading.Lock()

    def generate(self, req: AgentGenerateRequest) -> AgentGenerateResponse:
        """Run the full agent pipeline: plan → generate scenes → assemble."""
        with self._agent_lock:
            if self._agent_state.status in ("planning", "generating", "assembling"):
                return AgentGenerateResponse(
                    status="error",
                    error="Agent generation already in progress",
                )
            self._agent_state = AgentState(status="planning")

        try:
            # Phase 1: Plan scenes (try LLM first, fall back to rule-based)
            self._update_agent("planning", 0, 0, "Decomposing script into scenes...")
            scene_plans: list[AgentScenePlan] | None = None

            # Try LLM-based planning via NVIDIA API (uses LTX_DESKTOP_GEMINI_API env)
            self._update_agent("planning", 0, 0, "Using AI to plan scenes...")
            scene_plans = plan_scenes_with_llm(req)
            if scene_plans:
                logger.info("LLM director produced %d scenes", len(scene_plans))
            else:
                logger.info("LLM planning unavailable, using rule-based director")

            # Fall back to rule-based planning
            if scene_plans is None:
                scene_plans = plan_scenes(req)

            total = len(scene_plans)

            with self._agent_lock:
                self._agent_state.scene_plans = scene_plans
                self._agent_state.total_scenes = total

            logger.info("Agent planned %d scenes from script", total)

            # Phase 1.5: Warm up the model pipeline (so first scene doesn't pay loading cost)
            self._update_agent("generating", 0, total, "Loading video generation model...")
            try:
                # Trigger model load by accessing the pipeline handler
                # This pre-loads checkpoint to GPU before scene loop
                from handlers.pipelines_handler import PipelinesHandler
                pipelines: PipelinesHandler = self._video_gen._pipelines  # type: ignore[attr-defined]
                pipelines.load_gpu_pipeline(req.model)
                logger.info("Model pipeline warmed up")
            except Exception as e:
                logger.warning("Model warmup failed (will load on first generation): %s", e)

            # Phase 2: Generate scenes
            if req.parallel and total > 1:
                video_paths = self._generate_scenes_parallel(scene_plans, req, total)
            else:
                video_paths = self._generate_scenes_sequential(scene_plans, req, total)

            # Phase 3: Assemble final video
            if not video_paths:
                raise RuntimeError("All scenes failed to generate")

            self._update_agent("assembling", total, total, "Assembling final video...")
            final_path = self._assemble_videos(video_paths)

            with self._agent_lock:
                self._agent_state.status = "complete"
                self._agent_state.final_video_path = final_path
                self._agent_state.overall_progress = 100

            failed_count = total - len(video_paths)
            if failed_count > 0:
                logger.warning("Agent complete with %d/%d scenes failed", failed_count, total)
            else:
                logger.info("Agent complete. All %d scenes generated.", total)
            logger.info("Final video: %s", final_path)

            return AgentGenerateResponse(
                status="complete",
                video_paths=video_paths,
                scene_plans=scene_plans,
                final_video_path=final_path,
            )

        except Exception as e:
            error_msg = str(e)
            with self._agent_lock:
                self._agent_state.status = "error"
                self._agent_state.error = error_msg
            logger.exception("Agent generation failed: %s", error_msg)
            return AgentGenerateResponse(status="error", error=error_msg)

    def get_progress(self) -> AgentProgressResponse:
        with self._agent_lock:
            s = self._agent_state
            return AgentProgressResponse(
                status=s.status,
                current_scene=s.current_scene,
                total_scenes=s.total_scenes,
                scene_status=s.scene_status,
                overall_progress=s.overall_progress,
                completed_scenes=list(s.completed_scenes),
                failed_scenes=list(s.failed_scenes),
                scene_plans=list(s.scene_plans),
                video_paths=list(s.video_paths),
                final_video_path=s.final_video_path,
                error=s.error,
            )

    def plan_only(self, req: AgentGenerateRequest) -> list[AgentScenePlan]:
        """Return scene plans without generating any videos."""
        scene_plans = plan_scenes_with_llm(req)
        if scene_plans is None:
            scene_plans = plan_scenes(req)
        return scene_plans

    def cancel(self) -> None:
        with self._agent_lock:
            self._agent_state.cancelled = True
            self._agent_state.status = "cancelled"
        # Also cancel any in-progress generation
        self._generation.cancel_generation()

    def _generate_scenes_sequential(
        self,
        scene_plans: list[AgentScenePlan],
        req: AgentGenerateRequest,
        total: int,
    ) -> list[str]:
        """Generate scenes sequentially with I2V chaining for visual continuity."""
        self._update_agent("generating", 0, total, f"Generating {total} scenes...")
        video_paths: list[str] = []
        last_frame_path: str | None = None

        for i, scene in enumerate(scene_plans):
            if self._is_cancelled():
                break

            self._update_agent(
                "generating", i, total,
                f"Generating scene {i + 1}/{total}: {scene.description[:60]}...",
            )

            # Use last frame of previous scene as I2V conditioning
            if last_frame_path and scene.image_path is None:
                scene = AgentScenePlan(
                    scene_index=scene.scene_index,
                    description=scene.description,
                    prompt=scene.prompt,
                    duration=scene.duration,
                    camera_motion=scene.camera_motion,
                    image_path=last_frame_path,
                )

            try:
                video_path = self._generate_single_scene(scene, req)
            except Exception as scene_err:
                logger.error("Scene %d/%d failed: %s", i + 1, total, scene_err)
                with self._agent_lock:
                    self._agent_state.failed_scenes.append(i)
                    self._agent_state.overall_progress = int(((i + 1) / total) * 90)
                last_frame_path = None
                continue

            video_paths.append(video_path)

            try:
                last_frame_path = self._extract_last_frame(video_path, i)
            except Exception as e:
                logger.warning("Failed to extract last frame from scene %d: %s", i, e)
                last_frame_path = None

            with self._agent_lock:
                self._agent_state.video_paths.append(video_path)
                self._agent_state.completed_scenes.append(i)
                self._agent_state.overall_progress = int(((i + 1) / total) * 90)

            logger.info("Scene %d/%d generated: %s", i + 1, total, video_path)

        return video_paths

    def _generate_scenes_parallel(
        self,
        scene_plans: list[AgentScenePlan],
        req: AgentGenerateRequest,
        total: int,
    ) -> list[str]:
        """Generate all scenes in parallel (no I2V chaining)."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._update_agent("generating", 0, total, f"Generating {total} scenes in parallel...")
        # Map scene_index -> video_path
        results: dict[int, str] = {}

        def gen_one(idx: int, scene: AgentScenePlan) -> tuple[int, str]:
            return idx, self._generate_single_scene(scene, req)

        # Use up to 4 workers (limited by GPU memory)
        max_workers = min(total, 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(gen_one, i, scene): i
                for i, scene in enumerate(scene_plans)
            }
            for future in as_completed(futures):
                scene_idx = futures[future]
                try:
                    idx, video_path = future.result()
                    results[idx] = video_path
                    with self._agent_lock:
                        self._agent_state.video_paths.append(video_path)
                        self._agent_state.completed_scenes.append(idx)
                        done = len(self._agent_state.completed_scenes)
                        self._agent_state.overall_progress = int((done / total) * 90)
                        self._agent_state.scene_status = f"Completed scene {done}/{total}"
                    logger.info("Parallel scene %d completed: %s", idx + 1, video_path)
                except Exception as e:
                    logger.error("Parallel scene %d failed: %s", scene_idx + 1, e)
                    with self._agent_lock:
                        self._agent_state.failed_scenes.append(scene_idx)

        # Return in scene order
        video_paths = [results[i] for i in sorted(results.keys())]
        return video_paths

    def _generate_single_scene(
        self,
        scene: AgentScenePlan,
        req: AgentGenerateRequest,
    ) -> str:
        """Generate a single scene video using the existing video generation handler."""
        from api_types import VIDEO_STYLE_NEGATIVE_PROMPTS, DEFAULT_AGENT_NEGATIVE_PROMPT

        # Combine user negative prompt with style-specific one
        neg_parts: list[str] = []
        if req.negative_prompt:
            neg_parts.append(req.negative_prompt)
        style_neg = VIDEO_STYLE_NEGATIVE_PROMPTS.get(req.style, "")
        if style_neg:
            neg_parts.append(style_neg)
        neg_parts.append(DEFAULT_AGENT_NEGATIVE_PROMPT)
        combined_negative = ", ".join(neg_parts)

        video_req = GenerateVideoRequest(
            prompt=scene.prompt,
            resolution=req.resolution,
            model=req.model,
            cameraMotion=scene.camera_motion,
            negativePrompt=combined_negative,
            audio="true" if req.generate_audio else "false",
            duration=str(scene.duration),
            fps=req.fps,
            imagePath=scene.image_path,
            aspectRatio=req.aspect_ratio,
        )

        response = self._video_gen.generate(video_req)

        if response.status != "complete" or response.video_path is None:
            raise RuntimeError(f"Scene {scene.scene_index} generation failed: {response.status}")

        return response.video_path

    def _assemble_videos(self, video_paths: list[str], crossfade_duration: float = 0.5) -> str:
        """Concatenate scene videos with optional crossfade transitions using ffmpeg."""
        if not video_paths:
            raise RuntimeError("No videos to assemble")

        if len(video_paths) == 1:
            return video_paths[0]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.config.outputs_dir / f"agent_final_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"

        # Try crossfade assembly first, fall back to simple concat
        if crossfade_duration > 0 and len(video_paths) <= 20:
            try:
                self._assemble_with_crossfade(video_paths, str(final_path), crossfade_duration)
                return str(final_path)
            except Exception as e:
                logger.warning("Crossfade assembly failed, falling back to concat: %s", e)

        self._assemble_simple_concat(video_paths, str(final_path))
        return str(final_path)

    @staticmethod
    def _probe_duration(video_path: str) -> float:
        """Get video duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return 5.0  # fallback
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 5.0

    def _assemble_with_crossfade(
        self, video_paths: list[str], output_path: str, crossfade: float
    ) -> None:
        """Assemble videos with crossfade transitions using ffmpeg xfade filter."""
        n = len(video_paths)

        # Probe actual durations
        durations = [self._probe_duration(vp) for vp in video_paths]
        logger.info("Video durations for crossfade: %s", [f"{d:.1f}s" for d in durations])

        # Build ffmpeg command
        cmd: list[str] = ["ffmpeg", "-y"]
        for vp in video_paths:
            cmd.extend(["-i", vp])

        # Build xfade filter chain with correct offsets
        # offset[i] = sum(durations[0..i]) - i * crossfade_duration
        filter_parts: list[str] = []
        cumulative = durations[0]

        for i in range(1, n):
            offset = max(0, cumulative - crossfade)
            in_a = f"[{0 if i == 1 else ''}{'tmp' + str(i - 2) if i > 1 else '0'}:v]"
            in_b = f"[{i}:v]"
            if i == 1:
                in_a = "[0:v]"
            else:
                in_a = f"[tmp{i - 2}]"

            out_label = f"[tmp{i - 1}]" if i < n - 1 else "[outv]"
            filter_parts.append(
                f"{in_a}{in_b}xfade=transition=fade:duration={crossfade}:offset={offset:.3f}{out_label}"
            )
            cumulative = offset + durations[i]

        filter_str = ";".join(filter_parts)
        cmd.extend(["-filter_complex", filter_str, "-map", "[outv]"])
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23", output_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"xfade failed: {result.stderr[:300]}")

    def _assemble_simple_concat(self, video_paths: list[str], output_path: str) -> None:
        """Simple concatenation without transitions, preserving audio tracks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for vp in video_paths:
                f.write(f"file '{vp}'\n")
            concat_file = f.name

        try:
            # Try stream copy first (fastest, preserves audio)
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                # Re-encode fallback (handles mixed codecs/resolutions, maps audio)
                cmd_reencode = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", concat_file,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "192k",
                    "-map", "0:v?", "-map", "0:a?",  # Map video and audio if present
                    str(output_path),
                ]
                result = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg concat failed: {result.stderr[:500]}")
        finally:
            Path(concat_file).unlink(missing_ok=True)

    def _extract_last_frame(self, video_path: str, scene_index: int) -> str:
        """Extract the last frame from a video for scene-to-scene continuity."""
        frame_path = str(
            self.config.outputs_dir / f"_agent_frame_scene{scene_index}_{uuid.uuid4().hex[:6]}.png"
        )
        # Use ffmpeg to extract the last frame
        cmd = [
            "ffmpeg", "-y",
            "-sseof", "-0.1",  # seek to 0.1s before end
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            frame_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr[:200]}")
        if not Path(frame_path).exists():
            raise RuntimeError(f"Frame not written to {frame_path}")
        logger.info("Extracted last frame from scene %d -> %s", scene_index, frame_path)
        return frame_path

    def _update_agent(self, status: str, current: int, total: int, scene_status: str) -> None:
        with self._agent_lock:
            self._agent_state.status = status
            self._agent_state.current_scene = current
            self._agent_state.total_scenes = total
            self._agent_state.scene_status = scene_status
            if status == "planning":
                self._agent_state.overall_progress = 5
            elif status == "assembling":
                self._agent_state.overall_progress = 95

    def _is_cancelled(self) -> bool:
        with self._agent_lock:
            return self._agent_state.cancelled
