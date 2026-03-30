"""LLM-powered Director Agent using NVIDIA OpenAI-compatible API for scene planning.

Uses gcp/google/gemini-3-pro via NVIDIA inference API.
Falls back to rule-based director if no API key or on error.
"""

from __future__ import annotations

import json
import logging
import os
import re

from api_types import (
    AgentGenerateRequest,
    AgentScenePlan,
    VIDEO_STYLE_PROMPTS,
)

logger = logging.getLogger(__name__)

_VALID_CAMERA_MOTIONS: set[str] = {
    "none", "dolly_in", "dolly_out", "dolly_left", "dolly_right",
    "jib_up", "jib_down", "static", "focus_shift",
}

_SYSTEM_PROMPT = """You are a professional film director AI. Given a script/story, decompose it into a sequence of video scenes for AI video generation.

For each scene, provide:
1. "description": A brief description of what happens (1-2 sentences)
2. "prompt": A detailed, vivid video generation prompt describing visuals, lighting, mood, composition (2-4 sentences)
3. "duration": Duration in seconds (integer, min 2, max 10)
4. "camera_motion": One of: dolly_in, dolly_out, dolly_left, dolly_right, jib_up, jib_down, static, focus_shift

Guidelines:
- Create visually compelling scenes with variety in camera angles
- Maintain narrative coherence across scenes
- Use descriptive language that helps AI video generation: colors, lighting, textures, atmosphere
- Start with an establishing shot and end with a conclusion
- Each scene's prompt should be self-contained but fit the overall story
- Match the requested visual style throughout

IMPORTANT: Return ONLY a JSON array of scene objects. No markdown, no explanation, no code blocks.

Example:
[
  {
    "description": "A lone astronaut walks across a barren Mars landscape",
    "prompt": "An astronaut in a white spacesuit walks across a vast red Martian desert, footprints in fine dust, two pale moons visible in the pink sky, cinematic lighting with long shadows",
    "duration": 5,
    "camera_motion": "dolly_out"
  }
]"""

# NVIDIA inference API config
_NVIDIA_BASE_URL = "https://inference-api.nvidia.com"
_NVIDIA_MODEL = "gcp/google/gemini-2.5-pro"


def _get_api_key() -> str | None:
    """Get API key from env var LTX_DESKTOP_GEMINI_API or app settings."""
    return os.environ.get("LTX_DESKTOP_GEMINI_API") or None


def plan_scenes_with_llm(
    req: AgentGenerateRequest,
    api_key: str | None = None,
) -> list[AgentScenePlan] | None:
    """Use LLM to intelligently plan scenes from a script.

    Uses NVIDIA OpenAI-compatible API with Gemini 3 Pro.
    Returns None if LLM planning fails (caller should fall back to rule-based).
    """
    key = api_key or _get_api_key()
    if not key:
        logger.info("No LLM API key available, skipping LLM planning")
        return None

    style_desc = VIDEO_STYLE_PROMPTS.get(req.style, VIDEO_STYLE_PROMPTS["cinematic"])
    max_scenes = req.total_duration // req.duration_per_scene

    user_text = (
        f"Script: {req.script}\n\n"
        f"Visual style: {req.style} -- {style_desc}\n"
        f"Target: {max_scenes} scenes, {req.duration_per_scene}s each, {req.total_duration}s total\n"
        f"Resolution: {req.resolution}, Aspect ratio: {req.aspect_ratio}\n"
    )
    if req.negative_prompt:
        user_text += f"Avoid: {req.negative_prompt}\n"

    if req.scenes:
        user_text += "\nUser-specified scenes (use as constraints, fill in details):\n"
        for i, s in enumerate(req.scenes):
            user_text += f"  Scene {i + 1}: {s.description}"
            if s.duration:
                user_text += f" ({s.duration}s)"
            if s.camera_motion:
                user_text += f" [{s.camera_motion}]"
            user_text += "\n"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url=_NVIDIA_BASE_URL)
        response = client.chat.completions.create(
            model=_NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.8,
            max_tokens=4096,
            stream=False,
        )
        text = response.choices[0].message.content or ""
    except ImportError:
        logger.warning("openai package not installed, skipping LLM planning")
        return None
    except Exception as e:
        logger.warning("LLM API call failed: %s", e)
        return None

    if not text.strip():
        logger.warning("LLM returned empty response")
        return None

    # Parse JSON — handle markdown code blocks
    try:
        scenes_raw = json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            try:
                scenes_raw = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON: %s", text[:200])
                return None
        else:
            logger.warning("No JSON array found in LLM response: %s", text[:200])
            return None

    if not isinstance(scenes_raw, list) or len(scenes_raw) == 0:
        return None

    plans: list[AgentScenePlan] = []
    for i, scene_data in enumerate(scenes_raw):
        if not isinstance(scene_data, dict):
            continue

        description = str(scene_data.get("description", f"Scene {i + 1}"))
        prompt = str(scene_data.get("prompt", description))

        if req.style not in prompt.lower():
            prompt = f"{prompt}. {style_desc}"

        duration = scene_data.get("duration", req.duration_per_scene)
        if not isinstance(duration, int) or duration < 2:
            duration = req.duration_per_scene
        duration = min(duration, 10)

        camera = str(scene_data.get("camera_motion", "static"))
        if camera not in _VALID_CAMERA_MOTIONS:
            camera = "static"

        image_path: str | None = None
        if req.scenes and i < len(req.scenes) and req.scenes[i].image_path:
            image_path = req.scenes[i].image_path

        plans.append(
            AgentScenePlan(
                scene_index=i,
                description=description,
                prompt=prompt,
                duration=duration,
                camera_motion=camera,  # type: ignore[arg-type]
                image_path=image_path,
            )
        )

    if len(plans) == 0:
        return None

    logger.info("LLM Director planned %d scenes", len(plans))
    return plans
