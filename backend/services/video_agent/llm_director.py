"""LLM-powered Director Agent using Gemini API for intelligent scene planning.

Falls back to rule-based director if no API key or on error.
"""

from __future__ import annotations

import json
import logging
import re

from api_types import (
    AgentGenerateRequest,
    AgentScenePlan,
    VIDEO_STYLE_PROMPTS,
)
from services.interfaces import HTTPClient

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

IMPORTANT: Return ONLY a JSON array of scene objects. No markdown, no explanation.

Example:
[
  {
    "description": "A lone astronaut walks across a barren Mars landscape",
    "prompt": "An astronaut in a white spacesuit walks across a vast red Martian desert, footprints in fine dust, two pale moons visible in the pink sky, cinematic lighting with long shadows",
    "duration": 5,
    "camera_motion": "dolly_out"
  }
]"""


def plan_scenes_with_llm(
    req: AgentGenerateRequest,
    gemini_api_key: str,
    http: HTTPClient,
) -> list[AgentScenePlan] | None:
    """Use Gemini to intelligently plan scenes from a script.

    Returns None if LLM planning fails (caller should fall back to rule-based).
    """
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

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }

    try:
        response = http.post(
            gemini_url,
            headers={"Content-Type": "application/json", "x-goog-api-key": gemini_api_key},
            json_payload=payload,
            timeout=30,
        )
    except Exception as e:
        logger.warning("Gemini API call failed: %s", e)
        return None

    if response.status_code != 200:
        logger.warning("Gemini API returned %d: %s", response.status_code, response.text[:200])
        return None

    try:
        resp_data = response.json()
        text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
        scenes_raw = json.loads(text)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("Failed to parse Gemini response: %s", e)
        try:
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                scenes_raw = json.loads(json_match.group())
            else:
                return None
        except Exception:
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
