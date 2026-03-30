"""Director Agent: decomposes a script into a scene plan for video generation.

Currently uses rule-based decomposition. Can be swapped to LLM-based planning
by replacing `plan_scenes()` with an API call to Gemini/Claude/etc.
"""

from __future__ import annotations

import re

from api_types import (
    AgentGenerateRequest,
    AgentSceneInput,
    AgentScenePlan,
    VIDEO_STYLE_PROMPTS,
    VideoCameraMotion,
)

# ============================================================
# Camera motion heuristics
# ============================================================

# Keywords in scene descriptions that suggest a specific camera motion
_CAMERA_KEYWORDS: dict[str, list[str]] = {
    "dolly_in": ["approach", "close up", "zoom in", "examine", "discover", "notice", "look at", "enter"],
    "dolly_out": ["establishing", "wide shot", "overview", "reveal", "landscape", "city", "horizon", "panorama"],
    "dolly_left": ["follow", "walk", "run", "chase", "move along", "travel"],
    "dolly_right": ["pass by", "parade", "journey", "drift"],
    "jib_up": ["rise", "ascend", "fly up", "tower", "sky", "look up", "tall", "above", "sunrise"],
    "jib_down": ["descend", "land", "fall", "look down", "underwater", "ground", "below"],
    "static": ["observe", "watch", "sit", "stand", "pause", "contemplate", "wait", "still", "quiet"],
    "focus_shift": ["dramatic", "emotion", "intense", "face", "eyes", "react", "surprise", "shock", "sunset"],
}

_ORDERED_CAMERAS: list[VideoCameraMotion] = [
    "dolly_in",
    "static",
    "dolly_right",
    "jib_up",
    "dolly_out",
    "focus_shift",
    "jib_down",
    "dolly_left",
]


def _split_script_into_segments(script: str) -> list[str]:
    """Split script text into logical segments by sentences/punctuation."""
    # Split on Chinese and English sentence boundaries
    segments = re.split(r'[。！？.!?\n]+', script)
    segments = [s.strip() for s in segments if s.strip()]

    if not segments:
        return [script.strip()]

    return segments


def _pick_camera_motion(index: int, total: int, description: str) -> VideoCameraMotion:
    """Pick a camera motion that creates visual variety across scenes."""
    desc_lower = description.lower()

    # Check for keyword hints in description
    for motion, keywords in _CAMERA_KEYWORDS.items():
        for keyword in keywords:
            if keyword in desc_lower:
                return motion  # type: ignore[return-value]

    # First scene: establishing shot
    if index == 0:
        return "dolly_out"
    # Last scene: contemplative
    if index == total - 1:
        return "focus_shift"
    # Alternate for variety
    return _ORDERED_CAMERAS[index % len(_ORDERED_CAMERAS)]


def _build_prompt(
    description: str,
    style: str,
    negative_prompt: str,
    scene_index: int,
    total_scenes: int,
    prev_description: str | None,
    next_description: str | None,
) -> str:
    """Build a complete generation prompt with style and scene context."""
    style_suffix = VIDEO_STYLE_PROMPTS.get(style, VIDEO_STYLE_PROMPTS["cinematic"])  # type: ignore[arg-type]

    parts: list[str] = []

    # Main description — expand with visual detail
    parts.append(description)

    # Add continuity hints from neighboring scenes
    if prev_description and scene_index > 0:
        parts.append(f"Continuing from: {prev_description[:50]}")

    # Add scene position context
    if total_scenes > 1:
        if scene_index == 0:
            parts.append("Opening establishing shot")
        elif scene_index == total_scenes - 1:
            parts.append("Final concluding shot")

    # Add style
    parts.append(style_suffix)

    # Quality boosters
    parts.append("high quality, detailed, smooth fluid motion, consistent lighting")

    return ". ".join(parts)


def plan_scenes(req: AgentGenerateRequest) -> list[AgentScenePlan]:
    """Decompose a script into a list of scene plans.

    If the user provided explicit scenes, those are used directly.
    Otherwise, the script is auto-segmented by sentence boundaries.
    """
    style = req.style

    # If user provided explicit scenes, use them
    if req.scenes:
        return _plan_from_user_scenes(req.scenes, style, req.duration_per_scene, req.negative_prompt)

    # Auto-segment the script
    segments = _split_script_into_segments(req.script)

    # Calculate how many scenes we need
    max_scenes = req.total_duration // req.duration_per_scene

    # If we have more segments than needed, merge some
    if len(segments) > max_scenes:
        merged: list[str] = []
        chunk_size = len(segments) / max_scenes
        i = 0.0
        while i < len(segments):
            end = min(int(i + chunk_size), len(segments))
            start = int(i)
            merged.append(", ".join(segments[start:end]))
            i += chunk_size
        segments = merged[:max_scenes]

    total = len(segments)
    # Distribute duration evenly
    base_duration = min(req.duration_per_scene, req.total_duration // max(total, 1))
    base_duration = max(base_duration, 2)  # minimum 2 seconds

    plans: list[AgentScenePlan] = []
    for i, desc in enumerate(segments):
        camera = _pick_camera_motion(i, total, desc)
        prev_desc = segments[i - 1] if i > 0 else None
        next_desc = segments[i + 1] if i < total - 1 else None
        prompt = _build_prompt(desc, style, req.negative_prompt, i, total, prev_desc, next_desc)
        plans.append(
            AgentScenePlan(
                scene_index=i,
                description=desc,
                prompt=prompt,
                duration=base_duration,
                camera_motion=camera,
                image_path=None,
            )
        )

    return plans


def _plan_from_user_scenes(
    scenes: list[AgentSceneInput],
    style: str,
    default_duration: int,
    negative_prompt: str,
) -> list[AgentScenePlan]:
    """Build plans from user-provided scene overrides."""
    total = len(scenes)
    plans: list[AgentScenePlan] = []
    for i, scene in enumerate(scenes):
        duration = scene.duration if scene.duration is not None else default_duration
        duration = max(duration, 2)
        camera = scene.camera_motion if scene.camera_motion is not None else _pick_camera_motion(i, total, scene.description)
        prev_desc = scenes[i - 1].description if i > 0 else None
        next_desc = scenes[i + 1].description if i < total - 1 else None
        prompt = _build_prompt(scene.description, style, negative_prompt, i, total, prev_desc, next_desc)
        plans.append(
            AgentScenePlan(
                scene_index=i,
                description=scene.description,
                prompt=prompt,
                duration=duration,
                camera_motion=camera,
                image_path=scene.image_path,
            )
        )
    return plans
