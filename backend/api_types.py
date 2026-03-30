"""Pydantic request/response models and TypedDicts for ltx2_server."""

from __future__ import annotations

from typing import Literal, NamedTuple, TypeAlias, TypedDict
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ModelFileType = Literal[
    "checkpoint",
    "upsampler",
    "distilled_lora",
    "ic_lora",
    "depth_processor",
    "person_detector",
    "pose_processor",
    "text_encoder",
    "zit",
]


class ImageConditioningInput(NamedTuple):
    """Image conditioning triplet used by all video pipelines."""

    path: str
    frame_idx: int
    strength: float


# ============================================================
# TypedDicts for module-level state globals
# ============================================================


class GenerationState(TypedDict):
    id: str | None
    cancelled: bool
    result: str | list[str] | None
    error: str | None
    status: str  # "idle" | "running" | "complete" | "cancelled" | "error"
    phase: str
    progress: int
    current_step: int
    total_steps: int


JsonObject: TypeAlias = dict[str, object]
VideoCameraMotion = Literal[
    "none",
    "dolly_in",
    "dolly_out",
    "dolly_left",
    "dolly_right",
    "jib_up",
    "jib_down",
    "static",
    "focus_shift",
]


# ============================================================
# Response Models
# ============================================================


class ModelStatusItem(BaseModel):
    id: str
    name: str
    loaded: bool
    downloaded: bool


class GpuTelemetry(BaseModel):
    name: str
    vram: int
    vramUsed: int


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    active_model: str | None
    gpu_info: GpuTelemetry
    sage_attention: bool
    models_status: list[ModelStatusItem]


class GpuInfoResponse(BaseModel):
    cuda_available: bool
    mps_available: bool = False
    gpu_available: bool = False
    gpu_name: str | None
    vram_gb: int | None
    gpu_info: GpuTelemetry


class RuntimePolicyResponse(BaseModel):
    force_api_generations: bool
    pipeline_backend: str = "ltx"


class GenerationProgressResponse(BaseModel):
    status: str
    phase: str
    progress: int
    currentStep: int | None
    totalSteps: int | None


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str


class ModelFileStatus(BaseModel):
    id: ModelFileType
    name: str
    description: str
    downloaded: bool
    size: int
    expected_size: int
    required: bool = True
    is_folder: bool = False
    optional_reason: str | None = None


class TextEncoderStatus(BaseModel):
    downloaded: bool
    size_bytes: int
    size_gb: float
    expected_size_gb: float


class ModelsStatusResponse(BaseModel):
    models: list[ModelFileStatus]
    all_downloaded: bool
    total_size: int
    downloaded_size: int
    total_size_gb: float
    downloaded_size_gb: float
    models_path: str
    has_api_key: bool
    text_encoder_status: TextEncoderStatus
    use_local_text_encoder: bool


class DownloadProgressResponse(BaseModel):
    status: str
    current_downloading_file: ModelFileType | None
    current_file_progress: int
    total_progress: int
    total_downloaded_bytes: int
    expected_total_bytes: int
    completed_files: set[ModelFileType]
    all_files: set[ModelFileType]
    error: str | None
    speed_mbps: int


class SuggestGapPromptResponse(BaseModel):
    status: str = "success"
    suggested_prompt: str


class GenerateVideoResponse(BaseModel):
    status: str
    video_path: str | None = None


class GenerateImageResponse(BaseModel):
    status: str
    image_paths: list[str] | None = None


class CancelResponse(BaseModel):
    status: str
    id: str | None = None


class RetakeResponse(BaseModel):
    status: str
    video_path: str | None = None
    result: JsonObject | None = None


class IcLoraExtractResponse(BaseModel):
    conditioning: str
    original: str
    conditioning_type: Literal["canny", "depth"]
    frame_time: float


class IcLoraGenerateResponse(BaseModel):
    status: str
    video_path: str | None = None


class ModelDownloadStartResponse(BaseModel):
    status: str
    message: str | None = None
    sessionId: str | None = None


class TextEncoderDownloadResponse(BaseModel):
    status: str
    message: str | None = None
    sessionId: str | None = None


class StatusResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None


# ============================================================
# Request Models
# ============================================================


class GenerateVideoRequest(BaseModel):
    prompt: NonEmptyPrompt
    resolution: str = "512p"
    model: str = "fast"
    cameraMotion: VideoCameraMotion = "none"
    negativePrompt: str = ""
    duration: str = "2"
    fps: str = "24"
    audio: str = "false"
    imagePath: str | None = None
    audioPath: str | None = None
    aspectRatio: Literal["16:9", "9:16"] = "16:9"


class GenerateImageRequest(BaseModel):
    prompt: NonEmptyPrompt
    width: int = 1024
    height: int = 1024
    numSteps: int = 4
    numImages: int = 1


def _default_model_types() -> set[ModelFileType]:
    return set()


class ModelDownloadRequest(BaseModel):
    modelTypes: set[ModelFileType] = Field(default_factory=_default_model_types)


class RequiredModelsResponse(BaseModel):
    modelTypes: list[ModelFileType]


class SuggestGapPromptRequest(BaseModel):
    beforePrompt: str = ""
    afterPrompt: str = ""
    beforeFrame: str | None = None
    afterFrame: str | None = None
    gapDuration: float = 5
    mode: str = "t2v"
    inputImage: str | None = None


class RetakeRequest(BaseModel):
    video_path: str
    start_time: float
    duration: float
    prompt: str = ""
    mode: str = "replace_audio_and_video"


class IcLoraExtractRequest(BaseModel):
    video_path: str
    conditioning_type: Literal["canny", "depth"] = "canny"
    frame_time: float = 0


class IcLoraImageInput(BaseModel):
    path: str
    frame: int = 0
    strength: float = 1.0


def _default_ic_lora_images() -> list[IcLoraImageInput]:
    return []


class IcLoraGenerateRequest(BaseModel):
    video_path: str
    conditioning_type: Literal["canny", "depth"]
    prompt: NonEmptyPrompt
    conditioning_strength: float = 1.0
    num_inference_steps: int = 30
    cfg_guidance_scale: float = 1.0
    negative_prompt: str = ""
    images: list[IcLoraImageInput] = Field(default_factory=_default_ic_lora_images)


# ============================================================
# Video Agent Types
# ============================================================

VideoStyle = Literal[
    "cinematic",
    "anime",
    "realistic",
    "fantasy",
    "noir",
    "documentary",
]

VIDEO_STYLE_PROMPTS: dict[VideoStyle, str] = {
    "cinematic": "cinematic lighting, film grain, shallow depth of field, 35mm film, professional color grading, anamorphic lens flare, dramatic composition, movie quality",
    "anime": "anime style, vibrant saturated colors, cel shading, Japanese animation aesthetic, detailed line art, expressive character animation, Studio Ghibli quality",
    "realistic": "photorealistic, natural lighting, high detail, 8K UHD quality, lifelike textures, real-world physics, DSLR camera look, sharp focus",
    "fantasy": "fantasy art style, magical atmosphere, ethereal lighting, dreamlike quality, volumetric god rays, mystical particles, rich saturated palette",
    "noir": "film noir style, high contrast black and white, dramatic chiaroscuro shadows, moody atmosphere, venetian blinds lighting, 1940s aesthetic",
    "documentary": "documentary style, handheld camera, natural color grading, raw footage feel, available light, authentic atmosphere, observational",
}

VIDEO_STYLE_NEGATIVE_PROMPTS: dict[VideoStyle, str] = {
    "cinematic": "amateur, low quality, blurry, overexposed, shaky camera, bad composition",
    "anime": "3D render, photorealistic, uncanny valley, blurry, low resolution",
    "realistic": "cartoon, anime, painting, illustration, CGI look, artificial, fake",
    "fantasy": "mundane, boring, gray, desaturated, photorealistic, modern urban",
    "noir": "colorful, bright, cheerful, modern, low contrast, flat lighting",
    "documentary": "cinematic, staged, artificial, CGI, fantasy, unrealistic",
}

DEFAULT_AGENT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, pixelated, distorted, deformed, "
    "watermark, text overlay, logo, bad anatomy, extra limbs, flickering, "
    "inconsistent lighting, camera shake, compression artifacts"
)


class AgentSceneInput(BaseModel):
    """User-provided scene override (optional)."""

    description: str
    duration: int | None = None
    camera_motion: VideoCameraMotion | None = None
    image_path: str | None = None


def _default_agent_scenes() -> list[AgentSceneInput]:
    return []


class AgentGenerateRequest(BaseModel):
    script: NonEmptyPrompt
    style: VideoStyle = "cinematic"
    resolution: str = "720p"
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    duration_per_scene: int = 5
    total_duration: int = 60
    fps: str = "24"
    model: str = "fast"
    negative_prompt: str = ""
    generate_audio: bool = False
    parallel: bool = False  # If True, generate all scenes in parallel (no I2V chaining)
    scenes: list[AgentSceneInput] = Field(default_factory=_default_agent_scenes)


class AgentScenePlan(BaseModel):
    """A single planned scene from the Director Agent."""

    scene_index: int
    description: str
    prompt: str
    duration: int
    camera_motion: VideoCameraMotion
    image_path: str | None = None


class AgentGenerateResponse(BaseModel):
    status: str  # "complete", "error", "cancelled"
    video_paths: list[str] = Field(default_factory=list)
    scene_plans: list[AgentScenePlan] = Field(default_factory=list)
    final_video_path: str | None = None
    error: str | None = None


class AgentProgressResponse(BaseModel):
    status: str  # "idle", "planning", "generating", "assembling", "complete", "error", "cancelled"
    current_scene: int = 0
    total_scenes: int = 0
    scene_status: str = ""  # description of current scene
    overall_progress: int = 0  # 0-100
    completed_scenes: list[int] = Field(default_factory=list)
    failed_scenes: list[int] = Field(default_factory=list)
    scene_plans: list[AgentScenePlan] = Field(default_factory=list)
    video_paths: list[str] = Field(default_factory=list)
    final_video_path: str | None = None
    error: str | None = None
