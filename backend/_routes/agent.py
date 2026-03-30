"""Route handlers for /api/agent/* — video agent endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from api_types import (
    AgentGenerateRequest,
    AgentGenerateResponse,
    AgentProgressResponse,
    AgentScenePlan,
    StatusResponse,
)
from pydantic import BaseModel, Field
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/generate", response_model=AgentGenerateResponse)
def route_agent_generate(
    req: AgentGenerateRequest,
    background_tasks: BackgroundTasks,
    handler: AppHandler = Depends(get_state_service),
) -> AgentGenerateResponse:
    """POST /api/agent/generate — start agent video generation.

    Runs synchronously for now. For async operation, the frontend polls progress.
    """
    return handler.agent.generate(req)


@router.post("/generate/async")
def route_agent_generate_async(
    req: AgentGenerateRequest,
    background_tasks: BackgroundTasks,
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    """POST /api/agent/generate/async — start agent generation in background."""
    background_tasks.add_task(handler.agent.generate, req)
    return StatusResponse(status="started")


@router.get("/progress", response_model=AgentProgressResponse)
def route_agent_progress(
    handler: AppHandler = Depends(get_state_service),
) -> AgentProgressResponse:
    """GET /api/agent/progress — poll agent generation progress."""
    return handler.agent.get_progress()


@router.post("/cancel", response_model=StatusResponse)
def route_agent_cancel(
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    """POST /api/agent/cancel — cancel agent generation."""
    handler.agent.cancel()
    return StatusResponse(status="cancelling")


class PlanResponse(BaseModel):
    scene_plans: list[AgentScenePlan] = Field(default_factory=list)


@router.post("/plan", response_model=PlanResponse)
def route_agent_plan(
    req: AgentGenerateRequest,
    handler: AppHandler = Depends(get_state_service),
) -> PlanResponse:
    """POST /api/agent/plan — preview scene plan without generating videos."""
    plans = handler.agent.plan_only(req)
    return PlanResponse(scene_plans=plans)
