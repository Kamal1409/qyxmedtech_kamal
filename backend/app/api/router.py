"""
Root API router — aggregates all versioned sub-routers.
"""

from fastapi import APIRouter
from app.api.v1 import analyses, health, model_info

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(analyses.router, prefix="/analyses", tags=["Analyses"])
api_router.include_router(model_info.router, prefix="/model", tags=["Model"])
