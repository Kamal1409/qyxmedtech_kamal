"""
Health-check endpoints.
GET /api/v1/health        → simple liveness probe
GET /api/v1/health/ready  → readiness: pipeline code, config and detector weights present, GPU available
"""

from pathlib import Path

import yaml
from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/", summary="Liveness probe")
async def liveness():
    return {"status": "ok", "service": settings.APP_NAME}


@router.get("/ready", summary="Readiness probe")
async def readiness():
    checks = {"pipeline_code": (settings.TRAINING_DIR / "src" / "seg_pipeline.py").exists(),
              "config": settings.PIPELINE_CONFIG.exists(), "detector_weights": False, "gpu": False}
    if checks["config"]:
        w = yaml.safe_load(settings.PIPELINE_CONFIG.read_text(encoding="utf-8"))["segmentation"]["head"]["weights"]
        wp = Path(w) if Path(w).is_absolute() else settings.TRAINING_DIR / w
        checks["detector_weights"] = wp.exists()
    try:
        import torch
        checks["gpu"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    ready = all(v for k, v in checks.items() if k != "gpu")
    return {"status": "ready" if ready else "not_ready", "checks": checks}
