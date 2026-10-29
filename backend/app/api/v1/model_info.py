"""
GET /api/v1/model  -> the settings the analyses run with, and how the model
                      compares with the lab analyzer (from training/outputs/seg/validation).
"""

from __future__ import annotations

import json

import yaml
from fastapi import APIRouter

from app.config import get_settings
from app.models.analysis import ModelInfo

router = APIRouter()
settings = get_settings()

_METRICS = ["concentration", "total_motility", "progressive", "rapid_progressive", "slow_progressive",
            "non_progressive", "VAP_mean", "VSL_mean", "VCL_mean", "ALH_mean", "BCF_mean"]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _slim(report: dict | None) -> dict | None:
    """Per metric: typical error, bias, agreement and the per-video pairs."""
    if not report:
        return None
    return {m: {"mae": r["mae"], "bias": r["bland_altman"]["mean_diff"], "icc": r["icc"]["icc"],
                "pearson_r": r["pearson_r"], "n": r["n_videos"], "per_video": r["per_video"]}
            for m, r in report.items() if m in _METRICS}


@router.get("", response_model=ModelInfo, summary="Model settings and accuracy against the lab")
async def model_info():
    cfg = yaml.safe_load(settings.PIPELINE_CONFIG.read_text(encoding="utf-8"))
    k, c, s, cls = cfg.get("kinematics", {}), cfg.get("concentration", {}), cfg.get("segmentation", {}), \
        cfg.get("classification", {})
    val = _slim(_load(settings.VALIDATION_DIR / "validation_report.json"))
    base = _slim(_load(settings.VALIDATION_DIR / "baseline" / "validation_report.json"))
    split = _load(settings.TRAINING_DIR / "data" / "video_split.json") or {}
    changes_p = settings.VALIDATION_DIR / "baseline" / "CHANGES_PLAIN.txt"
    changes = [ln.strip() for ln in changes_p.read_text(encoding="utf-8").splitlines() if ln.strip()] \
        if changes_p.exists() else []
    return {
        "settings": {
            "detector": s.get("head", {}).get("weights", "").replace("\\", "/").split("/")[-1],
            "detector_confidence": s.get("head", {}).get("conf_threshold"),
            "pixel_to_micron": k.get("pixel_to_micron"),
            "capture_fps": k.get("capture_fps"),
            "smoothing_window": k.get("smoothing_window"),
            "tracker_search_radius_px": cfg.get("tracker", {}).get("search_radius_px"),
            "chamber_depth_um": c.get("chamber_depth_um"),
            "concentration_calibration": c.get("calibration_factor"),
            "rapid_min_vap": cls.get("progressive", {}).get("rapid_min_vap"),
            "immotile_below_vap": cls.get("non_progressive", {}).get("min_vap"),
        },
        "validation": val,
        "baseline": base,
        "n_videos": (val or {}).get("total_motility", {}).get("n"),
        "test_videos": split.get("test", []),
        "excluded_videos": split.get("excluded", []),
        "changes": changes,
    }
