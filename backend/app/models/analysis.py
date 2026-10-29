"""Schemas for the semen analysis API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class Headline(BaseModel):
    concentration: float | None = None
    total_motility: float | None = None
    progressive: float | None = None
    n_sperm: int | None = None
    flags: list[str] = []


class AnalysisMeta(BaseModel):
    id: str
    sample_id: str
    patient_ref: str | None = None
    volume_ml: float | None = None
    notes: str | None = None
    filename: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    status: Literal["queued", "running", "done", "failed"]
    stage: str | None = None
    stage_label: str | None = None
    progress: float = 0.0
    error: str | None = None
    queue_position: int | None = None
    headline: Headline | None = None


class AnalysisDetail(AnalysisMeta):
    result: dict[str, Any] | None = None


class ModelInfo(BaseModel):
    settings: dict[str, Any]
    validation: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    n_videos: int | None = None
    test_videos: list[str] = []
    excluded_videos: list[str] = []
    changes: list[str] = []
