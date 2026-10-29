"""
CASA analysis service: runs the training pipeline on an uploaded video.

Each analysis lives in its own folder under DATA_DIR/analyses/<id>/:

    meta.json          status, stage, progress, sample details
    input/sample.<ext> the uploaded video (the pipeline's video id is "sample")
    frames/sample/     extracted frames + frame_qc.json + metadata.json
    out/seg/...        heads, tracks, kinematics, classifications, results
    config.yaml        the pipeline config used (default.yaml with this job's paths)
    result.json        everything the report page shows
    overlay.png        tracked sperm paths drawn on a frame
    pipeline.log       stdout of the pipeline stages

One worker thread runs analyses one at a time (they share the GPU).  Jobs left
queued when the server stopped are queued again on start; a job that was mid-run
is marked failed, because its partial outputs cannot be trusted.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

VIDEO_ID = "sample"

# (key, label shown to the user, share of the total run time used for the progress bar)
STAGES = [
    ("frames",   "Extracting frames",             0.08),
    ("heads",    "Detecting sperm heads",         0.55),
    ("tracks",   "Following each sperm",          0.12),
    ("kinematic", "Measuring speed and path",     0.08),
    ("classify", "Classifying motility (WHO)",    0.05),
    ("report",   "Counting and building report",  0.12),
]

# Reference values printed on the LensHooke / WHO 6th edition report
WHO_REFERENCE = {
    "volume_ml":            (">=", 1.4, "mL"),
    "concentration":        (">=", 16.0, "M/mL"),
    "total_sperm_number":   (">=", 39.0, "M"),
    "total_motility":       (">=", 42.0, "%"),
    "progressive":          (">=", 30.0, "%"),
    "immotile":             ("<", 58.0, "%"),
}

# Histogram bins used by the lab report
DIST_BINS = {
    "VAP": ([0, 5, 15, 25, 35, 45, float("inf")], ["0-5", "5-15", "15-25", "25-35", "35-45", "45+"], "µm/s"),
    "VSL": ([0, 5, 15, 25, 35, 45, float("inf")], ["0-5", "5-15", "15-25", "25-35", "35-45", "45+"], "µm/s"),
    "STR": ([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100.0001],
            ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"], "%"),
    "ALH": ([0, 2, 4, 6, 8, 10, 12, float("inf")], ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12+"], "µm"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_pipeline_importable() -> None:
    p = str(settings.TRAINING_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


class _ProgressWriter(io.TextIOBase):
    """Tee pipeline stdout to a log file and pick up per-frame progress lines
    such as '[ 45.0%] 50/109 frames' printed by the head detector."""

    _pct = re.compile(r"\[\s*([\d.]+)%\]")

    def __init__(self, log, on_fraction):
        self._log, self._on = log, on_fraction

    def write(self, s: str) -> int:
        self._log.write(s)
        m = self._pct.search(s)
        if m:
            self._on(min(float(m.group(1)) / 100.0, 1.0))
        return len(s)

    def flush(self) -> None:
        self._log.flush()


class AnalysisStore:
    """Analyses on disk + the single background worker."""

    def __init__(self, root: Path | None = None, runner=None):
        self.root = Path(root or settings.DATA_DIR) / "analyses"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._worker: threading.Thread | None = None
        self._runner = runner or self._run_pipeline      # injectable for tests

    # -- persistence ---------------------------------------------------------
    def _dir(self, aid: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", aid):
            raise KeyError(aid)
        return self.root / aid

    def _read_meta(self, aid: str) -> dict:
        p = self._dir(aid) / "meta.json"
        if not p.exists():
            raise KeyError(aid)
        return json.loads(p.read_text(encoding="utf-8"))

    def _write_meta(self, aid: str, meta: dict) -> None:
        p = self._dir(aid) / "meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(p)

    def _update(self, aid: str, **fields) -> dict:
        with self._lock:
            meta = self._read_meta(aid)
            meta.update(fields)
            self._write_meta(aid, meta)
            return meta

    # -- public API ------------------------------------------------------------
    def create(self, upload_path: Path, filename: str, sample_id: str, patient_ref: str | None,
               volume_ml: float | None, notes: str | None) -> dict:
        aid = uuid.uuid4().hex[:12]
        d = self._dir(aid)
        (d / "input").mkdir(parents=True)
        ext = Path(filename).suffix.lower() or ".mp4"
        shutil.move(str(upload_path), d / "input" / f"{VIDEO_ID}{ext}")
        meta = {
            "id": aid, "sample_id": sample_id, "patient_ref": patient_ref, "volume_ml": volume_ml,
            "notes": notes, "filename": filename, "created_at": _now(), "started_at": None,
            "finished_at": None, "status": "queued", "stage": None, "stage_label": "Waiting to start",
            "progress": 0.0, "error": None,
        }
        with self._lock:
            self._write_meta(aid, meta)
        self._queue.put(aid)
        logger.info(f"Analysis {aid} queued ({filename}, sample {sample_id})")
        return meta

    def list(self) -> list[dict]:
        out = []
        for d in self.root.iterdir():
            if not d.is_dir() or not (d / "meta.json").exists():
                continue
            try:
                meta = self._read_meta(d.name)
            except (KeyError, ValueError):
                continue
            res = self.result(d.name)
            meta["headline"] = _headline(res) if res else None
            out.append(meta)
        return sorted(out, key=lambda m: m["created_at"], reverse=True)

    def get(self, aid: str) -> dict:
        meta = self._read_meta(aid)
        meta["queue_position"] = self._queue_position(aid) if meta["status"] == "queued" else None
        return meta

    def result(self, aid: str) -> dict | None:
        p = self._dir(aid) / "result.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def file(self, aid: str, name: str) -> Path:
        d = self._dir(aid)
        if name == "overlay":
            return d / "overlay.png"
        if name == "poster":
            return d / "poster.jpg"
        if name == "video":
            vids = list((d / "input").glob(f"{VIDEO_ID}.*"))
            if vids:
                return vids[0]
        raise KeyError(name)

    def delete(self, aid: str) -> None:
        meta = self._read_meta(aid)
        if meta["status"] == "running":
            raise RuntimeError("Analysis is running")
        with self._lock:
            shutil.rmtree(self._dir(aid), ignore_errors=True)

    def _queue_position(self, aid: str) -> int | None:
        with self._queue.mutex:
            items = list(self._queue.queue)
        return items.index(aid) + 1 if aid in items else None

    # -- worker ----------------------------------------------------------------
    def start(self) -> None:
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime):
            if not (d / "meta.json").exists():
                continue
            meta = self._read_meta(d.name)
            if meta["status"] == "running":
                self._update(d.name, status="failed", finished_at=_now(),
                             error="The server stopped while this analysis was running. Upload the video again.")
            elif meta["status"] == "queued":
                self._queue.put(d.name)
        self._worker = threading.Thread(target=self._loop, name="casa-worker", daemon=True)
        self._worker.start()

    def _loop(self) -> None:
        while True:
            aid = self._queue.get()
            try:
                if not self._dir(aid).exists():
                    continue                                 # deleted while queued
                self._update(aid, status="running", started_at=_now(), progress=0.0)
                self._runner(aid)
                self._update(aid, status="done", finished_at=_now(), progress=1.0,
                             stage=None, stage_label="Finished")
                logger.info(f"Analysis {aid} finished")
            except Exception as exc:                         # noqa: BLE001 -- report any failure to the user
                logger.error(f"Analysis {aid} failed: {exc}")
                with contextlib.suppress(Exception):
                    (self._dir(aid) / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                    self._update(aid, status="failed", finished_at=_now(), error=_friendly_error(exc))
            finally:
                self._queue.task_done()

    # -- the pipeline ----------------------------------------------------------
    def _set_stage(self, aid: str, i: int, fraction: float = 0.0) -> None:
        done = sum(w for _, _, w in STAGES[:i])
        self._update(aid, stage=STAGES[i][0], stage_label=STAGES[i][1],
                     progress=round(done + STAGES[i][2] * fraction, 3))

    def _run_pipeline(self, aid: str) -> None:
        _ensure_pipeline_importable()
        from src.stage1_extract_frames import extract_frames_from_video
        from src.seg_head import segment_heads
        from src.seg_track import track_heads
        from src.seg_kinematics import run_kinematics
        from src.seg_pipeline import _run_classify, _run_aggregate
        from src.seg_report import render_trajectory_overlay
        from src.frame_qc import _load_qc

        d = self._dir(aid)
        video = self.file(aid, "video")
        frames_root, out = d / "frames", d / "out"
        cfg = yaml.safe_load(settings.PIPELINE_CONFIG.read_text(encoding="utf-8"))
        cfg["paths"].update(frames_dir=str(frames_root), output_dir=str(out))
        cfg["segmentation"].update(head_masks_dir=str(out / "seg" / "head"),
                                   tail_masks_dir=str(out / "seg" / "tail"))
        weights = Path(cfg["segmentation"]["head"]["weights"])
        if not weights.is_absolute():
            cfg["segmentation"]["head"]["weights"] = str(settings.TRAINING_DIR / weights)
        cfg_path = d / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        c = str(cfg_path)

        with open(d / "pipeline.log", "w", encoding="utf-8") as log:
            writer = _ProgressWriter(log, lambda f: self._set_stage(aid, 1, f))
            with contextlib.redirect_stdout(writer):
                self._set_stage(aid, 0)
                meta = extract_frames_from_video(video, frames_root)
                if not meta.get("extracted_frames"):
                    raise ValueError("No frames could be read from this video file.")
                self._set_stage(aid, 1)
                segment_heads(c, VIDEO_ID)
                self._set_stage(aid, 2)
                track_heads(c, VIDEO_ID)
                self._set_stage(aid, 3)
                run_kinematics(c, VIDEO_ID)
                self._set_stage(aid, 4)
                _run_classify(c, VIDEO_ID)
                self._set_stage(aid, 5)
                _run_aggregate(c, VIDEO_ID)

        seg = out / "seg"
        tracks_p = seg / "tracks" / f"{VIDEO_ID}_tracks.json"
        cls_p = seg / "classifications" / f"{VIDEO_ID}_classification.json"
        summary_p = seg / "results" / f"{VIDEO_ID}_summary.json"
        if not summary_p.exists():
            raise RuntimeError("The pipeline finished without producing a summary.")
        W, H = meta["original_resolution"]
        png = render_trajectory_overlay(frames_root, VIDEO_ID, tracks_p, cls_p, out_size=(W, H))
        if png:
            (d / "overlay.png").write_bytes(base64.b64decode(png))
        qc = _load_qc(frames_root, VIDEO_ID, cfg)
        # a frame from the middle of the usable footage, shown before the video plays
        live = (qc or {}).get("valid_frames") or []
        frames = sorted((frames_root / VIDEO_ID).glob("*.jpg"))
        pick = [f for f in frames if int(f.stem.split("_")[-1]) in set(live)] or frames
        if pick:
            shutil.copy(pick[len(pick) // 2], d / "poster.jpg")
        result = build_result(
            summary=json.loads(summary_p.read_text()),
            classifications=json.loads(cls_p.read_text()) if cls_p.exists() else {},
            video_meta=meta, qc=qc, cfg=cfg, volume_ml=self._read_meta(aid).get("volume_ml"))
        (d / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # frames are only needed while the pipeline runs; keep the disk footprint small
        shutil.rmtree(frames_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Result assembly (pure functions, unit-tested)
# ---------------------------------------------------------------------------

def _headline(res: dict) -> dict:
    m = res.get("motility", {})
    return {
        "concentration": res.get("concentration_m_per_ml"),
        "total_motility": m.get("total_motility_pct"),
        "progressive": m.get("progressive_pct"),
        "n_sperm": res.get("n_sperm_analyzed"),
        "flags": [w["level"] for w in res.get("quality", {}).get("warnings", [])],
    }


def _who_row(key: str, label: str, value: float | None) -> dict:
    op, limit, unit = WHO_REFERENCE[key]
    if value is None:
        status = "na"
    elif op == ">=":
        status = "normal" if value >= limit else "low"
    else:
        status = "normal" if value < limit else "high"
    return {"key": key, "label": label, "value": value, "unit": unit,
            "reference": f"{'≥' if op == '>=' else '<'} {limit:g} {unit}", "status": status}


def _distribution(values: list[float], key: str) -> dict:
    edges, labels, unit = DIST_BINS[key]
    counts = [0] * len(labels)
    for v in values:
        for i in range(len(labels)):
            if edges[i] <= v < edges[i + 1]:
                counts[i] += 1
                break
    n = len(values)
    return {"labels": labels, "unit": unit, "n": n,
            "percent": [round(c / n * 100, 1) if n else 0.0 for c in counts]}


def build_result(summary: dict, classifications: dict, video_meta: dict, qc: dict | None,
                 cfg: dict, volume_ml: float | None) -> dict:
    mot = summary.get("motility", {})
    counts = summary.get("counts", {})
    n = int(sum(counts.values()))
    conc = summary.get("concentration_m_per_ml")
    total_number = round(conc * volume_ml, 2) if (conc is not None and volume_ml) else None

    # 95% margin of the motility % from the number of sperm analysed (binomial)
    p = (mot.get("total_motility_pct") or 0.0) / 100.0
    margin = round(1.96 * (p * (1 - p) / n) ** 0.5 * 100, 1) if n else None

    motile = [c for c in classifications.get("classifications", []) if c.get("motility_class") in ("PR", "NP")]
    distributions = {k: _distribution([float(c[k]) for c in motile if c.get(k) is not None], k)
                     for k in DIST_BINS}

    # -- quality and warnings ------------------------------------------------
    fps = float(video_meta.get("fps") or 0) or None
    live = len(qc.get("valid_frames", [])) if qc else video_meta.get("extracted_frames")
    capture_fps = cfg.get("kinematics", {}).get("capture_fps") or fps
    live_seconds = round(live / capture_fps, 1) if (live and capture_fps) else None
    warnings = []
    tier = summary.get("detection_quality", {}).get("tier")
    if n == 0:
        warnings.append({"level": "error", "text": "No sperm could be tracked in this video. Check that it is a "
                         "semen sample recorded by the analyzer and that the field is in focus."})
    elif n < 50:
        warnings.append({"level": "warning", "text": f"Only {n} sperm could be followed. Motility is uncertain by about "
                         f"±{margin:.0f} percentage points; WHO recommends assessing at least 200."})
    elif n < 200:
        warnings.append({"level": "info", "text": f"{n} sperm were followed (WHO recommends at least 200). "
                         f"Motility is accurate to about ±{margin:.0f} percentage points."})
    if tier and tier not in ("ok", "unknown"):
        warnings.append({"level": "warning", "text": summary["detection_quality"].get("message")
                         or "Many detections look like debris or mucus rather than sperm heads."})
    if qc and live is not None and live < 30:
        warnings.append({"level": "warning", "text": f"Only {live} usable frames ({live_seconds} s) after removing "
                         "dark gaps and frozen stills. Record a longer clip for a reliable result."})
    cv = summary.get("concentration_cv_pct")
    if cv is not None and cv > 20:
        warnings.append({"level": "info", "text": f"The count varies {cv:.0f}% from frame to frame (uneven field, "
                         "bubbles or drift), so concentration is less certain."})

    k = summary.get("motile_kinematics", {})
    kin = {p_: {"mean": k.get(p_, {}).get("mean"), "median": k.get(p_, {}).get("median")}
           for p_ in ("VCL", "VSL", "VAP", "LIN", "STR", "WOB", "ALH", "BCF")}

    kc, cc = cfg.get("kinematics", {}), cfg.get("concentration", {})
    return {
        "concentration_m_per_ml": conc,
        "total_sperm_number_m": total_number,
        "msc_m_per_ml": summary.get("msc_m_per_ml"),
        "pmsc_m_per_ml": summary.get("pmsc_m_per_ml"),
        "motility": mot,
        "counts": counts,
        "n_sperm_analyzed": n,
        "motility_margin_pp": margin,
        "kinematics": kin,
        "who": [
            _who_row("volume_ml", "Ejaculate volume", volume_ml),
            _who_row("concentration", "Concentration", conc),
            _who_row("total_sperm_number", "Total sperm number", total_number),
            _who_row("total_motility", "Total motility", mot.get("total_motility_pct")),
            _who_row("progressive", "Progressive motility", mot.get("progressive_pct")),
            _who_row("immotile", "Immotile", mot.get("immotile_pct")),
        ],
        "distributions": distributions,
        "quality": {
            "detection_tier": tier,
            "live_frames": live,
            "total_frames": video_meta.get("extracted_frames"),
            "live_seconds": live_seconds,
            "concentration_cv_pct": cv,
            "warnings": warnings,
        },
        "video": {"fps": fps, "resolution": video_meta.get("original_resolution"),
                  "frames": video_meta.get("total_frames")},
        "settings": {
            "pixel_to_micron": kc.get("pixel_to_micron"),
            "capture_fps": kc.get("capture_fps"),
            "concentration_calibration": cc.get("calibration_factor"),
            "chamber_depth_um": cc.get("chamber_depth_um"),
            "detector": Path(cfg.get("segmentation", {}).get("head", {}).get("weights", "")).name,
        },
    }


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    if "Cannot open video" in msg or "No frames" in msg:
        return "This file could not be read as a video. Upload an .mp4 recorded by the analyzer."
    if "CUDA out of memory" in msg or "OutOfMemory" in msg:
        return "The graphics card ran out of memory. Close other GPU programs and upload the video again."
    if "weights not found" in msg:
        return "The sperm detector model file is missing on the server. Check segmentation.head.weights in the config."
    return f"The analysis failed: {msg[:300]}"


_store: AnalysisStore | None = None


def get_store() -> AnalysisStore:
    global _store
    if _store is None:
        _store = AnalysisStore()
    return _store
