"""
Head Detection (YOLOv8) -- segmentation CASA pipeline.

Runs a fine-tuned YOLOv8 detector on the extracted video frames to produce
per-frame sperm-head detections with class labels (Sperm / Cluster / Pin-head).
Head centroids are derived from bounding-box centres and fed downstream into
tail-linking and trajectory tracking unchanged.

Outputs (per video, under segmentation.head_masks_dir):
    <video_id>_heads.json   -- per-frame head instances (centroid, bbox, area, cls)

Weights must be trained first:
    cd training && python train_yolo.py

Usage:
    python -m src.seg_head --config configs/default.yaml
    python -m src.seg_head --config configs/default.yaml --video-id 1
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_index(stem: str, fallback: int) -> int:
    """Parse frame index from a 'frame_000123' style filename stem."""
    try:
        return int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return fallback


def _load_model(weights: str):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("Run: pip install ultralytics") from e
    return YOLO(weights)


def _detect_frame(model, img_bgr: np.ndarray, conf: float, iou: float, imgsz: int) -> list[dict]:
    """Run YOLO on one frame and return instances in the pipeline's centroid format."""
    results = model(img_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    boxes = results.boxes
    instances: list[dict] = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())
        instances.append({
            "centroid": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
            "bbox":     [round(x1), round(y1), round(x2), round(y2)],
            "area":     int((x2 - x1) * (y2 - y1)),
            "cls":      int(boxes.cls[i]),
            "cls_name": results.names[int(boxes.cls[i])],
            "conf":     round(float(boxes.conf[i]), 4),
        })
    return instances


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_heads(
    config_path: str = "configs/default.yaml",
    video_id: str | None = None,
    save_masks: bool | None = None,  # unused; kept for API compatibility
    limit: int | None = None,
) -> None:
    """Detect sperm heads in every (or one) video's frames with YOLOv8."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    seg_cfg   = cfg.get("segmentation", {})
    head_cfg  = seg_cfg.get("head", {})

    frames_dir = Path(paths_cfg.get("frames_dir", "data/frames"))
    out_dir    = Path(seg_cfg.get("head_masks_dir", "outputs/seg/head"))
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_interval = int(seg_cfg.get("sample_interval", 1))
    weights = head_cfg.get("weights", "outputs/detection/yolo_trained/run/weights/best.pt")
    # resolve relative to the config file's directory so cwd doesn't matter
    weights_path = Path(weights)
    if not weights_path.is_absolute():
        weights_path = Path(config_path).parent.parent / weights_path
    conf    = float(head_cfg.get("conf_threshold", 0.35))
    iou     = float(head_cfg.get("iou_threshold",  0.45))
    imgsz   = int(head_cfg.get("imgsz", 640))

    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights_path}\n"
            "Train first:  python train_yolo.py"
        )

    if cv2 is None:
        raise ImportError("opencv-python is required.")

    model = _load_model(str(weights_path))

    print("=== Head Detection (YOLOv8) ===")
    print(f"Weights: {weights_path} | conf={conf} | iou={iou} | imgsz={imgsz} | "
          f"sample every {sample_interval} frame(s)")

    if video_id:
        frame_dirs = [frames_dir / str(video_id)]
    else:
        frame_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])

    if not frame_dirs:
        print(f"No frame directories in {frames_dir}. Run Stage 1 first.")
        return

    grand_total = 0
    for fdir in frame_dirs:
        if not fdir.is_dir():
            print(f"  [!] missing directory: {fdir}")
            continue

        vid = fdir.name
        frame_files = sorted(fdir.glob("*.jpg"))[::sample_interval]
        if limit:
            frame_files = frame_files[:limit]
        if not frame_files:
            print(f"  [!] no frames in {fdir}")
            continue

        print(f"  Video {vid}: detecting in {len(frame_files)} frame(s)...")
        per_frame: dict[str, list[dict]] = {}
        total_heads = 0

        for i, ff in enumerate(frame_files):
            img = cv2.imread(str(ff))
            if img is None:
                continue
            fidx = _frame_index(ff.stem, i)
            insts = _detect_frame(model, img, conf, iou, imgsz)
            per_frame[str(fidx)] = insts
            total_heads += len(insts)

            if (i + 1) % 50 == 0 or i == len(frame_files) - 1:
                pct = (i + 1) / len(frame_files) * 100
                print(f"    [{pct:5.1f}%] {i+1}/{len(frame_files)} frames | "
                      f"{total_heads} detections", flush=True)

        result = {
            "video_id":            vid,
            "detector":            "yolov8",
            "weights":             str(weights_path),
            "sample_interval":     sample_interval,
            "n_frames":            len(per_frame),
            "total_heads":         total_heads,
            "avg_heads_per_frame": round(total_heads / max(len(per_frame), 1), 1),
            "frames":              per_frame,
        }
        out_json = out_dir / f"{vid}_heads.json"
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    -> {total_heads} detections (avg {result['avg_heads_per_frame']}/frame) | {out_json.name}")
        grand_total += total_heads

    print(f"\nHead detection complete. Total: {grand_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sperm head detection with YOLOv8")
    parser.add_argument("--config",   default="configs/default.yaml")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--limit",    type=int, default=None, help="First N frames only")
    args = parser.parse_args()
    segment_heads(args.config, args.video_id, limit=args.limit)
