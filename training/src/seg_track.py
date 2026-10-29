"""
Multi-Object Head Tracking -- segmentation CASA pipeline.

Links per-frame head centroids (from seg_head.py) into persistent per-sperm
trajectories using trackpy's nearest-neighbour linker with a configurable
search radius and memory (gap-fill) parameter.

trackpy is purpose-built for microscopy particle tracking and outperforms
IoU-based ByteTrack for fast, tiny objects like sperm heads because it uses
Euclidean distance rather than bounding-box overlap.

Reads  : outputs/seg/head/<video_id>_heads.json
Writes : outputs/seg/tracks/<video_id>_tracks.json

Each trajectory entry:
    {
        "track_id": int,
        "n_frames":  int,
        "duration_sec": float,
        "first_frame": int,
        "last_frame":  int,
        "fps": float,
        "detections": [{"frame": int, "x": float, "y": float, "area": float}, ...]
    }

Usage:
    python -m src.seg_track --config configs/default.yaml
    python -m src.seg_track --config configs/default.yaml --video-id 1
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.frame_qc import load_live_segments
from src.head_filter import filter_heads, max_box_px


def _read_fps(frames_dir: Path, video_id: str) -> float:
    """Read FPS from per-video extraction metadata (Stage 1 output)."""
    meta = frames_dir / video_id / "metadata.json"
    if meta.exists():
        with open(meta) as f:
            d = json.load(f)
        fps = float(d.get("fps", 30.0))
        if fps > 0:
            return fps
    # Fallback: scan the extraction summary
    summary = frames_dir / "extraction_summary.json"
    if summary.exists():
        with open(summary) as f:
            for entry in json.load(f):
                if str(entry.get("video_id")) == str(video_id):
                    fps = float(entry.get("fps", 30.0))
                    if fps > 0:
                        return fps
    return 30.0  # last-resort default


def _link_trackpy(
    frames: dict[str, list[dict]],
    search_radius: float,
    memory: int,
    min_frames: int,
    adaptive_stop: float | None = None,
) -> list[dict]:
    """
    Link per-frame head centroids into trajectories with trackpy.

    Args:
        frames:        {frame_idx_str: [{"centroid": [x, y], "area": float}, ...]}
        search_radius: Max displacement between frames (px) to link the same head.
        memory:        Frames a track can be absent and still be linked (gap fill).
        min_frames:    Discard trajectories shorter than this.

    Returns:
        List of trajectory dicts.
    """
    try:
        import pandas as pd
        import trackpy as tp  # type: ignore
    except ImportError as e:
        raise ImportError(
            "trackpy and pandas are required. "
            "Install with: pip install trackpy pandas"
        ) from e

    tp.quiet()  # suppress verbose output

    rows = []
    for fk, heads in frames.items():
        fidx = int(fk)
        for h in heads:
            cx, cy = h["centroid"]
            rows.append({"frame": fidx, "x": cx, "y": cy, "area": float(h.get("area", 0))})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    if adaptive_stop is None:
        adaptive_stop = max(1.0, search_radius * 0.2)

    # Locate + link (use adaptive_stop to dynamically reduce search_range in dense subnets)
    linked = tp.link(
        df,
        search_range=search_radius,
        memory=memory,
        pos_columns=["x", "y"],
        adaptive_stop=adaptive_stop,
        adaptive_step=0.95,
    )
    # Filter short trajectories
    linked = tp.filter_stubs(linked, threshold=min_frames)

    if linked.empty:
        return []

    # trackpy sets 'frame' as an index level AND keeps it as a column, which
    # makes sort_values("frame") ambiguous in pandas.  Reset first.
    linked = linked.reset_index(drop=True)

    trajectories = []
    for tid, grp in linked.groupby("particle"):
        grp = grp.sort_values("frame")
        detections = [
            {
                "frame": int(r["frame"]),
                "x": round(float(r["x"]), 2),
                "y": round(float(r["y"]), 2),
                "area": round(float(r["area"]), 1),
            }
            for _, r in grp.iterrows()
        ]
        trajectories.append({
            "track_id": int(tid),
            "n_frames": len(detections),
            "first_frame": detections[0]["frame"],
            "last_frame": detections[-1]["frame"],
            "detections": detections,
        })

    return trajectories


def track_heads(
    config_path: str = "configs/default.yaml",
    video_id: str | None = None,
) -> None:
    """Link head centroids into per-sperm trajectories for all (or one) video."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    seg_cfg = cfg.get("segmentation", {})
    track_cfg = cfg.get("tracker", {})
    kin_cfg = cfg.get("kinematics", {})

    frames_dir = Path(paths_cfg.get("frames_dir", "data/frames"))
    head_dir = Path(seg_cfg.get("head_masks_dir", "outputs/seg/head"))
    out_dir = Path(paths_cfg.get("output_dir", "outputs")) / "seg" / "tracks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tracking hyper-params: generous defaults suit fast sperm heads
    search_radius = float(track_cfg.get("search_radius_px", 30))
    memory = int(track_cfg.get("memory_frames", 5))
    min_frames = int(kin_cfg.get("min_track_frames", 10))
    adaptive_stop_val = track_cfg.get("adaptive_stop", None)
    adaptive_stop = float(adaptive_stop_val) if adaptive_stop_val is not None else max(1.0, search_radius * 0.2)

    heads_files = ([head_dir / f"{video_id}_heads.json"] if video_id
                   else sorted(head_dir.glob("*_heads.json")))

    if not heads_files:
        print(f"No head JSON files found in {head_dir}. Run seg_head first.")
        return

    print("=== Multi-Object Head Tracking (trackpy) ===")
    print(f"search_radius={search_radius}px | memory={memory}fr | min_frames={min_frames} | adaptive_stop={adaptive_stop}px")

    for hf in heads_files:
        if not hf.exists():
            print(f"  [!] missing: {hf}")
            continue

        with open(hf) as f:
            hdata = json.load(f)

        vid = hdata["video_id"]
        fps = _read_fps(frames_dir, vid)
        print(f"  Video {vid} | fps={fps} | {hdata['n_frames']} frames")

        # Drop black-gap frames and frozen stills (see src/frame_qc.py): sperm in a
        # frozen still would be tracked as zero-velocity, i.e. immotile, sperm.
        # Each live clip is tracked on its own so nothing links across a cut.
        head_frames = hdata.get("frames", {})
        limit = max_box_px(cfg)
        if limit:
            n_before = sum(len(v) for v in head_frames.values())
            head_frames = filter_heads(head_frames, limit)
            n_dropped = n_before - sum(len(v) for v in head_frames.values())
            if n_dropped:
                print(f"    head filter: dropped {n_dropped} oversize (> {limit:.0f} px) boxes "
                      f"({n_dropped / max(n_before, 1) * 100:.1f}%)")
        segments = load_live_segments(frames_dir, vid, cfg)
        if segments is None:
            runs = [head_frames]
        else:
            runs = [{k: v for k, v in head_frames.items() if s <= int(k) <= e} for s, e in segments]
            n_used = sum(len(r) for r in runs)
            print(f"    frame QC: {n_used}/{len(head_frames)} frames used "
                  f"({len(head_frames) - n_used} dark/frozen dropped) in {len(runs)} live clip(s)")
            head_frames = {k: v for r in runs for k, v in r.items()}

        trajectories = []
        for run in runs:
            if not run:
                continue
            offset = 1 + max((t["track_id"] for t in trajectories), default=-1)
            for tr in _link_trackpy(
                run,
                search_radius=search_radius,
                memory=memory,
                min_frames=min_frames,
                adaptive_stop=adaptive_stop,
            ):
                tr["track_id"] += offset          # keep ids unique across clips
                trajectories.append(tr)

        # Annotate duration from actual per-video fps
        for tr in trajectories:
            span = tr["last_frame"] - tr["first_frame"] + 1
            tr["duration_sec"] = round(span / fps, 4)
            tr["fps"] = fps

        # Per-class preview counts (will be filled by kinematics later)
        out = {
            "video_id": vid,
            "fps": fps,
            "search_radius_px": search_radius,
            "memory_frames": memory,
            "min_track_frames": min_frames,
            "frames_used": len(head_frames),
            "total_tracks": len(trajectories),
            "tracks": trajectories,
        }

        out_path = out_dir / f"{vid}_tracks.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"    -> {len(trajectories)} tracks | {out_path.name}")

    print("Tracking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Head tracking with trackpy")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video-id", default=None, help="Single video id (e.g. '1')")
    args = parser.parse_args()
    track_heads(args.config, args.video_id)
