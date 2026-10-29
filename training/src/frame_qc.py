"""
Frame quality control -- find the frames of a video that are valid to analyse.

Why this exists
---------------
The microscope videos are not one continuous recording.  Almost every video is a
concatenation of clips separated by ~10 black frames, and most end with a long
still image (a *different* field of view whose sperm never move -- the frames
differ only by codec noise).  Left in, the pipeline detects and tracks every
sperm in the still and reports each as a zero-velocity (immotile) track.  In
video 1, 58 of 102 frames were such a still, so predicted immotile was ~78 %
against 20 % in the lab report, and no threshold or pixel-scale change could
fix it.  It also double-counts sperm in the concentration estimate.

Test (per segment between black gaps):
    dark    frame mean brightness < dark_fraction x video median
    frozen  segment with (a) essentially no pixels that change by a large amount
            between consecutive frames -- a moving sperm head changes tens of
            grey levels, codec noise never does -- AND (b) a mean frame-to-frame
            difference at noise level.  Condition (b) keeps a live but sparse
            field, whose sensor noise alone gives a mean diff >= 0.47.
            Measured: frozen tails have 0.00-0.52 x1e-4 large-change pixels and
            mean diff <= 0.19; live clips have >= 27 x1e-4 (one sparse clip
            0.16 x1e-4) and mean diff >= 0.47.
When in doubt a segment is kept -- discarding real data is the worse error.

Writes data/frames/<video_id>/frame_qc.json (cached; recomputed if the
parameters change).

Usage:
    python -m src.frame_qc --config configs/default.yaml
    python -m src.frame_qc --config configs/default.yaml --video-id 1
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULTS = {
    "enabled": True,
    "dark_fraction": 0.5,        # frame is dark if mean brightness < this x video median
    "min_segment_frames": 15,    # shorter segments are too brief to track
    "motion_threshold": 15,      # grey-level change that counts as "a pixel moved"
    "frozen_max_motion": 2e-4,   # frozen: fraction of moved pixels below this ...
    "frozen_max_diff": 0.3,      # ... and mean |frame diff| (grey levels, half-res) below this
}


def qc_params(config: dict | None) -> dict:
    p = dict(DEFAULTS)
    p.update((config or {}).get("frame_qc", {}) or {})
    return p


def _frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def analyse_video(video_frames_dir: Path, params: dict) -> dict:
    """Compute per-frame QC for one video's extracted frames."""
    if cv2 is None:
        raise ImportError("opencv-python is required for frame QC.")
    files = sorted(video_frames_dir.glob("*.jpg"), key=_frame_index)
    if not files:
        return {"params": params, "n_frames": 0, "segments": [], "valid_frames": []}

    thr = int(params["motion_threshold"])
    idx, bright = [], []
    d1 = {}                       # frame index -> mean |f[k]-f[k-1]|
    moved = {}                    # frame index -> fraction of pixels with |f[k]-f[k-1]| > thr
    prev = None
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.int16)
        k = _frame_index(f)
        idx.append(k)
        bright.append(float(g.mean()))
        if prev is not None:
            diff = np.abs(g - prev)
            d1[k] = float(diff.mean())
            moved[k] = float((diff > thr).mean())
        prev = g

    bright_a = np.array(bright)
    dark = bright_a < params["dark_fraction"] * np.median(bright_a)

    # Split into segments of consecutive non-dark frames.
    segments, start = [], None
    for i in range(len(idx) + 1):
        is_dark = True if i == len(idx) else bool(dark[i])
        if not is_dark and start is None:
            start = i
        elif is_dark and start is not None:
            segments.append((idx[start], idx[i - 1], start, i - 1))
            start = None

    seg_info, valid = [], []
    for (f0, f1, i0, i1) in segments:
        n = i1 - i0 + 1
        ks = idx[i0:i1 + 1]
        # skip the segment's first frame: its predecessor is a black frame
        seg_d1 = [d1[k] for k in ks[1:] if k in d1]
        seg_mv = [moved[k] for k in ks[1:] if k in moved]
        mean_d1 = float(np.mean(seg_d1)) if seg_d1 else float("nan")
        mean_mv = float(np.mean(seg_mv)) if seg_mv else float("nan")

        if n < params["min_segment_frames"]:
            verdict = "too_short"
        elif (not np.isnan(mean_d1) and mean_d1 < params["frozen_max_diff"]
              and mean_mv < params["frozen_max_motion"]):
            verdict = "frozen"
        else:
            verdict = "live"
        if verdict == "live":
            valid.extend(ks)
        seg_info.append({
            "start": f0, "end": f1, "n_frames": n,
            "mean_diff": None if np.isnan(mean_d1) else round(mean_d1, 3),
            "moved_fraction": None if np.isnan(mean_mv) else round(mean_mv, 6),
            "verdict": verdict,
        })

    return {
        "params": params,
        "n_frames": len(idx),
        "n_dark": int(dark.sum()),
        "segments": seg_info,
        "valid_frames": valid,
    }


def _load_qc(frames_dir: Path, video_id: str, config: dict | None) -> dict | None:
    """QC record for a video (cached on disk), or None if QC is off/unavailable."""
    params = qc_params(config)
    if not params["enabled"]:
        return None
    vdir = Path(frames_dir) / str(video_id)
    if not vdir.is_dir() or cv2 is None:
        return None

    cache = vdir / "frame_qc.json"
    if cache.exists():
        try:
            with open(cache) as f:
                data = json.load(f)
            if data.get("params") == params:
                return data
        except (OSError, ValueError):
            pass                         # unreadable / stale cache -> recompute

    data = analyse_video(vdir, params)
    if not data["n_frames"]:
        return None
    try:
        with open(cache, "w") as f:
            json.dump(data, f)
    except OSError:
        pass                             # read-only / full disk: still usable in memory
    return data


def load_valid_frames(frames_dir: Path, video_id: str, config: dict | None) -> set[int] | None:
    """Set of valid frame indices for a video, or None if QC is off / unavailable
    (callers must then treat every frame as valid)."""
    data = _load_qc(frames_dir, video_id, config)
    return None if data is None else set(data["valid_frames"])


def load_live_segments(frames_dir: Path, video_id: str, config: dict | None) -> list[tuple[int, int]] | None:
    """(start, end) frame ranges of the live clips, or None if QC is off.

    Tracking must run per segment: trackpy links on frame *order*, so across a
    dropped black gap it would link a sperm at the end of one clip to an
    unrelated sperm at the start of the next.
    """
    data = _load_qc(frames_dir, video_id, config)
    if data is None:
        return None
    return [(s["start"], s["end"]) for s in data["segments"] if s["verdict"] == "live"]


def run_frame_qc(config_path: str = "configs/default.yaml", video_id: str | None = None) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    frames_dir = Path(cfg.get("paths", {}).get("frames_dir", "data/frames"))
    vids = [str(video_id)] if video_id else sorted(
        [d.name for d in frames_dir.iterdir() if d.is_dir()], key=lambda x: (len(x), x))

    print("=== Frame QC (dark gaps / frozen stills) ===")
    for vid in vids:
        valid = load_valid_frames(frames_dir, vid, cfg)
        cache = frames_dir / vid / "frame_qc.json"
        if valid is None or not cache.exists():
            print(f"  Video {vid}: QC disabled or no frames")
            continue
        with open(cache) as f:
            d = json.load(f)
        segs = " ".join(f"[{s['start']}-{s['end']}:{s['verdict']}]" for s in d["segments"])
        print(f"  Video {vid}: {len(valid)}/{d['n_frames']} frames valid | dark={d['n_dark']} | {segs}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Flag dark / frozen frames")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--video-id", default=None)
    a = ap.parse_args()
    run_frame_qc(a.config, a.video_id)
