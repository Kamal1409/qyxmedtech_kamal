"""
Trajectory Kinematics -- segmentation CASA pipeline.

Computes all WHO/CASA velocity and motion-quality parameters for each
sperm trajectory produced by seg_track.py.

Key improvement over the old stage6_kinematics.py:
  * Uses the **head-mask centroid** (from Cellpose) instead of a
    loose bounding-box centre -- eliminates positional jitter from padding.
  * Reads per-video FPS from Stage-1 metadata (never hardcoded).
  * pixel_to_micron is read from config (must be calibrated per objective).

Parameters computed
-------------------
VCL  Curvilinear Velocity     total arc-length / duration                (µm/s)
VSL  Straight-Line Velocity   net displacement / duration                (µm/s)
VAP  Average Path Velocity    smoothed arc-length / duration             (µm/s)
LIN  Linearity                VSL / VCL × 100                            (%)
STR  Straightness             VSL / VAP × 100                            (%)
WOB  Wobble                   VAP / VCL × 100                            (%)
ALH  Amplitude Lateral Head   mean perpendicular distance to smooth path (µm)
BCF  Beat Cross Frequency     zero-crossings of lateral deviation / dur  (Hz)

Reads  : outputs/seg/tracks/<video_id>_tracks.json
Writes : outputs/seg/kinematics/<video_id>_kinematics.json

Usage:
    python -m src.seg_kinematics --config configs/default.yaml
    python -m src.seg_kinematics --config configs/default.yaml --video-id 1
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Math helpers (shared with old stage6; kept self-contained here)
# ---------------------------------------------------------------------------

def _fill_gaps(frames: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Linearly interpolate onto every frame between first and last detection.

    trackpy's `memory` lets a track skip frames.  Treating consecutive
    detections as consecutive time steps would count a k-frame jump as one
    step of arc length, so put every point on a uniform time grid first.
    """
    grid = np.arange(frames[0], frames[-1] + 1)
    if len(grid) == len(frames):
        return pts
    return np.column_stack([np.interp(grid, frames, pts[:, 0]),
                            np.interp(grid, frames, pts[:, 1])])


def _smooth(positions: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    """Centred moving average over fully-supported points only.

    Returns (smoothed_path, half_window).  smoothed_path[j] corresponds to
    positions[j + half_window].

    The previous implementation shrank the window at both ends of the track,
    which drags the smoothed endpoints towards the middle of the path.  That
    shortened the smoothed arc (VAP too low) and made VSL, measured between the
    raw endpoints, exceed VAP -- STR > 100 % is impossible by definition.  It also
    returned the raw path (VAP == VCL) for any track shorter than the window.
    The window is capped at half the track so short tracks are still smoothed.
    """
    n = len(positions)
    w = min(int(window), n // 2)
    if w % 2 == 0:
        w -= 1
    w = max(w, 3)
    if n < w + 1:
        return positions.copy(), 0
    kernel = np.ones(w) / w
    sm = np.column_stack([np.convolve(positions[:, 0], kernel, mode="valid"),
                          np.convolve(positions[:, 1], kernel, mode="valid")])
    return sm, w // 2


def _arc_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1)).sum())


def _displacement(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(pts[-1] - pts[0]))


def _alh(pts: np.ndarray, smooth: np.ndarray) -> float:
    """Mean distance from the actual path to the smooth path (same frames)."""
    if len(pts) < 3:
        return 0.0
    return float(np.sqrt(((pts - smooth) ** 2).sum(axis=1)).mean())


def _bcf(pts: np.ndarray, smooth: np.ndarray, duration: float) -> float:
    """Beat Cross Frequency: crossings of the average path per second."""
    if len(pts) < 4 or duration <= 0:
        return 0.0
    tangents = np.diff(smooth, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]]) / norms
    dev = pts[:-1] - smooth[:-1]
    signed = (dev * normals).sum(axis=1)
    # A deviation of exactly 0 is "on the path", not a side: drop it so a
    # -1 -> 0 -> +1 sequence counts as one crossing, not two.
    side = np.sign(signed)
    side = side[side != 0]
    crossings = int(np.sum(np.diff(side) != 0)) if len(side) > 1 else 0
    return round(crossings / duration, 3)


# ---------------------------------------------------------------------------
# Per-track analysis
# ---------------------------------------------------------------------------

def _analyse_track(track: dict, pixel_to_micron: float,
                   smoothing_window: int, min_frames: int) -> dict | None:
    dets = track.get("detections", [])
    if len(dets) < min_frames:
        return None

    pts_px = np.array([[d["x"], d["y"]] for d in dets])
    frames_arr = np.array([d["frame"] for d in dets])
    fps = float(track.get("fps", 30.0))

    # Uniform time grid, then micrometres.
    pts_um = _fill_gaps(frames_arr, pts_px) * pixel_to_micron
    n_steps = len(pts_um) - 1               # N points span N-1 frame intervals
    if n_steps < 1 or fps <= 0:
        return None
    duration = n_steps / fps

    smooth_um, half = _smooth(pts_um, smoothing_window)
    smooth_steps = len(smooth_um) - 1
    if smooth_steps < 1:
        return None
    smooth_duration = smooth_steps / fps
    raw_span = pts_um[half:len(pts_um) - half]   # raw points matching smooth_um

    vcl = _arc_length(pts_um) / duration
    vsl = _displacement(pts_um) / duration
    vap = _arc_length(smooth_um) / smooth_duration

    # By definition VSL <= VAP <= VCL, so LIN/STR/WOB <= 100 %.  Residual
    # excursions come from raw-endpoint jitter, so clamp rather than report
    # impossible values.
    lin = min(vsl / vcl * 100, 100.0) if vcl > 0 else 0.0
    str_val = min(vsl / vap * 100, 100.0) if vap > 0 else 0.0
    wob = min(vap / vcl * 100, 100.0) if vcl > 0 else 0.0
    alh = _alh(raw_span, smooth_um)
    bcf = _bcf(raw_span, smooth_um, smooth_duration)
    # Median speed over 5-frame steps.  A slow swimmer moves a little in most
    # frames; a still head that picks up one bad link from a neighbour (common
    # in crowded fields) has a high mean speed but a median near zero.
    k = 5
    vmed = (float(np.median(np.hypot(*(pts_um[k:] - pts_um[:-k]).T))) / k * fps
            if len(pts_um) > k else 0.0)

    return {
        "track_id": track["track_id"],
        "n_detections": len(dets),
        "duration_sec": round(duration, 4),
        "first_frame": int(frames_arr[0]),
        "last_frame": int(frames_arr[-1]),
        "fps": fps,
        "VCL": round(vcl, 2),
        "VSL": round(vsl, 2),
        "VAP": round(vap, 2),
        "LIN": round(lin, 1),
        "STR": round(str_val, 1),
        "WOB": round(wob, 1),
        "ALH": round(alh, 2),
        "BCF": round(bcf, 2),
        "VMED": round(vmed, 2),
    }


# ---------------------------------------------------------------------------
# Head shape (round cells / debris vs sperm heads)
# ---------------------------------------------------------------------------

def _blob_aspect(gray: np.ndarray, x: float, y: float, r: int = 18) -> float | None:
    """Long/short axis ratio of the dark blob at (x, y): a sperm head is an oval
    (>= ~1.4), a round cell or debris particle is close to 1."""
    import cv2
    H, W = gray.shape
    x0, y0 = max(int(x) - r, 0), max(int(y) - r, 0)
    p = gray[y0:min(int(y) + r, H), x0:min(int(x) + r, W)].astype(np.float32)
    if p.size == 0:
        return None
    n, lab, _, cen = cv2.connectedComponentsWithStats((p < p.mean() - p.std()).astype(np.uint8))
    if n < 2:
        return None
    j = 1 + int(np.argmin(np.hypot(cen[1:, 0] - (x - x0), cen[1:, 1] - (y - y0))))
    cnt, _ = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnt or len(cnt[0]) < 5:
        return None
    _, (a, b), _ = cv2.fitEllipse(cnt[0])
    return max(a, b) / max(min(a, b), 1e-6)


def _head_aspects(tracks: list[dict], frames_dir: Path, vid: str, per_track: int = 6) -> dict[int, float]:
    """Median blob aspect ratio of each track over up to `per_track` of its frames."""
    import cv2
    files = {int(p.stem.split("_")[-1]): p for p in (frames_dir / str(vid)).glob("*.jpg")}
    wanted: dict[int, list] = {}
    for t in tracks:
        dets = t.get("detections", [])
        for d in dets[::max(len(dets) // per_track, 1)][:per_track]:
            wanted.setdefault(d["frame"], []).append((t["track_id"], d["x"], d["y"]))
    vals: dict[int, list] = {}
    for fr, pts in wanted.items():
        if fr not in files:
            continue
        gray = cv2.imread(str(files[fr]), cv2.IMREAD_GRAYSCALE)
        for tid, x, y in pts:
            a = _blob_aspect(gray, x, y)
            if a is not None:
                vals.setdefault(tid, []).append(a)
    return {tid: round(float(np.median(v)), 2) for tid, v in vals.items()}


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_kinematics(
    config_path: str = "configs/default.yaml",
    video_id: str | None = None,
) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    kin_cfg = cfg.get("kinematics", {})

    tracks_dir = Path(paths_cfg.get("output_dir", "outputs")) / "seg" / "tracks"
    out_dir = Path(paths_cfg.get("output_dir", "outputs")) / "seg" / "kinematics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_to_micron = float(kin_cfg.get("pixel_to_micron", 0.5))
    smoothing = int(kin_cfg.get("smoothing_window", 15))   # matches default.yaml (RC-8)
    min_frames = int(kin_cfg.get("min_track_frames", 10))
    # Real capture rate when it differs from the rate stamped in the exported
    # video (the analyzer's exports are all 30 fps).  None = trust the video.
    capture_fps = kin_cfg.get("capture_fps")
    frames_dir = Path(paths_cfg.get("frames_dir", "data/frames"))

    track_files = ([tracks_dir / f"{video_id}_tracks.json"] if video_id
                   else sorted(tracks_dir.glob("*_tracks.json")))

    if not track_files:
        print(f"No track files in {tracks_dir}. Run seg_track first.")
        return

    print("=== Trajectory Kinematics ===")
    print(f"pixel_to_micron={pixel_to_micron} | smoothing={smoothing} | min_frames={min_frames}"
          f" | capture_fps={capture_fps or 'from video'}")

    for tf in track_files:
        if not tf.exists():
            print(f"  [!] missing: {tf}")
            continue
        with open(tf) as f:
            tdata = json.load(f)

        vid = tdata["video_id"]
        tracks = tdata.get("tracks", [])
        print(f"  Video {vid} | {len(tracks)} tracks | fps={tdata.get('fps', '?')}")

        aspects = _head_aspects(tracks, frames_dir, vid)
        profiles = []
        skipped = 0
        for tr in tracks:
            if capture_fps:
                tr = {**tr, "fps": float(capture_fps)}
            p = _analyse_track(tr, pixel_to_micron, smoothing, min_frames)
            if p:
                p["head_aspect"] = aspects.get(tr["track_id"])
                profiles.append(p)
            else:
                skipped += 1

        result = {
            "video_id": vid,
            "fps": float(capture_fps) if capture_fps else tdata.get("fps"),
            "video_fps": tdata.get("fps"),
            "pixel_to_micron": pixel_to_micron,
            "total_tracks": len(tracks),
            "analyzed_tracks": len(profiles),
            "skipped_tracks": skipped,
            "profiles": profiles,
        }
        out_path = out_dir / f"{vid}_kinematics.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"    -> {len(profiles)} profiles (skipped {skipped}) | {out_path.name}")

    print("Kinematics complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory kinematics from head tracks")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video-id", default=None, help="Single video id (e.g. '1')")
    args = parser.parse_args()
    run_kinematics(args.config, args.video_id)
