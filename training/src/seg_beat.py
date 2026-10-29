"""
Flagellar Beat Analysis -- segmentation CASA pipeline.

Extracts the full flagellar waveform per sperm per frame and computes:
  ALH_real  Amplitude of Lateral Head displacement -- mean signed amplitude
            of the wave relative to the principal axis              (µm)
  BCF_real  Beat Cross Frequency -- dominant frequency of the waveform (Hz)
  Waveform  Time-varying lateral displacement profile y(s, t) sampled at
            N arc-length points along the flagellum.  Stored per track for
            downstream phenotyping.

Pipeline per frame per sperm
-----------------------------
1. Extract tail mask for this sperm (from seg_tail output).
2. Skeletonise the mask (Zhang-Suen thinning via skimage.morphology.skeletonize).
3. Order the skeleton into a polyline starting from the head centroid.
4. Sample ``beat.arc_sample_points`` equidistant points along the arc.
5. For each arc point: compute lateral (perpendicular) displacement from the
   principal axis of the flagellum (connecting head centroid to tail tip).
6. Collect y(s, t) matrices across all frames of the track.
7. FFT along the time axis at each arc point -> dominant frequency (BCF_real).
8. Mean amplitude of the midpoint arc (ref_arc_fraction) -> ALH_real.

Reads  : outputs/seg/tail/<video_id>_tails.json
         outputs/seg/tail/<video_id>_tails_bootstrap.npz  (or _tails.npz)
         outputs/seg/head/<video_id>_heads.json
         outputs/seg/tracks/<video_id>_tracks.json
Writes : outputs/seg/beat/<video_id>_beat.json

Usage:
    python -m src.seg_beat --config configs/default.yaml
    python -m src.seg_beat --config configs/default.yaml --video-id 1
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Skeleton ordering
# ---------------------------------------------------------------------------

def _order_skeleton(skel_pts: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """
    Order a set of skeleton pixel coordinates into a polyline starting from
    ``seed`` (the head centroid mapped to the nearest skeleton pixel).

    Uses a greedy nearest-neighbour walk; good enough for short flagella.
    Returns an (N, 2) array in [x, y] order.
    """
    if len(skel_pts) == 0:
        return skel_pts

    pts = skel_pts.copy().astype(float)
    # Start from the skeleton pixel nearest to the head centroid
    dists = np.linalg.norm(pts - seed, axis=1)
    start = int(dists.argmin())
    ordered = [pts[start]]
    remaining = list(range(len(pts)))
    remaining.remove(start)

    while remaining:
        last = ordered[-1]
        rem_arr = pts[remaining]
        d = np.linalg.norm(rem_arr - last, axis=1)
        idx = int(d.argmin())
        # Stop if the nearest remaining point is too far (skeleton has a gap)
        if d[idx] > 8:
            break
        ordered.append(rem_arr[idx])
        remaining.pop(idx)

    return np.array(ordered)


def _sample_arc(polyline: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a polyline to exactly ``n`` equidistant arc-length points.
    Returns (n, 2) array.
    """
    if len(polyline) < 2:
        return polyline
    diffs = np.diff(polyline, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total == 0:
        return np.tile(polyline[0], (n, 1))
    targets = np.linspace(0, total, n)
    xs = np.interp(targets, cum, polyline[:, 0])
    ys = np.interp(targets, cum, polyline[:, 1])
    return np.column_stack([xs, ys])


def _lateral_displacement(arc_pts: np.ndarray, head_xy: np.ndarray) -> np.ndarray:
    """
    Compute signed lateral (perpendicular) displacement of each arc point
    from the principal axis (head centroid -> tail tip line).

    Returns 1-D array of length len(arc_pts).
    """
    tail_tip = arc_pts[-1]
    axis = tail_tip - head_xy
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return np.zeros(len(arc_pts))
    axis_unit = axis / axis_len
    # Normal to the axis (perpendicular, in 2-D)
    normal = np.array([-axis_unit[1], axis_unit[0]])
    # Displacement vector from the axis line
    vecs = arc_pts - head_xy
    return (vecs @ normal)  # signed projection onto normal


# ---------------------------------------------------------------------------
# Per-track beat analysis
# ---------------------------------------------------------------------------

def _analyse_beat(
    track: dict,
    head_centroids_by_frame: dict[str, list],
    tail_npz,
    tail_json_frames: dict,
    n_pts: int,
    ref_frac: float,
    pixel_to_micron: float,
    fps: float,
) -> dict | None:
    """Compute beat metrics for one sperm track across its frames."""
    try:
        from skimage.morphology import skeletonize  # type: ignore
    except ImportError as e:
        raise ImportError("scikit-image required. pip install scikit-image") from e

    dets = track.get("detections", [])
    track_id = track["track_id"]

    # Collect lateral wave y(s, t) for each frame this track appears
    wave_matrix: list[np.ndarray] = []   # each entry is (n_pts,)
    frame_times: list[float] = []

    for det in dets:
        frame_idx = det["frame"]
        fk = str(frame_idx)
        head_list = head_centroids_by_frame.get(fk, [])

        # Find which head in this frame matches the track's position
        if not head_list:
            continue
        track_xy = np.array([det["x"], det["y"]])
        head_arr = np.array([h["centroid"] for h in head_list])
        nearest_hi = int(np.linalg.norm(head_arr - track_xy, axis=1).argmin())
        head_xy = np.array(head_arr[nearest_hi])

        # Get the tail mask for this frame
        # Look up the tail json to find if this head has a tail this frame
        tail_frame = tail_json_frames.get(fk, [])
        head_tail = next((t for t in tail_frame if t.get("head_idx") == nearest_hi), None)
        if head_tail is None:
            continue

        # Find the corresponding frame stem in the npz
        # npz keys are like "frame_000000"
        stem = f"frame_{frame_idx:06d}"
        if tail_npz is None or stem not in tail_npz.files:
            continue
        tail_mask = tail_npz[stem]  # uint8

        # Skeletonize
        binary = (tail_mask > 0).astype(np.uint8)
        skel = skeletonize(binary)
        skel_ys, skel_xs = np.where(skel)
        if len(skel_xs) < 5:
            continue
        skel_pts = np.column_stack([skel_xs, skel_ys]).astype(float)

        ordered = _order_skeleton(skel_pts, head_xy)
        if len(ordered) < n_pts:
            continue

        arc = _sample_arc(ordered, n_pts)
        arc_um = arc * pixel_to_micron
        head_um = head_xy * pixel_to_micron
        lat = _lateral_displacement(arc_um, head_um)
        wave_matrix.append(lat)
        frame_times.append(frame_idx / fps)

    if len(wave_matrix) < 8:   # not enough frames for meaningful FFT
        return None

    wave = np.array(wave_matrix)      # (T, n_pts)
    ref_idx = int(ref_frac * (n_pts - 1))

    # ALH: mean peak-to-peak amplitude at the reference arc point
    ref_signal = wave[:, ref_idx]
    alh_real = round(float((ref_signal.max() - ref_signal.min()) / 2), 3)

    # BCF: dominant FFT frequency at the reference arc point
    T = len(ref_signal)
    dt = np.mean(np.diff(frame_times)) if len(frame_times) > 1 else 1.0 / fps
    freqs = np.fft.rfftfreq(T, d=dt)
    power = np.abs(np.fft.rfft(ref_signal - ref_signal.mean())) ** 2
    if power.max() > 0:
        bcf_real = round(float(freqs[power.argmax()]), 3)
    else:
        bcf_real = 0.0

    # Waveform summary: mean lateral profile across time
    mean_wave = wave.mean(axis=0).tolist()
    std_wave = wave.std(axis=0).tolist()

    return {
        "track_id": track_id,
        "n_frames_with_tail": len(wave_matrix),
        "ALH_real_um": alh_real,
        "BCF_real_hz": bcf_real,
        "mean_lateral_um": [round(v, 3) for v in mean_wave],
        "std_lateral_um": [round(v, 3) for v in std_wave],
        "arc_sample_points": n_pts,
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_beat_analysis(
    config_path: str = "configs/default.yaml",
    video_id: str | None = None,
) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    seg_cfg = cfg.get("segmentation", {})
    beat_cfg = seg_cfg.get("beat", {})
    kin_cfg = cfg.get("kinematics", {})

    head_dir = Path(seg_cfg.get("head_masks_dir", "outputs/seg/head"))
    tail_dir = Path(seg_cfg.get("tail_masks_dir", "outputs/seg/tail"))
    tracks_dir = Path(paths_cfg.get("output_dir", "outputs")) / "seg" / "tracks"
    out_dir = Path(paths_cfg.get("output_dir", "outputs")) / "seg" / "beat"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pts = int(beat_cfg.get("arc_sample_points", 20))
    ref_frac = float(beat_cfg.get("ref_arc_fraction", 0.5))
    pixel_to_micron = float(kin_cfg.get("pixel_to_micron", 0.5))

    track_files = ([tracks_dir / f"{video_id}_tracks.json"] if video_id
                   else sorted(tracks_dir.glob("*_tracks.json")))

    if not track_files:
        print(f"No track files in {tracks_dir}. Run seg_track first.")
        return

    print("=== Flagellar Beat Analysis ===")
    print(f"arc_sample_points={n_pts} | ref_arc_fraction={ref_frac} | pixel_to_micron={pixel_to_micron}")

    for tf in track_files:
        if not tf.exists():
            print(f"  [!] missing: {tf}")
            continue

        with open(tf) as f:
            tdata = json.load(f)

        vid = tdata["video_id"]
        fps = float(tdata.get("fps", 30.0))
        tracks = tdata.get("tracks", [])
        print(f"  Video {vid} | {len(tracks)} tracks | fps={fps}")

        # Load tail masks + json
        # Prefer unet masks, fall back to bootstrap
        npz_path = tail_dir / f"{vid}_tails.npz"
        if not npz_path.exists():
            npz_path = tail_dir / f"{vid}_tails_bootstrap.npz"
        tail_npz = np.load(npz_path) if npz_path.exists() else None

        tail_json_p = tail_dir / f"{vid}_tails.json"
        tail_json_frames: dict = {}
        if tail_json_p.exists():
            with open(tail_json_p) as f:
                tail_json_frames = json.load(f).get("frames", {})

        # Load head json for centroid lookup
        head_json_p = head_dir / f"{vid}_heads.json"
        head_centroids_by_frame: dict[str, list] = {}
        if head_json_p.exists():
            with open(head_json_p) as f:
                hd = json.load(f)
            head_centroids_by_frame = {k: v for k, v in hd.get("frames", {}).items()}

        if tail_npz is None:
            print(f"    [!] no tail masks for video {vid} -- run seg_tail first")
            continue

        results = []
        skipped = 0
        for tr in tracks:
            br = _analyse_beat(
                tr, head_centroids_by_frame, tail_npz, tail_json_frames,
                n_pts, ref_frac, pixel_to_micron, fps,
            )
            if br:
                results.append(br)
            else:
                skipped += 1

        out = {
            "video_id": vid,
            "fps": fps,
            "total_tracks": len(tracks),
            "analyzed_tracks": len(results),
            "skipped_tracks": skipped,
            "beat_profiles": results,
        }
        out_path = out_dir / f"{vid}_beat.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"    -> {len(results)} beat profiles (skipped {skipped}) | {out_path.name}")

    print("Beat analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flagellar beat analysis")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video-id", default=None, help="Single video id (e.g. '1')")
    args = parser.parse_args()
    run_beat_analysis(args.config, args.video_id)
