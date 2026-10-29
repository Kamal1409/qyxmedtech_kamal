"""
Stage 8: Statistical Aggregation

Aggregates per-sperm results into video-level summary statistics
including motility percentages and kinematic population metrics.

Usage:
    python -m src.stage8_aggregate --config configs/default.yaml --video 1
    python -m src.stage8_aggregate --config configs/default.yaml  # all videos
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.frame_qc import load_valid_frames
from src.head_filter import detection_quality, filter_heads, max_box_px


def compute_stats(values: list[float]) -> dict:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "median": round(float(np.median(arr)), 2),
        "min": round(float(arr.min()), 2),
        "max": round(float(arr.max()), 2),
        "n": len(values),
    }


def aggregate_video(classification_file: Path, kinematics_file: Path, config: dict, frames_dir: Path) -> dict:
    """
    Aggregate per-sperm data into video-level summary.

    Args:
        classification_file:  Path to classification JSON from Stage 7.
        kinematics_file:      Path to kinematics JSON from Stage 6.
        config:               Configuration dictionary.
        frames_dir:           Path to frames directory for metadata.

    Returns:
        dict with aggregated video-level metrics.
    """
    with open(classification_file) as f:
        cls_data = json.load(f)
    with open(kinematics_file) as f:
        kin_data = json.load(f)

    video_id = cls_data.get("video_id", "unknown")
    classifications = cls_data.get("classifications", [])
    not_sperm_ids = {t["track_id"] for t in cls_data.get("not_sperm", [])}
    profiles = [p for p in kin_data.get("profiles", []) if p["track_id"] not in not_sperm_ids]

    # -- Motility summary -----------------------------------------------
    motility = cls_data.get("percentages", {})

    # -- Kinematic population statistics --------------------------------
    param_names = ["VCL", "VSL", "VAP", "LIN", "STR", "WOB", "ALH", "BCF"]
    kinematic_stats = {}

    for param in param_names:
        values = [p[param] for p in profiles if param in p]
        kinematic_stats[param] = compute_stats(values)

    # -- Per-class kinematic breakdown ----------------------------------
    class_kinematics = {}
    cls_lookup = {c["track_id"]: c["motility_class"] for c in classifications}
    profile_lookup = {p["track_id"]: p for p in profiles}

    for motility_class in ["PR", "NP", "IM"]:
        class_profiles = [
            profile_lookup[tid]
            for tid, cls in cls_lookup.items()
            if cls == motility_class and tid in profile_lookup
        ]

        class_stats = {}
        for param in param_names:
            values = [p[param] for p in class_profiles if param in p]
            class_stats[param] = compute_stats(values)

        class_kinematics[motility_class] = {
            "count": len(class_profiles),
            "kinematics": class_stats,
        }

    # -- Motile-only kinematics -----------------------------------------
    # CASA reports VCL/VSL/VAP/LIN/STR/WOB/ALH/BCF over *motile* sperm.  The
    # lab values confirm it: video 4 is 31 % motile yet its lab VAP is 19 um/s,
    # the same as the 90 %-motile videos.  Averaging in the immotile tracks
    # (population_kinematics) drags every velocity down in proportion to the
    # immotile fraction, so compare against this block instead.
    motile_profiles = [
        profile_lookup[tid]
        for tid, cls in cls_lookup.items()
        if cls in ("PR", "NP") and tid in profile_lookup
    ]
    motile_kinematics = {
        param: compute_stats([p[param] for p in motile_profiles if param in p])
        for param in param_names
    }

    # -- Detection quality flag ------------------------------------------
    # Independent of `concentration.enabled`: this is the MVP safety gate that
    # keeps a video like 18/23 (detector fires on mucus/debris, not sperm) from
    # being reported as if it were a normal sample.  See src/head_filter.py.
    seg_cfg  = config.get("segmentation", {})
    head_dir = Path(seg_cfg.get("head_masks_dir", "outputs/seg/head"))
    heads_file = head_dir / f"{video_id}_heads.json"
    heads_data: dict | None = None
    quality = {"tier": "unknown", "oversize_pct": None, "message": "No head detection file found."}
    if heads_file.exists():
        with open(heads_file) as f:
            heads_data = json.load(f)
        quality = detection_quality(heads_data.get("frames", {}), max_box_px(config))
        if quality["tier"] != "ok":
            print(f"    [WARN] Video {video_id}: detection quality = {quality['tier']} "
                  f"(oversize {quality['oversize_pct']}%). {quality['message']}")

    # -- Concentration estimate -----------------------------------------
    total_analyzed = cls_data.get("total_analyzed", 0)

    # Calculate sperm concentration if enabled
    concentration_m_per_ml    = None
    concentration_estimated   = False   # True when falling back to track-count method
    concentration_cv_pct      = None    # Frame-to-frame coefficient of variation
    avg_sperm_per_frame       = None
    fov_source                = "config"
    conc_cfg = config.get("concentration", {})

    if conc_cfg.get("enabled", False):
        paths_cfg = config.get("paths", {})

        chamber_depth_um  = conc_cfg.get("chamber_depth_um",        20)
        fov_width_um      = conc_cfg.get("field_of_view_width_um",  1000)
        fov_height_um     = conc_cfg.get("field_of_view_height_um",  750)

        # Prefer this video's real frame size x the pixel_to_micron that produced
        # its velocities: frame size varies 856-880 x 592-618 px across videos, and
        # a hand-copied FOV constant silently goes stale whenever the scale changes.
        p2m = kin_data.get("pixel_to_micron") or config.get("kinematics", {}).get("pixel_to_micron")
        meta_file = frames_dir / str(video_id) / "metadata.json"
        if conc_cfg.get("fov_from_frame_size", True) and p2m and meta_file.exists():
            with open(meta_file) as f:
                res = json.load(f).get("original_resolution")
            if res and len(res) == 2:
                fov_width_um, fov_height_um = res[0] * p2m, res[1] * p2m
                fov_source = "frame_size"
        volume_per_frame_ul = (fov_width_um * fov_height_um * chamber_depth_um) / 1_000_000_000

        if heads_data is not None:
            # Per-frame head counts (valid frames only: a frozen still would count
            # the same sperm again, and black frames would count none)
            valid = load_valid_frames(frames_dir, video_id, config)
            head_frames = filter_heads(heads_data.get("frames", {}), max_box_px(config))
            per_frame_counts = [
                len(dets) for fk, dets in head_frames.items()
                if valid is None or int(fk) in valid
            ]
            if per_frame_counts:
                avg_sperm_per_frame = float(np.mean(per_frame_counts))
                # Round cells / debris that stage 7 dropped as not-sperm were
                # counted as heads in every frame they sat in: take them out.
                n_not_sperm = sum(t.get("n_detections", 0) for t in cls_data.get("not_sperm", []))
                if n_not_sperm:
                    avg_sperm_per_frame *= max(0.0, 1 - n_not_sperm / max(sum(per_frame_counts), 1))
                avg_sperm_per_frame = round(avg_sperm_per_frame, 2)
            else:
                avg_sperm_per_frame = heads_data.get("avg_heads_per_frame", 0)

            # Frame-to-frame CV: flags uneven distribution or FOV drift
            if len(per_frame_counts) > 1:
                arr = np.array(per_frame_counts, dtype=float)
                mean_c = arr.mean()
                if mean_c > 0:
                    concentration_cv_pct = round(float(arr.std() / mean_c * 100), 1)
                    if concentration_cv_pct > 20:
                        print(
                            f"    [WARN] Video {video_id}: high frame-to-frame concentration CV "
                            f"({concentration_cv_pct:.1f}%) — possible uneven sperm distribution, "
                            f"chamber bubble, or FOV drift. Concentration estimate is unreliable."
                        )

            if avg_sperm_per_frame and avg_sperm_per_frame > 0:
                # calibration_factor: empirical correction for the unknown
                # effective counting volume of the analyzer's cassette (its
                # depth / optics are not published).  1.0 = pure geometry.
                concentration_m_per_ml = round(
                    (avg_sperm_per_frame / volume_per_frame_ul) * 1000 / 1_000_000
                    * float(conc_cfg.get("calibration_factor", 1.0)), 2
                )
        else:
            # RC-3 FIX: was a silent null.  Now warn and fall back to a track-count
            # estimate so the report always has a concentration field.
            print(
                f"    [WARN] Video {video_id}: head detection file not found "
                f"({heads_file.name}). "
                f"Falling back to track-count concentration estimate (less accurate). "
                f"Re-run seg_head to get the accurate instantaneous-density metric."
            )
            # Fallback: total unique tracks / total processed frames
            # This underestimates because sperm swim in/out of FOV, but is
            # better than returning null and hiding the issue.
            if frames_dir.exists():
                vid_frames_dir = frames_dir / str(video_id)
                n_frames = len(list(vid_frames_dir.glob("*.jpg"))) if vid_frames_dir.exists() else 1
                avg_sperm_per_frame = round(total_analyzed / max(n_frames, 1), 2)
                if avg_sperm_per_frame > 0:
                    concentration_m_per_ml = round(
                        (avg_sperm_per_frame / volume_per_frame_ul) * 1000 / 1_000_000, 2
                    )
                    concentration_estimated = True

    # -- MSC / PMSC ---------------------------------------------------------
    # Motile / Progressively-Motile Sperm Concentration.  These were previously
    # only ever shown in the report from the lab's OCR'd reference value -- for
    # any video without a matching lab report (i.e. every real, new sample) they
    # were blank even though both inputs (concentration, motility %) are already
    # predicted here.  Derived, not independently measured -- inherits whatever
    # error concentration_m_per_ml and the motility percentages carry.
    msc_m_per_ml = pmsc_m_per_ml = None
    if concentration_m_per_ml is not None:
        msc_m_per_ml = round(concentration_m_per_ml * motility.get("total_motility_pct", 0.0) / 100, 2)
        pmsc_m_per_ml = round(concentration_m_per_ml * motility.get("progressive_pct", 0.0) / 100, 2)

    summary = {
        "video_id": video_id,
        "detection_quality": quality,   # {"tier": "ok"|"reduced_confidence"|"unreliable"|"unknown", ...}
        "total_sperm_analyzed": total_analyzed,
        "avg_sperm_per_frame": avg_sperm_per_frame,
        "concentration_m_per_ml": concentration_m_per_ml,
        "concentration_estimated": concentration_estimated,  # True = less accurate fallback
        "concentration_cv_pct": concentration_cv_pct,         # frame-to-frame variability
        "concentration_fov_source": fov_source,               # "frame_size" | "config"
        "msc_m_per_ml": msc_m_per_ml,     # predicted: concentration x total_motility_pct
        "pmsc_m_per_ml": pmsc_m_per_ml,   # predicted: concentration x progressive_pct
        "motility": {
            "total_motility_pct": motility.get("total_motility_pct", 0.0),
            "rapid_progressive_pct": motility.get("rapid_progressive_pct", 0.0),
            "slow_progressive_pct": motility.get("slow_progressive_pct", 0.0),
            "progressive_pct": motility.get("progressive_pct", 0.0),
            "non_progressive_pct": motility.get("non_progressive_pct", 0.0),
            "immotile_pct": motility.get("immotile_pct", 0.0),
        },
        "counts": cls_data.get("counts", {}),
        "population_kinematics": kinematic_stats,   # all tracks, immotile included
        "motile_kinematics": motile_kinematics,     # PR + NP only (CASA convention)
        "per_class_kinematics": class_kinematics,
    }

    return summary


def run_aggregation(config_path: str = "configs/default.yaml", video_id: str | None = None) -> None:
    """
    Aggregate classification and kinematic data for all videos.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    output_base = Path(paths_cfg.get("output_dir", "outputs"))
    frames_base = Path(paths_cfg.get("frames_dir", "data/frames"))
    cls_dir = output_base / "classifications"
    kin_dir = output_base / "kinematics"
    results_dir = output_base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    if video_id:
        cls_files = [cls_dir / f"{video_id}_classification.json"]
    else:
        cls_files = sorted(cls_dir.glob("*_classification.json"))

    if not cls_files:
        print(f"No classification files found in {cls_dir}")
        print("Run Stage 7 (classification) first.")
        return

    print(f"=== Stage 8: Statistical Aggregation ===")
    print(f"Processing {len(cls_files)} video(s)...\n")

    all_summaries = []

    for cf in cls_files:
        if not cf.exists():
            print(f"  [!] File not found: {cf}")
            continue

        vid = cf.stem.replace("_classification", "")
        kf = kin_dir / f"{vid}_kinematics.json"

        if not kf.exists():
            print(f"  [!] Matching kinematics file not found: {kf}")
            continue

        summary = aggregate_video(cf, kf, cfg, frames_base)
        all_summaries.append(summary)

        # Print summary
        m = summary["motility"]
        print(f"  Video {vid}:")
        print(f"    Sperm analyzed:  {summary['total_sperm_analyzed']} tracks")
        if summary.get('avg_sperm_per_frame'):
            print(f"    Avg per frame:   {summary['avg_sperm_per_frame']:.1f} sperm/frame")
        if summary.get('concentration_m_per_ml'):
            print(f"    Concentration:   {summary['concentration_m_per_ml']:.2f} M/mL")
        print(f"    Total Motility:  {m['total_motility_pct']:.1f}%")
        print(f"    Progressive:     {m['progressive_pct']:.1f}%")
        print(f"    Non-Progressive: {m['non_progressive_pct']:.1f}%")
        print(f"    Immotile:        {m['immotile_pct']:.1f}%")

        # Print mean kinematics
        ks = summary["population_kinematics"]
        print(f"    Mean VCL: {ks['VCL']['mean']:.1f} um/s | VAP: {ks['VAP']['mean']:.1f} um/s | VSL: {ks['VSL']['mean']:.1f} um/s")
        print()

        # Save individual summary
        out_path = results_dir / f"{vid}_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    # Save combined summary
    if all_summaries:
        combined_path = results_dir / "all_videos_summary.json"
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Combined summary: {combined_path}")

        flagged = [(s["video_id"], s["detection_quality"]) for s in all_summaries
                   if s.get("detection_quality", {}).get("tier") not in ("ok", None)]
        if flagged:
            print(f"\n[!] {len(flagged)}/{len(all_summaries)} video(s) flagged by the detection-quality gate:")
            for vid, q in flagged:
                print(f"    Video {vid}: {q['tier']} (oversize {q['oversize_pct']}%) -- {q['message']}")

    print(f"Stage 8 complete. Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8: Statistical aggregation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video", default=None, help="Video ID (e.g., '1')")
    args = parser.parse_args()
    run_aggregation(args.config, args.video)
