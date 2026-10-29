"""
Stage 7: WHO-Based Motility Classification

Classifies each tracked sperm cell into WHO 6th edition motility categories
based on its computed kinematic parameters from Stage 6.

Categories:
    PR (Progressive)      -- Forward progression with sufficient velocity & direction
    NP (Non-Progressive)  -- Movement but lacking forward progression
    IM (Immotile)         -- Negligible movement

Usage:
    python -m src.stage7_classify --config configs/default.yaml --video 1
    python -m src.stage7_classify --config configs/default.yaml  # all videos
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import yaml


# WHO motility category labels
PROGRESSIVE = "PR"
NON_PROGRESSIVE = "NP"
IMMOTILE = "IM"


NOT_SPERM = "XX"   # static object with no tail (round cell, debris): not counted at all


def classify_sperm(
    profile: dict,
    pr_min_str: float = 20.0,
    pr_min_lin: float = 0.0,
    np_min_vap: float = 5.0,
    np_min_vcl: float | None = None,
    not_sperm_max_aspect: float | None = None,
    min_vmed: float | None = None,
) -> str:
    """
    Classify a single sperm as PR / NP / IM.

    PR is *all* progressive sperm -- rapid (a) plus slow (b) -- which is how the
    lab report defines "Progressive" (progressive_pct == rapid + slow in every
    verified report).  The earlier rule required VAP >= 25 um/s for PR, which
    is only the *rapid* class, so every slow-progressive sperm was reported as
    NP.  Use `is_rapid()` for the rapid/slow split.

    Args:
        profile:     Kinematic profile dict from Stage 6.
        pr_min_str:  Minimum STR (%) for PR.  Low on purpose: WHO 6th Ed. defines
                     PR as active movement "linear or in a large circle,
                     regardless of speed"; it has no 80 % straightness cutoff
                     (that is a CASA-vendor convention).  Against the lab
                     reports an 80 % gate labels ~half of the motile sperm NP.
        pr_min_lin:  Minimum LIN (%) for PR.  0 = disabled.  A 25 % gate made
                     the NP share ~3x too large against the lab reports.
        np_min_vap:  Minimum VAP (um/s) for any motility (below this: IM).
        np_min_vcl:  A head below np_min_vap whose curvilinear speed still
                     reaches this (um/s) is NP, not IM: WHO NP is "any
                     activity without progression" -- a sperm beating in
                     place jiggles its head (VCL) without moving on (VAP).
                     None = off.
        not_sperm_max_aspect:  A static object (below np_min_vap) whose dark
                     blob is this round or rounder (profile["head_aspect"],
                     long/short axis) is a round cell or debris, not a sperm,
                     and is dropped from every count.  Moving objects are never
                     dropped: a sperm head seen end-on can look round.  None = off.
        min_vmed:    A track above np_min_vap whose median 5-frame step speed
                     (profile["VMED"], um/s) is below this is IM: its average
                     speed came from a few bad links to a neighbour, not from
                     swimming.  None = off.

    Returns:
        Motility category string: "PR", "NP", "IM" or "XX" (not a sperm).
    """
    vap     = profile.get("VAP", 0.0)
    str_val = profile.get("STR", 0.0)
    lin_val = profile.get("LIN", 0.0)

    if vap < np_min_vap:
        aspect = profile.get("head_aspect")
        if not_sperm_max_aspect is not None and aspect is not None and aspect <= not_sperm_max_aspect:
            return NOT_SPERM
        if np_min_vcl is not None and profile.get("VCL", 0.0) >= np_min_vcl:
            return NON_PROGRESSIVE
        return IMMOTILE
    if min_vmed is not None and profile.get("VMED") is not None and profile["VMED"] < min_vmed:
        return IMMOTILE
    if str_val >= pr_min_str and lin_val >= pr_min_lin:
        return PROGRESSIVE
    return NON_PROGRESSIVE


def is_rapid(profile: dict, rapid_min_vap: float = 25.0) -> bool:
    """True for the rapid-progressive (a) class: VAP >= 25 um/s (WHO 5th Ed.)."""
    return profile.get("VAP", 0.0) >= rapid_min_vap


def classify_video(kinematics_file: Path, config: dict) -> dict:
    """
    Classify all sperm in a video based on their kinematic profiles.

    Args:
        kinematics_file:  Path to the kinematics JSON from Stage 6.
        config:           Full CASA config dict.

    Returns:
        dict with per-sperm classifications and summary.
    """
    cls_cfg = config.get("classification", {})
    pr_cfg = cls_cfg.get("progressive", {})
    np_cfg = cls_cfg.get("non_progressive", {})

    rapid_min_vap = pr_cfg.get("rapid_min_vap", pr_cfg.get("min_vap", 25.0))  # legacy key: min_vap
    pr_min_str = pr_cfg.get("min_str", 20.0)
    pr_min_lin = pr_cfg.get("min_lin", 0.0)
    np_min_vap = np_cfg.get("min_vap", 5.0)
    np_min_vcl = np_cfg.get("min_vcl")
    not_sperm_max_aspect = (cls_cfg.get("not_sperm") or {}).get("max_head_aspect")
    min_vmed = np_cfg.get("min_vmed")

    with open(kinematics_file) as f:
        kin_data = json.load(f)

    video_id = kin_data.get("video_id", kinematics_file.stem)
    profiles = kin_data.get("profiles", [])

    print(f"  Classifying {len(profiles)} sperm cells from video {video_id}")
    print(f"  Thresholds -- PR: VAP>={np_min_vap} & STR>={pr_min_str} & LIN>={pr_min_lin} "
          f"(rapid: VAP>={rapid_min_vap}) | NP: VAP>={np_min_vap} otherwise | IM: VAP<{np_min_vap}")

    classifications = []
    counts = {PROGRESSIVE: 0, NON_PROGRESSIVE: 0, IMMOTILE: 0}
    n_rapid = 0
    not_sperm = []

    for profile in profiles:
        category = classify_sperm(profile, pr_min_str, pr_min_lin, np_min_vap,
                                  np_min_vcl, not_sperm_max_aspect, min_vmed)
        if category == NOT_SPERM:
            not_sperm.append({"track_id": profile["track_id"],
                              "n_detections": profile.get("n_detections", 0),
                              "head_aspect": profile.get("head_aspect")})
            continue
        counts[category] += 1
        rapid = category == PROGRESSIVE and is_rapid(profile, rapid_min_vap)
        n_rapid += rapid

        classifications.append({
            "track_id": profile["track_id"],
            "motility_class": category,
            "progressive_speed": ("rapid" if rapid else "slow") if category == PROGRESSIVE else None,
            "VAP": profile["VAP"],
            "VSL": profile["VSL"],
            "VCL": profile["VCL"],
            "STR": profile["STR"],
            "LIN": profile["LIN"],
            "WOB": profile["WOB"],
            "ALH": profile["ALH"],
            "BCF": profile["BCF"],
        })

    total = len(profiles) - len(not_sperm)
    n_slow = counts[PROGRESSIVE] - n_rapid
    percentages = {
        "rapid_progressive_pct": round(n_rapid / total * 100, 1) if total > 0 else 0.0,
        "slow_progressive_pct": round(n_slow / total * 100, 1) if total > 0 else 0.0,
        "progressive_pct": round(counts[PROGRESSIVE] / total * 100, 1) if total > 0 else 0.0,
        "non_progressive_pct": round(counts[NON_PROGRESSIVE] / total * 100, 1) if total > 0 else 0.0,
        "immotile_pct": round(counts[IMMOTILE] / total * 100, 1) if total > 0 else 0.0,
        "total_motility_pct": round((counts[PROGRESSIVE] + counts[NON_PROGRESSIVE]) / total * 100, 1) if total > 0 else 0.0,
    }

    print(f"  Results:")
    print(f"    Progressive (PR):     {counts[PROGRESSIVE]:4d} ({percentages['progressive_pct']:.1f}%)"
          f"  = rapid {percentages['rapid_progressive_pct']:.1f}% + slow {percentages['slow_progressive_pct']:.1f}%")
    print(f"    Non-Progressive (NP): {counts[NON_PROGRESSIVE]:4d} ({percentages['non_progressive_pct']:.1f}%)")
    print(f"    Immotile (IM):        {counts[IMMOTILE]:4d} ({percentages['immotile_pct']:.1f}%)")
    print(f"    Total Motility:       {percentages['total_motility_pct']:.1f}%")

    return {
        "video_id": video_id,
        "total_analyzed": total,
        "counts": counts,
        "percentages": percentages,
        "thresholds": {
            "rapid_min_vap": rapid_min_vap,
            "pr_min_str": pr_min_str,
            "pr_min_lin": pr_min_lin,
            "np_min_vap": np_min_vap,
            "np_min_vcl": np_min_vcl,
            "not_sperm_max_head_aspect": not_sperm_max_aspect,
            "min_vmed": min_vmed,
        },
        "classifications": classifications,
        "not_sperm": not_sperm,   # dropped: static and round (round cell / debris)
    }


def run_classification(config_path: str = "configs/default.yaml", video_id: str | None = None) -> None:
    """
    Run WHO motility classification on kinematic data.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    kin_dir = Path(paths_cfg.get("output_dir", "outputs")) / "kinematics"
    output_dir = Path(paths_cfg.get("output_dir", "outputs")) / "classifications"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find kinematic files
    if video_id:
        kin_files = [kin_dir / f"{video_id}_kinematics.json"]
    else:
        kin_files = sorted(kin_dir.glob("*_kinematics.json"))

    if not kin_files:
        print(f"No kinematic files found in {kin_dir}")
        print("Run Stage 6 (kinematics) first.")
        return

    print(f"=== Stage 7: WHO Motility Classification ===")
    print(f"Processing {len(kin_files)} video(s)...\n")

    for kf in kin_files:
        if not kf.exists():
            print(f"  [!] File not found: {kf}")
            continue

        result = classify_video(kf, cfg)

        out_path = output_dir / f"{result['video_id']}_classification.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path}\n")

    print(f"Stage 7 complete. Classifications saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 7: WHO motility classification")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video", default=None, help="Video ID (e.g., '1')")
    args = parser.parse_args()
    run_classification(args.config, args.video)
