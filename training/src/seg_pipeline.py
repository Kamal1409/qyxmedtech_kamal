"""
Segmentation CASA Pipeline Orchestrator

Runs the full segmentation-based analysis pipeline for one or all videos:

    Video frames
        ↓  seg_head   Head Detection       (YOLOv8 fine-tuned detector)
        ↓  seg_tail   Tail Segmentation    (ridge bootstrap → U-Net)
        ↓  seg_track  Multi-object Tracking (trackpy distance linking)
        ↓  seg_kinematics  Trajectory Extraction (VCL/VSL/VAP/LIN/STR/WOB)
        ↓  seg_beat   Tail Motion Analysis  (skeletonise → FFT → ALH/BCF)
        ↓  stage7     WHO Motility Classification (PR / NP / IM)
        ↓  stage8     Statistical Aggregation
        ↓  stage9     HTML Report

Usage:
    # Full segmentation pipeline for all videos
    python -m src.seg_pipeline --config configs/default.yaml

    # Single video
    python -m src.seg_pipeline --config configs/default.yaml --video-id 1

    # Run from a specific stage onwards
    python -m src.seg_pipeline --config configs/default.yaml --from-stage track

    # Skip tail (use only head kinematics)
    python -m src.seg_pipeline --config configs/default.yaml --no-tail

Stages
------
head        YOLOv8 head detection (train first: python train_yolo.py)
tail        Tail bootstrap + U-Net inference
track       trackpy head trajectory linking
kinematics  CASA velocity parameters from trajectories
beat        Flagellar beat analysis (ALH/BCF from flagella)
classify    WHO PR/NP/IM classification
aggregate   Video-level statistics
report      HTML report generation
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import yaml

STAGE_ORDER = ["head", "tail", "track", "kinematics", "beat", "classify", "aggregate", "report"]


def _banner(stage: str) -> None:
    print(f"\n{'='*60}")
    print(f"  seg_pipeline  >>  {stage.upper()}")
    print(f"{'='*60}")


def _read_video_ids(config_path: str, video_id: str | None) -> list[str]:
    """Return the list of video ids to process."""
    if video_id:
        return [str(video_id)]
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    frames_dir = Path(cfg.get("paths", {}).get("frames_dir", "data/frames"))
    return sorted([d.name for d in frames_dir.iterdir() if d.is_dir()])


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def _run_head(config_path: str, video_id: str | None, save_masks: bool) -> None:
    from src.seg_head import segment_heads
    segment_heads(config_path, video_id, save_masks=save_masks)


def _run_tail_bootstrap(config_path: str, video_id: str | None) -> None:
    from src.seg_tail import bootstrap_tails
    bootstrap_tails(config_path, video_id)


def _run_tail_train(config_path: str) -> None:
    from src.seg_tail import train_unet
    train_unet(config_path)


def _run_tail_infer(config_path: str, video_id: str | None) -> None:
    from src.seg_tail import infer_unet
    infer_unet(config_path, video_id)


def _run_track(config_path: str, video_id: str | None) -> None:
    from src.seg_track import track_heads
    track_heads(config_path, video_id)


def _run_kinematics(config_path: str, video_id: str | None) -> None:
    from src.seg_kinematics import run_kinematics
    run_kinematics(config_path, video_id)


def _run_beat(config_path: str, video_id: str | None) -> None:
    from src.seg_beat import run_beat_analysis
    run_beat_analysis(config_path, video_id)


def _run_classify(config_path: str, video_id: str | None) -> None:
    """WHO classification re-uses the existing stage7 with the seg kinematics output."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg.get("paths", {}).get("output_dir", "outputs"))

    # Point stage7 at the seg kinematics directory
    seg_kin_dir = output_dir / "seg" / "kinematics"
    cls_dir = output_dir / "seg" / "classifications"
    cls_dir.mkdir(parents=True, exist_ok=True)

    from src.stage7_classify import classify_video
    kin_files = ([seg_kin_dir / f"{video_id}_kinematics.json"] if video_id
                 else sorted(seg_kin_dir.glob("*_kinematics.json")))

    for kf in kin_files:
        if not kf.exists():
            print(f"  [!] missing: {kf}")
            continue
        result = classify_video(kf, cfg)
        vid = result["video_id"]

        # Merge ALH_real/BCF_real from beat analysis if available
        beat_p = output_dir / "seg" / "beat" / f"{vid}_beat.json"
        if beat_p.exists():
            with open(beat_p) as f:
                beat_data = json.load(f)
            beat_lookup = {b["track_id"]: b for b in beat_data.get("beat_profiles", [])}
            for cls_entry in result.get("classifications", []):
                tid = cls_entry["track_id"]
                if tid in beat_lookup:
                    b = beat_lookup[tid]
                    cls_entry["ALH_real"] = b.get("ALH_real_um")
                    cls_entry["BCF_real"] = b.get("BCF_real_hz")

        out_path = cls_dir / f"{vid}_classification.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {out_path.name}")


def _run_aggregate(config_path: str, video_id: str | None) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
    frames_dir = Path(cfg.get("paths", {}).get("frames_dir", "data/frames"))

    seg_kin_dir = output_dir / "seg" / "kinematics"
    cls_dir = output_dir / "seg" / "classifications"
    results_dir = output_dir / "seg" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    from src.stage8_aggregate import aggregate_video
    cls_files = ([cls_dir / f"{video_id}_classification.json"] if video_id
                 else sorted(cls_dir.glob("*_classification.json")))

    all_summaries = []
    for cf in cls_files:
        if not cf.exists():
            continue
        vid = cf.stem.replace("_classification", "")
        kf = seg_kin_dir / f"{vid}_kinematics.json"
        if not kf.exists():
            print(f"  [!] missing kinematics for {vid}")
            continue
        summary = aggregate_video(cf, kf, cfg, frames_dir)
        all_summaries.append(summary)
        out_path = results_dir / f"{vid}_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        m = summary["motility"]
        q = summary.get("detection_quality", {})
        flag = f"  [!] {q['tier'].upper()}" if q.get("tier") not in ("ok", "unknown", None) else ""
        print(f"  Video {vid}: Total Motility={m['total_motility_pct']:.1f}%  PR={m['progressive_pct']:.1f}%  NP={m['non_progressive_pct']:.1f}%  IM={m['immotile_pct']:.1f}%{flag}")

    if all_summaries:
        with open(results_dir / "all_videos_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2)
        flagged = [(s["video_id"], s["detection_quality"]) for s in all_summaries
                   if s.get("detection_quality", {}).get("tier") not in ("ok", None)]
        if flagged:
            print(f"\n[!] {len(flagged)}/{len(all_summaries)} video(s) flagged by the detection-quality gate:")
            for vid, q in flagged:
                print(f"    Video {vid}: {q['tier']} (oversize {q['oversize_pct']}%) -- {q['message']}")


def _run_report(config_path: str, video_id: str | None) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
    results_dir = output_dir / "seg" / "results"

    video_ids = ([video_id] if video_id
                 else [p.stem.replace("_summary", "")
                       for p in results_dir.glob("*_summary.json")
                       if p.name != "all_videos_summary.json"])

def _run_report(config_path: str, video_id: str | None) -> None:
    from src.seg_report import run_seg_report
    run_seg_report(config_path, video_id)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_seg_pipeline(
    config_path: str = "configs/default.yaml",
    video_id: str | None = None,
    from_stage: str = "head",
    no_tail: bool = False,
    save_masks: bool = True,
    tail_phase: str = "bootstrap",   # bootstrap | infer (after train)
) -> None:

    start = time.time()
    idx_from = STAGE_ORDER.index(from_stage) if from_stage in STAGE_ORDER else 0

    def should_run(stage: str) -> bool:
        return STAGE_ORDER.index(stage) >= idx_from

    print("\n+========================================================+")
    print("|  QYX MedTech -- Segmentation CASA Pipeline            |")
    print("|  Video -> Head/Tail Seg -> Track -> Kin -> Beat -> WHO    |")
    print("+========================================================+")

    # 1. HEAD SEGMENTATION
    if should_run("head"):
        _banner("head segmentation (Cellpose)")
        _run_head(config_path, video_id, save_masks=save_masks)

    # 2. TAIL SEGMENTATION
    if should_run("tail") and not no_tail:
        _banner("tail segmentation")
        from src.seg_tail import _find_weights, _resolve_dir
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        tail_out_dir = _resolve_dir(cfg.get("segmentation", {}).get("tail_masks_dir", "outputs/seg/tail"), config_path)
        weights_p = _find_weights(tail_out_dir, config_path)

        if tail_phase == "infer" or (tail_phase == "bootstrap" and weights_p.exists()):
            print(f"[+] Found trained U-Net weights ({weights_p}) — executing U-Net inference phase.")
            _run_tail_infer(config_path, video_id)
        else:
            print("[-] Executing Phase A Frangi bootstrap phase.")
            _run_tail_bootstrap(config_path, video_id)

    # 3. TRACKING
    if should_run("track"):
        _banner("multi-object head tracking (trackpy)")
        _run_track(config_path, video_id)

    # 4. KINEMATICS
    if should_run("kinematics"):
        _banner("trajectory kinematics (VCL/VSL/VAP/LIN/STR/WOB)")
        _run_kinematics(config_path, video_id)

    # 5. BEAT ANALYSIS
    if should_run("beat") and not no_tail:
        _banner("flagellar beat analysis (ALH/BCF/waveform)")
        _run_beat(config_path, video_id)

    # 6. WHO CLASSIFICATION
    if should_run("classify"):
        _banner("WHO motility classification (PR/NP/IM)")
        _run_classify(config_path, video_id)

    # 7. AGGREGATION
    if should_run("aggregate"):
        _banner("statistical aggregation")
        _run_aggregate(config_path, video_id)

    # 8. REPORT
    if should_run("report"):
        _banner("HTML report generation")
        try:
            _run_report(config_path, video_id)
        except Exception as e:
            print(f"  [!] Report generation failed: {e} (non-fatal)")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation CASA pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--from-stage", default="head", choices=STAGE_ORDER,
                        help="Start from this stage (skip earlier ones)")
    parser.add_argument("--no-tail", action="store_true",
                        help="Skip tail segmentation and beat analysis")
    parser.add_argument("--save-masks", action="store_true", default=True)
    parser.add_argument("--tail-phase", default="bootstrap", choices=["bootstrap", "infer"],
                        help="bootstrap=ridge labels (default) | infer=use trained U-Net")
    args = parser.parse_args()
    run_seg_pipeline(
        args.config, args.video_id, args.from_stage,
        args.no_tail, args.save_masks, args.tail_phase,
    )
