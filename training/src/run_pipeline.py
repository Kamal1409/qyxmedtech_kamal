"""
QYX MedTech — CASA Pipeline Runner

Segmentation-based Computer-Aided Sperm Analysis pipeline:

    Video
      ↓  stage1          Frame extraction
      ↓  seg_head        Head segmentation      (Cellpose)
      ↓  seg_tail        Tail segmentation      (Frangi bootstrap → U-Net)
      ↓  seg_track       Multi-object tracking  (trackpy)
      ↓  seg_kinematics  Trajectory kinematics  (VCL/VSL/VAP/LIN/STR/WOB)
      ↓  seg_beat        Flagellar beat          (skeleton → FFT → ALH/BCF)
      ↓  stage7          WHO classification     (PR / NP / IM)
      ↓  stage8          Statistical aggregation
      ↓  seg_report      Clinical HTML report

Usage
-----
# Full pipeline — one video
python -m src.run_pipeline --video data/vid/1.mp4

# Full pipeline — all videos in data/vid/
python -m src.run_pipeline --all

# Skip tail analysis (faster, head-only)
python -m src.run_pipeline --video data/vid/1.mp4 --no-tail

# Resume from a specific stage (avoids re-running earlier stages)
python -m src.run_pipeline --video-id 1 --from-stage track

# Tail pipeline stages
python -m src.run_pipeline --video-id 1 --from-stage tail --tail-phase bootstrap
python -m src.run_pipeline --tail-train          # train U-Net on bootstrap masks
python -m src.run_pipeline --all --tail-phase infer  # use trained U-Net

# Extract ground truth from lab reports + validate vs pipeline output
python -m src.run_pipeline --ocr
python -m src.run_pipeline --validate
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import yaml

STAGES = ["head", "tail", "track", "kinematics", "beat", "classify", "aggregate", "report"]


def _load_cfg(config_path: str) -> dict:
    if not Path(config_path).exists():
        sys.exit(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QYX MedTech CASA Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")

    # Video targeting
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--video", default=None, help="Path to a single video file (e.g. data/vid/1.mp4)")
    target.add_argument("--video-id", default=None, help="Video ID of already-extracted frames (e.g. 1)")
    target.add_argument("--all", action="store_true", help="Run on all videos in data/vid/")

    # Special modes
    special = parser.add_mutually_exclusive_group()
    special.add_argument("--tail-train", action="store_true", help="Train U-Net on bootstrap tail masks")
    special.add_argument("--ocr", action="store_true", help="Extract ground truth from lab report images")
    special.add_argument("--validate", action="store_true", help="Validate pipeline output vs lab reports")

    # Pipeline options
    parser.add_argument("--from-stage", default="head", choices=STAGES,
                        help="Start from this stage, skip earlier ones (default: head)")
    parser.add_argument("--no-tail", action="store_true",
                        help="Skip tail segmentation and beat analysis")
    parser.add_argument("--tail-phase", default="bootstrap", choices=["bootstrap", "infer"],
                        help="bootstrap (default) = Frangi labels | infer = use trained U-Net")

    args = parser.parse_args()
    cfg = _load_cfg(args.config)

    t0 = time.time()
    print("+========================================================+")
    print("|  QYX MedTech — CASA Pipeline                          |")
    print("|  Cellpose · trackpy · WHO 6th Edition                 |")
    print("+========================================================+")

    # --- Special modes ---
    if args.tail_train:
        from src.seg_tail import train_unet
        train_unet(args.config)

    elif args.ocr:
        from src.ocr_report import run_ocr
        run_ocr(args.config)

    elif args.validate:
        from src.validate import validate
        validate(args.config)

    # --- Main seg pipeline ---
    else:
        from src.seg_pipeline import run_seg_pipeline

        if args.all:
            video_dir = Path(cfg.get("paths", {}).get("video_dir", "data/vid"))
            videos = sorted(video_dir.glob("*.mp4"))
            if not videos:
                sys.exit(f"No .mp4 files found in {video_dir}")
            from src.stage1_extract_frames import extract_all
            extract_all(args.config)
            for vp in videos:
                print(f"\n>>> Video {vp.stem} <<<")
                run_seg_pipeline(
                    config_path=args.config, video_id=vp.stem,
                    from_stage=args.from_stage, no_tail=args.no_tail,
                    save_masks=True, tail_phase=args.tail_phase,
                )

        elif args.video:
            from src.stage1_extract_frames import extract_all
            extract_all(args.config, args.video)
            run_seg_pipeline(
                config_path=args.config, video_id=Path(args.video).stem,
                from_stage=args.from_stage, no_tail=args.no_tail,
                save_masks=True, tail_phase=args.tail_phase,
            )

        elif args.video_id:
            run_seg_pipeline(
                config_path=args.config, video_id=args.video_id,
                from_stage=args.from_stage, no_tail=args.no_tail,
                save_masks=True, tail_phase=args.tail_phase,
            )

        else:
            parser.print_help()

    elapsed = time.time() - t0
    print(f"\nTotal time: {int(elapsed // 60)}m {int(elapsed % 60)}s")


if __name__ == "__main__":
    main()
