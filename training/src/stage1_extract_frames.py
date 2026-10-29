"""
Stage 1: Video Frame Extraction

Extracts individual frames from microscopy videos for annotation
and subsequent YOLO detection training.

Usage:
    python -m src.stage1_extract_frames --config configs/default.yaml
    python -m src.stage1_extract_frames --config configs/default.yaml --video data/vid/1.mp4
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
import yaml


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 1,
    target_resolution: list[int] | None = None,
    output_format: str = "jpg",
    jpeg_quality: int = 95,
) -> dict:
    """
    Extract frames from a single video file.

    Args:
        video_path:         Path to the .mp4 video file.
        output_dir:         Directory to save extracted frames.
        frame_interval:     Extract every Nth frame (1 = all frames).
        target_resolution:  Optional [width, height] to resize frames.
        output_format:      Image format for saved frames ("jpg" or "png").
        jpeg_quality:       JPEG quality (1-100) if format is jpg.

    Returns:
        dict with extraction metadata (frame_count, fps, resolution, etc.)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Read video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_id = video_path.stem
    frames_out = output_dir / video_id
    frames_out.mkdir(parents=True, exist_ok=True)

    print(f"  Video: {video_path.name}")
    print(f"  Resolution: {width}x{height} | FPS: {fps:.1f} | Total frames: {total_frames}")

    extracted = 0
    frame_idx = 0

    encode_params = []
    ext = f".{output_format}"
    if output_format == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    elif output_format == "png":
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Optionally resize
            if target_resolution is not None:
                frame = cv2.resize(
                    frame,
                    (target_resolution[0], target_resolution[1]),
                    interpolation=cv2.INTER_AREA,
                )

            frame_name = f"frame_{frame_idx:06d}{ext}"
            frame_path = frames_out / frame_name
            cv2.imwrite(str(frame_path), frame, encode_params)
            extracted += 1

        frame_idx += 1

    cap.release()

    metadata = {
        "video_id": video_id,
        "video_file": video_path.name,
        "original_resolution": [width, height],
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "frame_interval": frame_interval,
        "extracted_frames": extracted,
        "output_dir": str(frames_out),
    }

    # Save metadata alongside frames
    with open(frames_out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Extracted {extracted} frames -> {frames_out}")
    return metadata


def extract_all(config_path: str = "configs/default.yaml", video_path: str | None = None) -> None:
    """
    Extract frames from all videos (or a single video) based on config.

    Args:
        config_path:  Path to the YAML config file.
        video_path:   Optional single video to process. If None, processes all.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    video_cfg = cfg.get("video", {})
    paths_cfg = cfg.get("paths", {})

    video_dir = Path(paths_cfg.get("video_dir", "data/vid"))
    frames_dir = Path(paths_cfg.get("frames_dir", "data/frames"))
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_interval = video_cfg.get("frame_interval", 1)
    target_resolution = video_cfg.get("target_resolution", None)
    output_format = video_cfg.get("output_format", "jpg")
    jpeg_quality = video_cfg.get("jpeg_quality", 95)

    # Determine which videos to process
    if video_path:
        videos = [Path(video_path)]
    else:
        videos = sorted(video_dir.glob("*.mp4"))

    if not videos:
        print(f"No .mp4 files found in {video_dir}")
        return

    print(f"=== Stage 1: Frame Extraction ===")
    print(f"Processing {len(videos)} video(s)...\n")

    all_metadata = []
    for vp in videos:
        meta = extract_frames_from_video(
            video_path=vp,
            output_dir=frames_dir,
            frame_interval=frame_interval,
            target_resolution=target_resolution,
            output_format=output_format,
            jpeg_quality=jpeg_quality,
        )
        all_metadata.append(meta)
        print()

    # Save summary
    summary_path = frames_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    total_extracted = sum(m["extracted_frames"] for m in all_metadata)
    print(f"Done. Total frames extracted: {total_extracted}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Extract frames from microscopy videos")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--video", default=None, help="Optional single video path")
    args = parser.parse_args()
    extract_all(args.config, args.video)
