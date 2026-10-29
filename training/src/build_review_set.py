"""
Build a small, pre-labelled review set to measure detector accuracy for real.

Why: the detector was trained and validated on auto-generated labels from videos
1-8 only, so its mAP measures agreement with the auto-annotator, and its accuracy
on videos 9-23 is unknown.  Correcting ~45 pre-labelled frames is far faster than
drawing boxes from scratch and gives a genuine held-out test set.

What it does
  * picks N live frames per video from videos NOT in the training set, spread over
    the live clips (never black frames or frozen stills -- see src/frame_qc.py);
  * writes the detector's current boxes (outputs/seg/head/<id>_heads.json) as
    YOLO-format pre-labels;
  * packages images/ (upload as a CVAT task) and annotations_yolo11.zip (import
    into that task as "YOLO 1.1"), plus selection.csv and instructions.

Usage (from training/):
    python -m src.build_review_set                       # 3 frames x videos 9-23
    python -m src.build_review_set --per-video 4 --out data/annotation_review

    # a 2nd, targeted batch after the 1st batch's F1 scores flagged which videos
    # actually need it -- avoids re-serving the same frames:
    python -m src.build_review_set --out data/annotation_review2 \\
        --videos 18,23 --counts 18=15,23=15 --exclude data/annotation_review
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import yaml

from src.frame_qc import _load_qc

INSTRUCTIONS = """\
# Review set: correct the pre-labels (about 1-2 hours)

Goal: measure how well the detector finds sperm heads on videos it never saw.
Correct the boxes; do NOT trust them - the pre-labels come from the detector, and
accepting a wrong box would hide exactly the errors we want to count.

1. In CVAT create a task with one label: Sperm (rectangle).
   Upload everything in `images/`.
2. Task -> Actions -> Upload annotations -> format "YOLO 1.1" -> annotations_yolo11.zip.
3. On each frame fix, in this order of priority:
   - MISSED sperm heads: draw a box (tight around the head, tail excluded - match
     the style of the existing boxes).
   - FALSE boxes on debris, round cells, pin-heads or tails: delete.
   - MERGED / SPLIT boxes: two boxes on one head or one box on two heads.
   Only *heads* count. Tails alone, debris and leukocytes (large round cells) do not.
   If you are unsure whether something is a sperm head, leave it out and note the
   frame in `notes.txt`.
4. Export: Task -> Actions -> Export task dataset -> "YOLO 1.1" (no images needed).
   Unzip into `corrected/` here so the labels are in corrected/obj_train_data/*.txt.
5. Score it:   python -m src.eval_detector --pred data/annotation_review/prelabels \\
                   --gt data/annotation_review/corrected/obj_train_data
"""


def _unseen_videos(yolo_dir: Path, frames_dir: Path) -> list[str]:
    seen = set()
    for split in ("train", "val"):
        for p in (yolo_dir / "images" / split).glob("*.jpg"):
            seen.add(p.stem.split("_")[0])
    vids = [d.name for d in frames_dir.iterdir() if d.is_dir() and d.name not in seen]
    return sorted(vids, key=int)


def _pick(valid: list[int], n: int, min_gap: int, avoid: set[int] = frozenset()) -> list[int]:
    """n frames at even quantiles of the valid frames, at least min_gap apart,
    preferring frames not in `avoid` (e.g. already annotated in a previous batch)."""
    if not valid or n <= 0:
        return []
    picks: list[int] = []
    for q in range(n):
        target = valid[int((q + 0.5) / n * len(valid))]
        cands = sorted(valid, key=lambda f: (f in avoid, abs(f - target), f))
        for cand in cands:
            if all(abs(cand - p) >= min_gap for p in picks):
                picks.append(cand)
                break
    return sorted(picks)


def build(config_path: str, out: Path, per_video: int, min_gap: int, yolo_dir: Path,
          videos: list[str] | None = None, counts: dict[str, int] | None = None,
          exclude_dirs: list[Path] = ()) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    frames_dir = Path(cfg["paths"]["frames_dir"])
    head_dir = Path(cfg["segmentation"]["head_masks_dir"])

    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "prelabels").mkdir()

    # Frames already picked in earlier batches (by video): never repeat one unless
    # a video runs out of fresh frames.
    already: dict[str, set[int]] = {}
    for ex in exclude_dirs:
        for p in (ex / "images").glob("*_frame_*.jpg") if ex.exists() else []:
            v, idx = p.stem.split("_frame_")
            already.setdefault(v, set()).add(int(idx))

    video_list = videos if videos is not None else _unseen_videos(yolo_dir, frames_dir)
    rows = []
    for vid in video_list:
        n_wanted = (counts or {}).get(vid, per_video)
        if n_wanted <= 0:
            continue
        qc = _load_qc(frames_dir, vid, cfg)
        heads_p = head_dir / f"{vid}_heads.json"
        meta_p = frames_dir / vid / "metadata.json"
        if qc is None or not heads_p.exists() or not meta_p.exists():
            print(f"  video {vid}: skipped (no QC / heads / metadata)")
            continue
        with open(heads_p) as f:
            heads = json.load(f)["frames"]
        with open(meta_p) as f:
            W, H = json.load(f)["original_resolution"]
        seg_of = {k: (s["start"], s["end"]) for s in qc["segments"]
                  if s["verdict"] == "live" for k in range(s["start"], s["end"] + 1)}

        avoid = already.get(vid, set())
        valid = sorted(qc["valid_frames"])
        fresh = [f for f in valid if f not in avoid]
        pool = fresh if len(fresh) >= n_wanted else valid   # fall back only if truly short
        for idx in _pick(pool, n_wanted, min_gap, avoid=avoid):
            src = frames_dir / vid / f"frame_{idx:06d}.jpg"
            if not src.exists():
                continue
            name = f"{vid}_frame_{idx:06d}"
            shutil.copy2(src, out / "images" / f"{name}.jpg")
            dets = heads.get(str(idx), [])
            lines = []
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                             f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            (out / "prelabels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            confs = [d.get("conf", 0) for d in dets]
            rows.append({"video": vid, "frame": idx, "live_clip": "%d-%d" % seg_of[idx],
                         "n_prelabels": len(lines),
                         "min_conf": round(min(confs), 3) if confs else "",
                         "mean_conf": round(sum(confs) / len(confs), 3) if confs else ""})

    with open(out / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "README_ANNOTATION.md").write_text(INSTRUCTIONS, encoding="utf-8")

    # CVAT "YOLO 1.1" annotation archive (labels only; images go in the task)
    zpath = out / "annotations_yolo11.zip"
    names = sorted(p.stem for p in (out / "prelabels").glob("*.txt"))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "Sperm\n")
        z.writestr("obj.data", "classes = 1\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{n}.jpg\n" for n in names))
        for n in names:
            z.write(out / "prelabels" / f"{n}.txt", f"obj_train_data/{n}.txt")

    print(f"{len(rows)} frames from {len({r['video'] for r in rows})} videos -> {out}")
    print(f"pre-labelled boxes: {sum(r['n_prelabels'] for r in rows)} "
          f"(mean {sum(r['n_prelabels'] for r in rows) / len(rows):.0f} per frame)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pre-labelled review set for detector evaluation")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="data/annotation_review")
    ap.add_argument("--per-video", type=int, default=3, help="default frame count; --counts overrides per video")
    ap.add_argument("--min-gap", type=int, default=15, help="min frames between picks in a video")
    ap.add_argument("--yolo-dir", default="data/yolo_dataset", help="training set, to exclude its videos")
    ap.add_argument("--videos", default=None, help="comma-separated video ids (default: all unseen by --yolo-dir)")
    ap.add_argument("--counts", default=None,
                    help="per-video frame counts as vid=n,vid=n,... (overrides --per-video for those ids)")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated earlier review dirs whose frames should not be repeated")
    a = ap.parse_args()
    videos = a.videos.split(",") if a.videos else None
    counts = dict(kv.split("=") for kv in a.counts.split(",")) if a.counts else {}
    counts = {k: int(v) for k, v in counts.items()}
    exclude = [Path(p) for p in a.exclude.split(",")] if a.exclude else []
    build(a.config, Path(a.out), a.per_video, a.min_gap, Path(a.yolo_dir), videos, counts, exclude)
