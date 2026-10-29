"""
Rebuild the YOLO dataset without duplicate frames or train/val leakage.

The original split (data/yolo_dataset) was random by frame.  Two problems:
  * ~21 of 23 videos end in a frozen still (see src/frame_qc.py), so a video's
    annotated frames from the still are near-copies of each other; a random split
    puts copies in both train and val.
  * consecutive live frames are near-duplicates too, and the same videos appear
    on both sides, so val mAP measures memorisation, not generalisation.

This builder
  * keeps every annotated *live* frame, but only ONE representative frame per
    frozen still, and drops black-gap frames;
  * splits by VIDEO: --holdout lists the videos whose frames go to `val`.

Usage (from training/):
    python -m src.build_yolo_dataset --holdout 3,8 --out data/yolo_holdout
    python -m src.build_yolo_dataset --holdout 3,8 --out data/yolo_full --all-train
        (--all-train puts every kept frame in train and mirrors the holdout frames
         into val only so ultralytics has a val set; its metrics are then in-sample)
"""
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path

import yaml

from src.frame_qc import _load_qc

NAME_RE = re.compile(r"^(?P<vid>\d+)_frame_(?P<idx>\d+)$")


def build(config_path: str, src: Path, out: Path, holdout: set[str], all_train: bool) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    frames_dir = Path(cfg["paths"]["frames_dir"])

    # (video, idx) -> (image path, label path)
    items = {}
    for split in ("train", "val"):
        for img in sorted((src / "images" / split).glob("*.jpg")):
            m = NAME_RE.match(img.stem)
            lbl = src / "labels" / split / f"{img.stem}.txt"
            if m and lbl.exists():
                items[(m["vid"], int(m["idx"]))] = (img, lbl)

    kept, dropped = {}, {"dark/too_short": 0, "frozen_duplicate": 0}
    for vid in sorted({v for v, _ in items}, key=int):
        qc = _load_qc(frames_dir, vid, cfg)
        idxs = sorted(i for v, i in items if v == vid)
        if qc is None:                                  # QC unavailable: keep all
            for i in idxs:
                kept[(vid, i)] = items[(vid, i)]
            continue
        for seg in qc["segments"]:
            in_seg = [i for i in idxs if seg["start"] <= i <= seg["end"]]
            if seg["verdict"] == "live":
                for i in in_seg:
                    kept[(vid, i)] = items[(vid, i)]
            elif seg["verdict"] == "frozen" and in_seg:
                mid = (seg["start"] + seg["end"]) / 2
                rep = min(in_seg, key=lambda i: abs(i - mid))   # one frame per still
                kept[(vid, rep)] = items[(vid, rep)]
                dropped["frozen_duplicate"] += len(in_seg) - 1
            else:
                dropped["dark/too_short"] += len(in_seg)
        in_any = {i for seg in qc["segments"] for i in idxs if seg["start"] <= i <= seg["end"]}
        dropped["dark/too_short"] += len([i for i in idxs if i not in in_any])

    if out.exists():
        shutil.rmtree(out)
    counts = {"train": 0, "val": 0}
    for (vid, idx), (img, lbl) in kept.items():
        is_val = vid in holdout
        targets = ["val"] if (is_val and not all_train) else ["train"]
        if is_val and all_train:
            targets = ["train", "val"]
        for split in targets:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(lbl, out / "labels" / split / lbl.name)
            counts[split] += 1

    names = yaml.safe_load(open(src / "dataset.yaml", encoding="utf-8"))["names"]
    with open(out / "dataset.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"path": str(out.resolve()), "train": "images/train", "val": "images/val",
                        "nc": len(names), "names": names}, f, sort_keys=False)
    return {"kept": len(kept), "of": len(items), "dropped": dropped, "written": counts}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Leak-free YOLO dataset")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--src", default="data/yolo_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", default="3,8", help="comma-separated video ids for val")
    ap.add_argument("--all-train", action="store_true")
    a = ap.parse_args()
    res = build(a.config, Path(a.src), Path(a.out), set(a.holdout.split(",")), a.all_train)
    print(res)
