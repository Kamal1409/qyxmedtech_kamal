"""
Cross-validation datasets for the detector: original (de-duplicated) training frames
+ the hand-corrected review frames, split by VIDEO so every review frame is tested by
a model that never saw its video.

    data/yolo_cv/A     train = 212 original frames + corrected frames of group A videos
    data/yolo_cv/B     train = 212 original frames + corrected frames of group B videos
    data/yolo_cv/full  train = 212 original frames + all corrected frames   (deployable model)

Model A is tested on group B, model B on group A; pooled, that scores "does adding the
corrections help on unseen videos".  `val` is a small in-sample slice used only for
monitoring -- evaluate `last.pt`, never a val-selected checkpoint.

    python -m src.build_cv_datasets
"""
from __future__ import annotations
import shutil
from pathlib import Path

import yaml

GROUP_A = {"9", "11", "13", "15", "17", "19", "21", "23"}
GROUP_B = {"10", "12", "14", "16", "18", "20", "22"}


def _vid(p: Path) -> str:
    return p.stem.split("_")[0]


def build(base: Path = Path("data/yolo_full"), review: Path = Path("data/annotation_review"),
          out_root: Path = Path("data/yolo_cv")) -> None:
    orig = sorted((base / "images" / "train").glob("*.jpg"))
    rev = sorted((review / "images").glob("*.jpg"))
    gt = review / "corrected" / "obj_train_data"
    rev = [p for p in rev if (gt / f"{p.stem}.txt").exists()]
    names = yaml.safe_load(open(base / "dataset.yaml", encoding="utf-8"))["names"]
    monitor = orig[::max(len(orig) // 20, 1)][:20]          # in-sample monitoring slice

    if out_root.exists():
        shutil.rmtree(out_root)
    for tag, vids in (("A", GROUP_A), ("B", GROUP_B), ("full", GROUP_A | GROUP_B)):
        d = out_root / tag
        for split in ("train", "val"):
            (d / "images" / split).mkdir(parents=True)
            (d / "labels" / split).mkdir(parents=True)
        for p in orig:
            shutil.copy2(p, d / "images" / "train" / p.name)
            shutil.copy2(base / "labels" / "train" / f"{p.stem}.txt", d / "labels" / "train" / f"{p.stem}.txt")
        n_rev = 0
        for p in rev:
            if _vid(p) in vids:
                shutil.copy2(p, d / "images" / "train" / p.name)
                shutil.copy2(gt / f"{p.stem}.txt", d / "labels" / "train" / f"{p.stem}.txt")
                n_rev += 1
        for p in monitor:
            shutil.copy2(p, d / "images" / "val" / p.name)
            shutil.copy2(base / "labels" / "train" / f"{p.stem}.txt", d / "labels" / "val" / f"{p.stem}.txt")
        with open(d / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"path": str(d.resolve()), "train": "images/train", "val": "images/val",
                            "nc": len(names), "names": names}, f, sort_keys=False)
        print(f"{tag:>4}: train {len(orig)} original + {n_rev} corrected = {len(orig) + n_rev} | val {len(monitor)} (monitor only)")


if __name__ == "__main__":
    build()
