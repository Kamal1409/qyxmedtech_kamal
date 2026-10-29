"""
Cross-validation datasets for the SECOND annotation batch (data/annotation_review2:
videos 18 + 23, focused on the two videos the first batch showed the detector
struggling with) -- combined with the first batch's corrections.

Unlike build_cv_datasets.py (fresh-retrain comparison), this is meant to feed
train_yolo.py's FINE-TUNE mode: start from the currently-deployed best.pt, not a
fresh COCO checkpoint, with a low --lr0.  See the top of train_yolo.py for why.

    python -m src.build_cv_datasets2

Produces data/yolo_cv2/{A,B,full} the same way as build_cv_datasets.py's groups,
folding in review2's corrections for whichever of 18/23 falls in each group
(18 is in GROUP_B, 23 is in GROUP_A -- see build_cv_datasets.GROUP_A/B).
"""
from __future__ import annotations
import shutil
from pathlib import Path

import yaml

from src.build_cv_datasets import GROUP_A, GROUP_B


def _vid(p: Path) -> str:
    return p.stem.split("_")[0]


def build(base: Path = Path("data/yolo_full"),
          reviews: tuple[Path, ...] = (Path("data/annotation_review"), Path("data/annotation_review2")),
          out_root: Path = Path("data/yolo_cv2")) -> None:
    orig = sorted((base / "images" / "train").glob("*.jpg"))
    names = yaml.safe_load(open(base / "dataset.yaml", encoding="utf-8"))["names"]
    monitor = orig[::max(len(orig) // 20, 1)][:20]

    rev_items: list[tuple[Path, Path]] = []   # (image, label)
    for rev in reviews:
        gt = rev / "corrected" / "obj_train_data"
        if not gt.exists():
            print(f"  [!] {gt} not found yet -- skipping {rev.name} (run its CVAT export first)")
            continue
        for p in sorted((rev / "images").glob("*.jpg")):
            lbl = gt / f"{p.stem}.txt"
            if lbl.exists():
                rev_items.append((p, lbl))
    if not rev_items:
        raise SystemExit("No corrected frames found in any review batch yet.")

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
        for img, lbl in rev_items:
            if _vid(img) in vids:
                shutil.copy2(img, d / "images" / "train" / img.name)
                shutil.copy2(lbl, d / "labels" / "train" / f"{img.stem}.txt")
                n_rev += 1
        for p in monitor:
            shutil.copy2(p, d / "images" / "val" / p.name)
            shutil.copy2(base / "labels" / "train" / f"{p.stem}.txt", d / "labels" / "val" / f"{p.stem}.txt")
        with open(d / "dataset.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"path": str(d.resolve()), "train": "images/train", "val": "images/val",
                            "nc": len(names), "names": names}, f, sort_keys=False)
        print(f"{tag:>4}: train {len(orig)} original + {n_rev} corrected (both batches) = "
              f"{len(orig) + n_rev} | val {len(monitor)} (monitor only)")


if __name__ == "__main__":
    build()
