"""
Score detector weights on the hand-corrected review frames and report whether the
retrained detector beats the current one.

    python -m src.eval_weights --baseline <best.pt> --foldA <A/last.pt> --foldB <B/last.pt> \
                               [--full <full/last.pt>] \
                               [--review-dirs data/annotation_review,data/annotation_review2]

Same settings for every model (pipeline settings): conf 0.35, iou 0.45, imgsz 640,
oversize-box cap from the config, IoU >= 0.5 matching.  Fold model A is scored on the
group-B videos and fold model B on group-A videos (see src/build_cv_datasets.py), so
the pooled fold score is a held-out score.  The `full` model has seen every review
frame; its score is reported only as an in-sample sanity check.

--review-dirs pools every corrected batch that exists (frame stems are unique across
batches by construction -- see build_review_set.py's --exclude), which matters most
for videos 18/23: batch 1 alone has only 3 frames of each, too few to trust a
precision/recall number; with batch 2 folded in that becomes ~18 each.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.build_cv_datasets import GROUP_A, GROUP_B
from src.eval_detector import _load, match
from src.head_filter import filter_heads, max_box_px

HARD = {"18", "23"}


def _index(review_dirs: list[Path]) -> dict[str, Path]:
    """{frame_stem: review_dir} for every frame that has a corrected label."""
    name_dir = {}
    for d in review_dirs:
        gt = d / "corrected" / "obj_train_data"
        if not gt.exists():
            print(f"  [!] {gt} not found -- skipping {d}")
            continue
        for p in gt.glob("*.txt"):
            if (d / "images" / f"{p.stem}.jpg").exists():
                name_dir[p.stem] = d
    return name_dir


def predict(weights: str, names: list[str], cfg: dict, name_dir: dict[str, Path]) -> dict[str, np.ndarray]:
    from ultralytics import YOLO
    head = cfg["segmentation"]["head"]
    model = YOLO(weights)
    limit = max_box_px(cfg)
    out = {}
    for n in names:
        im = cv2.imread(str(name_dir[n] / "images" / f"{n}.jpg"))
        H, W = im.shape[:2]
        r = model(im, conf=head["conf_threshold"], iou=head["iou_threshold"],
                  imgsz=head["imgsz"], verbose=False, max_det=1000)[0]
        b = r.boxes.xyxyn.cpu().numpy()
        dets = [{"bbox": [x1 * W, y1 * H, x2 * W, y2 * H]} for x1, y1, x2, y2 in b]
        kept = filter_heads({"f": dets}, limit)["f"]
        out[n] = np.array([[d["bbox"][0] / W, d["bbox"][1] / H, d["bbox"][2] / W, d["bbox"][3] / H]
                           for d in kept]).reshape(-1, 4)
    return out


def score(preds: dict[str, np.ndarray], names: list[str], name_dir: dict[str, Path]) -> dict[str, np.ndarray]:
    per = {}
    for n in names:
        g = _load(name_dir[n] / "corrected" / "obj_train_data" / f"{n}.txt")
        per[n] = np.array(match(preds[n], g, 0.5) + (len(preds[n]), len(g)), dtype=float)
    return per          # tp, fp, fn, n_pred, n_gt


def _fmt(label: str, per: dict, pick) -> str:
    a = sum((v for k, v in per.items() if pick(k.split("_")[0])), np.zeros(5))
    tp, fp, fn, npred, ngt = a
    if tp + fp + fn == 0:
        return f"{label:<34} (no frames)"
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    F = 2 * P * R / max(P + R, 1e-12)
    return (f"{label:<34} P {P:.3f}  R {R:.3f}  F1 {F:.3f} | pred {int(npred):>5} true {int(ngt):>5} "
            f"({(npred - ngt) / max(ngt, 1) * 100:+5.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--foldA", required=True, help="trained with group-A corrections; tested on group B")
    ap.add_argument("--foldB", required=True, help="trained with group-B corrections; tested on group A")
    ap.add_argument("--full", default=None)
    ap.add_argument("--review-dirs", default="data/annotation_review",
                    help="comma-separated review batches to pool (each needs corrected/obj_train_data)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    review_dirs = [Path(p) for p in a.review_dirs.split(",")]
    name_dir = _index(review_dirs)
    names = sorted(name_dir)
    if not names:
        raise SystemExit(f"No corrected frames found in {review_dirs}")
    A = [n for n in names if n.split("_")[0] in GROUP_A]
    B = [n for n in names if n.split("_")[0] in GROUP_B]

    base = score(predict(a.baseline, names, cfg, name_dir), names, name_dir)
    pa = predict(a.foldA, B, cfg, name_dir)
    pb = predict(a.foldB, A, cfg, name_dir)
    cv = {**score(pa, B, name_dir), **score(pb, A, name_dir)}   # held-out predictions only

    normal = lambda v: v not in HARD
    hard = lambda v: v in HARD
    everything = lambda v: True
    print(f"\n{len(names)} corrected frames | conf {cfg['segmentation']['head']['conf_threshold']} | "
          f"size cap {max_box_px(cfg)} px | IoU>=0.5\n")
    for title, sel in (("ALL 15 videos", everything), ("13 normal videos", normal), ("videos 18 + 23 (hard)", hard)):
        print(f"== {title}")
        print(_fmt("  current detector (baseline)", base, sel))
        print(_fmt("  retrained, HELD-OUT (2-fold CV)", cv, sel))
    if a.full:
        full = score(predict(a.full, names, cfg, name_dir), names, name_dir)
        print("\n== deployable model trained on everything -- IN-SAMPLE, not a fair score")
        print(_fmt("  full model (has seen these)", full, everything))
    d = lambda per, sel: (lambda x: 2 * x[0] / max(2 * x[0] + x[1] + x[2], 1))(
        sum((v for k, v in per.items() if sel(k.split('_')[0])), np.zeros(5)))
    print(f"\nF1 change on ALL: {d(cv, everything) - d(base, everything):+.3f} | "
          f"normal: {d(cv, normal) - d(base, normal):+.3f} | hard: {d(cv, hard) - d(base, hard):+.3f}")


if __name__ == "__main__":
    main()
