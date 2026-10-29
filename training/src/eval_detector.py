"""
Score detector boxes against hand-corrected boxes (YOLO txt format).

    python -m src.eval_detector --pred data/annotation_review/prelabels \
                                --gt   data/annotation_review/corrected/obj_train_data

Greedy one-to-one matching by IoU (highest first).  Reports precision, recall, F1
and the count error (predicted vs true heads per frame), overall and per video.
Note precision is only meaningful if the reviewer actually deleted wrong boxes:
the pre-labels are the detector's own output.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(p: Path) -> np.ndarray:
    if not p.exists():
        return np.zeros((0, 4))
    rows = [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
    a = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4)
    out = np.empty_like(a)                       # cx,cy,w,h -> x1,y1,x2,y2
    out[:, 0], out[:, 1] = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    out[:, 2], out[:, 3] = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def match(pred: np.ndarray, gt: np.ndarray, thr: float) -> tuple[int, int, int]:
    """(tp, fp, fn) with greedy one-to-one matching at IoU >= thr."""
    iou = _iou(pred, gt)
    tp = 0
    used_p, used_g = set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = divmod(int(flat), max(iou.shape[1], 1))
        if iou[i, j] < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    gt_dir, pred_dir = Path(a.gt), Path(a.pred)
    names = sorted(p.stem for p in gt_dir.glob("*.txt"))
    if not names:
        raise SystemExit(f"No ground-truth labels in {gt_dir}")

    tot = defaultdict(lambda: np.zeros(5))       # tp, fp, fn, n_pred, n_gt
    for n in names:
        p, g = _load(pred_dir / f"{n}.txt"), _load(gt_dir / f"{n}.txt")
        tp, fp, fn = match(p, g, a.iou)
        v = n.split("_")[0]
        for key in ("ALL", v):
            tot[key] += [tp, fp, fn, len(p), len(g)]

    def row(k: str) -> str:
        tp, fp, fn, npred, ngt = tot[k]
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); F = 2 * P * R / max(P + R, 1e-12)
        return (f"{k:>5} | P {P:5.3f}  R {R:5.3f}  F1 {F:5.3f} | "
                f"true {int(ngt):>5} pred {int(npred):>5} ({(npred - ngt) / max(ngt, 1) * 100:+5.1f}%) "
                f"| missed {int(fn):>4} false {int(fp):>4}")

    print(f"IoU >= {a.iou} | {len(names)} frames")
    for k in sorted([k for k in tot if k != 'ALL'], key=int):
        print(row(k))
    print("-" * 100)
    print(row("ALL"))


if __name__ == "__main__":
    main()
