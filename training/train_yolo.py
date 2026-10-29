"""
Fine-tune YOLOv8 on the sperm detection dataset, then validate.

Dataset: data/yolo_dataset  (Sperm / Cluster / Pin-head, 347 frames, 80/20 split)
Base:    yolov8s.pt          (small model — better capacity for dense microscopy)
         Use yolov8n.pt only if GPU memory < 4 GB.
Output:  outputs/detection/yolo_trained/

RC-9 FIX: The original script used yolov8n (nano, 3.2 M params).  Dense
sperm microscopy with 50-200 overlapping heads per frame exhausts the nano
model's capacity, causing a hard mAP ceiling (~87%).  yolov8s (11 M params)
provides ~20% mAP improvement on crowded small-object datasets with minimal
extra training time on a modern GPU.

Two ways to use a new batch of corrected frames -- pick one, don't mix:

  FRESH RETRAIN (--model yolov8n.pt/yolov8s.pt, a COCO checkpoint): relearns
  everything from the combined dataset.  Needs a dataset where the new frames
  are a healthy fraction of the whole, or the new examples get outweighed by
  epochs of re-deriving what the old data already taught it.  This is what
  produced a WORSE, held-out-tested detector from 18-22 correction frames
  against 212 original ones (F1 0.940 -> 0.898) -- see src/eval_weights.py.

  FINE-TUNE (--model <existing best.pt>, --lr0 1e-4, fewer epochs e.g. 20-30,
  --patience on): starts from a detector that already works and nudges it
  toward the new examples instead of re-deriving everything at full COCO-init
  learning rate.  Prefer this whenever the new batch is small relative to what
  the current weights were already trained on.

Run:
    cd training
    python train_yolo.py
    python train_yolo.py --epochs 100 --batch 8 --model yolov8s.pt

    # fine-tune the deployed detector on a new correction batch
    python train_yolo.py --model outputs/detection/yolo_trained/run/weights/best.pt \\
        --data data/yolo_cv/full --epochs 25 --lr0 0.0001 --patience 8 --name finetune
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

# absolute so ultralytics never prepends runs/detect/ to a relative project path
ROOT = Path(__file__).parent


def _sync_labels(dataset_root: Path) -> None:
    """Copy flat labels/ into labels/train/ + labels/val/ — YOLO expects mirrored structure."""
    for split in ("train", "val"):
        lbl_dir = dataset_root / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img in (dataset_root / "images" / split).glob("*.jpg"):
            src = dataset_root / "labels" / img.with_suffix(".txt").name
            dst = lbl_dir / img.with_suffix(".txt").name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
    print("Labels synced → labels/train/ and labels/val/")


def train(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Run: pip install ultralytics")

    dataset_root = Path(args.data) if Path(args.data).is_absolute() else ROOT / args.data
    if not dataset_root.exists():
        raise SystemExit(f"Dataset not found: {dataset_root}")
    _sync_labels(dataset_root)

    output_dir = ROOT / "outputs" / "detection" / args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(ROOT / args.model if not Path(args.model).is_absolute() else Path(args.model))
    model.train(
        data=str(dataset_root / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(output_dir),  # must be absolute to prevent runs/detect/ prefix
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=min(3, max(args.epochs // 4, 0)),  # don't warm up past the whole run on a short fine-tune
        patience=args.patience,   # early stopping: 0 = disabled
        # ----------------------------------------------------------------
        # Augmentation suite — tuned for dense sperm microscopy
        # ----------------------------------------------------------------
        mosaic=1.0,           # combine 4 images — key for crowded-scene generalisation
        copy_paste=0.3,       # RC-9 ADD: copies pasted objects, critical for overlapping
                              # sperm heads (dense clusters / pin-heads touching each other)
        mixup=0.1,            # RC-9 ADD: blends two images — helps with concentration
                              # variation across samples (9-275 M/mL range in ground truth)
        scale=0.5,            # random zoom in/out (handles FOV magnification variation)
        translate=0.1,        # RC-9 ADD: random shift — handles sperm at FOV edges
        degrees=10,           # rotation (sperm at any angle in the microscopy plane)
        flipud=0.5,           # vertical flip (microscopy has no canonical up-direction)
        fliplr=0.5,           # horizontal flip
        # HSV jitter — microscopy images vary with staining batch and exposure
        hsv_h=0.01,           # RC-9 ADD: small hue shift (CASA systems use visible light)
        hsv_s=0.3,            # RC-9 ADD: saturation jitter (different stains)
        hsv_v=0.3,            # RC-9 ADD: brightness jitter (condenser/illumination drift)
        # Loss weights — near-single-class detection; reduce cls penalty
        cls=0.3,              # RC-9 ADD: Sperm/Cluster/Pin-head — detection matters more
                              # than class label; lower cls weight prioritises box quality
        # Logging
        save_period=10,       # save checkpoint every N epochs (allows post-hoc review)
        plots=True,           # save training plots (loss curves, PR curves) to project dir
    )

    best = output_dir / args.name / "weights" / "best.pt"
    if best.exists():
        print(f"\nValidating {best}")
        val_result = YOLO(str(best)).val(
            data=str(dataset_root / "dataset.yaml"), imgsz=args.imgsz
        )
        print(f"  mAP50:    {val_result.box.map50:.4f}")
        print(f"  mAP50-95: {val_result.box.map:.4f}")
        print(f"  Precision:{val_result.box.mp:.4f}")
        print(f"  Recall:   {val_result.box.mr:.4f}")
    print(f"\nDone. Weights: {best}")
    print(f"Update configs/default.yaml -> segmentation.head.weights: \"{best}\"")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train YOLOv8 sperm detector")
    p.add_argument("--model",    default="yolov8s.pt",
                   help="Base weights file. Use yolov8n.pt if GPU memory < 4 GB.")
    p.add_argument("--epochs",   type=int, default=100,
                   help="Training epochs. 100 is recommended for yolov8s.")
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16,
                   help="Batch size. Lower to 8 if OOM on a 4 GB GPU.")
    p.add_argument("--workers",  type=int, default=4)
    p.add_argument("--data",     default="data/yolo_dataset",
                   help="Dataset root (dataset.yaml inside). Use data/yolo_holdout for a leak-free "
                        "video-level split (see src/build_yolo_dataset.py).")
    p.add_argument("--out",      default="yolo_trained", help="Folder under outputs/detection/")
    p.add_argument("--name",     default="run", help="Run name inside --out")
    p.add_argument("--patience", type=int, default=20,
                   help="Early-stopping patience (epochs without mAP50 improvement). 0=off. "
                        "Keep this on unless you specifically need every fold trained for an "
                        "identical epoch count (e.g. comparing architectures) -- disabling it is what let "
                        "a fresh-retrain fine-tune drift to a worse, held-out-tested model. See top of file.")
    p.add_argument("--lr0",      type=float, default=1e-3,
                   help="Initial LR. Use ~1e-4 when --model is an already-trained checkpoint "
                        "(fine-tune), not a fresh COCO one -- 1e-3 is tuned for training from scratch "
                        "and can overwrite what the checkpoint already learned.")
    p.add_argument("--lrf",      type=float, default=0.01, help="Final LR as a fraction of --lr0.")
    train(p.parse_args())
