# QYX_MedTech — CASA Training Pipeline

Computer-Aided Sperm Analysis (CASA) system using YOLOv8 detection,
ByteTrack multi-object tracking, and WHO 6th edition motility classification.

## Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Stage 1    │    │   Stage 2    │    │     Stage 3      │    │     Stage 4      │
│  Extract    │───▶│  CVAT→YOLO   │───▶│  Dataset Split   │───▶│  Train YOLOv8    │
│  Frames     │    │  Conversion  │    │  + Augmentation   │    │  Detector        │
└─────────────┘    └──────────────┘    └──────────────────┘    └──────────────────┘
                                                                        │
┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌───────▼──────────┐
│  Stage 9    │    │   Stage 8    │    │     Stage 7      │    │     Stage 5      │
│  Report     │◀───│  Statistical │◀───│  WHO Motility    │◀───│  ByteTrack       │
│  Generation │    │  Aggregation │    │  Classification   │    │  Tracking        │
└─────────────┘    └──────────────┘    └──────────────────┘    └──────────────────┘
                                                                        │
                                                               ┌───────▼──────────┐
                                                               │     Stage 6      │
                                                               │  Kinematic       │
                                                               │  Analysis        │
                                                               └──────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
cd training
pip install -r requirements.txt

# 2. Extract frames from videos
python -m src.stage1_extract_frames --config configs/default.yaml

# 3. Annotate frames in CVAT (draw bounding boxes around sperm cells)
#    Export annotations as CVAT XML

# 4. Convert annotations to YOLO format
python -m src.stage2_cvat_to_yolo --config configs/default.yaml

# 5. Prepare dataset (train/val split)
python -m src.stage3_prepare_dataset --config configs/default.yaml

# 6. Train the sperm detection model
python -m src.stage4_train_detector --config configs/default.yaml

# 7. Run inference pipeline on a video
python -m src.run_pipeline --config configs/default.yaml --infer --video data/vid/1.mp4

# 8. Extract ground truth from lab reports & validate
python -m src.run_pipeline --config configs/default.yaml --validate
```

## Pipeline Modes

```bash
# Data preparation (Stages 1-3)
python -m src.run_pipeline --prepare

# Training (Stage 4)
python -m src.run_pipeline --train
python -m src.run_pipeline --train --resume    # resume from checkpoint

# Inference on a video (Stages 5-9)
python -m src.run_pipeline --infer --video data/vid/1.mp4

# Single stage
python -m src.run_pipeline --stage 6 --video-id 1

# Validation (OCR + comparison)
python -m src.run_pipeline --validate

# Full pipeline
python -m src.run_pipeline --all
```

## CASA Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| VCL | Curvilinear Velocity | µm/s |
| VSL | Straight Line Velocity | µm/s |
| VAP | Average Path Velocity | µm/s |
| LIN | Linearity (VSL/VCL) | % |
| STR | Straightness (VSL/VAP) | % |
| WOB | Wobble (VAP/VCL) | % |
| ALH | Amplitude of Lateral Head Displacement | µm |
| BCF | Beat Cross Frequency | Hz |

## Motility Classification

Matches the lab report, where `progressive = rapid + slow` (checked for every video in
`data/ground_truth_verified.json`). Thresholds live in `configs/default.yaml`.

| Category | Criteria |
|----------|----------|
| Progressive (PR) | VAP ≥ 5 µm/s AND STR ≥ 20% (LIN gate disabled) |
| ↳ Rapid (a) | PR and VAP ≥ 25 µm/s |
| ↳ Slow (b) | PR and VAP < 25 µm/s |
| Non-Progressive (NP) | VAP ≥ 5 µm/s but not PR |
| Immotile (IM) | VAP < 5 µm/s |

WHO 6th Ed. defines PR as active movement "linear or in a large circle, regardless of
speed". It has no STR ≥ 80% rule (that is a CASA-vendor convention), and VAP ≥ 25 µm/s is
only the *rapid* class. Using either as the PR test put predicted PR 26–38 pp below the lab.

## Frame QC (`src/frame_qc.py`)

The videos are clips separated by ~10 black frames, and 21 of 23 end in a long *frozen
still* (a different field whose sperm never move). Tracking it produced a zero-velocity
"immotile" track for every sperm. Dark and frozen segments are now dropped before tracking
and before the concentration average; each live clip is tracked separately so nothing
links across a cut. Inspect the verdicts with `python -m src.frame_qc`; results are cached
in `data/frames/<id>/frame_qc.json`. Disable with `frame_qc.enabled: false`.

## Known open issue: `pixel_to_micron`

Still unverified (0.45 is a placeholder). The lab-derived estimates disagree: motile
velocities imply ≈ 0.7 µm/px, concentration (10 µm chamber) implies ≈ 0.4. Measure it with
a stage micrometer; do not fit it to the lab report. Concentration and all velocities
scale with it.

## Project Structure

```
training/
├── configs/
│   └── default.yaml              # CASA pipeline configuration
├── data/
│   ├── vid/                      # Input microscopy videos
│   ├── report/                   # Lab report images (ground truth)
│   ├── frames/                   # Extracted video frames
│   ├── annotations/              # CVAT XML exports
│   └── yolo_dataset/             # Prepared YOLO dataset
│       ├── images/{train,val}/
│       ├── labels/{train,val}/
│       └── dataset.yaml
├── outputs/
│   ├── detection/                # YOLO training outputs
│   ├── tracks/                   # Tracking results (JSON)
│   ├── kinematics/               # Kinematic profiles (JSON)
│   ├── classifications/          # Motility classifications (JSON)
│   ├── results/                  # Aggregated summaries (JSON)
│   ├── reports/                  # Generated HTML reports
│   └── validation/               # Validation results + plots
├── requirements.txt
└── src/
    ├── frame_qc.py               # Dark-gap / frozen-still detection (run before tracking)
    ├── stage1_extract_frames.py  # Video → frames
    ├── stage2_cvat_to_yolo.py    # CVAT XML → YOLO labels
    ├── stage3_prepare_dataset.py # Train/val split
    ├── stage4_train_detector.py  # YOLOv8 training
    ├── stage5_track.py           # ByteTrack tracking
    ├── stage6_kinematics.py      # CASA parameters
    ├── stage7_classify.py        # WHO classification
    ├── stage8_aggregate.py       # Video-level stats
    ├── stage9_report.py          # HTML report generation
    ├── ocr_report.py             # OCR ground truth extraction
    ├── validate.py               # Prediction vs lab comparison
    └── run_pipeline.py           # Pipeline orchestrator
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Annotation | CVAT |
| Detection | YOLOv8n (ultralytics) |
| Tracking | ByteTrack (custom implementation) |
| Kinematics | NumPy, SciPy |
| Classification | WHO 6th edition rules |
| OCR | Tesseract (pytesseract) |
| Visualization | Plotly |
| Validation | MAE, Pearson r, Bland-Altman |

## Validation Strategy

Lab reports (JPG images) are parsed via OCR to extract reference values.
System predictions are compared against these using:
- **MAE** (Mean Absolute Error)
- **Percentage Difference**
- **Pearson Correlation**
- **Bland-Altman Agreement**

