# QYX MedTech — CASA Pipeline (training/)

Computer-Aided Sperm Analysis for LensHooke analyzer videos: YOLO head detection,
Frangi/U-Net tail segmentation, trackpy tracking, CASA kinematics and WHO 6th-edition
motility classification. This folder holds the pipeline the website runs, plus the
scripts used to train the models and validate them against the lab reports.

## Pipeline

```
 video
   │
   ▼
 stage1_extract_frames   frames + capture metadata
   │
   ▼
 frame_qc                drop black gaps and frozen stills, split into live clips
   │
   ▼
 seg_head                YOLO sperm-head detector  → <id>_heads.json
   │      (head_filter drops oversized boxes: mucus, debris, clumps)
   │
   ├──────────────────────────────────────┐
   │                                      ▼
   │                               seg_tail   (optional)
   │                                 Phase A  Frangi ridge filter → bootstrap tail masks
   │                                 Phase B  U-Net (ResNet-34) trained on those masks
   │                                      │
   ▼                                      │
 seg_track               trackpy linking → per-sperm trajectories
   │                                      │
   ▼                                      │
 seg_kinematics          VCL VSL VAP LIN STR WOB, ALH/BCF from the head path
   │                                      ▼
   │                               seg_beat   (optional)
   │                                 skeletonise tail → FFT → ALH_real, BCF_real, waveform
   ▼
 stage7_classify         WHO PR (rapid / slow) / NP / IM
   │
   ▼
 stage8_aggregate        video-level summary + concentration
   │
   ▼
 seg_report              HTML report in the lab's layout (CLI only)
```

### What the website runs

The backend (`backend/app/services/casa.py`) runs the **head-only path**:
frames → `seg_head` → `seg_track` → `seg_kinematics` → `stage7_classify` →
`stage8_aggregate`, then draws the trajectory overlay. ALH and BCF on the website come
from the head trajectory (lateral deviation from the smoothed path), not from the tail.

The **tail branch** (`seg_tail` + `seg_beat`) is research code run from the command line.
It measures the flagellum directly (true beat amplitude/frequency and the waveform) but is
not yet validated against the lab, and it adds a lot of GPU time per video.

## Setup

```bash
pip install -r requirements.txt        # from training/, inside the repo's .venv
```

Requires a CUDA GPU for reasonable speed (the U-Net trains with `unet_batch_size: 4` on a
4 GB card). `pytesseract` needs the Tesseract binary only for `--ocr`.

Data that is **not in the repo** (git-ignored) must be copied in before running the CLI:

| Path | Contents |
|---|---|
| `data/vid/` | Analyzer videos `1.mp4` … `23.mp4` |
| `outputs/seg/tail/unet_tail.pt` | Trained tail U-Net (98 MB, not distributed; train it as in [Tail segmentation](#tail-segmentation-u-net)) |

Everything below is run from `training/`.

## Running the pipeline

```bash
# One video, full pipeline (head + tail; tail uses the U-Net if its weights exist)
python -m src.run_pipeline --video data/vid/1.mp4

# All videos
python -m src.run_pipeline --all

# Head-only, the same path as the website (faster)
python -m src.run_pipeline --all --no-tail

# Re-run from a stage on already-extracted frames
python -m src.run_pipeline --video-id 1 --from-stage track
```

Stages for `--from-stage`: `head tail track kinematics beat classify aggregate report`.
Outputs go to `outputs/seg/{head,tail,tracks,kinematics,classifications,results,reports}/`.

`src.seg_pipeline` is the same orchestrator without frame extraction / OCR / validation
(`python -m src.seg_pipeline --video-id 1`).

## Tail segmentation (U-Net)

`src/seg_tail.py` builds tail masks in two phases, because there are no hand-drawn tail
labels:

1. **Bootstrap (Phase A).** A Frangi ridge filter (`ridge_filter`, `ridge_sigmas`,
   `ridge_threshold`) picks out thin flagellar strands. Each strand is linked to the
   nearest detected head within `max_link_dist_px`. This gives noisy but free labels:
   `outputs/seg/tail/<id>_tails_bootstrap.npz`.
2. **U-Net (Phase B).** A `segmentation-models-pytorch` U-Net (ResNet-34 encoder,
   ImageNet init, grayscale input) is trained on the bootstrap masks, then run on every
   frame to give cleaner masks: `<id>_tails.npz`. Weights: `outputs/seg/tail/unet_tail.pt`.

```bash
# 1. Bootstrap masks for every video (heads must exist first)
python -m src.run_pipeline --all --from-stage tail --tail-phase bootstrap

#    Check outputs/seg/tail/ before training: the U-Net learns whatever the bootstrap got wrong.

# 2. Train the U-Net on the bootstrap masks
python -m src.run_pipeline --tail-train

# 3. Re-run tails with the trained U-Net, then beat analysis onward
python -m src.run_pipeline --all --from-stage tail --tail-phase infer
```

Settings live under `segmentation.tail` in `configs/default.yaml` (`unet_encoder`,
`unet_epochs`, `unet_batch_size`, `unet_lr`, `device`). Once `unet_tail.pt` exists, the
default `--tail-phase bootstrap` also picks it up automatically.

`src/seg_beat.py` then skeletonises each sperm's tail mask, samples
`beat.arc_sample_points` points along it and takes an FFT over time to get `ALH_real`,
`BCF_real` and the per-track waveform.

## Head detector

The deployed detector is `weights/sperm_detector.pt` (set in `segmentation.head.weights`).
Classes: Sperm / Cluster / Pin-head.

```bash
# Rebuild a leak-free dataset (split by video, one frame per frozen still)
python -m src.build_yolo_dataset --holdout 3,8 --out data/yolo_holdout

# Train (fresh from a COCO checkpoint) …
python train_yolo.py --data data/yolo_holdout --model yolov8s.pt --epochs 100 --batch 8

# … or fine-tune the deployed model on new corrected frames (low LR, few epochs)
python train_yolo.py --model weights/sperm_detector.pt --lr0 1e-4 --epochs 25
```

See the docstring at the top of `train_yolo.py` for when to use which; don't mix them.
Runs are written to `outputs/detection/<out>/<name>/`; copy the chosen `last.pt` to
`weights/` and point the config at it.

### Measuring detector accuracy

Detector mAP on auto-generated labels only measures agreement with the auto-annotator, so
accuracy is measured on hand-corrected frames from videos the detector never trained on:

| Script | Purpose |
|---|---|
| `src.build_review_set` | Pick frames from unseen videos, pre-label them, package for CVAT |
| `src.eval_detector` | Score boxes against the corrected labels (precision, recall, F1, count error) |
| `src.build_cv_datasets` / `build_cv_datasets2` | Two-fold, video-split datasets that fold in the corrections |
| `src.eval_weights` | Compare the deployed weights with retrained fold models on held-out review frames |

The hand-corrected test set is in `data/annotation_review/corrected/` (committed).

## Validation against the lab

```bash
python -m src.run_pipeline --ocr                    # (re)transcribe data/report/*.jpg
python -m src.validate --split test                 # held-out videos only
python -m src.validate --split dev
```

`data/ground_truth_verified.json` (hand-checked) is authoritative; OCR output never
overwrites it. `data/video_split.json` fixes the dev/test split: thresholds are tuned on
dev, and only test numbers are reported as accuracy. Metrics: MAE, percentage difference,
Pearson r, ICC, Bland-Altman.

## CASA parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| VCL | Curvilinear velocity | µm/s |
| VSL | Straight-line velocity | µm/s |
| VAP | Average path velocity | µm/s |
| LIN | Linearity (VSL/VCL) | % |
| STR | Straightness (VSL/VAP) | % |
| WOB | Wobble (VAP/VCL) | % |
| ALH | Amplitude of lateral head displacement | µm |
| BCF | Beat-cross frequency | Hz |

## Motility classification

Matches the lab report, where `progressive = rapid + slow` (checked for every video in
`data/ground_truth_verified.json`). Thresholds live under `classification` in
`configs/default.yaml`; current values:

| Category | Criteria |
|----------|----------|
| Progressive (PR) | VAP ≥ 3 µm/s AND STR ≥ 20% (LIN gate disabled) |
| ↳ Rapid (a) | PR and VAP ≥ 25 µm/s |
| ↳ Slow (b) | PR and VAP < 25 µm/s |
| Non-progressive (NP) | VAP ≥ 3 µm/s but not PR |
| Immotile (IM) | VAP < 3 µm/s |

WHO 6th Ed. defines PR as active movement "linear or in a large circle, regardless of
speed". It has no STR ≥ 80% rule (that is a CASA-vendor convention), and VAP ≥ 25 µm/s is
only the *rapid* class. Using either as the PR test put predicted PR 26–38 pp below the lab.

## Calibration

Two values are not published for the analyzer and are fitted on the **dev** videos only:

- `kinematics.capture_fps` — exported videos play at 30 fps; the real capture rate (19)
  was fitted. Replace it with the spec value if the manufacturer provides one.
- `concentration.calibration_factor` — the cassette's effective counting volume.

`kinematics.pixel_to_micron` (currently 0.5) is **unverified**. Lab-derived estimates
disagree (motile velocities imply ≈ 0.7 µm/px, concentration with a 10 µm chamber implies
≈ 0.4). Measure it with a stage micrometer rather than fitting it to the lab report;
concentration and all velocities scale with it.

## Frame QC (`src/frame_qc.py`)

The videos are clips separated by ~10 black frames, and 21 of 23 end in a long *frozen
still* (a different field whose sperm never move). Tracking it produced a zero-velocity
"immotile" track for every sperm. Dark and frozen segments are dropped before tracking
and before the concentration average; each live clip is tracked separately so nothing
links across a cut. Inspect the verdicts with `python -m src.frame_qc`; results are cached
in `data/frames/<id>/frame_qc.json`. Disable with `frame_qc.enabled: false`.

## Layout

```
training/
├── configs/default.yaml        every pipeline setting (also used by the website)
├── weights/sperm_detector.pt   deployed head detector
├── train_yolo.py               detector training / fine-tuning
├── data/
│   ├── report/                 lab report images (ground truth source)
│   ├── ground_truth*.json      transcribed lab values (verified = authoritative)
│   ├── video_split.json        dev/test split
│   ├── annotation_review*/     detector review sets (corrected/ = hand-corrected test set)
│   └── vid/ frames/ yolo_*/    videos, frames, generated datasets  (git-ignored)
├── outputs/                    all pipeline / training outputs     (git-ignored)
├── src/
│   ├── run_pipeline.py         CLI entry point (extract, run, OCR, validate, tail training)
│   ├── seg_pipeline.py         stage orchestrator
│   ├── stage1_extract_frames.py
│   ├── frame_qc.py             black-gap / frozen-still detection
│   ├── seg_head.py             YOLO head detection
│   ├── head_filter.py          oversize-box filter
│   ├── seg_tail.py             Frangi bootstrap + U-Net tail segmentation
│   ├── seg_track.py            trackpy tracking
│   ├── seg_kinematics.py       CASA velocities, ALH/BCF from head path
│   ├── seg_beat.py             flagellar beat analysis from tail masks
│   ├── stage7_classify.py      WHO motility classes
│   ├── stage8_aggregate.py     video-level statistics, concentration
│   ├── seg_report.py           HTML report + trajectory overlay
│   ├── ocr_report.py           lab report OCR
│   ├── validate.py             predictions vs lab
│   ├── build_yolo_dataset.py   leak-free detector dataset
│   ├── build_review_set.py     pre-labelled review frames for CVAT
│   ├── build_cv_datasets*.py   cross-validation datasets with corrections
│   ├── eval_detector.py        score boxes vs corrected labels
│   └── eval_weights.py         compare detector weights on review frames
└── tests/test_casa.py          kinematics and classification rules
```

## Tests

```bash
pytest tests
```

Run `pytest tests`, not bare `pytest`: a bare run also collects `scratch/` (git-ignored
dev scripts), which fail to import.

## Future work: self-learning detector

Detector errors cause most of the remaining error (missed heads in viscous samples such as
videos 18 and 23; false detections on mucus and debris), and every new video the website
analyses is unlabelled training data. The plan is to automate the loop that is currently
done by hand (`build_review_set` → correct in CVAT → `train_yolo.py` fine-tune →
`eval_weights`):

1. **Collect.** After each website analysis, keep the detector's boxes and confidences for
   a sample of live frames (frames that passed `frame_qc`).
2. **Pick frames to review (active learning).** Queue the frames the model is least sure
   about: many boxes near `conf_threshold`, head counts that change sharply between
   frames, or quality warnings on the analysis. A reviewer corrects them in the app or in
   CVAT; these become new hand-corrected labels.
3. **Pseudo-label the rest (semi-supervised).** Frames where the detector is confident and
   its boxes agree with the tracker (they belong to long, consistent tracks) are used as
   pseudo-labels with a lower training weight.
4. **Fine-tune.** Periodically fine-tune from the deployed weights (low `--lr0`, few epochs,
   as in `train_yolo.py`'s fine-tune mode). Fine-tuning keeps what the model has already
   learned instead of retraining from scratch on a dataset dominated by new frames.
5. **Gate before deploying.** Score the candidate with `eval_weights` on the hand-corrected
   test set and re-run `validate --split test`. Promote it to `weights/` only if it beats
   the current model on both; otherwise keep the current one. Keep every past model
   version so it can be rolled back.

Constraints:

- **No silent model changes.** Results must be reproducible and comparable over time, so
  each analysis records the detector version it used, and a reviewer approves each
  promotion.
- **Test data stays held out.** Frames from the test-split videos and the corrected test
  set are never used for training.
- **Consent first.** Using patient videos for training needs consent and data-handling
  rules before any of this is switched on.

## Technology

| Component | Technology |
|-----------|------------|
| Head detection | YOLO (ultralytics) |
| Tail segmentation | Frangi ridge filter (scikit-image) → U-Net (segmentation-models-pytorch) |
| Tracking | trackpy |
| Kinematics / beat | NumPy, SciPy (FFT) |
| Classification | WHO 6th edition rules |
| Annotation | CVAT (YOLO 1.1 format) |
| OCR | Tesseract (pytesseract) |
| Reports / plots | Plotly, HTML |
