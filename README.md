# QYX MedTech — Semen Analysis

Computer-aided semen analysis (CASA) for videos recorded by the LensHooke analyzer.
Upload a video in the browser and get concentration, WHO motility classes
(rapid/slow progressive, non-progressive, immotile), swimming speed (VCL/VSL/VAP),
path shape (LIN/STR/WOB), ALH and BCF, checked against WHO reference values.

> **Research use only. Not a diagnostic device.** The analyzer's capture rate and
> counting-chamber volume are not published, so two calibration values are estimated
> from 21 lab-verified samples (see *Model accuracy* in the app).

## Run it

Needs Python 3.11+ (an NVIDIA GPU with CUDA is recommended) and Node 18+.

First-time setup after cloning:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r training/requirements.txt -r backend/requirements.txt
```

Then:

```bat
start.bat
```

That builds the website the first time, starts the server and opens
http://localhost:8000. Close the window to stop it. To rebuild after changing the
frontend: `set REBUILD=1` then `start.bat`.

Manual equivalent:

```bash
pip install -r training/requirements.txt -r backend/requirements.txt
cd frontend && npm install && npm run build && cd ..
cd backend && uvicorn app.main:app --port 8000
```

For frontend development, run the backend as above and `npm run dev` in `frontend/`
(http://localhost:5173, API calls are proxied to port 8000).

## What the website does

| Page | |
|---|---|
| Dashboard | Counts, averages, samples below a WHO reference, recent analyses |
| New analysis | Upload a video with sample ID, optional patient reference, semen volume and notes |
| Analysis report | Live progress while it runs; then the lab-style report: key results, WHO table, motility breakdown, tracked-sperm image, the video, speed and path statistics with histograms, quality warnings. Printable. |
| Analyses | Search, filter by status, open or delete past analyses |
| Model accuracy | Agreement with the lab analyzer per measurement, lab-vs-ours scatter plots, settings in use |

Analyses run one at a time on the GPU (about 20–60 s per video). Each is stored in
`backend/data/analyses/<id>/` (video, results, config used, pipeline log). That
folder holds patient data and is git-ignored.

## Repository structure

```
QYX_MedTech/
├── backend/                 FastAPI: upload + job queue that runs the pipeline, serves the built site
│   ├── app/                 API routes, CASA job service, settings
│   └── tests/
├── frontend/                React 18 + Vite + TypeScript website
├── training/                the CASA pipeline, detector training, validation
│   ├── src/                 pipeline stages (detect → track → kinematics → classify → aggregate)
│   ├── configs/default.yaml every setting the website uses
│   ├── weights/sperm_detector.pt   the deployed YOLO sperm-head detector
│   ├── data/                ground truth and evaluation sets (see below)
│   ├── train_yolo.py        detector training / fine-tuning
│   └── tests/
└── start.bat                one-command start (Windows)
```

## Model and data

The repository includes what the website needs to run and what is needed to check
its accuracy:

| Included | |
|---|---|
| `training/weights/sperm_detector.pt` | Deployed detector (referenced from `configs/default.yaml`) |
| `training/data/report/` | Lab analyzer reports, cropped to the results (no patient identifiers) |
| `training/data/ground_truth*.json` | Values transcribed from those reports; `ground_truth_verified.json` is authoritative |
| `training/data/video_split.json` | Video-level dev/test split used for tuning vs. held-out evaluation |
| `training/data/annotation_review/corrected/` | Hand-corrected detector test frames |
| `training/outputs/seg/validation/` | Accuracy vs. the lab: metrics JSON and Bland-Altman / comparison graphs (results table in [training/README.md](training/README.md#evaluation-metrics)) |

Not included (git-ignored, kept locally): the raw analyzer videos (`training/data/vid/`),
extracted frames, generated YOLO datasets (`yolo_dataset`, `yolo_cv`, `yolo_full`,
`yolo_holdout`), training runs and checkpoints (`runs/`, the rest of `training/outputs/`), base
COCO/YOLO weights (Ultralytics downloads these on demand), and every uploaded analysis
(`backend/data/`). Retraining or re-running validation therefore needs the videos copied
into `training/data/vid/` first.

## API

`POST /api/v1/analyses` (multipart: `file`, `sample_id`, optional `patient_ref`,
`volume_ml`, `notes`) · `GET /api/v1/analyses` · `GET /api/v1/analyses/{id}` ·
`GET /api/v1/analyses/{id}/overlay|poster|video` · `DELETE /api/v1/analyses/{id}` ·
`GET /api/v1/model` · `GET /api/v1/health/ready`. Interactive docs at `/docs`.

## Tests

```bash
cd backend && pytest        # API, job queue, report assembly (pipeline stubbed)
cd training && pytest       # kinematics and classification rules
```

## Not included yet

- Login / user accounts (single-user local tool for now; `backend/app/core/security.py`
  has JWT helpers to build on)
- Hosting online (the analysis needs this machine's GPU and model files)
- Morphology (normal forms) — not measured by this pipeline
- Self-learning detector — the detector would keep improving from the videos analysed
  in the app: uncertain frames are queued for review, the corrections are fed back into
  training, and a new model goes live only if it beats the current one on the
  hand-corrected test set (see [training/README.md](training/README.md#future-work-self-learning-detector))
