# QYX MedTech — backend

FastAPI server that queues uploaded videos, runs the CASA pipeline from `../training/src` on them one at a time, and serves the built website.

- Code: `app/services/casa.py` (job queue, pipeline run, report assembly), `app/api/v1/analyses.py` (endpoints)
- Settings: `app/config.py` (override with `.env`, see `.env.example`)
- Data: `data/analyses/<id>/` (git-ignored; contains patient videos)
- Tests: `pytest`

See the main [README](../README.md) for how to run everything.
