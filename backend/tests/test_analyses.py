"""API tests with the GPU pipeline replaced by a stand-in that writes a result."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from app.services import casa
from app.services.casa import AnalysisStore, build_result

SUMMARY = {
    "concentration_m_per_ml": 40.0, "msc_m_per_ml": 24.0, "pmsc_m_per_ml": 20.0,
    "concentration_cv_pct": 8.0, "detection_quality": {"tier": "ok"},
    "motility": {"total_motility_pct": 60.0, "progressive_pct": 50.0, "non_progressive_pct": 10.0,
                 "immotile_pct": 40.0, "rapid_progressive_pct": 20.0, "slow_progressive_pct": 30.0},
    "counts": {"PR": 50, "NP": 10, "IM": 40},
    "motile_kinematics": {"VAP": {"mean": 18.0, "median": 17.0}},
}
CLASSIFICATIONS = {"classifications": [
    {"track_id": i, "motility_class": "PR", "VAP": 20.0, "VSL": 15.0, "STR": 85.0, "ALH": 1.5} for i in range(5)
] + [{"track_id": 99, "motility_class": "IM", "VAP": 1.0, "VSL": 0.5, "STR": 50.0, "ALH": 0.1}]}
CFG = {"kinematics": {"pixel_to_micron": 0.5, "capture_fps": 19}, "concentration": {"calibration_factor": 1.583},
       "segmentation": {"head": {"weights": "x/best.pt"}}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    def fake_runner(aid):
        d = store._dir(aid)
        res = build_result(SUMMARY, CLASSIFICATIONS, {"fps": 30, "extracted_frames": 90, "total_frames": 90,
                                                      "original_resolution": [868, 598]},
                           {"valid_frames": list(range(60))}, CFG, store._read_meta(aid)["volume_ml"])
        (d / "result.json").write_text(json.dumps(res))

    store = AnalysisStore(root=tmp_path, runner=fake_runner)
    monkeypatch.setattr(casa, "_store", store)
    from app.main import app
    with TestClient(app) as c:
        yield c


def _upload(client, name="clip.mp4", **form):
    data = {"sample_id": "S-001", **form}
    return client.post("/api/v1/analyses", data=data, files={"file": (name, b"fake video bytes", "video/mp4")})


def _wait_done(client, aid, timeout=5.0):
    t = time.time()
    while time.time() - t < timeout:
        r = client.get(f"/api/v1/analyses/{aid}").json()
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.05)
    raise AssertionError("analysis did not finish")


def test_upload_runs_and_returns_result(client):
    r = _upload(client, volume_ml="2.5")
    assert r.status_code == 201
    aid = r.json()["id"]
    done = _wait_done(client, aid)
    assert done["status"] == "done" and done["progress"] == 1.0
    res = done["result"]
    assert res["concentration_m_per_ml"] == 40.0
    assert res["total_sperm_number_m"] == 100.0                     # 40 M/mL x 2.5 mL
    assert res["n_sperm_analyzed"] == 100
    who = {w["key"]: w for w in res["who"]}
    assert who["total_motility"]["status"] == "normal" and who["immotile"]["status"] == "normal"
    listed = client.get("/api/v1/analyses").json()
    assert listed[0]["id"] == aid and listed[0]["headline"]["total_motility"] == 60.0


def test_rejects_non_video_and_missing_sample(client):
    assert _upload(client, name="notes.txt").status_code == 415
    r = client.post("/api/v1/analyses", files={"file": ("a.mp4", b"x", "video/mp4")})
    assert r.status_code == 422


def test_delete_and_not_found(client):
    aid = _upload(client).json()["id"]
    _wait_done(client, aid)
    assert client.delete(f"/api/v1/analyses/{aid}").status_code == 204
    assert client.get(f"/api/v1/analyses/{aid}").status_code == 404
    assert client.get("/api/v1/analyses/not-an-id").status_code == 404


def test_failed_run_reports_error(tmp_path, monkeypatch):
    def broken(aid):
        raise RuntimeError("Cannot open video: sample.mp4")
    store = AnalysisStore(root=tmp_path, runner=broken)
    monkeypatch.setattr(casa, "_store", store)
    from app.main import app
    with TestClient(app) as c:
        aid = _upload(c).json()["id"]
        r = _wait_done(c, aid)
    assert r["status"] == "failed" and "could not be read as a video" in r["error"]


def test_build_result_flags_small_samples_and_low_values():
    summary = {**SUMMARY, "concentration_m_per_ml": 10.0,
               "motility": {**SUMMARY["motility"], "total_motility_pct": 30.0}, "counts": {"PR": 5, "NP": 1, "IM": 14}}
    res = build_result(summary, CLASSIFICATIONS, {"fps": 30, "extracted_frames": 20}, {"valid_frames": list(range(20))},
                       CFG, None)
    who = {w["key"]: w for w in res["who"]}
    assert who["concentration"]["status"] == "low" and who["total_motility"]["status"] == "low"
    assert who["volume_ml"]["status"] == "na" and res["total_sperm_number_m"] is None
    levels = [w["level"] for w in res["quality"]["warnings"]]
    assert "warning" in levels                                   # only 20 sperm, and only 20 live frames
    assert res["distributions"]["VAP"]["percent"][2] == 100.0      # all motile at 20 um/s -> 15-25 bin


def test_model_info(client):
    r = client.get("/api/v1/model")
    assert r.status_code == 200
    assert "capture_fps" in r.json()["settings"]
