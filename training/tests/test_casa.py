"""Regression tests for the CASA accuracy fixes (kinematics, classification, frame QC).

Run from training/:   python -m pytest tests -q
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.seg_kinematics import _analyse_track          # noqa: E402
from src.stage7_classify import classify_sperm, is_rapid  # noqa: E402

FPS = 30.0
UM_PER_PX = 0.5


def _track(xy, frames=None):
    f = np.arange(len(xy)) if frames is None else np.asarray(frames)
    return {"track_id": 1, "fps": FPS,
            "detections": [{"frame": int(a), "x": float(x), "y": float(y)} for a, (x, y) in zip(f, xy)]}


def _kin(track, window=15):
    return _analyse_track(track, UM_PER_PX, window, 10)


def test_straight_line_has_equal_velocities():
    # 2 px/frame * 0.5 um/px * 30 fps = 30 um/s.  Duration must be N-1 intervals.
    p = _kin(_track([(2 * i, 0) for i in range(45)]))
    assert p["VCL"] == pytest.approx(30, abs=0.01)
    assert p["VAP"] == pytest.approx(30, abs=0.01)
    assert p["VSL"] == pytest.approx(30, abs=0.01)
    assert p["STR"] == 100.0


def test_track_shorter_than_window_is_still_smoothed():
    # Used to return the raw path (VAP == VCL, WOB == 100 %) for len < window.
    rng = np.random.default_rng(0)
    p = _kin(_track([(2 * i + rng.normal(0, 1), rng.normal(0, 1)) for i in range(12)]))
    assert p["VAP"] < p["VCL"]
    assert p["WOB"] < 100


def test_frame_gaps_do_not_change_speed():
    full = [(2 * i, 0) for i in range(45)]
    keep = [i for i in range(45) if not 10 <= i <= 14]          # trackpy `memory` gap
    p = _kin(_track([full[i] for i in keep], keep))
    assert p["VCL"] == pytest.approx(30, abs=0.01)
    assert p["VSL"] == pytest.approx(30, abs=0.01)


def test_ratios_never_exceed_100_percent():
    rng = np.random.default_rng(1)
    for _ in range(200):
        walk = np.cumsum(rng.normal(0.8, 2.0, (rng.integers(12, 80), 2)), axis=0)
        p = _kin(_track(walk))
        assert p["LIN"] <= 100 and p["STR"] <= 100 and p["WOB"] <= 100


def test_static_track_is_all_zero():
    p = _kin(_track([(5, 5)] * 30))
    assert p["VAP"] == 0 and p["LIN"] == 0 and p["STR"] == 0


def test_classification_progressive_is_rapid_plus_slow():
    assert classify_sperm({"VAP": 30, "STR": 90, "LIN": 60}) == "PR"
    assert is_rapid({"VAP": 30})
    # slow progressive (5 <= VAP < 25) is PR too, matching the lab's progressive = rapid + slow
    assert classify_sperm({"VAP": 12, "STR": 70, "LIN": 50}) == "PR"
    assert not is_rapid({"VAP": 12})
    assert classify_sperm({"VAP": 12, "STR": 5, "LIN": 5}) == "NP"
    assert classify_sperm({"VAP": 3, "STR": 90, "LIN": 90}) == "IM"


def test_head_jiggling_in_place_is_non_progressive():
    still = {"VAP": 1.0, "VCL": 2.0, "STR": 50, "LIN": 50}
    beating = {"VAP": 1.0, "VCL": 9.0, "STR": 50, "LIN": 50}
    assert classify_sperm(still, np_min_vcl=6.0) == "IM"
    assert classify_sperm(beating, np_min_vcl=6.0) == "NP"
    assert classify_sperm(beating) == "IM"          # rule off by default


def test_static_round_object_is_not_counted_as_sperm():
    round_static = {"VAP": 1.0, "VCL": 2.0, "STR": 50, "LIN": 50, "head_aspect": 1.05}
    oval_static = {**round_static, "head_aspect": 1.8}
    round_moving = {"VAP": 30.0, "VCL": 40.0, "STR": 90, "LIN": 80, "head_aspect": 1.05}
    assert classify_sperm(round_static, not_sperm_max_aspect=1.2) == "XX"
    assert classify_sperm(oval_static, not_sperm_max_aspect=1.2) == "IM"
    assert classify_sperm(round_moving, not_sperm_max_aspect=1.2) == "PR"   # end-on view of a swimmer
    assert classify_sperm({**round_static, "head_aspect": None}, not_sperm_max_aspect=1.2) == "IM"


def test_median_step_rule_demotes_glitch_motility():
    glitch = {"VAP": 6.0, "VCL": 9.0, "STR": 80, "LIN": 60, "VMED": 1.0}   # one bad link
    swimmer = {**glitch, "VMED": 6.0}                                      # moves every frame
    assert classify_sperm(glitch) == "PR"                                  # rule off by default
    assert classify_sperm(glitch, min_vmed=3.0) == "IM"
    assert classify_sperm(swimmer, min_vmed=3.0) == "PR"


def test_capture_fps_scales_velocities_not_distances():
    pts = [{"frame": i, "x": 10.0 + 2 * i, "y": 5.0} for i in range(30)]
    at30 = _analyse_track({"track_id": 0, "fps": 30.0, "detections": pts}, 0.5, 15, 10)
    at20 = _analyse_track({"track_id": 0, "fps": 20.0, "detections": pts}, 0.5, 15, 10)
    assert at20["VSL"] == pytest.approx(at30["VSL"] * 20 / 30, rel=1e-3)
    assert at20["ALH"] == pytest.approx(at30["ALH"])


def test_frame_qc_drops_black_gap_and_frozen_still(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from src.frame_qc import analyse_video, qc_params

    rng = np.random.default_rng(2)
    vdir = tmp_path / "v"
    vdir.mkdir()
    bg = np.full((120, 160), 150, np.uint8)

    def live(k):                                   # a dark blob sweeping across the field
        im = bg.copy()
        cv2.circle(im, (10 + 3 * k, 60), 6, 40, -1)
        return im

    frames = [live(k) for k in range(40)]                             # 0-39   live clip
    frames += [np.zeros_like(bg) for _ in range(10)]                  # 40-49  black gap
    still = bg.copy()
    cv2.circle(still, (80, 60), 6, 40, -1)
    # a frozen still is the SAME image re-encoded: at most a codec-level flicker
    # of +-1 grey level on ~1 % of pixels
    frames += [np.clip(still + (rng.random(still.shape) < 0.01) * rng.integers(-1, 2, still.shape),
                       0, 255).astype(np.uint8) for _ in range(30)]   # 50-79  frozen still
    for k, im in enumerate(frames):
        cv2.imwrite(str(vdir / f"frame_{k:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])

    qc = analyse_video(vdir, qc_params({}))
    verdicts = [(s["start"], s["end"], s["verdict"]) for s in qc["segments"]]
    assert verdicts == [(0, 39, "live"), (50, 79, "frozen")]
    assert set(qc["valid_frames"]) == set(range(40))


def test_frame_qc_keeps_live_field_with_no_motion(tmp_path):
    # Sensor noise alone (independent every frame) is a live camera, not a frozen
    # still, even if nothing in the field moves (e.g. an all-immotile sample).
    cv2 = pytest.importorskip("cv2")
    from src.frame_qc import analyse_video, qc_params

    rng = np.random.default_rng(3)
    base = np.full((120, 160), 150, np.int16)
    cv2.circle(base, (80, 60), 6, 40, -1)
    for k in range(40):
        im = np.clip(base + rng.integers(-2, 3, base.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{k:06d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])

    qc = analyse_video(tmp_path, qc_params({}))
    assert [s["verdict"] for s in qc["segments"]] == ["live"]
