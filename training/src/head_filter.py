"""
Head-size sanity filter for detector output.

A sperm head is ~23-32 px across in every video (measured on hand-corrected boxes,
videos 9-23).  In viscous / debris-laden samples the detector also fires on mucus
strands and cell aggregates with boxes 50-100+ px wide (video 23: median 73 px vs a
true head of 23 px).  Dropping boxes whose longest side exceeds `max_box_px`
(default config: 60 px, the 99th percentile of the training boxes) raised F1 on the
reviewed frames from 0.926 to 0.939 (precision 0.928 -> 0.959) with no change on
normal videos.  It cannot rescue samples where the detector misses the real heads
altogether (videos 18/23) -- those need training data.

Applied when heads are *consumed* (tracking, concentration), so no re-detection is
needed.  Set segmentation.head.max_box_px to null/0 to disable.
"""
from __future__ import annotations


def max_box_px(config: dict | None) -> float | None:
    v = ((config or {}).get("segmentation", {}).get("head", {}) or {}).get("max_box_px")
    return float(v) if v else None


def filter_heads(frames: dict, limit: float | None) -> dict:
    """{frame: [det, ...]} without detections whose bbox longest side > limit (px)."""
    if not limit:
        return frames
    out = {}
    for k, dets in frames.items():
        keep = []
        for d in dets:
            b = d.get("bbox")
            if b is None or max(b[2] - b[0], b[3] - b[1]) <= limit:
                keep.append(d)
        out[k] = keep
    return out


# Thresholds from the full 23-video pipeline run cross-checked against the hand-
# corrected review set (data/annotation_review): 21 of 23 videos have an oversize-box
# rate under 7% and scored F1 0.94-0.97 on hand-corrected boxes; videos 18 and 23
# (mucus/debris-heavy) hit 24.7% and 84.3% and scored F1 0.10 and 0.00.  There is a
# large, unobserved gap between 7% and 25% -- the 15% cut below is placed in the
# middle of that gap, not measured, so it may misclassify a video that actually
# falls in between. Both known-bad videos clear it; treat "reduced_confidence" as
# "no data either way" rather than "measured to be borderline".
QUALITY_TIERS = (
    (0.15, "unreliable", "Detector mistook debris/mucus for sperm heads on a large share "
                          "of boxes. Numbers for this video are unreliable -- do not report "
                          "without manual review."),
    (0.07, "reduced_confidence", "Oversize/debris detection rate is above the range seen on "
                                  "clean videos (0-6.5%). Review before reporting."),
    (0.0, "ok", None),
)


def detection_quality(heads_frames: dict, limit: float | None) -> dict:
    """Detection-confidence tier for one video from its raw (unfiltered) heads.

    Returns {"tier": "ok"|"reduced_confidence"|"unreliable", "oversize_pct": float,
    "message": str|None}.  `oversize_pct` is what filter_heads() already removes
    downstream, so this makes that silent cleanup visible as a reportable flag.
    """
    if not limit:
        return {"tier": "ok", "oversize_pct": 0.0, "message": None}
    total = sum(len(v) for v in heads_frames.values())
    kept = sum(len(v) for v in filter_heads(heads_frames, limit).values())
    pct = round((total - kept) / total * 100, 1) if total else 0.0
    for cutoff, tier, msg in QUALITY_TIERS:
        if pct >= cutoff * 100:
            return {"tier": tier, "oversize_pct": pct, "message": msg}
    return {"tier": "ok", "oversize_pct": pct, "message": None}
