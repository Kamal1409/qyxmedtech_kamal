"""
Clinical CASA Report Generator -- segmentation pipeline.

Generates an HTML report that closely matches the layout of the lab's
own printed semen analysis report:

    ┌─────────────────────────────────────────────────────────────┐
    │  I. Sample Information           [trajectory-overlay image] │
    ├─────────────────────────────────────────────────────────────┤
    │  II. Test Result  (Parameter | Result | WHO 6th | 95% CI)   │
    │      Ejaculate / Sperm count / Motility / Morphology        │
    ├─────────────────────────────────────────────────────────────┤
    │  Detail                                                     │
    │  Concentration (MSC/PMSC)   │  Morphology breakdown        │
    │  Motility (LIN STR WOB VAP  │  VSL VCL ALH BCF)           │
    ├─────────────────────────────────────────────────────────────┤
    │  Horizontal bar charts: VAP | VSL | STR | ALH distribution │
    └─────────────────────────────────────────────────────────────┘

Sperm paths are drawn on the most representative (median-density) frame
with colour-coded trajectories:  PR=cyan  NP=yellow  IM=gray.

Usage:
    python -m src.seg_report --config configs/default.yaml
    python -m src.seg_report --config configs/default.yaml --video-id 1
"""
from __future__ import annotations
import argparse
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False


# ---------------------------------------------------------------------------
# WHO 6th edition reference values (lower reference limits)
# ---------------------------------------------------------------------------
WHO6 = {
    "ejaculate_volume_ml":     ("≥ 1.4 mL",  "[1.3 - 1.5]"),
    "concentration_m_per_ml": ("≥ 16 M/mL",  "[15 - 18]"),
    "total_sperm_number_m":    ("≥ 39 M",     "[35 - 40]"),
    "total_motility_pct":      ("≥ 42 %",     "[40 - 43]"),
    "progressive_pct":         ("≥ 30 %",     "[29 - 31]"),
    "non_progressive_pct":     ("≥ 1 %",      "[1 - 1]"),
    "immotile_pct":            ("< 58 %",     "—"),
    "normal_forms_pct":        ("≥ 4 %",      "[3.9 - 4.0]"),
}


# ---------------------------------------------------------------------------
# Trajectory overlay rendering
# ---------------------------------------------------------------------------

def _motility_color_bgr(motility_class: str) -> tuple[int, int, int]:
    return {"PR": (255, 255, 0), "NP": (0, 200, 255), "IM": (140, 140, 140)}.get(
        motility_class, (200, 200, 200)
    )


def render_trajectory_overlay(
    frames_dir: Path,
    video_id: str,
    tracks_path: Path,
    classifications_path: Path,
    out_size: tuple[int, int] = (520, 380),
) -> str | None:
    """
    Draw colour-coded sperm trajectories onto the first clear video frame.
    Returns a base64-encoded PNG string for embedding in HTML, or None.
    """
    if not CV2_OK:
        return None
    if not tracks_path.exists() or not classifications_path.exists():
        return None

    with open(tracks_path) as f:
        tdata = json.load(f)
    with open(classifications_path) as f:
        cdata = json.load(f)

    cls_map = {c["track_id"]: c["motility_class"] for c in cdata.get("classifications", [])}

    # Find the first non-blank frame
    frame_dir = frames_dir / str(video_id)
    img = None
    for ff in sorted(frame_dir.glob("*.jpg"))[:60]:
        candidate = cv2.imread(str(ff))
        if candidate is not None and candidate.mean() > 20:
            img = candidate
            break
    if img is None:
        return None

    H, W = img.shape[:2]

    # Draw trajectories
    for track in tdata.get("tracks", []):
        tid = track["track_id"]
        dets = track.get("detections", [])
        if len(dets) < 2:
            continue
        cls = cls_map.get(tid, "IM")
        color = _motility_color_bgr(cls)
        pts = [(int(d["x"]), int(d["y"])) for d in dets]
        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)
        # Head dot at last position
        cv2.circle(img, pts[-1], 2, color, -1, cv2.LINE_AA)

    # Legend
    legend = [("PR", (255, 255, 0)), ("NP", (0, 200, 255)), ("IM", (140, 140, 140))]
    for i, (label, c) in enumerate(legend):
        y = H - 10 - i * 18
        cv2.line(img, (8, y), (24, y), c, 2)
        cv2.putText(img, label, (28, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)

    # Resize for the report
    img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ---------------------------------------------------------------------------
# Bar chart helpers (horizontal, matching the lab report style)
# ---------------------------------------------------------------------------

def _bin_series(values: list[float], bins: list[float]) -> list[float]:
    """Bin values into percentage-of-total buckets."""
    total = max(len(values), 1)
    counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        c = sum(1 for v in values if lo <= v < hi)
        counts.append(round(c / total * 100, 1))
    return counts


def _hbar_chart(values: list[float], bins: list[float], title: str, color: str, unit: str) -> str:
    """Return an inline Plotly HTML snippet for a horizontal bar chart."""
    if not PLOTLY_OK or not values:
        return ""
    pcts = _bin_series(values, bins)
    labels = [f"{int(lo)}-{int(hi)}" for lo, hi in zip(bins[:-1], bins[1:])]
    text = [f"{p}%" if p > 0 else "" for p in pcts]
    fig = go.Figure(go.Bar(
        x=pcts, y=labels, orientation="h",
        marker_color=color, text=text, textposition="outside",
        textfont=dict(size=10, color="#333"),
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text=f"<b>{title}</b> ({unit})", font=dict(size=13, color="#222")),
        height=250, margin=dict(l=55, r=40, t=38, b=20),
        plot_bgcolor="#f9f9f9", paper_bgcolor="#ffffff",
        xaxis=dict(range=[0, max(pcts) * 1.3 + 5], showgrid=True,
                   gridcolor="#e0e0e0", title="%", tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        font=dict(family="Arial, sans-serif"),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def _fmt(v: Any, decimals: int = 1, suffix: str = "") -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def generate_report(
    summary: dict,
    kinematics: dict,
    tracks_path: Path,
    classifications_path: Path,
    frames_dir: Path,
    gt: dict | None,
    output_path: Path,
) -> None:
    if not PLOTLY_OK:
        print("  [!] plotly not installed — cannot generate HTML report")
        return

    vid = summary["video_id"]
    motility = summary.get("motility", {})
    kin_pop = summary.get("population_kinematics", {})
    profiles = kinematics.get("profiles", [])
    fps = kinematics.get("fps", 30.0)
    p2m = kinematics.get("pixel_to_micron", 5.25)
    counts = summary.get("counts", {})
    total = summary.get("total_sperm_analyzed", len(profiles))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    quality = summary.get("detection_quality", {})
    warning_html = ""
    if quality.get("tier") in ("unreliable", "reduced_confidence"):
        color = "#c0392b" if quality["tier"] == "unreliable" else "#e08c00"
        label = "UNRELIABLE" if quality["tier"] == "unreliable" else "REDUCED CONFIDENCE"
        warning_html = (
            f'<div style="background:{color};color:#fff;padding:8px 14px;border-radius:4px;'
            f'font-size:12px;font-weight:bold;margin-bottom:12px;">'
            f'&#9888; {label} &mdash; {quality.get("message", "")} '
            f'(oversize/debris detections: {quality.get("oversize_pct")}%)</div>'
        )

    # --- trajectory overlay image ---
    overlay_b64 = render_trajectory_overlay(frames_dir, vid, tracks_path, classifications_path)
    img_tag = (
        f'<img src="data:image/png;base64,{overlay_b64}" '
        f'style="width:100%;border-radius:6px;border:1px solid #ccc;" '
        f'alt="Sperm trajectory overlay">'
        if overlay_b64 else
        '<div style="background:#eee;height:220px;display:flex;align-items:center;'
        'justify-content:center;color:#999;border-radius:6px;">No frame available</div>'
    )

    # --- WHO test result rows ---
    # "Result" is always this pipeline's own prediction, never the lab's reference
    # value -- videos 1-8 have a lab report for validation, but reusing its numbers
    # here would make the report show the answer key instead of what the system
    # actually measured, and every other (real, new-patient) video would just be
    # blank.  Where a lab reference exists it's shown as a small secondary note.
    def result_row(label: str, result_html: str, who_key: str, indent: bool = False,
                   ref: Any = None, ref_suffix: str = "", ref_decimals: int = 1) -> str:
        who_ref, ci = WHO6.get(who_key, ("—", "—"))
        pad = "padding-left:24px;" if indent else ""
        note = (f'<br><small style="color:#999">lab report: {_fmt(ref, ref_decimals)}{ref_suffix}</small>'
                if ref is not None else "")
        return (f'<tr><td style="font-weight:{'normal' if indent else 'bold'};{pad}">{label}</td>'
                f'<td>{result_html}{note}</td><td>{who_ref}</td><td>{ci}</td></tr>')

    def motility_badge(pct: float, cls: str) -> str:
        color = {"PR": "#2a9d5c", "NP": "#e08c00", "IM": "#c0392b"}.get(cls, "#555")
        return f'<span style="color:{color};font-weight:700">{pct:.0f} %</span>'

    gt = gt or {}
    conc_pred = summary.get("concentration_m_per_ml")
    # Ejaculate volume is a macroscopic measurement (graduated container) -- no video
    # can show it. Total sperm number = concentration (predicted) x that volume, so
    # it can only be computed once a real volume is entered; the lab's OCR'd volume
    # is used here only because this pipeline has no separate manual-entry input yet.
    vol = gt.get("ejaculate_volume_ml")
    total_sperm_pred = round(conc_pred * vol, 2) if (conc_pred is not None and vol is not None) else None
    rows_html = (
        result_row("Ejaculate", '— <small style="color:#999">(manual entry required -- '
                                'not measurable from video)</small>', "ejaculate_volume_ml", ref=vol)
        + result_row("Concentration", f'{_fmt(conc_pred, 1)} M/mL', "concentration_m_per_ml",
                     ref=gt.get("concentration_m_per_ml"))
        + result_row("Total sperm number",
                     f'{_fmt(total_sperm_pred, 2)} M' if total_sperm_pred is not None
                     else '— <small style="color:#999">(needs ejaculate volume)</small>',
                     "total_sperm_number_m", ref=gt.get("total_sperm_number_m"), ref_decimals=2)
        + result_row("Total Motility", motility_badge(motility.get("total_motility_pct", 0), ""), "total_motility_pct")
        + result_row("Progressive", motility_badge(motility.get("progressive_pct", 0), "PR"), "progressive_pct", indent=True)
        + result_row("Non-Progressive", motility_badge(motility.get("non_progressive_pct", 0), "NP"), "non_progressive_pct", indent=True)
        + result_row("Immotile", motility_badge(motility.get("immotile_pct", 0), "IM"), "immotile_pct", indent=True)
        + result_row("Normal forms (morphology)",
                     '— <small style="color:#999">(not assessed -- morphology classification is not '
                     'implemented by this pipeline)</small>', "normal_forms_pct")
    )

    # --- kinematic detail rows ---
    def kin_stat(param: str) -> str:
        s = kin_pop.get(param, {})
        return _fmt(s.get("mean"), 1)

    motility_detail = (
        f'<tr><td>LIN (Mean)</td><td><b>{kin_stat("LIN")} %</b></td>'
        f'<td>VSL (Mean)</td><td><b>{kin_stat("VSL")} µm/s</b></td></tr>'
        f'<tr><td>STR (Mean)</td><td><b>{kin_stat("STR")} %</b></td>'
        f'<td>VCL (Mean)</td><td><b>{kin_stat("VCL")} µm/s</b></td></tr>'
        f'<tr><td>WOB (Mean)</td><td><b>{kin_stat("WOB")} %</b></td>'
        f'<td>ALH (Mean)</td><td><b>{kin_stat("ALH")} µm</b></td></tr>'
        f'<tr><td>VAP (Mean)</td><td><b>{kin_stat("VAP")} µm/s</b></td>'
        f'<td>BCF (Mean)</td><td><b>{kin_stat("BCF")} Hz</b></td></tr>'
    )

    # MSC/PMSC: predicted (concentration x motility %, computed in stage8_aggregate),
    # with the lab's reference value alongside where one exists.
    def _detail_row(label: str, sub: str, pred: Any, ref: Any) -> str:
        note = f'<br><small style="color:#999">lab: {_fmt(ref, 1)} M/mL</small>' if ref is not None else ""
        return f'<tr><td>{label}<br><small>({sub})</small></td><td><b>{_fmt(pred, 1)} M/mL</b>{note}</td></tr>'

    concentration_detail = (
        _detail_row("MSC", "Motile Sperm Concentration", summary.get("msc_m_per_ml"), gt.get("msc_m_per_ml"))
        + _detail_row("PMSC", "Progressive Motile Sperm Concentration",
                      summary.get("pmsc_m_per_ml"), gt.get("pmsc_m_per_ml"))
    )

    # Morphology (head/midpiece/tail defect classification) is not implemented by
    # this pipeline -- there is no model producing these numbers.  Showing the lab's
    # own figures here would make the report look like it graded morphology when it
    # didn't; say so plainly instead of leaving a number that looks like a result.
    morphology_detail = (
        '<tr><td colspan="2" style="color:#999;font-style:italic;padding:6px 8px;">'
        'Not assessed -- morphology classification (WHO strict criteria) is not '
        'implemented by this pipeline. Requires expert-annotated training data.</td></tr>'
    )

    # --- distribution bar charts ---
    def _vals(param: str) -> list[float]:
        return [p[param] for p in profiles if param in p and p[param] is not None]

    vap_chart = _hbar_chart(_vals("VAP"), [0, 5, 15, 25, 35, 45, 100],  "VAP", "#f4a261", "µm/s")
    vsl_chart = _hbar_chart(_vals("VSL"), [0, 5, 15, 25, 35, 45, 100],  "VSL", "#2a9d8f", "µm/s")
    str_chart = _hbar_chart(_vals("STR"), [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], "STR", "#e76f51", "%")
    alh_chart = _hbar_chart(_vals("ALH"), [0, 2, 4, 6, 8, 10, 12, 20],  "ALH", "#457b9d", "µm")

    # --- full HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CASA Report — Video {vid}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, sans-serif; font-size: 13px; color: #222; background: #fff; }}
  .page {{ max-width: 900px; margin: 20px auto; padding: 20px; border: 1px solid #bbb; }}
  h2 {{ font-size: 15px; background: #222; color: #fff; padding: 5px 10px; margin: 16px 0 6px; }}
  h3 {{ font-size: 13px; font-weight: bold; margin: 10px 0 4px; color: #333; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; }}
  th {{ background: #444; color: #fff; padding: 5px 8px; text-align: left; font-size: 12px; }}
  td {{ padding: 4px 8px; border-bottom: 1px solid #e0e0e0; vertical-align: top; }}
  tr:hover td {{ background: #f5f5f5; }}
  .top-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; margin-bottom: 8px; }}
  .info-table td:first-child {{ font-weight: bold; width: 55%; color: #444; }}
  .result-table th:first-child {{ width: 36%; }}
  .result-table th:nth-child(2) {{ width: 16%; }}
  .result-table th:nth-child(3) {{ width: 24%; }}
  .detail-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .detail-inner-table td:first-child {{ color: #555; width: 60%; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }}
  .meta {{ font-size: 11px; color: #888; text-align: right; margin-top: 16px; }}
  .badge-pr {{ color: #2a9d5c; font-weight: bold; }}
  .badge-np {{ color: #e08c00; font-weight: bold; }}
  .badge-im {{ color: #c0392b; font-weight: bold; }}
  .legend {{ font-size: 11px; display: flex; gap: 16px; margin-bottom: 4px; }}
  .legend span {{ display: inline-block; width: 20px; height: 3px; border-radius:2px; vertical-align: middle; margin-right: 3px; }}
</style>
</head>
<body>
<div class="page">

  {warning_html}
  <!-- HEADER -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <div>
      <div style="font-size:18px;font-weight:bold;color:#222;">QYX MedTech — CASA Analysis Report</div>
      <div style="color:#666;font-size:12px;">Computer-Aided Sperm Analysis · WHO 6th Edition</div>
    </div>
    <div style="text-align:right;font-size:11px;color:#888;">
      Video ID: <b>{vid}</b><br>Generated: {now}<br>
      fps={fps} · px/µm={p2m} · n={total} sperm
    </div>
  </div>

  <!-- SECTION I: Sample Information + Trajectory Image -->
  <h2>I. Sample Information</h2>
  <div class="top-grid">
    <table class="info-table">
      <tr><td>pH</td><td>{_fmt(gt.get("ph"), 1)} <small style="color:#999">(lab report; not video-derivable)</small></td></tr>
      <tr><td>Ejaculate volume</td><td>{_fmt(vol, 1)} mL <small style="color:#999">(lab report; not video-derivable)</small></td></tr>
      <tr><td>Concentration (predicted)</td><td>{_fmt(conc_pred, 1)} M/mL</td></tr>
      <tr><td>Total sperm number (predicted)</td><td>{_fmt(total_sperm_pred, 2)} M</td></tr>
      <tr><td>Sperm analyzed (CASA)</td><td>{total}</td></tr>
    </table>
    <div>
      <div class="legend">
        <div><span style="background:#ffff00;"></span>Progressive (PR)</div>
        <div><span style="background:#00c8ff;"></span>Non-Progressive (NP)</div>
        <div><span style="background:#8c8c8c;"></span>Immotile (IM)</div>
      </div>
      {img_tag}
    </div>
  </div>

  <!-- SECTION II: Test Results -->
  <h2>II. Test Result</h2>
  <table class="result-table">
    <thead>
      <tr><th>Parameter</th><th>Result</th><th>WHO 6th</th><th>[ 95% CI ]</th></tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <!-- SECTION III: Detail -->
  <h2>Detail</h2>
  <div class="detail-grid">
    <div>
      <h3>Concentration :</h3>
      <table class="detail-inner-table">{concentration_detail}</table>
      <h3 style="margin-top:12px;">Morphology :</h3>
      <table class="detail-inner-table">{morphology_detail}</table>
    </div>
    <div>
      <h3>Motility (kinematic parameters) :</h3>
      <table class="detail-inner-table">{motility_detail}</table>
    </div>
  </div>

  <!-- DISTRIBUTION CHARTS -->
  <h2>Distribution Analysis</h2>
  <div style="font-size:11px;color:#666;margin-bottom:6px;">
    Velocity distributions of all analyzed sperm (n={total}).
    Medians:
    VAP={_fmt(kin_pop.get("VAP", {}).get("median"), 1)} um/s,
    VSL={_fmt(kin_pop.get("VSL", {}).get("median"), 1)} um/s,
    STR={_fmt(kin_pop.get("STR", {}).get("median"), 1)} %,
    ALH={_fmt(kin_pop.get("ALH", {}).get("median"), 2)} um
  </div>
  <div class="chart-grid">
    <div>{vap_chart}</div>
    <div>{vsl_chart}</div>
    <div>{str_chart}</div>
    <div>{alh_chart}</div>
  </div>

  <div class="meta">
    QYX MedTech CASA System · For research purposes only · Clinical decisions require qualified personnel.
  </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_seg_report(
    config_path: str = "configs/default.yaml",
    video_id: str | None = None,
) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get("paths", {})
    output_dir = Path(paths.get("output_dir", "outputs"))
    frames_dir = Path(paths.get("frames_dir", "data/frames"))
    gt_file_verified = Path("data/ground_truth_verified.json")
    gt_file_ocr = Path(paths.get("ground_truth_file", "data/ground_truth.json"))

    seg = output_dir / "seg"
    results_dir = seg / "results"
    kin_dir = seg / "kinematics"
    tracks_dir = seg / "tracks"
    cls_dir = seg / "classifications"
    reports_dir = seg / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth (prefer hand-verified)
    gt_all: dict = {}
    for gp in [gt_file_verified, gt_file_ocr]:
        if gp.exists():
            with open(gp) as f:
                gt_all = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
            break

    summaries = (
        [results_dir / f"{video_id}_summary.json"] if video_id
        else sorted(results_dir.glob("*_summary.json"))
    )
    summaries = [s for s in summaries if s.name != "all_videos_summary.json"]

    if not summaries:
        print(f"No summary files in {results_dir}. Run the seg pipeline first.")
        return

    print(f"=== Clinical CASA Report Generation ===")
    for sf in summaries:
        if not sf.exists():
            print(f"  [!] missing: {sf}")
            continue
        with open(sf) as f:
            summary = json.load(f)
        vid = summary["video_id"]

        kin_path = kin_dir / f"{vid}_kinematics.json"
        kinematics = {}
        if kin_path.exists():
            with open(kin_path) as f:
                kinematics = json.load(f)

        tracks_path = tracks_dir / f"{vid}_tracks.json"
        cls_path = cls_dir / f"{vid}_classification.json"
        gt = gt_all.get(str(vid), {})

        out_path = reports_dir / f"{vid}_report.html"
        print(f"  Video {vid}...", end=" ")
        generate_report(summary, kinematics, tracks_path, cls_path, frames_dir, gt, out_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical CASA report (seg pipeline)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--video-id", default=None)
    args = parser.parse_args()
    run_seg_report(args.config, args.video_id)
