"""
Validation: Compare Predictions vs Laboratory Reports

Compares CASA pipeline predictions against OCR-extracted ground truth
from laboratory reports using multiple evaluation metrics:
    - Mean Absolute Error (MAE)
    - Percentage Difference
    - Pearson Correlation
    - Bland-Altman Agreement Analysis

Usage:
    python -m src.validate --config configs/default.yaml
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def mean_absolute_error(predicted: list[float], actual: list[float]) -> float:
    """Compute Mean Absolute Error."""
    if not predicted or not actual:
        return float("nan")
    return float(np.mean(np.abs(np.array(predicted) - np.array(actual))))


def percentage_difference(predicted: list[float], actual: list[float]) -> list[float]:
    """Compute percentage difference for each pair."""
    diffs = []
    for p, a in zip(predicted, actual):
        if a != 0:
            diffs.append(abs(p - a) / abs(a) * 100)
        else:
            diffs.append(0.0 if p == 0 else float("inf"))
    return diffs


def pearson_correlation(predicted: list[float], actual: list[float]) -> tuple[float, float]:
    """Compute Pearson correlation coefficient and p-value."""
    if not SCIPY_AVAILABLE or len(predicted) < 3:
        return (float("nan"), float("nan"))
    r, p = stats.pearsonr(predicted, actual)
    return (round(float(r), 4), round(float(p), 6))


def bland_altman_analysis(predicted: list[float], actual: list[float]) -> dict:
    """
    Perform Bland-Altman agreement analysis.

    Returns:
        dict with mean_diff, sd_diff, upper_loa, lower_loa
    """
    if not predicted or not actual:
        return {"mean_diff": 0, "sd_diff": 0, "upper_loa": 0, "lower_loa": 0}

    pred = np.array(predicted)
    act = np.array(actual)
    diffs = pred - act
    means = (pred + act) / 2

    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0

    return {
        "mean_diff": round(mean_diff, 3),
        "sd_diff": round(sd_diff, 3),
        "upper_loa": round(mean_diff + 1.96 * sd_diff, 3),
        "lower_loa": round(mean_diff - 1.96 * sd_diff, 3),
        "means": means.tolist(),
        "diffs": diffs.tolist(),
    }


def icc(
    predicted: list[float], actual: list[float], model: str = "twoway", typ: str = "agreement"
) -> dict:
    """
    Intraclass Correlation Coefficient (ICC) — the medical-device standard for
    method-comparison studies (ISO 5725, CLSI EP09-A3).

    Uses the two-way mixed-effects model (ICC(2,1) for agreement), which is
    appropriate when one "rater" is the CASA system and the other is the lab.

    Returns a dict with: icc, lower_ci, upper_ci, f_stat, df1, df2, p_value
    """
    if len(predicted) < 3:
        return {"icc": float("nan"), "lower_ci": float("nan"), "upper_ci": float("nan")}

    n = len(predicted)
    # Build n x 2 matrix (subjects x raters)
    data = np.column_stack([np.array(predicted), np.array(actual)])
    k = 2  # two raters

    # Grand mean
    grand_mean = data.mean()
    # Subject means, rater means
    subj_means  = data.mean(axis=1)          # (n,)
    rater_means = data.mean(axis=0)          # (k,)

    # Sum of squares
    SS_total  = ((data - grand_mean) ** 2).sum()
    SS_subj   = k * ((subj_means  - grand_mean) ** 2).sum()
    SS_rater  = n * ((rater_means - grand_mean) ** 2).sum()
    SS_error  = SS_total - SS_subj - SS_rater

    # Mean squares
    MS_subj  = SS_subj  / (n - 1)
    MS_rater = SS_rater / (k - 1)
    MS_error = SS_error / ((n - 1) * (k - 1))

    # ICC(2,1) agreement
    icc_val = (MS_subj - MS_error) / (MS_subj + (k - 1) * MS_error + k / n * (MS_rater - MS_error))

    # 95% CI via F-distribution
    F_val = MS_subj / MS_error
    df1   = n - 1
    df2   = (n - 1) * (k - 1)

    if SCIPY_AVAILABLE:
        F_lower = F_val / stats.f.ppf(0.975, df1, df2)
        F_upper = F_val / stats.f.ppf(0.025, df1, df2)
        lower_ci = (F_lower - 1) / (F_lower + k - 1)
        upper_ci = (F_upper - 1) / (F_upper + k - 1)
        p_value  = float(1 - stats.f.cdf(F_val, df1, df2))
    else:
        lower_ci = upper_ci = p_value = float("nan")

    return {
        "icc":      round(float(icc_val),  4),
        "lower_ci": round(float(lower_ci), 4),
        "upper_ci": round(float(upper_ci), 4),
        "f_stat":   round(float(F_val),    4),
        "df1": df1, "df2": df2,
        "p_value":  round(p_value, 6) if not np.isnan(p_value) else float("nan"),
    }


def who_confusion_matrix(
    predictions: dict, ground_truth: dict,
    pr_min_vap: float = 25.0, pr_min_str: float = 80.0, pr_min_lin: float = 25.0,
    np_min_vap: float = 5.0,
) -> dict | None:
    """
    Build a WHO PR/NP/IM confusion matrix comparing predicted vs GT motility %.

    Since we don't have per-cell GT labels, this approximates the confusion by
    computing which WHO bucket the video-level percentages fall into and
    flagging discordant videos.  This shows which boundary is causing errors.

    Returns a dict with per-metric error distribution or None if insufficient data.
    """
    common_vids = sorted(set(ground_truth.keys()) & set(predictions.keys()))
    if len(common_vids) < 2:
        return None

    errors = {"PR": [], "NP": [], "IM": [], "total_motility": []}
    for vid in common_vids:
        gt  = ground_truth[vid]
        pr  = predictions[vid].get("motility", {})
        errors["PR"].append(          pr.get("progressive_pct",     0) - float(gt.get("progressive_pct",     0)))
        errors["NP"].append(          pr.get("non_progressive_pct", 0) - float(gt.get("non_progressive_pct", 0)))
        errors["IM"].append(          pr.get("immotile_pct",        0) - float(gt.get("immotile_pct",        0)))
        errors["total_motility"].append(pr.get("total_motility_pct", 0) - float(gt.get("total_motility_pct", 0)))

    result: dict = {}
    for cat, errs in errors.items():
        arr = np.array(errs)
        result[cat] = {
            "mean_error":   round(float(arr.mean()), 2),
            "std_error":    round(float(arr.std()),  2),
            "max_overest":  round(float(arr.max()),  2),
            "max_underest": round(float(arr.min()),  2),
            "n_overestimated":  int((arr > 5).sum()),   # >5 pp over GT
            "n_underestimated": int((arr < -5).sum()),  # >5 pp under GT
            "n_within_5pp":     int((np.abs(arr) <= 5).sum()),
        }
    return result


def create_bland_altman_plot(ba_result: dict, metric_name: str) -> go.Figure | None:
    """Create a Bland-Altman plot for a metric."""
    if not PLOTLY_AVAILABLE:
        return None

    means = ba_result.get("means", [])
    diffs = ba_result.get("diffs", [])

    if not means:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=means, y=diffs,
        mode="markers",
        marker=dict(size=10, color="#3498db"),
        name="Data points",
    ))

    # Mean difference line
    fig.add_hline(y=ba_result["mean_diff"], line_dash="dash", line_color="#2ecc71",
                  annotation_text=f"Mean: {ba_result['mean_diff']:.2f}")

    # Limits of agreement
    fig.add_hline(y=ba_result["upper_loa"], line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"+1.96 SD: {ba_result['upper_loa']:.2f}")
    fig.add_hline(y=ba_result["lower_loa"], line_dash="dot", line_color="#e74c3c",
                  annotation_text=f"-1.96 SD: {ba_result['lower_loa']:.2f}")

    fig.update_layout(
        title=f"Bland-Altman: {metric_name}",
        xaxis_title="Mean of Predicted & Actual",
        yaxis_title="Difference (Predicted - Actual)",
        template="plotly_dark",
        height=400,
    )

    return fig


def validate(config_path: str = "configs/default.yaml", split: str = "all") -> dict:
    """
    Compare CASA predictions against lab report ground truth.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    output_dir = Path(paths_cfg.get("output_dir", "outputs"))
    gt_file = Path(paths_cfg.get("ground_truth_file", "data/ground_truth.json"))
    # Prefer a hand-verified ground truth (immune to OCR errors) if present.
    verified_file = gt_file.with_name("ground_truth_verified.json")
    if verified_file.exists():
        gt_file = verified_file
    results_dir = output_dir / "results"
    seg_res_dir = output_dir / "seg" / "results"
    if seg_res_dir.exists():
        seg_files = [f for f in seg_res_dir.glob("*_summary.json") if f.name != "all_videos_summary.json"]
        if seg_files:
            results_dir = seg_res_dir

    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    # -- Load ground truth ----------------------------------------------
    if not gt_file.exists():
        print(f"[!] Ground truth file not found: {gt_file}")
        print("  Run OCR first: python -m src.ocr_report --config configs/default.yaml")
        return {}

    print(f"Ground truth source: {gt_file.name}")
    with open(gt_file) as f:
        ground_truth = json.load(f)
    # Videos without a hand-verified entry fall back to the transcribed reports
    # (data/ground_truth_transcribed.json, not yet human-checked).
    transcribed_file = gt_file.with_name("ground_truth_transcribed.json")
    if transcribed_file.exists():
        with open(transcribed_file) as f:
            transcribed = json.load(f)
        added = [k for k in transcribed if not k.startswith("_") and k not in ground_truth]
        ground_truth.update({k: transcribed[k] for k in added})
        if added:
            print(f"  + {len(added)} video(s) from {transcribed_file.name} (unreviewed transcription)")
    # Drop any non-video metadata entries (e.g. "_meta")
    ground_truth = {k: v for k, v in ground_truth.items() if not k.startswith("_")}

    # Drop excluded videos, then restrict to one side of the video-level split
    # (data/video_split.json)
    split_file = gt_file.with_name("video_split.json")
    if split_file.exists():
        with open(split_file) as f:
            split_def = json.load(f)
        excluded = set(split_def.get("excluded", []))
        ground_truth = {k: v for k, v in ground_truth.items() if k not in excluded}
        if excluded:
            print(f"  - excluded videos: {', '.join(sorted(excluded, key=int))}")
        if split != "all":
            ground_truth = {k: v for k, v in ground_truth.items() if k in set(split_def[split])}
    elif split != "all":
        raise SystemExit(f"--split {split} needs {split_file}")
    print(f"Split: {split} ({len(ground_truth)} videos)")

    # -- Load predictions -----------------------------------------------
    summary_files = sorted(results_dir.glob("*_summary.json"))
    summary_files = [f for f in summary_files if f.name != "all_videos_summary.json"]

    if not summary_files:
        print(f"[!] No prediction summaries found in {results_dir}")
        print("  Run Stages 5-8 first to generate predictions.")
        return {}

    predictions = {}
    for sf in summary_files:
        with open(sf) as f:
            data = json.load(f)
            predictions[data["video_id"]] = data

    print(f"=== Validation: Predictions vs Lab Reports ===")
    print(f"Ground truth videos: {len(ground_truth)}")
    print(f"Predicted videos: {len(predictions)}")

    # -- Compare metrics ------------------------------------------------
    # Metrics to compare
    def _kin(param: str) -> list[tuple]:
        # Lab CASA velocities are over motile sperm; older summaries only have
        # the all-tracks block, so fall back to it.
        return [("motile_kinematics", param, "mean"), ("population_kinematics", param, "mean")]

    comparison_metrics = {
        "concentration": {
            "gt_key": "concentration_m_per_ml",
            "pred_path": ("concentration_m_per_ml",),
        },
        "total_motility": {
            "gt_key": "total_motility_pct",
            "pred_path": ("motility", "total_motility_pct"),
        },
        "progressive": {
            "gt_key": "progressive_pct",
            "pred_path": ("motility", "progressive_pct"),
        },
        "rapid_progressive": {
            "gt_key": "rapid_progressive_pct",
            "pred_path": ("motility", "rapid_progressive_pct"),
        },
        "slow_progressive": {
            "gt_key": "slow_progressive_pct",
            "pred_path": ("motility", "slow_progressive_pct"),
        },
        "non_progressive": {
            "gt_key": "non_progressive_pct",
            "pred_path": ("motility", "non_progressive_pct"),
        },
        "immotile": {
            "gt_key": "immotile_pct",
            "pred_path": ("motility", "immotile_pct"),
        },
        "VAP_mean": {"gt_key": "VAP_um_per_s", "pred_paths": _kin("VAP")},
        "VSL_mean": {"gt_key": "VSL_um_per_s", "pred_paths": _kin("VSL")},
        "VCL_mean": {"gt_key": "VCL_um_per_s", "pred_paths": _kin("VCL")},
        "LIN_mean": {"gt_key": "LIN_pct",      "pred_paths": _kin("LIN")},
        "STR_mean": {"gt_key": "STR_pct",      "pred_paths": _kin("STR")},
        "WOB_mean": {"gt_key": "WOB_pct",      "pred_paths": _kin("WOB")},
        "ALH_mean": {"gt_key": "ALH_um",       "pred_paths": _kin("ALH")},
        "BCF_mean": {"gt_key": "BCF_hz",       "pred_paths": _kin("BCF")},
    }

    validation_results = {}

    for metric_name, metric_info in comparison_metrics.items():
        gt_values = []
        pred_values = []
        video_pairs = []

        for vid in sorted(set(ground_truth.keys()) & set(predictions.keys())):
            gt_val = ground_truth[vid].get(metric_info["gt_key"])

            # Navigate nested prediction path (first alternative that resolves)
            pred_val = None
            for path in metric_info.get("pred_paths") or [metric_info["pred_path"]]:
                node = predictions[vid]
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
                if node is not None:
                    pred_val = node
                    break

            if gt_val is not None and pred_val is not None:
                gt_values.append(float(gt_val))
                pred_values.append(float(pred_val))
                video_pairs.append(vid)

        if not gt_values:
            continue

        mae = mean_absolute_error(pred_values, gt_values)
        pct_diffs = percentage_difference(pred_values, gt_values)
        r, p_val = pearson_correlation(pred_values, gt_values)
        ba = bland_altman_analysis(pred_values, gt_values)

        icc_res = icc(pred_values, gt_values)
        validation_results[metric_name] = {
            "n_videos": len(gt_values),
            "mae": round(mae, 3),
            "mean_pct_diff": round(float(np.mean(pct_diffs)), 2),
            "pearson_r": r,
            "pearson_p": p_val,
            "icc": icc_res,
            "bland_altman": {
                "mean_diff": ba["mean_diff"],
                "sd_diff": ba["sd_diff"],
                "upper_loa": ba["upper_loa"],
                "lower_loa": ba["lower_loa"],
            },
            "per_video": [
                {"video": vid, "predicted": round(p, 2), "actual": round(a, 2), "diff": round(p - a, 2)}
                for vid, p, a in zip(video_pairs, pred_values, gt_values)
            ],
        }

    # -- Print results --------------------------------------------------
    print(f"\n{'Metric':<20} {'MAE':>8} {'Mean %Diff':>12} {'Bias':>8} {'Pearson r':>10} {'ICC':>7} {'N':>4}")
    print("-" * 74)
    for metric_name, result in validation_results.items():
        print(
            f"{metric_name:<20} "
            f"{result['mae']:>8.3f} "
            f"{result['mean_pct_diff']:>11.2f}% "
            f"{result['bland_altman']['mean_diff']:>8.2f} "
            f"{result['pearson_r']:>10.4f} "
            f"{result['icc']['icc']:>7.3f} "
            f"{result['n_videos']:>4}"
        )

    # Per-category error distribution (which motility bucket carries the error)
    motility_errors = who_confusion_matrix(predictions, ground_truth)
    if motility_errors:
        validation_results["_motility_error_distribution"] = motility_errors
        print("\nMotility error by category (pred - lab, percentage points):")
        for cat, e in motility_errors.items():
            print(f"  {cat:<15} mean {e['mean_error']:>7.2f}  "
                  f"within +/-5pp: {e['n_within_5pp']}  over: {e['n_overestimated']}  under: {e['n_underestimated']}")

    # -- Save validation report -----------------------------------------
    report_path = validation_dir / ("validation_report.json" if split == "all"
                                    else f"validation_report_{split}.json")
    with open(report_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    print(f"\nValidation report saved: {report_path}")

    # -- Generate Bland-Altman plots ------------------------------------
    if PLOTLY_AVAILABLE:
        for metric_name, result in validation_results.items():
            if metric_name.startswith("_"):      # non-metric entries
                continue
            pred_vals = [v["predicted"] for v in result["per_video"]]
            act_vals = [v["actual"] for v in result["per_video"]]
            ba = bland_altman_analysis(pred_vals, act_vals)
            fig = create_bland_altman_plot(ba, metric_name)
            if fig:
                plot_path = validation_dir / f"bland_altman_{metric_name}.html"
                # cdn: ~10 KB per file instead of ~4.8 MB (plotly.js is inlined otherwise)
                fig.write_html(str(plot_path), include_plotlyjs="cdn")

        print(f"Bland-Altman plots saved to {validation_dir}")

    return validation_results


# ---------------------------------------------------------------------------
# Before / after comparison report
# ---------------------------------------------------------------------------

# metric -> (display name, unit, group)
_METRIC_INFO = {
    "concentration":     ("Concentration",          "M/mL", "Count"),
    "total_motility":    ("Total motility (PR+NP)", "pp",   "Motility"),
    "progressive":       ("Progressive (PR)",       "pp",   "Motility"),
    "rapid_progressive": ("Rapid progressive",      "pp",   "Motility"),
    "slow_progressive":  ("Slow progressive",       "pp",   "Motility"),
    "non_progressive":   ("Non-progressive (NP)",   "pp",   "Motility"),
    "immotile":          ("Immotile (IM)",          "pp",   "Motility"),
    "VAP_mean":          ("VAP",                    "µm/s", "Kinematics"),
    "VSL_mean":          ("VSL",                    "µm/s", "Kinematics"),
    "VCL_mean":          ("VCL",                    "µm/s", "Kinematics"),
    "LIN_mean":          ("LIN",                    "%",    "Kinematics"),
    "STR_mean":          ("STR",                    "%",    "Kinematics"),
    "WOB_mean":          ("WOB",                    "%",    "Kinematics"),
    "ALH_mean":          ("ALH",                    "µm",   "Kinematics"),
    "BCF_mean":          ("BCF",                    "Hz",   "Kinematics"),
}
_KEY_METRICS = ["concentration", "total_motility", "progressive", "non_progressive", "VAP_mean", "VCL_mean"]


def _fmt(x: float | None, nd: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _change_cell(before: float, after: float, higher_is_better: bool, nd: int = 2) -> str:
    """'-1.23 (-45%)' coloured good/bad; neutral when either side is missing."""
    if before is None or after is None or not (np.isfinite(before) and np.isfinite(after)):
        return '<td class="num">n/a</td>'
    d = after - before
    better = d > 0 if higher_is_better else d < 0
    cls = "same" if abs(d) < 10 ** -nd else ("good" if better else "bad")
    pct = f" ({d / abs(before) * 100:+.0f}%)" if abs(before) > 1e-9 else ""
    return f'<td class="num {cls}">{d:+.{nd}f}{pct}</td>'


def comparison_report(before: dict, after: dict, out_path: Path, split: str,
                      before_label: str = "Before", after_label: str = "After",
                      notes: list[str] | None = None) -> Path:
    """Write a self-contained HTML report comparing two validation_report JSONs
    (as written by validate()) metric by metric and video by video."""
    from datetime import datetime
    from html import escape

    metrics = [m for m in _METRIC_INFO if m in before and m in after]
    n_vid = after[metrics[0]]["n_videos"] if metrics else 0
    improved = [m for m in metrics if after[m]["mae"] < before[m]["mae"]]
    worse = [m for m in metrics if after[m]["mae"] > before[m]["mae"]]

    # -- summary table -------------------------------------------------
    rows, group = [], None
    for m in metrics:
        name, unit, g = _METRIC_INFO[m]
        if g != group:
            rows.append(f'<tr class="group"><td colspan="11">{g}</td></tr>')
            group = g
        b, a = before[m], after[m]
        bb, ab = b["bland_altman"]["mean_diff"], a["bland_altman"]["mean_diff"]
        rows.append(
            f"<tr><td>{name}</td><td>{unit}</td>"
            f'<td class="num">{_fmt(b["mae"])}</td><td class="num">{_fmt(a["mae"])}</td>'
            + _change_cell(b["mae"], a["mae"], higher_is_better=False)
            + f'<td class="num">{_fmt(bb)}</td><td class="num">{_fmt(ab)}</td>'
            + f'<td class="num">{_fmt(b["pearson_r"], 3)} → {_fmt(a["pearson_r"], 3)}</td>'
            + f'<td class="num">{_fmt(b["icc"]["icc"], 3)}</td><td class="num">{_fmt(a["icc"]["icc"], 3)}</td>'
            + _change_cell(b["icc"]["icc"], a["icc"]["icc"], higher_is_better=True, nd=3)
            + "</tr>")

    # -- WHO motility categories ---------------------------------------
    cat_rows = []
    bd, ad = before.get("_motility_error_distribution", {}), after.get("_motility_error_distribution", {})
    for cat in ("total_motility", "PR", "NP", "IM"):
        if cat in bd and cat in ad:
            b, a = bd[cat], ad[cat]
            cat_rows.append(
                f"<tr><td>{cat.replace('_', ' ')}</td>"
                f'<td class="num">{b["mean_error"]:+.2f}</td><td class="num">{a["mean_error"]:+.2f}</td>'
                f'<td class="num">{b["n_within_5pp"]}/{n_vid}</td><td class="num">{a["n_within_5pp"]}/{n_vid}</td>'
                + _change_cell(b["n_within_5pp"], a["n_within_5pp"], higher_is_better=True, nd=0)
                + f'<td class="num">{b["n_overestimated"]} / {b["n_underestimated"]}</td>'
                f'<td class="num">{a["n_overestimated"]} / {a["n_underestimated"]}</td></tr>')

    # -- per-video tables ----------------------------------------------
    per_video = []
    for m in _KEY_METRICS:
        if m not in metrics:
            continue
        name, unit, _ = _METRIC_INFO[m]
        bv = {r["video"]: r for r in before[m]["per_video"]}
        av = {r["video"]: r for r in after[m]["per_video"]}
        body = []
        for v in sorted(set(bv) & set(av), key=lambda s: int(s) if s.isdigit() else s):
            eb, ea = bv[v]["diff"], av[v]["diff"]
            body.append(
                f'<tr><td>{escape(v)}</td><td class="num">{_fmt(av[v]["actual"], 1)}</td>'
                f'<td class="num">{_fmt(bv[v]["predicted"], 1)}</td><td class="num">{_fmt(av[v]["predicted"], 1)}</td>'
                f'<td class="num">{eb:+.1f}</td><td class="num">{ea:+.1f}</td>'
                + _change_cell(abs(eb), abs(ea), higher_is_better=False, nd=1) + "</tr>")
        per_video.append(
            f"<details><summary>{name} ({unit}) — per video</summary><div class='scroll'><table>"
            "<thead><tr><th>Video</th><th>Lab</th><th>Before</th><th>After</th>"
            "<th>Error before</th><th>Error after</th><th>|Error| change</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div></details>")

    # -- Bland-Altman before vs after (key metrics) --------------------
    plots = []
    if PLOTLY_AVAILABLE:
        for m in _KEY_METRICS:
            if m not in metrics:
                continue
            fig = go.Figure()
            for res, label, colour in ((before[m], before_label, "#9aa4b2"), (after[m], after_label, "#2f7fd8")):
                pv = [r["predicted"] for r in res["per_video"]]
                av_ = [r["actual"] for r in res["per_video"]]
                ba = bland_altman_analysis(pv, av_)
                fig.add_trace(go.Scatter(
                    x=ba.get("means", []), y=ba.get("diffs", []), mode="markers", name=label,
                    marker=dict(size=9, color=colour),
                    text=[f"video {r['video']}" for r in res["per_video"]]))
                fig.add_hline(y=ba["mean_diff"], line_dash="dash", line_color=colour,
                              annotation_text=f"{label} bias {ba['mean_diff']:+.2f}")
            fig.add_hline(y=0, line_color="#888", line_width=1)
            name, unit, _ = _METRIC_INFO[m]
            fig.update_layout(title=f"{name}: predicted − lab vs mean ({unit})", height=360,
                              margin=dict(l=50, r=20, t=50, b=40),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#8a93a0"), legend=dict(orientation="h", y=-0.2))
            fig.update_xaxes(gridcolor="rgba(128,128,128,.2)")
            fig.update_yaxes(gridcolor="rgba(128,128,128,.2)", zerolinecolor="rgba(128,128,128,.4)")
            plots.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if not plots else False))

    notes_html = "".join(f"<li>{escape(n)}</li>" for n in (notes or []))
    headline = (f"Error (MAE) went down on <b>{len(improved)}</b> of {len(metrics)} metrics"
                f" and up on <b>{len(worse)}</b>"
                + (f" ({', '.join(_METRIC_INFO[m][0] for m in worse)})" if worse else "") + ".")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CASA Before/After Comparison</title>
<style>
:root{{--bg:#f7f8fa;--card:#fff;--fg:#1d232b;--muted:#5d6773;--line:#e2e6eb;--good:#1d7a45;--good-bg:#e5f4ea;--bad:#b3261e;--bad-bg:#fbe7e5;--accent:#2f7fd8}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#12161b;--card:#1a2027;--fg:#e6e9ed;--muted:#9aa4b2;--line:#2c343d;--good:#6fd39a;--good-bg:#173524;--bad:#ff8a80;--bad-bg:#3d1d1b;--accent:#6aa8f0}}}}
:root[data-theme="dark"]{{--bg:#12161b;--card:#1a2027;--fg:#e6e9ed;--muted:#9aa4b2;--line:#2c343d;--good:#6fd39a;--good-bg:#173524;--bad:#ff8a80;--bad-bg:#3d1d1b;--accent:#6aa8f0}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 48px}}
h1{{font-size:1.6rem;margin:0 0 4px}}h2{{font-size:1.15rem;margin:28px 0 10px}}
.meta{{color:var(--muted);font-size:.9rem}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
th{{color:var(--muted);font-weight:600}}td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
tr.group td{{font-weight:700;color:var(--accent);padding-top:14px}}
.good{{color:var(--good);background:var(--good-bg)}}.bad{{color:var(--bad);background:var(--bad-bg)}}.same{{color:var(--muted)}}
details{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:8px 0}}
summary{{cursor:pointer;font-weight:600}}dl{{display:grid;grid-template-columns:max-content 1fr;gap:6px 14px;margin:0}}
dt{{font-weight:600}}dd{{margin:0;color:var(--muted)}}code{{font-size:.85em}}
</style></head><body><main>
<h1>CASA pipeline: before vs after</h1>
<div class="meta">Split: <b>{escape(split)}</b> · {n_vid} videos with lab reports · generated {datetime.now():%Y-%m-%d %H:%M}
 · <b>{escape(before_label)}</b> vs <b>{escape(after_label)}</b></div>
<div class="card"><b>Summary.</b> {headline}</div>
{f'<div class="card"><b>What changed</b><ul>{notes_html}</ul></div>' if notes_html else ''}

<h2>How each number is calculated</h2>
<div class="card"><dl>
<dt>Error</dt><dd>predicted − lab value, per video (lab = LensHooke report).</dd>
<dt>MAE</dt><dd>mean of |error| over the videos — the typical size of a miss, in the metric's own unit. Lower is better.</dd>
<dt>Bias</dt><dd>mean of the signed error — whether the pipeline systematically reads high (+) or low (−). Closer to 0 is better.</dd>
<dt>Pearson r</dt><dd>how well predictions rise and fall with the lab values (ranking), ignoring any constant offset.</dd>
<dt>ICC(2,1)</dt><dd>agreement: like r but also penalises offsets and scale errors. &gt;0.9 excellent, 0.75–0.9 good, &lt;0.5 poor.</dd>
<dt>Within ±5 pp</dt><dd>videos whose motility category is within 5 percentage points of the lab.</dd>
</dl></div>

<h2>All metrics</h2>
<div class="card scroll"><table><thead><tr><th>Metric</th><th>Unit</th>
<th class="num">MAE before</th><th class="num">MAE after</th><th class="num">MAE change</th>
<th class="num">Bias before</th><th class="num">Bias after</th><th class="num">Pearson r</th>
<th class="num">ICC before</th><th class="num">ICC after</th><th class="num">ICC change</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>

<h2>WHO motility categories</h2>
<div class="card scroll"><table><thead><tr><th>Category</th>
<th class="num">Mean error before</th><th class="num">Mean error after</th>
<th class="num">Within ±5 pp before</th><th class="num">Within ±5 pp after</th><th class="num">Change</th>
<th class="num">Over / under before</th><th class="num">Over / under after</th></tr></thead>
<tbody>{''.join(cat_rows)}</tbody></table></div>

<h2>Per video</h2>
{''.join(per_video)}

<h2>Bland–Altman: before vs after</h2>
<div class="meta">Each dot is a video: vertical position = predicted − lab. Dots hugging 0 = agreement; dashed lines = bias.</div>
{''.join(f'<div class="card">{p}</div>' for p in plots)}
</main></body></html>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


# Plain-language rows for the stakeholder summary: metric -> (name, what it means, unit)
_PLAIN = [
    ("concentration",     "Sperm count",           "millions of sperm per mL of semen",            "M/mL"),
    ("total_motility",    "Total motility",        "share of sperm that move at all",              "points"),
    ("progressive",       "Progressive motility",  "share of sperm swimming forward",              "points"),
    ("rapid_progressive", "Fast swimmers",         "share swimming forward quickly",               "points"),
    ("slow_progressive",  "Slow swimmers",         "share swimming forward slowly",                "points"),
    ("VAP_mean",          "Swimming speed",        "average speed along the path, micrometres per second", "µm/s"),
    ("non_progressive",   "Moving in place",       "share that move without getting anywhere",     "points"),
]
# metric -> (name, unit, decimals) for the sample-by-sample table
_SAMPLE_COLS = [("concentration", "Sperm count", "M/mL", 1),
                ("total_motility", "Total motility", "%", 0),
                ("VAP_mean", "Swimming speed", "µm/s", 1)]


def _gap(rows: list[dict], vids: set[str] | None = None) -> float | None:
    d = [abs(r["diff"]) for r in rows if vids is None or r["video"] in vids]
    return float(np.mean(d)) if d else None


def stakeholder_report(before: dict, after: dict, out_path: Path, test_ids: set[str],
                       notes: list[str] | None = None, device: str = "lab analyzer") -> Path:
    """One-page, jargon-free before/after summary for non-technical readers.

    Every number is an average absolute difference from the lab report, in the
    measurement's own unit, so it reads as "we were off by X on average".
    """
    from datetime import datetime
    from html import escape

    n = after["total_motility"]["n_videos"]
    n_test = len({r["video"] for r in after["total_motility"]["per_video"]} & test_ids)

    def pct(b, a):
        return (b - a) / b * 100 if b else 0.0

    # -- measurement rows with to-scale bars ---------------------------
    rows, better, worse = [], [], []
    for key, name, meaning, unit in _PLAIN:
        if key not in before or key not in after:
            continue
        b, a = _gap(before[key]["per_video"]), _gap(after[key]["per_video"])
        bt, at = _gap(before[key]["per_video"], test_ids), _gap(after[key]["per_video"], test_ids)
        scale = max(b, a, 1e-9)
        improved = a < b
        (better if improved else worse).append(name)
        change = pct(b, a)
        verdict = (f'<span class="pill good">{change:.0f}% closer</span>' if improved
                   else f'<span class="pill warn">{-change:.0f}% further</span>')
        test_line = ""
        if bt is not None and at is not None:
            test_line = f"On the {n_test} samples kept aside for checking: {bt:.1f} → {at:.1f} {unit}"
            if at > bt:
                test_line = f'<span class="warn-t">{test_line} (worse)</span>'
        rows.append(f"""
<div class="row">
  <div class="what"><div class="name">{name}</div><div class="meaning">{meaning}</div></div>
  <div class="bars">
    <div class="bar-line"><span class="lbl">Before</span><span class="track"><span class="fill before" style="width:{b / scale * 100:.1f}%"></span></span><span class="val">{b:.1f} <small>{unit}</small></span></div>
    <div class="bar-line"><span class="lbl">After</span><span class="track"><span class="fill {'after' if improved else 'after-worse'}" style="width:{a / scale * 100:.1f}%"></span></span><span class="val">{a:.1f} <small>{unit}</small></span></div>
    <div class="sub">{verdict} <span class="muted">{test_line}</span></div>
  </div>
</div>""")

    # -- headline numbers ----------------------------------------------
    conc_b, conc_a = _gap(before["concentration"]["per_video"]), _gap(after["concentration"]["per_video"])
    spd_b, spd_a = _gap(before["VAP_mean"]["per_video"]), _gap(after["VAP_mean"]["per_video"])
    mot_b, mot_a = _gap(before["total_motility"]["per_video"]), _gap(after["total_motility"]["per_video"])
    w5 = lambda rep: rep.get("_motility_error_distribution", {}).get("total_motility", {}).get("n_within_5pp")
    w5b, w5a = w5(before), w5(after)

    # -- sample table ---------------------------------------------------
    by = {k: ({r["video"]: r for r in before[k]["per_video"]}, {r["video"]: r for r in after[k]["per_video"]})
          for k, *_ in _SAMPLE_COLS if k in before and k in after}
    vids = sorted({v for b, _ in by.values() for v in b}, key=lambda s: int(s) if s.isdigit() else s)
    head = "".join(f'<th colspan="3" class="grp">{nm} <small>({u})</small></th>' for _, nm, u, _ in _SAMPLE_COLS)
    sub = "".join('<th class="num">Lab</th><th class="num">Before</th><th class="num">After</th>' for _ in _SAMPLE_COLS)
    body = []
    for v in vids:
        cells = []
        for k, _, _, nd in _SAMPLE_COLS:
            b, a = by[k][0].get(v), by[k][1].get(v)
            if not (b and a):
                cells.append('<td class="num">–</td>' * 3)
                continue
            closer = abs(a["diff"]) < abs(b["diff"])
            cells.append(f'<td class="num lab">{a["actual"]:.{nd}f}</td><td class="num">{b["predicted"]:.{nd}f}</td>'
                         f'<td class="num {"good-t" if closer else "warn-t"}">{a["predicted"]:.{nd}f}</td>')
        tag = ' <span class="tag">check</span>' if v in test_ids else ""
        body.append(f'<tr><td class="sample">{escape(v)}{tag}</td>{"".join(cells)}</tr>')

    notes_html = "".join(f"<li>{escape(x)}</li>" for x in (notes or []))
    worse_txt = (f" It got slightly worse on {', '.join(n.lower() for n in worse)}." if worse else "")

    html = f"""<title>Sperm Analysis Accuracy</title>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap">
<style>
:root{{--bg:#f4f7f8;--card:#ffffff;--ink:#15232a;--muted:#5a6b73;--line:#d9e2e6;--accent:#0f7c86;
--before:#b9c6cc;--good:#1e7a4c;--good-bg:#e3f3ea;--warn:#a55f00;--warn-bg:#fbefdc;color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0f171b;--card:#162128;--ink:#e3ebee;--muted:#93a5ad;--line:#26343b;--accent:#4fc1cb;--before:#4a5a62;--good:#6cd29a;--good-bg:#173427;--warn:#f0b35a;--warn-bg:#3a2a12;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0f171b;--card:#162128;--ink:#e3ebee;--muted:#93a5ad;--line:#26343b;--accent:#4fc1cb;--before:#4a5a62;--good:#6cd29a;--good-bg:#173427;--warn:#f0b35a;--warn-bg:#3a2a12;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:860px;margin:0 auto;padding-inline:16px;padding-block:32px 56px;display:flex;flex-direction:column;gap:28px}}
h1{{font-size:clamp(1.6rem,4vw,2.1rem);line-height:1.2;margin:0;text-wrap:balance}}
h2{{font-size:1.2rem;margin:0 0 4px;text-wrap:balance}}
p{{margin:0;max-width:65ch}}.muted{{color:var(--muted)}}small{{font-size:.78em;color:var(--muted)}}
.eyebrow{{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);font-weight:600}}
.answer{{font-size:1.1rem}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:4px}}
.stat .k{{font-size:.85rem;color:var(--muted)}}
.stat .v{{font:500 1.5rem/1.2 "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.stat .v .arrow{{color:var(--muted);font-size:1rem;margin:0 4px}}
section{{display:flex;flex-direction:column;gap:12px}}
.rows{{background:var(--card);border:1px solid var(--line);border-radius:10px}}
.row{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.6fr);gap:16px;padding:14px 16px;border-top:1px solid var(--line)}}
.row:first-child{{border-top:0}}
.name{{font-weight:600}}.meaning{{font-size:.88rem;color:var(--muted)}}
.bars{{display:flex;flex-direction:column;gap:6px}}
.bar-line{{display:grid;grid-template-columns:3.6rem minmax(0,1fr) 7.5rem;align-items:center;gap:10px;font-size:.88rem}}
.lbl{{color:var(--muted)}}.track{{height:10px;background:color-mix(in srgb,var(--line) 60%,transparent);border-radius:5px;overflow:hidden}}
.fill{{display:block;height:100%;border-radius:5px}}.fill.before{{background:var(--before)}}.fill.after{{background:var(--good)}}.fill.after-worse{{background:var(--warn)}}
.val{{font:500 .95rem "IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;text-align:right}}
.sub{{font-size:.84rem;display:flex;flex-wrap:wrap;gap:8px;align-items:center}}
.pill{{font-weight:600;font-size:.8rem;padding:1px 8px;border-radius:999px}}.pill.good{{color:var(--good);background:var(--good-bg)}}.pill.warn{{color:var(--warn);background:var(--warn-bg)}}
.scroll{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}}
table{{border-collapse:collapse;width:100%;font-size:.9rem}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}}
th{{font-weight:600;color:var(--muted);text-align:left}}th.grp{{text-align:center;color:var(--ink);border-left:1px solid var(--line)}}
td.num,th.num{{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
td.lab{{border-left:1px solid var(--line);font-weight:500}}
.good-t{{color:var(--good);font-weight:600}}.warn-t{{color:var(--warn);font-weight:600}}
.tag{{font-size:.7rem;letter-spacing:.05em;text-transform:uppercase;color:var(--accent);border:1px solid currentColor;border-radius:4px;padding:0 4px;margin-left:4px}}
ul{{margin:0;padding-left:1.2em;display:flex;flex-direction:column;gap:6px;max-width:70ch}}
@media (max-width:620px){{.row{{grid-template-columns:1fr;gap:8px}}.bar-line{{grid-template-columns:3.2rem minmax(0,1fr) 6.5rem}}}}
</style>
<main>
<header style="display:flex;flex-direction:column;gap:10px">
  <div class="eyebrow">Sperm analysis software · accuracy update · {datetime.now():%d %b %Y}</div>
  <h1>{"Our results are now closer to the " + escape(device) if len(better) > len(worse) else "Accuracy against the " + escape(device)}</h1>
  <p class="answer">We compared our software with the lab report on <b>{n} semen samples</b>. After the update it is closer to
  the lab on <b>{len(better)} of {len(better) + len(worse)}</b> measurements: sperm count moved from {conc_b:.1f} to {conc_a:.1f} million/mL off,
  swimming speed from {spd_b:.1f} to {spd_a:.1f} µm/s off.{worse_txt}</p>
</header>

<div class="stats">
  <div class="stat"><span class="k">Sperm count: average difference from the lab</span>
    <span class="v">{conc_b:.1f}<span class="arrow">→</span>{conc_a:.1f} <small>M/mL</small></span></div>
  <div class="stat"><span class="k">Swimming speed: average difference from the lab</span>
    <span class="v">{spd_b:.1f}<span class="arrow">→</span>{spd_a:.1f} <small>µm/s</small></span></div>
  <div class="stat"><span class="k">Total motility: average difference from the lab</span>
    <span class="v">{mot_b:.1f}<span class="arrow">→</span>{mot_a:.1f} <small>points</small></span></div>
  {f'<div class="stat"><span class="k">Samples within 5 points of the lab on motility</span><span class="v">{w5b}<span class="arrow">→</span>{w5a} <small>of {n}</small></span></div>' if w5b is not None else ''}
</div>

<section>
  <div><h2>How far off we are, measurement by measurement</h2>
  <p class="muted">Each number is how far our result was from the lab report on average. Shorter bars are better. "Points" means percentage points: if the lab says 60% and we say 55%, that is 5 points.</p></div>
  <div class="rows">{''.join(rows)}</div>
</section>

<section>
  <div><h2>Sample by sample</h2>
  <p class="muted">The lab's number next to ours before and after the update. Green = closer to the lab than before, amber = further.
  Samples marked <span class="tag">check</span> were kept aside and not used while adjusting the software, so they show how it does on new samples.</p></div>
  <div class="scroll"><table><thead><tr><th rowspan="2">Sample</th>{head}</tr><tr>{sub}</tr></thead>
  <tbody>{''.join(body)}</tbody></table></div>
</section>

{f'<section><h2>What we changed</h2><ul>{notes_html}</ul></section>' if notes_html else ''}

<section>
  <h2>What this does not show yet</h2>
  <ul>
    <li>This is {n} samples. Before clinical use we need a larger set and a second lab's reports.</li>
    <li>The lab analyzer itself varies by about 8–10% between repeat runs of the same sample, so small differences are within its own noise.</li>
    <li>"Moving in place" is still unreliable and is not ready to report on its own.</li>
    <li>Two settings were estimated from our samples because the analyzer's maker does not publish them (its camera speed and counting chamber). Replacing them with the maker's official values is the next step.</li>
  </ul>
</section>
</main>"""
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate predictions against lab reports")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="all", choices=["all", "dev", "test"],
                        help="score only this side of data/video_split.json")
    parser.add_argument("--compare", default=None,
                        help="baseline validation_report JSON: also write a before/after HTML report")
    parser.add_argument("--before-label", default="Before")
    parser.add_argument("--after-label", default="After")
    parser.add_argument("--notes", default=None,
                        help="text file, one line per change, shown under 'What changed'")
    parser.add_argument("--summary-notes", default=None,
                        help="with --compare: plain-language change list; also writes summary_report.html "
                             "(one page for non-technical readers)")
    args = parser.parse_args()
    results = validate(args.config, args.split)
    if args.compare and results:
        with open(args.compare, encoding="utf-8") as f:
            baseline = json.load(f)
        notes = None
        if args.notes:
            with open(args.notes, encoding="utf-8") as f:
                notes = [ln.strip() for ln in f if ln.strip()]
        with open(args.config, encoding="utf-8") as f:
            out_dir = Path(yaml.safe_load(f).get("paths", {}).get("output_dir", "outputs")) / "validation"
        out = comparison_report(baseline, results, out_dir / (
            "comparison_report.html" if args.split == "all" else f"comparison_report_{args.split}.html"),
            args.split, args.before_label, args.after_label, notes)
        print(f"Comparison report saved: {out}")
        if args.summary_notes:
            with open(args.summary_notes, encoding="utf-8") as f:
                plain = [ln.strip() for ln in f if ln.strip()]
            split_file = Path("data/video_split.json")
            test_ids = set(json.load(open(split_file))["test"]) if split_file.exists() else set()
            summ = stakeholder_report(baseline, results, out.with_name("summary_report.html"),
                                      test_ids, plain, device="LensHooke lab analyzer")
            print(f"Stakeholder summary saved: {summ}")
