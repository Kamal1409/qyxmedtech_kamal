"""
OCR Report Parser -- Extract Ground Truth from Lab Report Images

Parses the structured semen analysis report JPG images using OCR
(Tesseract) to extract quantitative values for validation.

The lab reports have a consistent layout with sections:
    I. Sample Information
    II. Test Results (concentration, motility, morphology)
    Detail section (MSC, PMSC, morphology breakdown, kinematics)

Usage:
    python -m src.ocr_report --config configs/default.yaml
    python -m src.ocr_report --config configs/default.yaml --report data/report/1.jpg
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import yaml

try:
    from PIL import Image, ImageOps
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def _preprocess_for_ocr(image_path: Path, upscale: float = 2.0) -> "Image.Image":
    """
    Prepare a report image for more accurate OCR.

    The lab reports are rendered at a fairly low resolution, which makes
    Tesseract drop decimal points (e.g. "18.8" -> "188", "4.6" -> "46").
    Up-scaling + grayscale + autocontrast recovers most of those decimals.
    """
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")  # grayscale
    if upscale and upscale != 1.0:
        w, h = img.size
        img = img.resize((int(w * upscale), int(h * upscale)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    return img


def _normalize_ocr_text(text: str) -> str:
    """Map micro-sign variants to ASCII 'u' so unit regexes match
    (µm/s, μm/s -> um/s) and normalise odd whitespace."""
    return (
        text.replace("\u00b5", "u")  # MICRO SIGN
        .replace("\u03bc", "u")        # GREEK SMALL LETTER MU
        .replace("\u2009", " ")        # thin space
        .replace("\u00a0", " ")        # non-breaking space
    )


def _result_part(line: str) -> str:
    """
    Return only the leftmost 'Result' portion of a report row, discarding the
    'WHO 6th' reference and '95% CI' columns. Those columns always begin with
    one of >=, <=, < or '[', so the actual measured value is everything to the
    left of the first such marker. This stops reference thresholds (e.g. the
    "16" in ">= 16 M/mL") from being mistaken for the measured value.
    """
    cut = len(line)
    for marker in ("\u2265", "\u2264", "<", "["):  # >=, <=, <, [
        idx = line.find(marker)
        if idx != -1 and idx < cut:
            cut = idx
    return line[:cut]


def _validate_extracted(result: dict) -> list[str]:
    """Sanity-check extracted values and return a list of human-readable
    warnings for any that look physically implausible."""
    warnings: list[str] = []

    for key, val in result.items():
        if key.endswith("_pct") and val is not None and not (0.0 <= val <= 100.0):
            warnings.append(f"{key}={val} outside [0, 100]")

    vcl = result.get("VCL_um_per_s")
    vap = result.get("VAP_um_per_s")
    vsl = result.get("VSL_um_per_s")
    if None not in (vcl, vap, vsl) and not (vcl >= vap >= vsl):
        warnings.append(f"velocity ordering VCL>=VAP>=VSL violated (VCL={vcl}, VAP={vap}, VSL={vsl})")

    vol = result.get("ejaculate_volume_ml")
    if vol is not None and not (0.1 <= vol <= 12.0):
        warnings.append(f"ejaculate_volume_ml={vol} implausible (expected ~0.1-12 mL)")

    return warnings


def extract_number(text: str) -> float | None:
    """Extract the first number from a text string."""
    match = re.search(r"(\d+\.?\d*)", text.replace(",", ""))
    if match:
        return float(match.group(1))
    return None


def extract_percentage(text: str) -> float | None:
    """Extract a percentage value from text."""
    match = re.search(r"(\d+\.?\d*)\s*%", text)
    if match:
        return float(match.group(1))
    return None


def extract_value_with_unit(text: str, unit_pattern: str) -> float | None:
    """Extract a numeric value followed by a specific unit pattern."""
    pattern = rf"(\d+\.?\d*)\s*{unit_pattern}"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def parse_report_image(image_path: Path, tesseract_cmd: str | None = None, lang: str = "eng", psm: int = 6) -> dict:
    """
    Parse a semen analysis report image using OCR.

    Args:
        image_path:     Path to the report JPG image.
        tesseract_cmd:  Path to tesseract executable (None = system PATH).
        lang:           OCR language.
        psm:            Page segmentation mode.

    Returns:
        dict with extracted ground truth values.
    """
    if not OCR_AVAILABLE:
        raise ImportError("Install pytesseract and Pillow: pip install pytesseract Pillow")

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    # Up-scale + grayscale the image so Tesseract reliably reads decimals.
    img = _preprocess_for_ocr(image_path)

    # Run OCR
    custom_config = f"--psm {psm} -l {lang}"
    text = pytesseract.image_to_string(img, config=custom_config)
    text = _normalize_ocr_text(text)

    lines = text.split("\n")
    lines = [line.strip() for line in lines if line.strip()]

    # Initialize result
    result = {
        "video_id": image_path.stem,
        "source_file": str(image_path),
        "raw_ocr_text": text,
        # Sample info
        "ejaculate_volume_ml": None,
        "ph": None,
        # Concentration
        "concentration_m_per_ml": None,
        "total_sperm_number_m": None,
        # Motility
        "total_motility_pct": None,
        "progressive_pct": None,
        "rapid_progressive_pct": None,
        "slow_progressive_pct": None,
        "non_progressive_pct": None,
        "immotile_pct": None,
        # Morphology
        "normal_forms_pct": None,
        "abnormal_heads_pct": None,
        "abnormal_midpieces_pct": None,
        "abnormal_tails_pct": None,
        # Concentration details
        "msc_m_per_ml": None,
        "pmsc_m_per_ml": None,
        # Kinematics
        "LIN_pct": None,
        "STR_pct": None,
        "WOB_pct": None,
        "VAP_um_per_s": None,
        "VSL_um_per_s": None,
        "VCL_um_per_s": None,
        "ALH_um": None,
        "BCF_hz": None,
    }

    # -- Parse line by line ---------------------------------------------
    for i, line in enumerate(lines):
        line_lower = line.lower()
        # For rows in the "II. Test Result" table, parse ONLY the result
        # column so the WHO reference / 95% CI columns can't leak in.
        rline = _result_part(line)

        # Ejaculate volume
        if "ejaculate volume" in line_lower or "volume" in line_lower:
            val = extract_value_with_unit(rline, r"m[lL]")
            if val:
                result["ejaculate_volume_ml"] = val

        # pH (may be printed as "> 8.0") -- search the full line
        if line_lower.startswith("ph") or " ph" in line_lower:
            val = extract_number(line.split("ph")[-1] if "ph" in line_lower else line)
            if val and 5.0 <= val <= 10.0:
                result["ph"] = val

        # Concentration
        if "concentration" in line_lower and "motile" not in line_lower and "progressive" not in line_lower:
            val = extract_value_with_unit(rline, r"M/m[lL]")
            if val:
                result["concentration_m_per_ml"] = val

        # Total sperm number
        if "total sperm number" in line_lower:
            val = extract_value_with_unit(rline, r"M\b")
            if val:
                result["total_sperm_number_m"] = val

        # Total Motility
        if "total motility" in line_lower:
            val = extract_percentage(rline)
            if val is not None:
                result["total_motility_pct"] = val

        # Progressive (but not rapid/slow/non)
        if "progressive" in line_lower and "rapid" not in line_lower and "slow" not in line_lower and "non" not in line_lower:
            val = extract_percentage(rline)
            if val is not None:
                result["progressive_pct"] = val

        # Rapid Progressive
        if "rapid" in line_lower and "progressive" in line_lower:
            val = extract_percentage(rline)
            if val is not None:
                result["rapid_progressive_pct"] = val

        # Slow Progressive
        if "slow" in line_lower and "progressive" in line_lower:
            val = extract_percentage(rline)
            if val is not None:
                result["slow_progressive_pct"] = val

        # Non-Progressive
        if "non-progressive" in line_lower or "non progressive" in line_lower:
            val = extract_percentage(rline)
            if val is not None:
                result["non_progressive_pct"] = val

        # Immotile
        if "immotile" in line_lower:
            val = extract_percentage(rline)
            if val is not None:
                result["immotile_pct"] = val

        # Normal forms (morphology)
        if "normal form" in line_lower:
            val = extract_percentage(rline)
            if val is not None:
                result["normal_forms_pct"] = val

        # Abnormal heads
        if "abnormal head" in line_lower:
            val = extract_percentage(line)
            if val is not None:
                result["abnormal_heads_pct"] = val

        # Abnormal midpieces
        if "abnormal midpiece" in line_lower:
            val = extract_percentage(line)
            if val is not None:
                result["abnormal_midpieces_pct"] = val

        # Abnormal tails
        if "abnormal tail" in line_lower:
            val = extract_percentage(line)
            if val is not None:
                result["abnormal_tails_pct"] = val

        # MSC
        if line_lower.startswith("msc") or "motile sperm concentration" in line_lower:
            val = extract_value_with_unit(line, r"M/m[lL]")
            if val:
                result["msc_m_per_ml"] = val

        # PMSC
        if line_lower.startswith("pmsc") or "progressive motile sperm" in line_lower:
            val = extract_value_with_unit(line, r"M/m[lL]")
            if val:
                result["pmsc_m_per_ml"] = val

        # Kinematics (Detail section -- single value column, use full line)
        if "lin" in line_lower and "mean" in line_lower:
            val = extract_percentage(line)
            if val is not None:
                result["LIN_pct"] = val

        if "str" in line_lower and "mean" in line_lower and "straight" not in line_lower:
            val = extract_percentage(line)
            if val is not None:
                result["STR_pct"] = val

        if "wob" in line_lower and "mean" in line_lower:
            val = extract_percentage(line)
            if val is not None:
                result["WOB_pct"] = val

        if "vap" in line_lower and "mean" in line_lower:
            val = extract_value_with_unit(line, r"um/s")
            if val:
                result["VAP_um_per_s"] = val

        if "vsl" in line_lower and "mean" in line_lower:
            val = extract_value_with_unit(line, r"um/s")
            if val:
                result["VSL_um_per_s"] = val

        if "vcl" in line_lower and "mean" in line_lower:
            val = extract_value_with_unit(line, r"um/s")
            if val:
                result["VCL_um_per_s"] = val

        if "alh" in line_lower and "mean" in line_lower:
            val = extract_value_with_unit(line, r"um\b")
            if val:
                result["ALH_um"] = val

        if "bcf" in line_lower and "mean" in line_lower:
            val = extract_value_with_unit(line, r"Hz")
            if val:
                result["BCF_hz"] = val

    # Remove raw OCR text from the structured output (keep it separate)
    ocr_text = result.pop("raw_ocr_text")

    # Sanity-check the extracted values and attach any warnings
    warnings = _validate_extracted(result)
    result["_warnings"] = warnings

    # Count extracted fields
    extracted = sum(1 for v in result.values() if v is not None and v != image_path.stem and v != str(image_path))
    print(f"  Report {image_path.name}: extracted {extracted} fields", end="")
    if warnings:
        print(f"  [!] {len(warnings)} warning(s): {'; '.join(warnings)}")
    else:
        print()

    result["raw_ocr_text"] = ocr_text
    return result


def run_ocr(config_path: str = "configs/default.yaml", report_path: str | None = None) -> None:
    """
    Run OCR on all report images (or a single report).
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg.get("paths", {})
    ocr_cfg = cfg.get("ocr", {})

    report_dir = Path(paths_cfg.get("report_dir", "data/report"))
    gt_file = Path(paths_cfg.get("ground_truth_file", "data/ground_truth.json"))

    tesseract_cmd = ocr_cfg.get("tesseract_cmd", None)
    lang = ocr_cfg.get("lang", "eng")
    psm = ocr_cfg.get("psm", 6)

    # Find report images
    if report_path:
        report_files = [Path(report_path)]
    else:
        report_files = sorted(report_dir.glob("*.jpg")) + sorted(report_dir.glob("*.png"))

    if not report_files:
        print(f"No report images found in {report_dir}")
        return

    print(f"=== OCR: Ground Truth Extraction ===")
    print(f"Processing {len(report_files)} report image(s)...\n")

    ground_truth = {}
    for rf in report_files:
        try:
            result = parse_report_image(rf, tesseract_cmd, lang, psm)
            video_id = result["video_id"]
            ground_truth[video_id] = result
        except Exception as e:
            print(f"  [!] Error processing {rf.name}: {e}")

    # Save ground truth (without raw OCR text for cleaner output)
    gt_clean = {}
    for vid, data in ground_truth.items():
        gt_clean[vid] = {k: v for k, v in data.items() if k != "raw_ocr_text"}

    with open(gt_file, "w") as f:
        json.dump(gt_clean, f, indent=2)

    print(f"\nGround truth saved: {gt_file}")
    print(f"Extracted data for {len(gt_clean)} video(s)")

    # Print summary
    print("\n-- Extracted Values Summary --")
    for vid, data in gt_clean.items():
        print(f"\n  Video {vid}:")
        for key, val in data.items():
            if val is not None and key not in ("video_id", "source_file"):
                print(f"    {key}: {val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR: Extract ground truth from lab report images")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report", default=None, help="Path to a specific report image")
    args = parser.parse_args()
    run_ocr(args.config, args.report)
