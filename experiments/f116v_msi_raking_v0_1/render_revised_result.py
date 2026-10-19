#!/usr/bin/env python3
"""Render a conservative combined f116v result after all controls.

The post-hoc morphology audit is diagnostic only. It may demote an automatic
positive to 'not validated'; it may not promote a result or establish absence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def candidate_audit(out: Path) -> dict:
    candidate = load_mask(out / "candidate_mask.png")
    h, w = candidate.shape

    # Acquisition-specific page rectangle, then a conservative 8% inset.
    x0, x1 = int(0.133 * w), int(0.925 * w)
    y0, y1 = int(0.31 * h), h
    mx, my = int(0.08 * (x1 - x0)), int(0.08 * (y1 - y0))
    page = np.zeros_like(candidate)
    page[y0:y1, x0:x1] = True
    interior = np.zeros_like(candidate)
    interior[y0 + my:y1 - my, x0 + mx:x1 - mx] = True

    # The question concerns the otherwise blank field, not the known marginalia.
    central = interior.copy()
    central[:, :int(x0 + 0.18 * (x1 - x0))] = False
    central[:, int(x0 + 0.82 * (x1 - x0)):] = False
    tested = candidate & central

    # Diagnostic grouping in the writing direction. A line-like group must be
    # materially wider than high after closing only small horizontal gaps.
    closed = cv2.morphologyEx(
        tested.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8)
    )
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    line_groups = []
    for i in range(1, n):
        x, y, width, height, area = [int(v) for v in stats[i]]
        ratio = float(width / max(1, height))
        if area >= 20 and width >= 30 and height <= 24 and ratio >= 3.0:
            line_groups.append(
                {"x": x, "y": y, "width": width, "height": height, "area": area, "width_height_ratio": ratio}
            )

    raw = int(candidate.sum())
    outside_or_border = int((candidate & (~interior)).sum())
    metrics = {
        "status": "POST_HOC_DIAGNOSTIC_ONLY",
        "raw_candidate_pixels": raw,
        "candidate_pixels_outside_or_in_page_border_band": outside_or_border,
        "outside_or_border_fraction": float(outside_or_border / raw) if raw else 0.0,
        "central_blank_field_candidate_pixels": int(tested.sum()),
        "horizontal_line_group_count": len(line_groups),
        "horizontal_line_groups": line_groups,
        "interpretation": (
            "The automatic candidates are dominated by page-edge/margin structure and do not form "
            "horizontal line groups in the central blank-field test region. Because this audit was "
            "introduced after viewing the real result, it can invalidate a positive but cannot prove absence."
        ),
    }
    (out / "posthoc_candidate_audit.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.result_dir

    original = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    recto = json.loads((out / "recto_control.json").read_text(encoding="utf-8"))
    audit = candidate_audit(out)

    recto_verdict = recto["verdict"]
    if recto_verdict == "NO_RECTO_INDEPENDENT_ERASED_TEXT_SIGNAL_AT_PREFLIGHT_RESOLUTION":
        combined = "NO_VALIDATED_ERASED_TEXT_SIGNAL_AT_PREFLIGHT_RESOLUTION"
        recto_text = "The recto control explains the preregistered candidate sufficiently to reject it at preflight resolution."
    else:
        combined = "ERASED_TEXT_NOT_VALIDATED_RECTO_CONTROL_INCONCLUSIVE"
        recto_text = (
            "The lower-resolution recto control is inconclusive. It therefore cannot establish whether every residual "
            "candidate is independent of f116r show-through."
        )

    report = f"""# Final controlled f116v preflight result

## Answers

### 1. Surviving visible text

Verdict: `{original.get('visible_text_status', 'UNKNOWN')}`

The multiband grouped-holdout F1 was {original['ink_model']['metrics']['f1']:.4f}, versus {original['ink_model']['metrics']['best_single_band_f1']:.4f} for the best individual band. The fusion therefore did not produce a validated quantitative gain. Deterministic panels remain useful inspection aids, but no additional reading should be claimed from this preflight alone.

### 2. Erased or washed text

Raw detector: `{original.get('hidden_text_status', 'UNKNOWN')}`

Final controlled status: `{combined}`

The physically constrained f116r comparison explains {recto['recto_explained_fraction']:.1%} of candidate pixels in its test region. {recto_text}

The post-hoc geometry audit found that {audit['outside_or_border_fraction']:.1%} of raw candidate pixels lie outside the inset page interior or in its border band. In the central blank-field region, {audit['central_blank_field_candidate_pixels']} candidate pixels remain, but they form **{audit['horizontal_line_group_count']} horizontal line groups** under the diagnostic rule. This demotes the raw automatic positive: it does not validate erased writing. Because the geometry audit was added after inspection, it is not evidence of absence.

### 3. Physical imprint or indentation

Verdict: `{original.get('physical_imprint_status', 'UNKNOWN')}`

The inventory contains spectral reflectance and transmitted-light captures but no matched opposite-direction raking-light pair. Relief or indentation is therefore not identifiable from this image set.

## Required next evidence

A decisive second stage needs the native-resolution matching f116r MegaVision cube for sheet-through registration and newly acquired calibrated multi-light/raking images from at least four directions. Full-resolution learned registration and restoration may improve visualization, but no generative output may count as recovered text unless the stroke is independently supported by acquired bands.
"""
    (out / "FINAL_CONTROLLED_RESULT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"combined_status": combined, "line_groups": audit["horizontal_line_group_count"]}, indent=2))


if __name__ == "__main__":
    main()
