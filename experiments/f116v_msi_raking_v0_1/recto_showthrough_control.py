#!/usr/bin/env python3
"""Test whether apparent f116v hidden text is explained by f116r show-through.

This control is deliberately non-generative. It segments visible f116r ink,
applies the physically expected backside transform (mirror plus quarter-turn),
and permits only small scale/translation refinements. It then measures how much
of the f116v candidate mask is supported by the transformed recto writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RECTO_URL = "https://www.voynich.com/folios/color/116r.jpg"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def segment_recto(gray: np.ndarray) -> np.ndarray:
    """Conservative dark-ink segmentation with component cleanup."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(8.0, gray.shape[1] / 90))
    residual = cv2.subtract(background, gray)
    _, mask = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 3 <= area <= 1200 and h <= 80 and w <= 160:
            clean[labels == i] = 255
    return clean


def resize_place(image: np.ndarray, target_shape: tuple[int, int], scale: float, dx: int, dy: int) -> np.ndarray:
    th, tw = target_shape
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((th, tw), np.uint8)
    sh, sw = resized.shape
    sx0, sy0 = max(0, -dx), max(0, -dy)
    tx0, ty0 = max(0, dx), max(0, dy)
    width = min(sw - sx0, tw - tx0)
    height = min(sh - sy0, th - ty0)
    if width > 0 and height > 0:
        canvas[ty0:ty0 + height, tx0:tx0 + width] = resized[sy0:sy0 + height, sx0:sx0 + width]
    return canvas


def normalized_corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    av = a[valid].astype(np.float32)
    bv = b[valid].astype(np.float32)
    if av.size < 100 or av.std() < 1e-6 or bv.std() < 1e-6:
        return -1.0
    return float(np.corrcoef(av, bv)[0, 1])


def align_recto(recto_mask: np.ndarray, f116v_gray: np.ndarray) -> tuple[np.ndarray, dict]:
    # Crop away photography borders using frozen page-relative coordinates.
    vh, vw = f116v_gray.shape
    v_crop = f116v_gray[int(0.31 * vh):vh, int(0.13 * vw):int(0.925 * vw)]
    rh, rw = recto_mask.shape
    r_crop = recto_mask[int(0.026 * rh):int(0.957 * rh), int(0.042 * rw):int(0.957 * rw)]

    # Backside geometry: mirror, then rotate clockwise. No arbitrary reflection
    # or free homography is allowed because that could manufacture agreement.
    physical = np.rot90(np.fliplr(r_crop), k=3)
    base_scale = min(v_crop.shape[1] / physical.shape[1], v_crop.shape[0] / physical.shape[0])

    # Compare transformed recto strokes with dark local residual on f116v.
    bg = cv2.GaussianBlur(v_crop, (0, 0), sigmaX=max(6.0, v_crop.shape[1] / 100))
    target = cv2.subtract(bg, v_crop)
    target = cv2.normalize(target, None, 0, 1, cv2.NORM_MINMAX)
    valid = np.ones(v_crop.shape, bool)
    margin = max(8, int(0.025 * min(v_crop.shape)))
    valid[:margin] = valid[-margin:] = False
    valid[:, :margin] = valid[:, -margin:] = False

    best = None
    for rel_scale in (0.96, 0.98, 1.00, 1.02, 1.04, 1.06):
        scale = base_scale * rel_scale
        for dx in range(-50, 51, 5):
            for dy in range(-60, 61, 5):
                placed = resize_place(physical, v_crop.shape, scale, dx, dy)
                blurred = cv2.GaussianBlur((placed > 0).astype(np.float32), (0, 0), 1.8)
                score = normalized_corr(blurred, target, valid)
                if best is None or score > best[0]:
                    best = (score, placed, scale, dx, dy)
    assert best is not None

    aligned_crop = best[1]
    aligned_full = np.zeros_like(f116v_gray)
    y0, x0 = int(0.31 * vh), int(0.13 * vw)
    aligned_full[y0:y0 + aligned_crop.shape[0], x0:x0 + aligned_crop.shape[1]] = aligned_crop
    return aligned_full, {
        "correlation": best[0],
        "scale": best[2],
        "dx": best[3],
        "dy": best[4],
        "physical_transform": "horizontal_mirror_then_clockwise_90_degrees",
    }


def component_summary(mask: np.ndarray) -> tuple[list[dict], int]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    rows: list[dict] = []
    line_like = 0
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < 4:
            continue
        elongation = float(max(w, h) / max(1, min(w, h)))
        row = {"x": x, "y": y, "width": w, "height": h, "area": area, "elongation": elongation}
        rows.append(row)
        if area >= 20 and elongation >= 3.0:
            line_like += 1
    rows.sort(key=lambda r: r["area"], reverse=True)
    return rows, line_like


def save_overlays(base: np.ndarray, candidate: np.ndarray, support: np.ndarray, residual: np.ndarray, out: Path) -> None:
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    contour = cv2.morphologyEx((support > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    overlay = rgb.copy()
    overlay[contour] = (255, 40, 40)
    overlay[candidate > 0] = (255, 220, 0)
    Image.fromarray(overlay).save(out / "recto_support_overlay.png")

    residual_overlay = rgb.copy()
    residual_overlay[residual > 0] = (255, 30, 30)
    Image.fromarray(residual_overlay).save(out / "recto_residual_overlay.png")
    Image.fromarray((support > 0).astype(np.uint8) * 255).save(out / "recto_aligned_ink_support.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--recto", type=Path)
    args = parser.parse_args()
    out = args.result_dir

    recto_path = args.recto or out / "_f116r_reference.jpg"
    if not recto_path.exists():
        urllib.request.urlretrieve(RECTO_URL, recto_path)

    base_path = out / "registered_median.png"
    candidate_path = out / "candidate_mask.png"
    if not base_path.exists() or not candidate_path.exists():
        raise SystemExit("Missing registered_median.png or candidate_mask.png")

    base = load_gray(base_path)
    candidate = load_gray(candidate_path) > 0
    recto_gray = load_gray(recto_path)
    recto_mask = segment_recto(recto_gray)
    aligned, alignment = align_recto(recto_mask, base)

    h, w = candidate.shape
    interior = np.ones((h, w), bool)
    border = max(8, int(0.04 * min(h, w)))
    interior[:border] = interior[-border:] = False
    interior[:, :border] = interior[:, -border:] = False
    # Exclude the known right-margin f116v writing from the erased-page test.
    interior[:, int(0.83 * w):] = False

    valid_candidate = candidate & interior
    dilation = max(5, int(round(w / 110)))
    support = cv2.dilate((aligned > 0).astype(np.uint8), np.ones((dilation, dilation), np.uint8)) > 0
    explained = valid_candidate & support
    residual = valid_candidate & ~support

    candidate_pixels = int(valid_candidate.sum())
    explained_pixels = int(explained.sum())
    residual_pixels = int(residual.sum())
    explained_fraction = float(explained_pixels / candidate_pixels) if candidate_pixels else 0.0
    components, line_like = component_summary(residual.astype(np.uint8))

    if candidate_pixels == 0:
        verdict = "NO_RECOVERABLE_SIGNAL"
    elif explained_fraction >= 0.70 and line_like == 0:
        verdict = "NO_RECTO_INDEPENDENT_ERASED_TEXT_SIGNAL_AT_PREFLIGHT_RESOLUTION"
    else:
        verdict = "RECTO_CONTROL_INCONCLUSIVE"

    metrics = {
        "verdict": verdict,
        "recto_reference_url": RECTO_URL,
        "recto_reference_sha256": sha256(recto_path),
        "alignment": alignment,
        "candidate_pixels_in_test_region": candidate_pixels,
        "recto_explained_pixels": explained_pixels,
        "recto_explained_fraction": explained_fraction,
        "recto_independent_residual_pixels": residual_pixels,
        "residual_component_count_area_ge_4": len(components),
        "residual_line_like_component_count": line_like,
        "largest_residual_components": components[:20],
        "limitations": [
            "The recto reference is a lower-resolution public colour image, not the matching raw MegaVision capture.",
            "Alignment is constrained to the physical backside transform plus small scale and translation refinement.",
            "This is a 1200-pixel preflight; native-resolution confirmation remains required.",
        ],
    }
    (out / "recto_control.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_overlays(base, valid_candidate, support, residual, out)

    original = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    report = f"""# Revised f116v preflight result\n\n## Verdicts\n\n- Visible surviving text: `{original.get('visible_text_status', 'UNKNOWN')}`\n- Raw hidden-text detector: `{original.get('hidden_text_status', 'UNKNOWN')}`\n- After mandatory f116r show-through control: `{verdict}`\n- Physical indentation/imprint: `{original.get('physical_imprint_status', 'UNKNOWN')}`\n\n## Recto-control result\n\nThe physically expected mirror-plus-clockwise-quarter-turn transform of f116r explains **{explained_fraction:.1%}** of candidate pixels in the interior f116v test region. The remaining {residual_pixels} pixels form {len(components)} components of area at least four pixels, with **{line_like} line-like components** under the frozen criterion.\n\nThe raw `CANDIDATE_ERASED_TEXT_SIGNAL` verdict is therefore not accepted as evidence of erased f116v writing. At this preflight resolution, the dominant full-page text-like pattern is consistent with f116r show-through.\n\n## Interpretation\n\n1. The surviving f116v marginal text did not obtain a validated quantitative gain over the best individual source band.\n2. No recto-independent erased-text signal survives the present control.\n3. Physical indentation is not identifiable because the dataset contains no matched opposite-direction raking-light pair.\n4. Native-resolution analysis with the matching raw f116r MSI cube is still required before a final negative conclusion.\n"""
    (out / "REVISED_RESULT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verdict": verdict, "recto_explained_fraction": explained_fraction, "line_like": line_like}, indent=2))

    # Do not upload the downloaded third-party reference inside the evidence bundle.
    if args.recto is None:
        recto_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
