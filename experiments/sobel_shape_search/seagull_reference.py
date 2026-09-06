#!/usr/bin/env python3
"""Extract the red f1r doodles directly from the Yale IIIF scan.

This is deliberately a reference-QA step, not a semantic classifier.  It finds
red-painted connected components, then derives an ink mask locally around each
component so the Sobel search can use the original manuscript strokes rather
than a modern redraw.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

DEFAULT_URL = "https://collections.library.yale.edu/iiif/2/1006076/full/2400,/0/default.jpg"


def download_rgb(url: str) -> np.ndarray:
    r = requests.get(url, timeout=(10, 60), headers={"User-Agent": "ManuComp-SeagullReference/0.1 (+research)"})
    r.raise_for_status()
    return np.array(Image.open(__import__('io').BytesIO(r.content)).convert("RGB"))


def red_mask(rgb: np.ndarray) -> np.ndarray:
    # Combine HSV saturation/hue evidence with a simple red-excess test.  The
    # pigment is faded/uneven, so requiring either criterion is safer than one
    # brittle RGB threshold.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    R, G, B = [rgb[..., i].astype(np.int16) for i in range(3)]
    hue_red = ((h <= 18) | (h >= 165)) & (s >= 38) & (v >= 55)
    excess = (R >= 80) & ((R - G) >= 16) & ((R - B) >= 10)
    m = (hue_red | excess).astype(np.uint8) * 255
    # Remove isolated pigment/noise and bridge small cracks within painted glyphs.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


def local_ink_mask(rgb_crop: np.ndarray, red_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    # Otsu catches the black outline; a dilated pigment support prevents nearby
    # body text from becoming part of the glyph reference.
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    support = cv2.dilate((red_crop > 0).astype(np.uint8) * 255, np.ones((17, 17), np.uint8), iterations=1)
    ink = ((dark > 0) & (support > 0)) | (red_crop > 0)
    ink = ink.astype(np.uint8) * 255
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Tight crop.
    ys, xs = np.where(ink > 0)
    if len(xs):
        ink = ink[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return ink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--outdir", default="seagull_reference")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rgb = download_rgb(args.url)
    H, W = rgb.shape[:2]
    Image.fromarray(rgb).save(out / "f1r_source.jpg", quality=92)

    rm = red_mask(rgb)
    cv2.imwrite(str(out / "f1r_red_mask.png"), rm)

    n, labels, stats, cents = cv2.connectedComponentsWithStats((rm > 0).astype(np.uint8), 8)
    comps = []
    # Broad area floor; true red doodles are large at 2400-px page width.
    for lab in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[lab]]
        if area < 120 or w < 8 or h < 8:
            continue
        # Ignore page-edge stains; f1r glyphs occur in the written field.
        if x < 0.015 * W or x + w > 0.985 * W or y < 0.015 * H or y + h > 0.985 * H:
            continue
        comps.append((lab, x, y, w, h, area, float(cents[lab][0]), float(cents[lab][1])))

    # Keep the visually substantive red components, sorted top-to-bottom.  This
    # preserves all f1r doodles rather than assuming in code which is &253.
    comps.sort(key=lambda z: z[5], reverse=True)
    comps = comps[:12]
    comps.sort(key=lambda z: (z[2], z[1]))

    annotated = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(annotated)
    records = []
    tiles = []
    for idx, (lab, x, y, w, h, area, cx, cy) in enumerate(comps, 1):
        pad = max(14, int(round(max(w, h) * 0.22)))
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(W, x+w+pad), min(H, y+h+pad)
        crop = rgb[y0:y1, x0:x1]
        comp_red = ((labels[y0:y1, x0:x1] == lab).astype(np.uint8) * 255)
        ink = local_ink_mask(crop, comp_red)

        Image.fromarray(crop).save(out / f"component_{idx:02d}_crop.jpg", quality=95)
        cv2.imwrite(str(out / f"component_{idx:02d}_red.png"), comp_red)
        cv2.imwrite(str(out / f"component_{idx:02d}_ink_mask.png"), ink)

        # Shape descriptors useful for identifying the squiggle-bearing glyph.
        ys, xs = np.where(ink > 0)
        ink_h, ink_w = ink.shape[:2]
        aspect = ink_h / max(1.0, ink_w)
        fill = float((ink > 0).mean())
        # Central-column occupancy versus side occupancy: a vertical central
        # squiggle should raise this without us using it as a hard selector.
        bw = max(1, ink_w // 5)
        mid = ink[:, max(0, ink_w//2-bw//2):min(ink_w, ink_w//2+(bw+1)//2)] > 0
        sides = np.concatenate([(ink[:, :bw] > 0), (ink[:, -bw:] > 0)], axis=1) if ink_w >= 2*bw else (ink > 0)
        central_occupancy = float(mid.mean()) if mid.size else 0.0
        side_occupancy = float(sides.mean()) if sides.size else 0.0

        rec = {
            "component": idx,
            "label": lab,
            "bbox": [x, y, w, h],
            "bbox_norm": [x/W, y/H, w/W, h/H],
            "red_area": area,
            "centroid": [cx, cy],
            "ink_size": [ink_w, ink_h],
            "ink_aspect_h_over_w": aspect,
            "ink_fill": fill,
            "central_occupancy": central_occupancy,
            "side_occupancy": side_occupancy,
            "crop_file": f"component_{idx:02d}_crop.jpg",
            "mask_file": f"component_{idx:02d}_ink_mask.png",
        }
        records.append(rec)

        draw.rectangle((x0, y0, x1, y1), outline=(0, 80, 255), width=5)
        draw.text((x0+4, y0+4), f"C{idx}", fill=(0, 40, 220))

        # Contact-sheet tile: crop on left, mask on right.
        cimg = Image.fromarray(crop).convert("RGB")
        mimg = Image.fromarray(ink).convert("RGB")
        tile_h = 260
        scale = min(1.0, tile_h / max(cimg.height, 1))
        cimg = cimg.resize((max(1, int(cimg.width*scale)), max(1, int(cimg.height*scale))))
        mscale = min(1.0, tile_h / max(mimg.height, 1))
        mimg = mimg.resize((max(1, int(mimg.width*mscale)), max(1, int(mimg.height*mscale))))
        tile_w = max(540, cimg.width + mimg.width + 30)
        tile = Image.new("RGB", (tile_w, 300), "white")
        td = ImageDraw.Draw(tile)
        td.text((8, 6), f"C{idx}  bbox={x},{y},{w},{h}  red_area={area}", fill="black")
        tile.paste(cimg, (8, 32))
        tile.paste(mimg, (18+cimg.width, 32))
        tiles.append(tile)

    annotated.save(out / "f1r_components_annotated.jpg", quality=92)
    if tiles:
        sheet_w = max(t.width for t in tiles)
        sheet = Image.new("RGB", (sheet_w, sum(t.height for t in tiles)), "white")
        yy = 0
        for t in tiles:
            sheet.paste(t, (0, yy)); yy += t.height
        sheet.save(out / "component_contact_sheet.jpg", quality=94)

    payload = {
        "source_url": args.url,
        "source_width": W,
        "source_height": H,
        "method": "HSV/red-excess connected components + local ink support",
        "note": "No component is auto-labelled as the seagull; verify contact sheet before search.",
        "components": records,
    }
    (out / "components.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
