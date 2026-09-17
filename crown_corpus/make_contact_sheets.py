#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

A0_LANDSCAPE_150DPI = (7016, 4967)


def fit_on_white(path: Path, box_w: int, box_h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((box_w, box_h))
    canvas = Image.new("RGB", (box_w, box_h), "white")
    x = (box_w - im.width) // 2
    y = (box_h - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def make_sheet(rows, out_path: Path, title: str, cols=12, grid_rows=8):
    W, H = A0_LANDSCAPE_150DPI
    margin, gap, title_h = 80, 16, 70
    cell_w = (W - 2 * margin - (cols - 1) * gap) // cols
    cell_h = (H - 2 * margin - title_h - (grid_rows - 1) * gap) // grid_rows
    sheet = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 22), title, fill="black")
    for j, row in enumerate(rows):
        r, c = divmod(j, cols)
        x = margin + c * (cell_w + gap)
        y = margin + title_h + r * (cell_h + gap)
        p = Path(row["crop_path"])
        tile = fit_on_white(p, cell_w, cell_h - 28)
        sheet.paste(tile, (x, y))
        # Blind ID only; no provenance/date/subject.
        draw.text((x + 4, y + cell_h - 24), row["blind_id"], fill="black")
    sheet.save(out_path, quality=92, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, default=Path("crown_corpus/output/detections.json"))
    ap.add_argument("--out", type=Path, default=Path("crown_corpus/output/a0"))
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--min-score", type=float, default=0.18)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = json.loads(args.detections.read_text(encoding="utf-8"))
    rows = [r for r in rows if float(r.get("score", 0)) >= args.min_score and Path(r["crop_path"]).exists()]
    # Primary sheet uses only top-ranked detection per source image to reduce repeated false positives.
    top = [r for r in rows if int(r.get("rank", 99)) == 1]
    rng = random.Random(args.seed)
    rng.shuffle(top)
    for i, r in enumerate(top, 1):
        r["blind_id"] = f"C{i:04d}"

    per_sheet = 12 * 8
    for s in range(0, len(top), per_sheet):
        chunk = top[s:s + per_sheet]
        make_sheet(chunk, args.out / f"blind_random_{s // per_sheet + 1:02d}.jpg",
                   f"Voynich crown null — blind randomized sheet {s // per_sheet + 1}")

    # Separate confidence-sorted QC sheets; still provenance blind.
    conf = sorted(top, key=lambda r: float(r.get("score", 0)), reverse=True)
    for s in range(0, len(conf), per_sheet):
        chunk = conf[s:s + per_sheet]
        make_sheet(chunk, args.out / f"qc_confidence_{s // per_sheet + 1:02d}.jpg",
                   f"Crown detector QC — confidence sheet {s // per_sheet + 1}")

    # Key is intentionally separate from sheets so blind scoring can happen first.
    key = [{"blind_id": r["blind_id"], "object_url": r.get("object_url"), "image_path": r.get("image_path"),
            "crop_path": r.get("crop_path"), "score": r.get("score"), "label": r.get("label")}
           for r in top]
    (args.out / "blind_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"top_detection_crops": len(top), "random_a0_sheets": math.ceil(len(top) / per_sheet) if top else 0}, indent=2))


if __name__ == "__main__":
    main()
