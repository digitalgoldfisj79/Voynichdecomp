#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def atomic_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def expand_box(box, w, h, pad=0.18):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    return [
        max(0, int(x1 - bw * pad)),
        max(0, int(y1 - bh * pad)),
        min(w, int(x2 + bw * pad)),
        min(h, int(y2 + bh * pad)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("crown_corpus/output/manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("crown_corpus/output"))
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--box-threshold", type=float, default=0.18)
    ap.add_argument("--text-threshold", type=float, default=0.18)
    ap.add_argument("--max-per-image", type=int, default=5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out / "crops"
    crops_dir.mkdir(exist_ok=True)
    state_path = args.out / "detection_state.json"
    state = {"model": args.model, "records": {}, "failures": []}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    model.eval()
    # Broad prompts deliberately favour recall; later A0 audit removes false positives.
    text = "crown. coronet. royal crown. imperial crown. tiara."

    for rec in rows:
        key = rec.get("object_url") or rec.get("image_path")
        if not key or key in state["records"]:
            continue
        outrec = {"object_url": rec.get("object_url"), "image_path": rec.get("image_path"), "detections": []}
        try:
            p = Path(rec.get("image_path", ""))
            if not p.exists():
                outrec["status"] = "image_missing"
                state["records"][key] = outrec
                atomic_json(state, state_path)
                continue
            image = Image.open(p).convert("RGB")
            # Keep detector input bounded while retaining small crown structure.
            det_image = image.copy()
            det_image.thumbnail((1280, 1280))
            inputs = processor(images=det_image, text=text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[det_image.size[::-1]],
            )[0]
            sx = image.width / det_image.width
            sy = image.height / det_image.height
            items = []
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                b = box.tolist()
                full = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                items.append((float(score), str(label), full))
            items.sort(reverse=True, key=lambda t: t[0])
            items = items[: args.max_per_image]

            digest = hashlib.sha1(key.encode()).hexdigest()[:14]
            for j, (score, label, box) in enumerate(items, 1):
                eb = expand_box(box, image.width, image.height)
                crop = image.crop(tuple(eb))
                cpath = crops_dir / f"{digest}_{j:02d}.jpg"
                crop.save(cpath, quality=94)
                outrec["detections"].append({
                    "rank": j,
                    "score": score,
                    "label": label,
                    "box": [round(v, 2) for v in box],
                    "expanded_box": eb,
                    "crop_path": str(cpath),
                    "crop_width": crop.width,
                    "crop_height": crop.height,
                })
            outrec["status"] = "detected" if items else "no_detection"
        except Exception as exc:
            outrec["status"] = "error"
            outrec["error"] = repr(exc)
            state["failures"].append({"key": key, "error": repr(exc)})
        state["records"][key] = outrec
        atomic_json(state, state_path)

    flat = []
    for r in state["records"].values():
        for d in r.get("detections", []):
            flat.append({**{k: r.get(k) for k in ["object_url", "image_path"]}, **d})
    atomic_json(flat, args.out / "detections.json")
    print(json.dumps({
        "images_seen": len(state["records"]),
        "images_with_detections": sum(bool(r.get("detections")) for r in state["records"].values()),
        "candidate_crops": len(flat),
        "images_without_detection": sum(r.get("status") == "no_detection" for r in state["records"].values()),
        "detector_errors": len(state["failures"]),
    }, indent=2))


if __name__ == "__main__":
    main()
