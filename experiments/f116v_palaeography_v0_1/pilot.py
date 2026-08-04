#!/usr/bin/env python3
"""One-line f116v palaeography pilot.

Downloads three public f116v views, registers them at a common inspection scale,
crops the strongest second marginal line without OCR, and runs CATMuS/Kraken and
an independent TrOCR ensemble. No lexicon or language-model correction is used.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import cv2
import gdown
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import requests
import tifffile

DRIVE = {
    "true": ("Lab_true_color_TIFF/Voynich_116v_PSC.tif", "1EwdxnZURhNOjLwCTiaVZVMPW0UDeNPIK"),
    "bw": ("Processed_Images/Voynich_116v/Voynich_116v_bands01-22_RF+FL_cal_faded_text_RB+UV_FL_8bands_PCA_R-1G-2B3_hue+20b_r90_BW.tif", "16SuJ5R7RpPKXRnySPv8Pn1tNE0WouTGF"),
    "pca": ("Processed_Images/Voynich_116v/Voynich_116v_bands01-22_RF+FL_cal_faded_text_RB+UV_FL_8bands_PCA_R1G2B3.tif", "1Ed7oVeeOSEawpizLi8eOu47ZFR6WYQsg"),
}
CATMUS_URL = "https://zenodo.org/api/records/21488839/files/catmus-medieval-1.6.0.mlmodel/content"
TROCR_MODELS = [
    "medieval-data/trocr-medieval-base",
    "medieval-data/trocr-medieval-cursiva",
    "medieval-data/trocr-medieval-textualis",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    a = tifffile.imread(path)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a[:3], 0, -1)
    elif a.ndim == 3:
        a = a[..., :3]
    else:
        raise ValueError(f"Unsupported image shape {a.shape}")
    a = a.astype(np.float32)
    out = np.empty_like(a, dtype=np.uint8)
    for c in range(3):
        lo, hi = np.percentile(a[..., c], [0.5, 99.5])
        out[..., c] = np.clip((a[..., c] - lo) * 255.0 / (hi - lo + 1e-6), 0, 255).astype(np.uint8)
    return out


def resize_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA if arr.shape[0] > h else cv2.INTER_CUBIC)


def edge(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    e = cv2.magnitude(gx, gy)
    e /= float(e.max() + 1e-6)
    return e.astype(np.float32)


def align_to_reference(src_rgb: np.ndarray, ref_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    rh, rw = ref_rgb.shape[:2]
    refg = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    for k in range(4):
        rot = np.rot90(src_rgb, k).copy()
        cand = resize_to(rot, (rh, rw))
        cg = cv2.cvtColor(cand, cv2.COLOR_RGB2GRAY)
        # Constrain alignment to a small affine correction after discrete orientation.
        W = np.eye(2, 3, dtype=np.float32)
        try:
            corr, W = cv2.findTransformECC(edge(refg), edge(cg), W, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6),
                                           None, 3)
            warped = cv2.warpAffine(cand, W, (rw, rh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REFLECT)
            score = float(corr)
        except cv2.error:
            warped = cand
            score = -1.0
        meta = {"rotation_k": k, "ecc": score, "warp": W.tolist()}
        if best is None or score > best[0]:
            best = (score, warped, meta)
    assert best is not None
    return best[1], best[2]


def save_line_variants(rgb: np.ndarray, stem: Path) -> list[Path]:
    im = Image.fromarray(rgb)
    # 3x enlargement gives recognition models enough vertical samples without generating new structure.
    im = im.resize((im.width * 3, im.height * 3), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(im)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 8)).apply(np.array(gray))
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 11)
    paths = []
    for suffix, obj in [("rgb", im), ("gray", gray), ("clahe", Image.fromarray(clahe)),
                        ("binary", Image.fromarray(adaptive))]:
        p = stem.with_name(stem.name + f"_{suffix}.png")
        obj.save(p)
        paths.append(p)
    return paths


def run_kraken(image_path: Path, model_path: Path, out_dir: Path) -> dict[str, Any]:
    stem = out_dir / image_path.stem
    txt = stem.with_suffix(".kraken.txt")
    alto = stem.with_suffix(".kraken.alto.xml")
    attempts = []
    cmds = [
        ["kraken", "-i", str(image_path), str(txt), "binarize", "segment", "ocr", "-m", str(model_path)],
        ["kraken", "-i", str(image_path), str(txt), "segment", "ocr", "-m", str(model_path)],
    ]
    ok = False
    for cmd in cmds:
        cp = subprocess.run(cmd, text=True, capture_output=True)
        attempts.append({"cmd": cmd, "returncode": cp.returncode, "stdout": cp.stdout[-3000:], "stderr": cp.stderr[-3000:]})
        if cp.returncode == 0 and txt.exists():
            ok = True
            break
    # ALTO is attempted separately to retain geometry/confidence when supported.
    if ok:
        cp = subprocess.run(["kraken", "-i", str(image_path), str(alto), "-a", "binarize", "segment", "ocr", "-m", str(model_path)],
                            text=True, capture_output=True)
        attempts.append({"cmd": "alto", "returncode": cp.returncode, "stdout": cp.stdout[-3000:], "stderr": cp.stderr[-3000:]})
    text = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
    return {"ok": ok, "text": text, "txt": str(txt), "alto": str(alto) if alto.exists() else None, "attempts": attempts}


def normalized_similarity(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein
    aa = re.sub(r"\s+", " ", a.strip())
    bb = re.sub(r"\s+", " ", b.strip())
    if not aa and not bb:
        return 1.0
    return 1.0 - Levenshtein.normalized_distance(aa, bb)


def upload_tmp(path: Path) -> str | None:
    try:
        with path.open("rb") as f:
            r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": (path.name, f)}, timeout=180)
        if r.ok:
            return r.json()["data"]["url"]
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    src_dir = out / "source"
    crop_dir = out / "crops"
    raw_dir = out / "raw_outputs"
    for d in (out, src_dir, crop_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    raw_rgb = {}
    for key, (source_path, file_id) in DRIVE.items():
        p = src_dir / f"{key}.tif"
        gdown.download(id=file_id, output=str(p), quiet=True)
        raw_rgb[key] = read_rgb(p)
        manifest.append({"key": key, "source_path": source_path, "drive_file_id": file_id,
                         "bytes": p.stat().st_size, "sha256": sha256(p),
                         "shape": list(raw_rgb[key].shape), "dtype": str(raw_rgb[key].dtype)})
        print("ACQUIRED", key, manifest[-1], flush=True)

    # Expert monochrome image is already oriented with marginal lines horizontal.
    ref_full = raw_rgb["bw"]
    scale = 2200.0 / max(ref_full.shape[:2])
    ref = cv2.resize(ref_full, (int(ref_full.shape[1] * scale), int(ref_full.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    aligned = {"bw": ref}
    registration = {"bw": {"rotation_k": 0, "ecc": 1.0, "warp": [[1, 0, 0], [0, 1, 0]]}}
    for key in ("true", "pca"):
        src = cv2.resize(raw_rgb[key], (int(raw_rgb[key].shape[1] * min(1.0, 2200.0 / max(raw_rgb[key].shape[:2])),
                                       int(raw_rgb[key].shape[0] * min(1.0, 2200.0 / max(raw_rgb[key].shape[:2])))), interpolation=cv2.INTER_AREA)
        aligned[key], registration[key] = align_to_reference(src, ref)
        print("REGISTERED", key, registration[key], flush=True)

    # Frozen line groups derived from non-OCR projection peaks on the rotated BW composite.
    # x covers the complete marginal text; line 2 is strongest and used for the pilot.
    boxes = {
        "line1": [420, 160, 1645, 232],
        "line2": [420, 232, 1645, 302],
        "line3": [420, 300, 1645, 342],
        "line4": [420, 342, 1645, 382],
        "blank": [420, 700, 1645, 770],
    }
    line_paths: dict[str, dict[str, list[Path]]] = {}
    for region in ("line2", "blank"):
        x0, y0, x1, y1 = boxes[region]
        line_paths[region] = {}
        for key, arr in aligned.items():
            crop = arr[y0:y1, x0:x1]
            line_paths[region][key] = save_line_variants(crop, crop_dir / f"{region}_{key}")

    # Synthetic fading uses acquired line pixels blended toward a local bright background.
    base = Image.open(crop_dir / "line2_bw_gray.png").convert("L")
    bg = Image.new("L", base.size, color=int(np.percentile(np.array(base), 85)))
    faded_paths = []
    for amplitude in (0.75, 0.5, 0.25):
        faded = Image.blend(bg, base, amplitude)
        p = crop_dir / f"line2_bw_fade_{amplitude:.2f}.png"
        faded.save(p)
        faded_paths.append(p)

    model_path = out / "catmus-medieval-1.6.0.mlmodel"
    r = requests.get(CATMUS_URL, timeout=180)
    r.raise_for_status()
    model_path.write_bytes(r.content)
    model_manifest: dict[str, Any] = {
        "kraken_version": subprocess.run(["kraken", "--version"], text=True, capture_output=True).stdout.strip(),
        "catmus": {"record_version": "1.6.2", "filename": model_path.name, "sha256": sha256(model_path), "url": CATMUS_URL},
        "trocr": [],
    }

    results: dict[str, Any] = {"kraken": {}, "trocr": {}, "controls": {}, "registration": registration, "boxes": boxes}
    # Evidence views use grayscale/CLAHE only. Binary is retained as a segmentation audit, not the primary recognition input.
    evidence_inputs = [crop_dir / f"line2_{key}_clahe.png" for key in ("bw", "pca", "true")]
    blank_inputs = [crop_dir / f"blank_{key}_clahe.png" for key in ("bw", "pca", "true")]
    for p in evidence_inputs + blank_inputs + faded_paths:
        results["kraken"][p.name] = run_kraken(p, model_path, raw_dir)
        print("KRAKEN", p.name, repr(results["kraken"][p.name]["text"]), flush=True)

    import torch
    from huggingface_hub import model_info
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    trocr_inputs = evidence_inputs + [blank_inputs[0]] + faded_paths
    for model_id in TROCR_MODELS:
        info = model_info(model_id)
        model_manifest["trocr"].append({"id": model_id, "revision": info.sha})
        processor = TrOCRProcessor.from_pretrained(model_id, revision=info.sha)
        model = VisionEncoderDecoderModel.from_pretrained(model_id, revision=info.sha)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        model_out = {}
        for p in trocr_inputs:
            im = Image.open(p).convert("RGB")
            px = processor(images=im, return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                gen = model.generate(px, num_beams=5, num_return_sequences=5, max_new_tokens=128,
                                     return_dict_in_generate=True, output_scores=True)
            texts = processor.batch_decode(gen.sequences, skip_special_tokens=True)
            scores = gen.sequences_scores.detach().cpu().tolist() if gen.sequences_scores is not None else [None] * len(texts)
            model_out[p.name] = [{"text": t, "sequence_score": s} for t, s in zip(texts, scores)]
            print("TROCR", model_id, p.name, json.dumps(model_out[p.name], ensure_ascii=False), flush=True)
        results["trocr"][model_id] = model_out
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Stability metrics: view consistency, blank suppression, and fading retention.
    metrics: dict[str, Any] = {}
    kr_texts = [results["kraken"][p.name]["text"] for p in evidence_inputs]
    metrics["kraken_cross_view_similarity"] = [normalized_similarity(kr_texts[i], kr_texts[j]) for i in range(3) for j in range(i + 1, 3)]
    metrics["kraken_blank_lengths"] = [len(results["kraken"][p.name]["text"]) for p in blank_inputs]
    metrics["kraken_fade_similarity_to_bw"] = {p.name: normalized_similarity(kr_texts[0], results["kraken"][p.name]["text"]) for p in faded_paths}
    metrics["trocr"] = {}
    for model_id, mo in results["trocr"].items():
        tops = [mo[p.name][0]["text"] for p in evidence_inputs]
        metrics["trocr"][model_id] = {
            "cross_view_similarity": [normalized_similarity(tops[i], tops[j]) for i in range(3) for j in range(i + 1, 3)],
            "blank_length": len(mo[blank_inputs[0].name][0]["text"]),
            "fade_similarity_to_bw": {p.name: normalized_similarity(tops[0], mo[p.name][0]["text"]) for p in faded_paths},
        }

    cross_arch = []
    k_ref = kr_texts[0]
    for model_id, mo in results["trocr"].items():
        cross_arch.append({"model": model_id, "similarity_to_kraken_bw": normalized_similarity(k_ref, mo[evidence_inputs[0].name][0]["text"])})
    metrics["cross_architecture"] = cross_arch

    # Pilot is a stability gate, not a claim that any model label is correct.
    view_scores = metrics["kraken_cross_view_similarity"] + [x for m in metrics["trocr"].values() for x in m["cross_view_similarity"]]
    fade_scores = list(metrics["kraken_fade_similarity_to_bw"].values()) + [x for m in metrics["trocr"].values() for x in m["fade_similarity_to_bw"].values()]
    blank_ok = max(metrics["kraken_blank_lengths"] + [m["blank_length"] for m in metrics["trocr"].values()]) <= 40
    pilot_pass = (max(view_scores, default=0) >= 0.45 and max(fade_scores, default=0) >= 0.35 and blank_ok)
    verdict = "PILOT_STABILITY_GATE_PASS" if pilot_pass else "PILOT_STABILITY_GATE_FAIL"

    (out / "DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "MODEL_MANIFEST.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    (out / "raw_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "status": "COMPLETE", "verdict": verdict, "pilot_pass": pilot_pass,
        "line": "line2", "metrics": metrics,
        "kraken_top": {p.name: results["kraken"][p.name]["text"] for p in evidence_inputs},
        "trocr_top": {m: {p.name: mo[p.name][0] for p in evidence_inputs} for m, mo in results["trocr"].items()},
        "limitations": [
            "Model agreement measures stability, not palaeographic correctness.",
            "The pilot uses expert-derived composites and true colour, not all 46 raw bands.",
            "No lexicon or language model is used.",
        ],
    }
    (out / "PILOT_RESULT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# f116v palaeography pilot", "", f"- Verdict: **{verdict}**", "- Pilot line: second surviving marginal line", "- Views: expert BW PCA, colour PCA, true colour", "", "## Kraken top outputs"]
    for k, v in summary["kraken_top"].items():
        md.append(f"- `{k}`: `{v}`")
    md += ["", "## TrOCR top outputs"]
    for m, vals in summary["trocr_top"].items():
        md.append(f"### {m}")
        for k, v in vals.items():
            md.append(f"- `{k}`: `{v['text']}` (sequence score {v['sequence_score']})")
    md += ["", "## Interpretation", "A pass means the acquired line contains enough repeatable visual structure to justify full four-line extraction. It does not validate the character labels. A fail means current recognition outputs are too unstable or insufficiently separated from controls to scale without revising preprocessing or segmentation."]
    (out / "PILOT_RESULT.md").write_text("\n".join(md), encoding="utf-8")

    # Public temporary links are an execution handoff only; source TIFFs are not uploaded.
    uploaded = {}
    for p in [out / "PILOT_RESULT.json", out / "PILOT_RESULT.md", out / "metrics.json", crop_dir / "line2_bw_clahe.png", crop_dir / "line2_pca_clahe.png", crop_dir / "line2_true_clahe.png"]:
        uploaded[p.name] = upload_tmp(p)
    summary["temporary_uploads"] = uploaded
    print("PILOT_RESULT_JSON=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
