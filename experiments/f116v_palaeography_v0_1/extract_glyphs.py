#!/usr/bin/env python3
"""Baseline-constrained f116v glyph extraction.

This pipeline never asks a recognizer to locate page text. Four physical line
bands are fixed from the expert monochrome composite. CATMuS is applied through
Kraken's baseline recognizer, preserving character cuts and confidences.
Aligned true-colour and PCA views are treated as independent observations.
DINOv3 is used only for visual-shape comparison; TrOCR is a hostile independent
recognition control. No lexicon or language model is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any
import unicodedata

import cv2
import gdown
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import requests
import tifffile

DRIVE = {
    "true": ("Lab_true_color_TIFF/Voynich_116v_PSC.tif", "1EwdxnZURhNOjLwCTiaVZVMPW0UDeNPIK"),
    "bw": ("Processed_Images/Voynich_116v/Voynich_116v_bands01-22_RF+FL_cal_faded_text_RB+UV_FL_8bands_PCA_R-1G-2B3_hue+20b_r90_BW.tif", "16SuJ5R7RpPKXRnySPv8Pn1tNE0WouTGF"),
    "pca": ("Processed_Images/Voynich_116v/Voynich_116v_bands01-22_RF+FL_cal_faded_text_RB+UV_FL_8bands_PCA_R1G2B3.tif", "1Ed7oVeeOSEawpizLi8eOu47ZFR6WYQsg"),
}
CATMUS_URL = "https://zenodo.org/api/records/21488839/files/catmus-medieval-1.6.0.mlmodel/content"
DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
TROCR_MODEL = "medieval-data/trocr-medieval-base"
# Coordinates in the common 2200-pixel-high expert-composite frame.
X0, X1 = 600, 1605
LINE_BANDS = {
    "line1": (165, 225),
    "line2": (238, 300),
    "line3": (300, 340),
    "line4": (340, 385),
}
BLANK_BAND = (600, 662)
TARGET_WIDTH = 2010
TARGET_HEIGHT = 124


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
    else:
        a = a[..., :3]
    a = a.astype(np.float32)
    out = np.empty_like(a, dtype=np.uint8)
    for c in range(3):
        lo, hi = np.percentile(a[..., c], [0.5, 99.5])
        out[..., c] = np.clip((a[..., c] - lo) * 255.0 / (hi - lo + 1e-6), 0, 255).astype(np.uint8)
    scale = 2200.0 / max(out.shape[:2])
    return cv2.resize(out, (int(out.shape[1] * scale), int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def crop_line(arr: np.ndarray, band: tuple[int, int]) -> Image.Image:
    y0, y1 = band
    crop = Image.fromarray(arr[y0:y1, X0:X1])
    crop = crop.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(ImageOps.grayscale(crop)).enhance(1.35)


def normalize_char(ch: str) -> str:
    return unicodedata.normalize("NFC", ch).casefold()


@dataclass
class CharRecord:
    char: str
    confidence: float
    center: float
    left: float
    right: float


def records_from_prediction(prediction: str, confidences: list[float], cuts: list[Any]) -> list[CharRecord]:
    centers = []
    for cut in cuts:
        xs = [float(p[0]) for p in cut]
        centers.append(float(np.mean(xs)))
    n = min(len(prediction), len(confidences), len(centers))
    prediction = prediction[:n]
    confidences = confidences[:n]
    centers = centers[:n]
    recs = []
    for i, (ch, conf, center) in enumerate(zip(prediction, confidences, centers)):
        left = 0.0 if i == 0 else (centers[i - 1] + center) / 2.0
        right = float(TARGET_WIDTH - 1) if i == n - 1 else (center + centers[i + 1]) / 2.0
        recs.append(CharRecord(ch, float(conf), center, left, right))
    return recs


def catmus_recognize(net: Any, im: Image.Image, name: str) -> tuple[str, list[CharRecord], dict[str, Any]]:
    from kraken import rpred
    from kraken.containers import BaselineLine, Segmentation

    base_y = int(im.height * 0.72)
    line = BaselineLine(
        id="l1",
        base_dir="L",
        baseline=[(0, base_y), (im.width - 1, base_y)],
        boundary=[(0, 0), (im.width - 1, 0), (im.width - 1, im.height - 1), (0, im.height - 1)],
    )
    seg = Segmentation(
        type="baselines",
        imagename=name,
        text_direction="horizontal-lr",
        script_detection=False,
        lines=[line],
    )
    rec = next(rpred.rpred(net, im, seg, pad=0, bidi_reordering="L"))
    records = records_from_prediction(rec.prediction, rec.confidences, rec.cuts)
    raw = {"prediction": rec.prediction, "confidences": list(map(float, rec.confidences)), "cuts": rec.cuts}
    return rec.prediction, records, raw


def pair_records(a: list[CharRecord], b: list[CharRecord], tolerance: float = 13.0) -> list[tuple[CharRecord, CharRecord]]:
    pairs = []
    used = set()
    for ra in a:
        candidates = [(abs(ra.center - rb.center), j, rb) for j, rb in enumerate(b) if j not in used]
        if not candidates:
            continue
        d, j, rb = min(candidates, key=lambda x: x[0])
        if d <= tolerance:
            pairs.append((ra, rb))
            used.add(j)
    return pairs


def ink_response(im: Image.Image) -> np.ndarray:
    g = np.asarray(im, dtype=np.float32)
    bg = cv2.GaussianBlur(g, (0, 0), 7.0)
    r = np.maximum(bg - g, 0.0)
    lo, hi = np.percentile(r, [50, 99.5])
    return np.clip((r - lo) / (hi - lo + 1e-6), 0, 1)


def interval_effect(response: np.ndarray, left: float, right: float) -> float:
    l = max(0, int(math.floor(left)))
    r = min(response.shape[1], int(math.ceil(right)))
    if r <= l:
        return 0.0
    pad = max(4, int((r - l) * 0.75))
    ring = np.concatenate([
        response[:, max(0, l - pad):l].ravel(),
        response[:, r:min(response.shape[1], r + pad)].ravel(),
    ])
    inside = response[:, l:r].ravel()
    if inside.size == 0 or ring.size < 8:
        return 0.0
    return float((inside.mean() - ring.mean()) / (ring.std() + 1e-6))


def edge_correlation(a: Image.Image, b: Image.Image, left: float, right: float) -> float:
    l = max(0, int(left) - 3)
    r = min(a.width, int(right) + 3)
    if r - l < 3:
        return 0.0
    def e(im: Image.Image) -> np.ndarray:
        x = np.asarray(im, dtype=np.float32)[:, l:r]
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
        z = cv2.magnitude(gx, gy).ravel()
        return (z - z.mean()) / (z.std() + 1e-6)
    x, y = e(a), e(b)
    return float(np.mean(x * y)) if x.size == y.size and x.size else 0.0


def longest_common_substring(a: str, b: str) -> str:
    aa = re.sub(r"\s+", " ", unicodedata.normalize("NFC", a).casefold())
    bb = re.sub(r"\s+", " ", unicodedata.normalize("NFC", b).casefold())
    table = [0] * (len(bb) + 1)
    best_len = 0
    best_end = 0
    for i, ca in enumerate(aa, 1):
        new = [0] * (len(bb) + 1)
        for j, cb in enumerate(bb, 1):
            if ca == cb:
                new[j] = table[j - 1] + 1
                if new[j] > best_len:
                    best_len = new[j]
                    best_end = i
        table = new
    return aa[best_end - best_len:best_end]


def fade_image(im: Image.Image, amplitude: float) -> Image.Image:
    a = np.asarray(im, dtype=np.float32)
    bg = cv2.GaussianBlur(a, (0, 0), 17.0)
    out = bg + amplitude * (a - bg)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def prepare_patch(im: Image.Image, left: float, right: float) -> Image.Image:
    l = max(0, int(left) - 8)
    r = min(im.width, int(right) + 8)
    patch = im.crop((l, 0, r, im.height))
    w, h = patch.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), "white")
    rgb = ImageOps.colorize(patch, black="black", white="white")
    canvas.paste(rgb, ((side - w) // 2, (side - h) // 2))
    return canvas


def union_find_clusters(sim: np.ndarray, threshold: float) -> list[int]:
    n = sim.shape[0]
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)
    roots = {}
    labels = []
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels.append(roots[r])
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    src = out / "source"
    crops = out / "crops"
    for d in (out, src, crops):
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    views: dict[str, np.ndarray] = {}
    for key, (source_path, file_id) in DRIVE.items():
        p = src / f"{key}.tif"
        gdown.download(id=file_id, output=str(p), quiet=True)
        views[key] = read_rgb(p)
        manifest.append({"key": key, "source_path": source_path, "drive_file_id": file_id,
                         "sha256": sha256(p), "bytes": p.stat().st_size, "shape": list(views[key].shape)})
        print("ACQUIRED", key, flush=True)

    catmus_path = out / "catmus-medieval-1.6.0.mlmodel"
    r = requests.get(CATMUS_URL, timeout=180)
    r.raise_for_status()
    catmus_path.write_bytes(r.content)
    from kraken.lib.models import load_any
    net = load_any(catmus_path)

    line_images: dict[str, dict[str, Image.Image]] = {}
    raw: dict[str, Any] = {"catmus": {}, "trocr": {}}
    records: dict[str, dict[str, list[CharRecord]]] = {}
    for line_name, band in LINE_BANDS.items():
        line_images[line_name] = {}
        records[line_name] = {}
        raw["catmus"][line_name] = {}
        for view, arr in views.items():
            im = crop_line(arr, band)
            p = crops / f"{line_name}_{view}.png"
            im.save(p)
            line_images[line_name][view] = im
            pred, recs, rr = catmus_recognize(net, im, f"{line_name}_{view}")
            records[line_name][view] = recs
            raw["catmus"][line_name][view] = rr
            print("CATMUS", line_name, view, repr(pred), flush=True)

    blank_images = {view: crop_line(arr, BLANK_BAND) for view, arr in views.items()}
    raw["catmus"]["blank"] = {}
    blank_records = {}
    for view, im in blank_images.items():
        pred, recs, rr = catmus_recognize(net, im, f"blank_{view}")
        blank_records[view] = recs
        raw["catmus"]["blank"][view] = rr
        print("CATMUS blank", view, repr(pred), flush=True)

    blank_max = max((r.confidence for recs in blank_records.values() for r in recs), default=0.0)
    confidence_gate = max(0.70, blank_max + 0.05)

    # Fading sensitivity on the strongest BW pilot line.
    fade_control = {}
    original_bw = line_images["line2"]["bw"]
    original_recs = records["line2"]["bw"]
    for amp in (0.75, 0.50, 0.25):
        fim = fade_image(original_bw, amp)
        pred, frecs, rr = catmus_recognize(net, fim, f"line2_bw_fade_{amp:.2f}")
        pairs = pair_records(original_recs, frecs)
        eligible = [a for a in original_recs if a.char.strip() and a.confidence >= confidence_gate]
        retained = [
            (a, b) for a, b in pairs
            if a.char.strip() and normalize_char(a.char) == normalize_char(b.char)
            and a.confidence >= confidence_gate and b.confidence >= confidence_gate
        ]
        fade_control[f"{amp:.2f}"] = {
            "prediction": pred,
            "eligible_original": len(eligible),
            "retained_exact_high_confidence": len(retained),
            "retention": len(retained) / max(1, len(eligible)),
            "raw": rr,
        }
        print("FADE", amp, fade_control[f"{amp:.2f}"]["retention"], repr(pred), flush=True)

    # Independent TrOCR control, exact physical crops only.
    import torch
    from huggingface_hub import model_info
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    trocr_info = model_info(TROCR_MODEL)
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL, revision=trocr_info.sha)
    trocr = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL, revision=trocr_info.sha)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr.to(device).eval()
    for line_name in LINE_BANDS:
        raw["trocr"][line_name] = {}
        for view, im in line_images[line_name].items():
            px = processor(images=im.convert("RGB"), return_tensors="pt").pixel_values.to(device)
            with torch.inference_mode():
                gen = trocr.generate(px, num_beams=5, num_return_sequences=5, max_new_tokens=96,
                                     return_dict_in_generate=True, output_scores=True)
            texts = processor.batch_decode(gen.sequences, skip_special_tokens=True)
            scores = gen.sequences_scores.detach().cpu().tolist() if gen.sequences_scores is not None else [None] * len(texts)
            raw["trocr"][line_name][view] = [{"text": t, "score": s} for t, s in zip(texts, scores)]
            print("TROCR", line_name, view, repr(texts[0]), flush=True)
    del trocr, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    responses = {line: {view: ink_response(im) for view, im in ims.items()} for line, ims in line_images.items()}
    apparatus = []
    position_id = 0
    for line_name in LINE_BANDS:
        true_recs = records[line_name]["true"]
        bw_recs = records[line_name]["bw"]
        pca_recs = records[line_name]["pca"]
        for a, b in pair_records(true_recs, bw_recs):
            if not a.char.strip() and not b.char.strip():
                continue
            position_id += 1
            center = (a.center + b.center) / 2.0
            left = min(a.left, b.left)
            right = max(a.right, b.right)
            pca_near = min(pca_recs, key=lambda r: abs(r.center - center), default=None)
            pca_match = pca_near is not None and abs(pca_near.center - center) <= 13.0
            exact = normalize_char(a.char) == normalize_char(b.char)
            high = a.confidence >= confidence_gate and b.confidence >= confidence_gate
            effects = {v: interval_effect(responses[line_name][v], left, right) for v in ("true", "bw", "pca")}
            edge_corr = edge_correlation(line_images[line_name]["true"], line_images[line_name]["bw"], left, right)
            physical_views = sum(effects[v] > 0.15 for v in effects)
            if exact and high and physical_views >= 2:
                status = "PROBABLE_CROSS_VIEW_SINGLE_ARCH"
            elif high and physical_views >= 2:
                status = "AMBIGUOUS_LABEL"
            elif exact and physical_views >= 2:
                status = "LOW_CONFIDENCE_CROSS_VIEW"
            else:
                status = "MODEL_ONLY_OR_WEAK"
            apparatus.append({
                "position_id": position_id,
                "line": line_name,
                "center": center,
                "left": left,
                "right": right,
                "true": asdict(a),
                "bw": asdict(b),
                "pca": asdict(pca_near) if pca_match and pca_near is not None else None,
                "exact_true_bw": exact,
                "confidence_gate": confidence_gate,
                "ink_effect": effects,
                "true_bw_edge_correlation": edge_corr,
                "physical_view_count": physical_views,
                "status": status,
            })

    probable = [x for x in apparatus if x["status"] == "PROBABLE_CROSS_VIEW_SINGLE_ARCH"]

    # DINOv3 shape embeddings for probable glyph positions only.
    dino_meta: dict[str, Any] = {"model": DINO_MODEL, "available": False}
    if probable:
        try:
            from transformers import AutoImageProcessor, AutoModel
            dino_info = model_info(DINO_MODEL)
            dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL, revision=dino_info.sha)
            dino = AutoModel.from_pretrained(DINO_MODEL, revision=dino_info.sha).to(device).eval()
            dino_meta.update({"available": True, "revision": dino_info.sha})
            embeddings: dict[int, dict[str, np.ndarray]] = {}
            with torch.inference_mode():
                for item in probable:
                    line_name = item["line"]
                    patches = [prepare_patch(line_images[line_name][v], item["left"], item["right"]) for v in ("true", "bw", "pca")]
                    inputs = dino_processor(images=patches, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    output = dino(**inputs)
                    feat = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0]
                    feat = torch.nn.functional.normalize(feat, dim=-1).cpu().numpy()
                    embeddings[item["position_id"]] = {v: feat[i] for i, v in enumerate(("true", "bw", "pca"))}
                    item["dino_same_position_cosine"] = {
                        "true_bw": float(feat[0] @ feat[1]),
                        "true_pca": float(feat[0] @ feat[2]),
                        "bw_pca": float(feat[1] @ feat[2]),
                    }
            ids = [x["position_id"] for x in probable]
            mean_feat = np.stack([np.mean(np.stack(list(embeddings[i].values())), axis=0) for i in ids])
            mean_feat /= np.linalg.norm(mean_feat, axis=1, keepdims=True) + 1e-8
            sim = mean_feat @ mean_feat.T
            offdiag = sim[~np.eye(len(ids), dtype=bool)] if len(ids) > 1 else np.array([])
            cluster_threshold = max(0.94, float(np.percentile(offdiag, 95)) if offdiag.size else 0.94)
            labels = union_find_clusters(sim, cluster_threshold)
            for item, label in zip(probable, labels):
                item["dino_cluster"] = int(label)
            dino_meta.update({"cluster_threshold": cluster_threshold, "cluster_count": len(set(labels))})
            del dino, dino_processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            dino_meta["error"] = repr(exc)

    # Line-level apparatus strings: only cross-view positions are represented.
    line_summaries = {}
    for line_name in LINE_BANDS:
        items = sorted([x for x in apparatus if x["line"] == line_name], key=lambda x: x["center"])
        tokens = []
        for item in items:
            a, b = item["true"], item["bw"]
            if item["status"] == "PROBABLE_CROSS_VIEW_SINGLE_ARCH":
                tokens.append(a["char"])
            elif item["status"] == "AMBIGUOUS_LABEL":
                tokens.append(f"[{a['char']}|{b['char']}]")
            elif item["status"] == "LOW_CONFIDENCE_CROSS_VIEW":
                tokens.append(f"<{a['char']}?>")
            else:
                tokens.append("<?>")
        cat_true = raw["catmus"][line_name]["true"]["prediction"]
        cat_bw = raw["catmus"][line_name]["bw"]["prediction"]
        trocr_texts = [raw["trocr"][line_name][v][0]["text"] for v in ("true", "bw", "pca")]
        lcs = max((longest_common_substring(cat_true, t) for t in trocr_texts), key=len, default="")
        line_summaries[line_name] = {
            "apparatus": "".join(tokens),
            "catmus_true": cat_true,
            "catmus_bw": cat_bw,
            "catmus_pca": raw["catmus"][line_name]["pca"]["prediction"],
            "trocr_top": dict(zip(("true", "bw", "pca"), trocr_texts)),
            "longest_catmus_trocr_common_substring": lcs,
            "probable_positions": sum(x["status"] == "PROBABLE_CROSS_VIEW_SINGLE_ARCH" for x in items),
            "ambiguous_positions": sum(x["status"] == "AMBIGUOUS_LABEL" for x in items),
        }

    # Strict pilot: line 2 must have a nontrivial probable core, no high-confidence blank chars,
    # and retain at least some high-confidence positions at 75% amplitude.
    blank_high = sum(r.confidence >= confidence_gate and r.char.strip() for recs in blank_records.values() for r in recs)
    line2_probable = line_summaries["line2"]["probable_positions"]
    fade75 = fade_control["0.75"]["retention"]
    independent_lcs = len(line_summaries["line2"]["longest_catmus_trocr_common_substring"])
    pilot_pass = line2_probable >= 8 and blank_high == 0 and fade75 >= 0.10
    verdict = "GLYPH_EXTRACTION_PILOT_PASS" if pilot_pass else "GLYPH_EXTRACTION_PILOT_INCONCLUSIVE"

    result = {
        "status": "COMPLETE",
        "verdict": verdict,
        "pilot_pass": pilot_pass,
        "confidence_gate": confidence_gate,
        "blank_max_character_confidence": blank_max,
        "blank_high_confidence_nonspace_characters": int(blank_high),
        "fade_control": fade_control,
        "line_summaries": line_summaries,
        "apparatus": apparatus,
        "dino": dino_meta,
        "independent_architecture_line2_common_substring_length": independent_lcs,
        "interpretation": [
            "PROBABLE_CROSS_VIEW_SINGLE_ARCH means CATMuS agreed positionally on two acquired views; it is not a fully supported palaeographic reading.",
            "No position is upgraded to SUPPORTED without an independent recognition architecture or blinded palaeographic confirmation.",
            "DINOv3 clusters compare visual form only and do not assign letters.",
            "No absence claim is made for unrecognized or missing positions.",
        ],
    }

    model_manifest = {
        "kraken_version": subprocess.run(["kraken", "--version"], text=True, capture_output=True).stdout.strip(),
        "catmus": {"zenodo_record_version": "1.6.2", "model_file": catmus_path.name,
                   "sha256": sha256(catmus_path), "url": CATMUS_URL},
        "trocr": {"id": TROCR_MODEL, "revision": trocr_info.sha},
        "dino": dino_meta,
    }
    (out / "DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "MODEL_MANIFEST.json").write_text(json.dumps(model_manifest, indent=2), encoding="utf-8")
    (out / "RAW_OUTPUTS.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "GLYPH_APPARATUS.json").write_text(json.dumps(apparatus, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = [
        "# f116v baseline-constrained glyph extraction",
        "",
        f"- Verdict: **{verdict}**",
        f"- CATMuS confidence gate: **{confidence_gate:.3f}**",
        f"- Highest blank-control character confidence: **{blank_max:.3f}**",
        f"- High-confidence blank non-space characters: **{blank_high}**",
        f"- DINOv3 available: **{dino_meta.get('available')}**",
        "",
        "## Line apparatus",
    ]
    for line, summary in line_summaries.items():
        md += [
            f"### {line}",
            f"- Cross-view apparatus: `{summary['apparatus']}`",
            f"- CATMuS true colour: `{summary['catmus_true']}`",
            f"- CATMuS BW PCA: `{summary['catmus_bw']}`",
            f"- CATMuS colour PCA: `{summary['catmus_pca']}`",
            f"- Probable positional agreements: **{summary['probable_positions']}**",
            f"- Ambiguous high-confidence positions: **{summary['ambiguous_positions']}**",
            f"- Longest CATMuS/TrOCR common substring: `{summary['longest_catmus_trocr_common_substring']}`",
            "",
        ]
    md += [
        "## Interpretation",
        "",
        "A `PROBABLE` position is a source-aligned CATMuS agreement across true-colour and expert BW-PCA views with physical contrast in at least two views. It remains one recognition architecture and is not a final palaeographic reading. TrOCR is retained as an independent hostile control. DINOv3 is used only for shape comparison.",
        "",
        "No lexicon, language model, word completion, abbreviation expansion, OCR-guided line detection, diffusion restoration, or semantic inpainting was used.",
    ]
    (out / "RESULT.md").write_text("\n".join(md), encoding="utf-8")

    # Small visual atlas for temporary review.
    atlas_items = [x for x in apparatus if x["line"] == "line2" and x["status"] in {"PROBABLE_CROSS_VIEW_SINGLE_ARCH", "AMBIGUOUS_LABEL"}]
    if atlas_items:
        cell_w, cell_h = 140, 170
        canvas = Image.new("RGB", (cell_w * 5, cell_h * math.ceil(len(atlas_items) / 5)), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, item in enumerate(atlas_items):
            row, col = divmod(idx, 5)
            x, y = col * cell_w, row * cell_h
            patch = prepare_patch(line_images["line2"]["true"], item["left"], item["right"]).resize((100, 100))
            canvas.paste(patch, (x + 20, y + 5))
            label = f"{item['position_id']} {item['true']['char']}|{item['bw']['char']}\n{item['status'][:9]}"
            draw.multiline_text((x + 5, y + 110), label, fill="black")
        atlas_path = out / "LINE2_GLYPH_ATLAS.jpg"
        canvas.save(atlas_path, quality=92)
        try:
            with atlas_path.open("rb") as f:
                up = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": (atlas_path.name, f)}, timeout=180)
            result["temporary_line2_atlas"] = up.json()["data"]["url"] if up.ok else None
        except Exception:
            result["temporary_line2_atlas"] = None

    print("GLYPH_RESULT_JSON=" + json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str), flush=True)


if __name__ == "__main__":
    main()
