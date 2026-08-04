#!/usr/bin/env python3
"""DINOv3 extension for f116v palaeographic extraction.

Purpose
-------
Compare DINOv3 and DINOv2 as *unlabelled visual encoders* on the same frozen
f116v line crops and candidate glyph windows.  OCR labels are never used to
form embeddings, select nearest neighbours, or determine clusters.  The
provisional CATMuS labels are consulted only after retrieval as a secondary
consistency audit.

Primary tests
-------------
1. Exact-position cross-view retrieval: does each true-colour glyph patch
   retrieve the corresponding BW-PCA patch rather than another position?
2. Positive-vs-mismatched AUC: are aligned true/BW glyph pairs more similar
   than all nonmatching pairs?
3. Dense line correspondence: do mutual nearest-neighbour patch tokens follow
   the known image alignment on ink-rich patches?
4. Post-hoc label consistency: do visually nearest positions tend to share the
   frozen provisional CATMuS label?  This is diagnostic, not validation.

No OCR, lexicon, language model, generative restoration, or semantic inpainting
is used in this extension.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import gdown
import numpy as np
from PIL import Image, ImageOps
import tifffile
from sklearn.metrics import roc_auc_score

DRIVE = {
    "true": ("Lab_true_color_TIFF/Voynich_116v_PSC.tif", "1EwdxnZURhNOjLwCTiaVZVMPW0UDeNPIK"),
    "bw": ("expert BW multispectral PCA TIFF", "16SuJ5R7RpPKXRnySPv8Pn1tNE0WouTGF"),
    "pca": ("expert colour PCA TIFF", "1Ed7oVeeOSEawpizLi8eOu47ZFR6WYQsg"),
}
MODELS = {
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov2": "facebook/dinov2-base",
}
X0, X1 = 600, 1605
LINE_BANDS = {
    "line1": (165, 225),
    "line2": (238, 300),
    "line3": (300, 340),
    "line4": (340, 385),
}
TARGET_WIDTH = 2010
TARGET_HEIGHT = 124

# Frozen probable positions from the corrected baseline-constrained CATMuS run.
# Labels are used only for the post-hoc consistency audit.
POSITIONS = [
    # line 2
    ("line2", 281.5, 306.0, "c"), ("line2", 306.0, 330.5, "h"),
    ("line2", 330.5, 355.5, "i"), ("line2", 355.5, 376.5, "c"),
    ("line2", 376.5, 401.5, "o"), ("line2", 475.5, 498.5, "l"),
    ("line2", 496.5, 523.5, "a"), ("line2", 521.5, 554.5, "d"),
    ("line2", 550.0, 583.5, "a"), ("line2", 579.0, 608.0, "b"),
    ("line2", 608.0, 636.5, "a"), ("line2", 905.5, 940.75, "s"),
    ("line2", 1315.5, 1348.5, "r"), ("line2", 1348.5, 1385.5, "e"),
    ("line2", 1514.0, 1551.0, "o"),
    # line 3
    ("line3", 343.0, 384.5, "a"), ("line3", 384.5, 413.5, "r"),
    ("line3", 413.5, 446.5, "i"), ("line3", 591.0, 632.5, "o"),
    ("line3", 632.5, 678.0, "u"),
    # line 4
    ("line4", 440.5, 471.5, "p"), ("line4", 469.5, 508.5, "a"),
    ("line4", 558.5, 581.25, "a"), ("line4", 595.5, 616.5, "u"),
    ("line4", 645.5, 674.5, "r"), ("line4", 674.5, 699.0, "e"),
    ("line4", 786.0, 798.0, "o"), ("line4", 940.75, 967.5, "g"),
    ("line4", 967.5, 996.5, "a"),
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
    else:
        a = a[..., :3]
    a = a.astype(np.float32)
    out = np.empty_like(a, dtype=np.uint8)
    for c in range(3):
        lo, hi = np.percentile(a[..., c], [0.5, 99.5])
        out[..., c] = np.clip((a[..., c] - lo) * 255.0 / (hi - lo + 1e-6), 0, 255).astype(np.uint8)
    scale = 2200.0 / max(out.shape[:2])
    return cv2.resize(out, (int(out.shape[1] * scale), int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def line_image(arr: np.ndarray, line: str) -> Image.Image:
    y0, y1 = LINE_BANDS[line]
    crop = Image.fromarray(arr[y0:y1, X0:X1])
    crop = crop.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    return crop.convert("RGB")


def glyph_patch(im: Image.Image, left: float, right: float) -> Image.Image:
    # Include enough neighbouring pen context to preserve cursive shape while
    # keeping the central glyph dominant.  The window definition is identical
    # for all models and views.
    center = 0.5 * (left + right)
    width = max(72.0, 3.0 * (right - left))
    x0 = max(0, int(round(center - width / 2)))
    x1 = min(im.width, int(round(center + width / 2)))
    patch = im.crop((x0, 0, x1, im.height))
    # Pad to square without geometric distortion.
    side = max(patch.size)
    canvas = Image.new("RGB", (side, side), tuple(np.asarray(patch).reshape(-1, 3).mean(axis=0).astype(int)))
    canvas.paste(patch, ((side - patch.width) // 2, (side - patch.height) // 2))
    return canvas


def manual_tensor(images: list[Image.Image], processor: Any, height: int, width: int, device: str):
    import torch
    mean = np.asarray(processor.image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(processor.image_std, dtype=np.float32).reshape(1, 1, 3)
    arrs = []
    for im in images:
        im = im.resize((width, height), Image.Resampling.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
        a = (a - mean) / std
        arrs.append(np.moveaxis(a, -1, 0))
    return torch.from_numpy(np.stack(arrs)).to(device)


def forward_model(model: Any, pixels: Any):
    try:
        return model(pixel_values=pixels, interpolate_pos_encoding=True)
    except TypeError:
        return model(pixel_values=pixels)


def global_features(model: Any, processor: Any, images: list[Image.Image], device: str, batch: int = 24) -> np.ndarray:
    import torch
    feats = []
    with torch.inference_mode():
        for start in range(0, len(images), batch):
            px = manual_tensor(images[start:start + batch], processor, 224, 224, device)
            out = forward_model(model, px)
            hidden = out.last_hidden_state
            # Blend the global token with mean dense features.  This avoids
            # depending solely on either the CLS token or parchment background.
            offset = max(1, hidden.shape[1] - (224 // 16) * (224 // 16))
            cls = hidden[:, 0]
            dense = hidden[:, offset:].mean(dim=1)
            f = torch.nn.functional.normalize(cls + dense, dim=-1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def dense_features(model: Any, processor: Any, images: list[Image.Image], device: str):
    import torch
    h, w = 224, 896
    gh, gw = h // 16, w // 16
    px = manual_tensor(images, processor, h, w, device)
    with torch.inference_mode():
        out = forward_model(model, px)
        hidden = out.last_hidden_state
        offset = hidden.shape[1] - gh * gw
        if offset < 1:
            raise RuntimeError(f"Cannot infer dense-token offset from {hidden.shape}")
        tok = hidden[:, offset:offset + gh * gw]
        tok = torch.nn.functional.normalize(tok, dim=-1)
    return tok.cpu().numpy().reshape(len(images), gh, gw, -1)


def retrieval_metrics(true_f: np.ndarray, bw_f: np.ndarray, labels: list[str]) -> dict[str, Any]:
    sim = true_f @ bw_f.T
    n = len(labels)
    positives = np.diag(sim)
    negatives = sim[~np.eye(n, dtype=bool)]
    y = np.r_[np.ones_like(positives), np.zeros_like(negatives)]
    scores = np.r_[positives, negatives]
    auc = float(roc_auc_score(y, scores))
    top1 = np.argmax(sim, axis=1)
    exact_position = float(np.mean(top1 == np.arange(n)))
    # Label audit is post-hoc only and excludes singleton labels.
    counts = {x: labels.count(x) for x in set(labels)}
    eligible = [i for i, x in enumerate(labels) if counts[x] > 1]
    label_hits = [labels[top1[i]] == labels[i] for i in eligible]
    # Within-view leave-one-out visual retrieval.
    mean_f = true_f + bw_f
    mean_f /= np.linalg.norm(mean_f, axis=1, keepdims=True) + 1e-8
    within = mean_f @ mean_f.T
    np.fill_diagonal(within, -np.inf)
    nn = np.argmax(within, axis=1)
    within_hits = [labels[nn[i]] == labels[i] for i in eligible]
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same if labels[i] == labels[j] else diff).append(float(within[i, j]))
    label_auc = None
    if same and diff:
        label_auc = float(roc_auc_score(np.r_[np.ones(len(same)), np.zeros(len(diff))], np.r_[same, diff]))
    return {
        "position_retrieval_top1": exact_position,
        "aligned_vs_mismatched_auc": auc,
        "aligned_similarity_median": float(np.median(positives)),
        "mismatched_similarity_median": float(np.median(negatives)),
        "aligned_similarity_margin": float(np.median(positives) - np.median(negatives)),
        "posthoc_cross_view_label_top1": float(np.mean(label_hits)) if label_hits else None,
        "posthoc_within_view_label_top1": float(np.mean(within_hits)) if within_hits else None,
        "posthoc_same_label_auc": label_auc,
        "eligible_repeated_label_queries": len(eligible),
    }


def ink_mask(im: Image.Image, gh: int, gw: int) -> np.ndarray:
    g = cv2.cvtColor(np.asarray(im.resize((896, 224), Image.Resampling.BICUBIC)), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    pooled = mag.reshape(gh, mag.shape[0] // gh, gw, mag.shape[1] // gw).mean(axis=(1, 3))
    return pooled >= np.percentile(pooled, 70)


def dense_metrics(t_true: np.ndarray, t_other: np.ndarray, true_im: Image.Image, other_im: Image.Image) -> dict[str, Any]:
    gh, gw, dim = t_true.shape
    m = ink_mask(true_im, gh, gw) | ink_mask(other_im, gh, gw)
    coords = np.argwhere(m)
    a = t_true[m]
    b = t_other[m]
    sim = a @ b.T
    ab = np.argmax(sim, axis=1)
    ba = np.argmax(sim, axis=0)
    mutual = np.array([ba[j] == i for i, j in enumerate(ab)])
    if not mutual.any():
        return {"ink_patch_count": int(len(coords)), "mutual_match_count": 0, "identity_fraction": 0.0}
    src = coords[mutual]
    dst = coords[ab[mutual]]
    dr = np.abs(src[:, 0] - dst[:, 0])
    dc = np.abs(src[:, 1] - dst[:, 1])
    identity = (dr <= 1) & (dc <= 1)
    # Deterministic mismatch control: rotate destination columns by one quarter line.
    shifted_dc = np.abs(src[:, 1] - ((dst[:, 1] + gw // 4) % gw))
    shifted_identity = (dr <= 1) & (shifted_dc <= 1)
    return {
        "ink_patch_count": int(len(coords)),
        "mutual_match_count": int(mutual.sum()),
        "identity_fraction": float(identity.mean()),
        "shifted_control_identity_fraction": float(shifted_identity.mean()),
        "median_abs_row_error": float(np.median(dr)),
        "median_abs_column_error": float(np.median(dc)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_dir = args.output / "sources"
    source_dir.mkdir(exist_ok=True)

    arrays, source_meta = {}, {}
    for key, (name, fid) in DRIVE.items():
        path = source_dir / f"{key}.tif"
        if not path.exists():
            gdown.download(id=fid, output=str(path), quiet=True)
        arrays[key] = read_rgb(path)
        source_meta[key] = {"name": name, "drive_id": fid, "sha256": sha256(path)}
        print("ACQUIRED", key, flush=True)

    lines = {line: {view: line_image(arrays[view], line) for view in arrays} for line in LINE_BANDS}
    labels = [x[3] for x in POSITIONS]
    patches = {
        view: [glyph_patch(lines[line][view], left, right) for line, left, right, _ in POSITIONS]
        for view in arrays
    }

    import torch
    from huggingface_hub import model_info
    from transformers import AutoImageProcessor, AutoModel
    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: dict[str, Any] = {
        "status": "COMPLETE",
        "sources": source_meta,
        "positions": len(POSITIONS),
        "device": device,
        "models": {},
        "interpretation": [
            "Exact-position retrieval and aligned-vs-mismatched AUC are independent of OCR labels.",
            "Post-hoc label metrics measure consistency with provisional CATMuS labels, not truth.",
            "Dense correspondence tests local patch alignment on ink-rich regions and cannot assign letters.",
            "DINOv3 does not by itself validate a transcription.",
        ],
    }

    for model_key, repo in MODELS.items():
        info = model_info(repo, token=token)
        processor = AutoImageProcessor.from_pretrained(repo, revision=info.sha, token=token)
        model = AutoModel.from_pretrained(repo, revision=info.sha, token=token).to(device).eval()
        print("MODEL", model_key, repo, info.sha, flush=True)
        feats = {view: global_features(model, processor, patches[view], device) for view in arrays}
        model_result = {
            "repo": repo,
            "revision": info.sha,
            "true_bw_retrieval": retrieval_metrics(feats["true"], feats["bw"], labels),
            "true_pca_retrieval": retrieval_metrics(feats["true"], feats["pca"], labels),
            "dense": {},
        }
        for line in LINE_BANDS:
            ims = [lines[line]["true"], lines[line]["bw"], lines[line]["pca"]]
            toks = dense_features(model, processor, ims, device)
            model_result["dense"][line] = {
                "true_bw": dense_metrics(toks[0], toks[1], ims[0], ims[1]),
                "true_pca": dense_metrics(toks[0], toks[2], ims[0], ims[2]),
            }
        results["models"][model_key] = model_result
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Direct comparison; no threshold was chosen using the desired outcome.
    d3 = results["models"]["dinov3"]["true_bw_retrieval"]
    d2 = results["models"]["dinov2"]["true_bw_retrieval"]
    results["comparison"] = {
        "dinov3_minus_dinov2_position_top1": d3["position_retrieval_top1"] - d2["position_retrieval_top1"],
        "dinov3_minus_dinov2_auc": d3["aligned_vs_mismatched_auc"] - d2["aligned_vs_mismatched_auc"],
        "dinov3_minus_dinov2_margin": d3["aligned_similarity_margin"] - d2["aligned_similarity_margin"],
    }

    out_json = args.output / "dinov3_result.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# f116v DINOv3 visual-consistency extension", "",
        f"- Positions tested: **{len(POSITIONS)}**", f"- Device: **{device}**", "",
    ]
    for key in ("dinov3", "dinov2"):
        r = results["models"][key]["true_bw_retrieval"]
        md += [
            f"## {key}",
            f"- Exact-position true→BW retrieval: **{r['position_retrieval_top1']:.3f}**",
            f"- Aligned-vs-mismatched AUC: **{r['aligned_vs_mismatched_auc']:.3f}**",
            f"- Median aligned similarity: **{r['aligned_similarity_median']:.3f}**",
            f"- Median mismatched similarity: **{r['mismatched_similarity_median']:.3f}**",
            f"- Median similarity margin: **{r['aligned_similarity_margin']:.3f}**",
            f"- Post-hoc repeated-label top-1: **{r['posthoc_within_view_label_top1']:.3f}**",
            f"- Post-hoc same-label AUC: **{r['posthoc_same_label_auc']:.3f}**", "",
        ]
    c = results["comparison"]
    md += [
        "## DINOv3 minus DINOv2",
        f"- Position top-1 difference: **{c['dinov3_minus_dinov2_position_top1']:+.3f}**",
        f"- AUC difference: **{c['dinov3_minus_dinov2_auc']:+.3f}**",
        f"- Similarity-margin difference: **{c['dinov3_minus_dinov2_margin']:+.3f}**", "",
        "## Interpretation", "",
        "DINO models test visual correspondence and repeated-form structure. They do not assign historically valid characters. OCR labels were consulted only after retrieval for a secondary consistency audit.",
    ]
    (args.output / "DINOV3_RESULT.md").write_text("\n".join(md), encoding="utf-8")
    print("DINOV3_RESULT_JSON=" + json.dumps(results, ensure_ascii=False), flush=True)
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
