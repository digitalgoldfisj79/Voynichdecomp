#!/usr/bin/env python3
"""Apply a selected v0.2 checkpoint to aligned f116v views.

The script reconstructs the split checkpoint written by train_v02.py, loads the
frozen DINOv3 encoder and the selected CATMuS CTC head, and fuses per-view CTC
posteriors before greedy decoding. Individual-view predictions are retained for
audit. No dictionary or language model is used.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import cv2
import gdown
import numpy as np
from PIL import Image
import tifffile
import torch

import train_v02 as base

DRIVE = {
    "true": ("Voynich_116v_PSC.tif", "1EwdxnZURhNOjLwCTiaVZVMPW0UDeNPIK"),
    "bw": ("Voynich_116v_expert_BW_PCA.tif", "16SuJ5R7RpPKXRnySPv8Pn1tNE0WouTGF"),
    "pca": ("Voynich_116v_expert_colour_PCA.tif", "1Ed7oVeeOSEawpizLi8eOu47ZFR6WYQsg"),
}
X0, X1 = 600, 1605
LINE_BANDS = {
    "line1": (165, 225),
    "line2": (238, 300),
    "line3": (300, 340),
    "line4": (340, 385),
}


def read_rgb(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr[:3], 0, -1)
    else:
        arr = arr[..., :3]
    arr = arr.astype(np.float32)
    out = np.empty_like(arr, dtype=np.uint8)
    for channel in range(3):
        lo, hi = np.percentile(arr[..., channel], [0.5, 99.5])
        out[..., channel] = np.clip(
            (arr[..., channel] - lo) * 255.0 / (hi - lo + 1e-6), 0, 255
        )
    scale = 2200.0 / max(out.shape[:2])
    return cv2.resize(
        out,
        (int(out.shape[1] * scale), int(out.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )


def load_split_checkpoint(result_dir: Path) -> dict:
    manifest_path = result_dir / "f116v-dinov3-catmus-htr-v0.2.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = b""
    for part in manifest["parts"]:
        path = result_dir / part["name"]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != part["bytes"] or digest != part["sha256"]:
            raise RuntimeError(f"Checkpoint part verification failed: {path.name}")
        data += payload
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != manifest["combined_bytes"] or digest != manifest["combined_sha256"]:
        raise RuntimeError("Combined checkpoint verification failed")
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


def decode(logits: torch.Tensor, valid_steps: int, id_to_char: dict[int, str]) -> str:
    return base.greedy_decode(logits[:valid_steps].argmax(-1).detach().cpu().tolist(), id_to_char)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoint = load_split_checkpoint(args.result_dir)
    best_arm = checkpoint["best_arm"]
    if best_arm == "PIXEL_ONLY":
        raise RuntimeError("Refusing f116v inference because no DINOv3 arm won")
    char_to_id = checkpoint["char_to_id"]
    id_to_char = {value: key for key, value in char_to_id.items()}
    state = checkpoint["state_dict"]
    dino_dim = int(state["dino_proj.1.weight"].shape[1])
    model = base.HTRModel(dino_dim, len(char_to_id) + 1, best_arm)
    model.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    threshold = float(checkpoint["metrics"][best_arm]["ink_threshold"])

    from transformers import AutoImageProcessor, AutoModel

    dino_repo = checkpoint["architecture"]["dino_repo"]
    dino_revision = checkpoint["architecture"]["dino_revision"]
    processor = AutoImageProcessor.from_pretrained(
        dino_repo, revision=dino_revision, token=os.environ.get("HF_TOKEN")
    )
    encoder = AutoModel.from_pretrained(
        dino_repo, revision=dino_revision, token=os.environ.get("HF_TOKEN")
    ).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    sources: dict[str, np.ndarray] = {}
    source_manifest = {}
    for view, (name, file_id) in DRIVE.items():
        path = args.output / name
        gdown.download(id=file_id, output=str(path), quiet=True)
        sources[view] = read_rgb(path)
        source_manifest[view] = {
            "drive_id": file_id,
            "name": name,
            "shape": list(sources[view].shape),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    line_names = list(LINE_BANDS)
    actual_views = np.zeros(
        (len(line_names), base.VIEWS, base.HEIGHT, base.WIDTH, 3), dtype=np.uint8
    )
    valid_lengths = torch.zeros(len(line_names), dtype=torch.long)
    view_names = list(DRIVE)
    for line_index, line_name in enumerate(line_names):
        y0, y1 = LINE_BANDS[line_name]
        lengths = []
        for view_index, view_name in enumerate(view_names):
            crop = sources[view_name][y0:y1, X0:X1]
            prepared, valid = base.prepare_line(Image.fromarray(crop))
            actual_views[line_index, view_index] = prepared
            lengths.append(valid)
            Image.fromarray(prepared).save(args.output / f"{line_name}_{view_name}_input.png")
        # All physical views use the common conservative valid extent.
        valid_lengths[line_index] = min(lengths)

    features = base.extract_dino_features(
        encoder, processor, actual_views, valid_lengths, device, batch_size=12
    )
    raw: dict[str, dict] = {}
    with torch.inference_mode():
        for line_index, line_name in enumerate(line_names):
            logits_by_view = []
            ink_by_view = []
            individual = {}
            for view_index, view_name in enumerate(view_names):
                pixels = base.pixel_batch(actual_views, [line_index], [view_index], device)
                dino = features[line_index, view_index].unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                length = valid_lengths[line_index : line_index + 1].to(device)
                logits, ink = model(pixels, dino, length)
                logits_by_view.append(logits[0])
                ink_by_view.append(torch.sigmoid(ink[0]))
                individual[view_name] = decode(
                    logits[0], int(valid_lengths[line_index]), id_to_char
                )
            # Geometric posterior mean, equivalent to averaging log-posteriors.
            fused_log_probs = torch.stack(
                [x.log_softmax(-1) for x in logits_by_view], dim=0
            ).mean(0)
            fused_ink = float(torch.stack(ink_by_view).mean().cpu())
            fused = "" if fused_ink < threshold else decode(
                fused_log_probs, int(valid_lengths[line_index]), id_to_char
            )
            raw[line_name] = {
                "individual_predictions": individual,
                "fused_prediction": fused,
                "fused_ink_probability": fused_ink,
                "ink_threshold": threshold,
                "valid_steps": int(valid_lengths[line_index]),
            }
            print("F116V_V02", line_name, json.dumps(raw[line_name], ensure_ascii=False), flush=True)

    result = {
        "status": "COMPLETE",
        "verdict": "F116V_V02_POSTERIOR_FUSION_HYPOTHESES_ONLY",
        "best_arm": best_arm,
        "training_test_metrics": checkpoint["metrics"][best_arm]["test"],
        "sources": source_manifest,
        "lines": raw,
        "limitations": [
            "Fused strings are model hypotheses, not palaeographic transcriptions.",
            "CATMuS supervision does not make the model independent of Kraken-CATMuS.",
            "No dictionary, language model, abbreviation expansion, or word correction was used.",
        ],
    }
    (args.output / "F116V_V02_RESULT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("F116V_V02_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
