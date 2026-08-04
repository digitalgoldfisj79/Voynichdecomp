#!/usr/bin/env python3
"""Retrain the passing hybrid arm deterministically and apply it to f116v views."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import cv2
import gdown
import numpy as np
from PIL import Image
import tifffile
import torch

import train_pilot as base
import train_pilot_v2 as corrected_sampler
import train_hybrid as hybrid

base.collect_samples = corrected_sampler.collect_samples_balanced

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
        out[..., channel] = np.clip((arr[..., channel] - lo) * 255.0 / (hi - lo + 1e-6), 0, 255)
    scale = 2200.0 / max(out.shape[:2])
    return cv2.resize(out, (int(out.shape[1] * scale), int(out.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def lcs(a: str, b: str) -> str:
    table = [[""] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            if ca == cb:
                table[i][j] = table[i - 1][j - 1] + ca
            else:
                left, up = table[i][j - 1], table[i - 1][j]
                table[i][j] = left if len(left) >= len(up) else up
    return table[-1][-1]


def decode_with_confidence(logits: torch.Tensor, valid_steps: int, id_to_char: dict[int, str]):
    probs = logits.softmax(-1)[:valid_steps]
    ids = probs.argmax(-1).detach().cpu().tolist()
    max_probs = probs.max(-1).values.detach().cpu().tolist()
    records = []
    previous = None
    run = []
    run_id = None
    run_start = 0
    for t, (idx, confidence) in enumerate(zip(ids, max_probs)):
        if idx != previous:
            if run_id not in (None, 0):
                records.append({
                    "char": id_to_char.get(run_id, "�"),
                    "start_step": run_start,
                    "end_step": t,
                    "confidence": max(run),
                })
            run_id, run, run_start = idx, [confidence], t
        else:
            run.append(confidence)
        previous = idx
    if run_id not in (None, 0):
        records.append({
            "char": id_to_char.get(run_id, "�"),
            "start_step": run_start,
            "end_step": valid_steps,
            "confidence": max(run),
        })
    return "".join(r["char"] for r in records), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("f116v_hybrid_result"))
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from huggingface_hub import HfApi, create_repo, dataset_info, model_info
    from transformers import AutoImageProcessor, AutoModel

    data_meta = dataset_info(base.DATA_REPO)
    dino_meta = model_info(base.DINO_REPO)
    samples, manifest = base.collect_samples(512, 96, 96, args.seed, 48, 60000)
    char_to_id, id_to_char = base.make_vocab(samples)

    processor = AutoImageProcessor.from_pretrained(base.DINO_REPO, revision=dino_meta.sha, token=os.environ.get("HF_TOKEN"))
    encoder = AutoModel.from_pretrained(base.DINO_REPO, revision=dino_meta.sha, token=os.environ.get("HF_TOKEN")).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    features, lengths = {}, {}
    for split in ("train", "dev", "test"):
        features[split], lengths[split] = base.extract_features(
            encoder, processor, samples[split], device, 8
        )

    model, training_result, _, _ = hybrid.train_arm(
        "CNN_DINOV3",
        True,
        features,
        lengths,
        samples,
        char_to_id,
        id_to_char,
        device,
        30,
        12,
        args.seed,
    )

    source_arrays = {}
    source_manifest = {}
    for view, (name, file_id) in DRIVE.items():
        path = args.output / name
        gdown.download(id=file_id, output=str(path), quiet=True)
        source_arrays[view] = read_rgb(path)
        source_manifest[view] = {"drive_id": file_id, "name": name, "shape": list(source_arrays[view].shape)}
        print("SOURCE", view, source_manifest[view], flush=True)

    inference_samples = []
    inference_keys = []
    for line_name, (y0, y1) in LINE_BANDS.items():
        for view in ("true", "bw", "pca"):
            crop = source_arrays[view][y0:y1, X0:X1]
            prepared, valid_steps = base.prepare_line(Image.fromarray(crop))
            inference_samples.append(
                base.Sample("f116v", f"f116v:{view}", "", prepared, valid_steps, 15, "unknown", "unknown")
            )
            inference_keys.append((line_name, view))
            Image.fromarray(prepared).save(args.output / f"{line_name}_{view}_input.png")

    inference_features, inference_lengths = base.extract_features(
        encoder, processor, inference_samples, device, 6
    )
    model.eval()
    raw = {line: {} for line in LINE_BANDS}
    with torch.inference_mode():
        for index, (line_name, view) in enumerate(inference_keys):
            images = hybrid.image_batch(inference_samples, [index], device)
            dino = inference_features[index:index + 1].to(device=device, dtype=torch.float32)
            lens = inference_lengths[index:index + 1].to(device)
            logits = model(images, dino, lens)[0]
            prediction, records = decode_with_confidence(
                logits, int(inference_lengths[index]), id_to_char
            )
            raw[line_name][view] = {
                "prediction": prediction,
                "characters": records,
                "valid_steps": int(inference_lengths[index]),
            }
            print("F116V", line_name, view, repr(prediction), flush=True)

    consensus = {}
    for line_name in LINE_BANDS:
        true = raw[line_name]["true"]["prediction"]
        bw = raw[line_name]["bw"]["prediction"]
        pca = raw[line_name]["pca"]["prediction"]
        true_bw = lcs(true, bw)
        all_views = lcs(true_bw, pca)
        consensus[line_name] = {
            "true_bw_lcs": true_bw,
            "all_three_lcs": all_views,
            "true_bw_normalized_length": len(true_bw) / max(1, min(len(true), len(bw))),
            "all_three_normalized_length": len(all_views) / max(1, min(len(true), len(bw), len(pca))),
        }

    checkpoint = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "char_to_id": char_to_id,
        "dino_repo": base.DINO_REPO,
        "dino_revision": dino_meta.sha,
        "dataset_repo": base.DATA_REPO,
        "dataset_revision": data_meta.sha,
        "training_result": training_result,
    }
    checkpoint_path = args.output / "f116v_dinov3_catmus_hybrid_head.pt"
    torch.save(checkpoint, checkpoint_path)

    upload = {"status": "NOT_ATTEMPTED"}
    try:
        repo_id = "Digitalgoldfish79/f116v-dinov3-catmus-htr-v0.1"
        create_repo(repo_id, repo_type="model", exist_ok=True, token=os.environ.get("HF_TOKEN"))
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.upload_file(path_or_fileobj=str(checkpoint_path), path_in_repo=checkpoint_path.name, repo_id=repo_id, repo_type="model")
        upload = {"status": "UPLOADED", "repo_id": repo_id, "filename": checkpoint_path.name}
    except Exception as exc:
        upload = {"status": "FAILED", "error": repr(exc)}

    result = {
        "status": "COMPLETE",
        "verdict": "F116V_HYBRID_MODEL_HYPOTHESES_ONLY",
        "training": training_result,
        "dataset": {"repo": base.DATA_REPO, "revision": data_meta.sha, "manifest": manifest},
        "dino": {"repo": base.DINO_REPO, "revision": dino_meta.sha},
        "sources": source_manifest,
        "raw_predictions": raw,
        "cross_view_subsequences": consensus,
        "checkpoint_upload": upload,
        "limitations": [
            "Held-out test CER is approximately 0.595, so output is not a transcription.",
            "Kraken-CATMuS and this model share CATMuS supervision and are not independent palaeographic witnesses.",
            "Longest-common-subsequence summaries preserve order but do not prove character identity or word boundaries.",
            "No dictionary or language model was used.",
        ],
    }
    (args.output / "F116V_INFERENCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# f116v DINOv3–CATMuS hybrid inference",
        "",
        "- Verdict: **F116V_HYBRID_MODEL_HYPOTHESES_ONLY**",
        f"- Held-out hybrid test CER: **{training_result['test']['cer']:.4f}**",
        f"- Checkpoint upload: **{upload['status']}**",
        "",
        "| Line | True colour | BW PCA | Colour PCA | True/BW LCS | All-view LCS |",
        "|---|---|---|---|---|---|",
    ]
    for line_name in LINE_BANDS:
        md.append(
            f"| {line_name} | `{raw[line_name]['true']['prediction']}` | `{raw[line_name]['bw']['prediction']}` | "
            f"`{raw[line_name]['pca']['prediction']}` | `{consensus[line_name]['true_bw_lcs']}` | "
            f"`{consensus[line_name]['all_three_lcs']}` |"
        )
    md += [
        "",
        "These are model hypotheses, not a transcription. No spaces, words or language should be inferred from the table.",
    ]
    (args.output / "F116V_RESULT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("F116V_HYBRID_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
