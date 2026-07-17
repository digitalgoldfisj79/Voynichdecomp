#!/usr/bin/env python3
"""Bounded operational smoke test; never loads Voynich hand labels."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path


def main() -> int:
    import torch
    from PIL import Image
    from huggingface_hub import HfApi
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    token = os.environ.get("HF_TOKEN")
    root = Path(os.environ.get("VDINO_ROOT", "/vdino3"))
    output_repo = os.environ.get(
        "OUTPUT_REPO", "Digitalgoldfish79/blind-scribal-hands-v1"
    )
    report: dict = {
        "schema": "blind-palaeography-preflight-fast-v1",
        "timestamp_unix": int(time.time()),
        "davis_labels_loaded": False,
        "platform": platform.platform(),
        "root_exists": root.is_dir(),
        "status": "STARTED",
    }
    try:
        if not root.is_dir():
            raise RuntimeError(f"missing mounted dataset root: {root}")
        top = sorted(root.iterdir(), key=lambda p: p.name)
        report["root_entries"] = [
            {"name": p.name, "is_file": p.is_file(), "size": p.stat().st_size if p.is_file() else None}
            for p in top[:100]
        ]
        required = [
            "corpus_crop_manifest.jsonl",
            "corpus_embeddings_full.npz",
        ]
        report["required_assets"] = {
            name: {
                "exists": (root / name).is_file(),
                "size": (root / name).stat().st_size if (root / name).is_file() else None,
            }
            for name in required
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"
        report["device"] = device
        report["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        image = Image.new("RGB", (224, 224), "white")

        dino_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        processor = AutoImageProcessor.from_pretrained(dino_id, token=token)
        model = AutoModel.from_pretrained(
            dino_id,
            token=token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**{k: v.to(device) for k, v in inputs.items()})
            hidden = outputs.last_hidden_state
        report["dinov3"] = {
            "model": dino_id,
            "shape": list(hidden.shape),
            "finite": bool(torch.isfinite(hidden).all().item()),
        }
        del model, processor, inputs, outputs, hidden
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        htr_id = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
        htr = VisionEncoderDecoderModel.from_pretrained(htr_id, token=token)
        try:
            hproc = TrOCRProcessor.from_pretrained(htr_id, token=token)
            processor_source = htr_id
        except Exception:
            processor_source = "microsoft/trocr-base-handwritten"
            hproc = TrOCRProcessor.from_pretrained(processor_source, token=token)
        encoder = htr.encoder.to(device).eval()
        with torch.inference_mode():
            pixels = hproc(images=image, return_tensors="pt").pixel_values.to(device)
            hhidden = encoder(pixel_values=pixels).last_hidden_state
        report["historical_htr"] = {
            "model": htr_id,
            "processor": processor_source,
            "shape": list(hhidden.shape),
            "finite": bool(torch.isfinite(hhidden).all().item()),
        }
        del htr, hproc, encoder, pixels, hhidden
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        api = HfApi(token=token)
        report["whoami"] = api.whoami(token=token)
        api.create_repo(
            output_repo,
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token=token,
        )
        probe = json.dumps(report, sort_keys=True, default=str).encode()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp.write(probe)
            tmp_path = tmp.name
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="preflight/fast_access_probe.json",
            repo_id=output_repo,
            repo_type="dataset",
            token=token,
            commit_message="Blind palaeography bounded access probe",
        )
        report["hub_write"] = {
            "ok": True,
            "repo": output_repo,
            "path": "preflight/fast_access_probe.json",
        }
        report["report_sha256"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, default=str).encode()
        ).hexdigest()
        report["status"] = "PASS"
    except Exception as exc:
        import traceback

        report["status"] = "FAIL"
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
    print("BLIND_PALAEOGRAPHY_FAST_PREFLIGHT " + json.dumps(report, sort_keys=True, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
