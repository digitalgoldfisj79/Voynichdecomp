#!/usr/bin/env python3
"""Non-scientific access and asset smoke test for blind palaeography v1.

This script must not load any Davis hand map or perform model selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_inventory(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if not root.exists():
        return {"exists": False, "root": str(root), "records": []}
    for p in sorted(root.rglob("*"), key=lambda x: str(x)):
        if p.is_file():
            st = p.stat()
            records.append({
                "path": str(p.relative_to(root)),
                "size_bytes": st.st_size,
            })
    return {
        "exists": True,
        "root": str(root),
        "file_count": len(records),
        "total_bytes": sum(r["size_bytes"] for r in records),
        "records": records[:5000],
        "truncated": len(records) > 5000,
    }


def inspect_known_assets(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    jsonl = root / "corpus_crop_manifest.jsonl"
    if jsonl.exists():
        rows = []
        with jsonl.open("r", encoding="utf-8") as f:
            for _ in range(3):
                line = f.readline()
                if not line:
                    break
                rows.append(json.loads(line))
        out["corpus_crop_manifest"] = {
            "path": str(jsonl),
            "size_bytes": jsonl.stat().st_size,
            "sample_keys": [sorted(r.keys()) for r in rows],
            "sample_rows": rows,
        }
    npz = root / "corpus_embeddings_full.npz"
    if npz.exists():
        import numpy as np

        with np.load(npz, mmap_mode="r", allow_pickle=False) as z:
            out["corpus_embeddings_full"] = {
                "path": str(npz),
                "size_bytes": npz.stat().st_size,
                "keys": list(z.files),
                "arrays": {
                    k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)}
                    for k in z.files
                },
            }
    for candidate in (
        "voynichese_folios_registration.csv",
        "corpus_registration_manifest.jsonl",
        "physical_bifolia.json",
        "folio_metadata.csv",
    ):
        p = root / candidate
        if p.exists():
            out[candidate] = {
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
    return out


def test_models(token: str | None) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, VisionEncoderDecoderModel

    result: dict[str, Any] = {"cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        result["gpu_count"] = torch.cuda.device_count()
        result["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.new("RGB", (224, 224), "white")

    dino_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    proc = AutoImageProcessor.from_pretrained(dino_id, token=token)
    model = AutoModel.from_pretrained(dino_id, token=token, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    with torch.inference_mode():
        inp = proc(images=image, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        y = model(**inp).last_hidden_state
    result["dinov3"] = {
        "model": dino_id,
        "shape": list(y.shape),
        "finite": bool(torch.isfinite(y).all().item()),
        "dtype": str(y.dtype),
    }
    del model, proc, inp, y
    if device == "cuda":
        torch.cuda.empty_cache()

    htr_id = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
    htr = VisionEncoderDecoderModel.from_pretrained(htr_id, token=token)
    enc = htr.encoder.to(device).eval()
    from transformers import TrOCRProcessor

    hproc = TrOCRProcessor.from_pretrained(htr_id, token=token)
    with torch.inference_mode():
        px = hproc(images=image, return_tensors="pt").pixel_values.to(device)
        hy = enc(pixel_values=px).last_hidden_state
    result["historical_htr_encoder"] = {
        "model": htr_id,
        "shape": list(hy.shape),
        "finite": bool(torch.isfinite(hy).all().item()),
        "dtype": str(hy.dtype),
    }
    return result


def test_hub_write(token: str | None, report: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami(token=token)
    requested = os.environ.get("OUTPUT_REPO", "Digitalgoldfish79/blind-scribal-hands-v1")
    fallback = os.environ.get("FALLBACK_OUTPUT_REPO", "Digitalgoldfish79/v060-terminal-checkpoints")
    result: dict[str, Any] = {"whoami": who, "requested_repo": requested}
    payload = json.dumps({
        "kind": "blind_scribal_hands_v1_preflight",
        "timestamp_unix": int(time.time()),
        "python": sys.version,
        "report_digest": hashlib.sha256(json.dumps(report, sort_keys=True, default=str).encode()).hexdigest(),
    }, indent=2, sort_keys=True).encode()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name
    try:
        api.create_repo(requested, repo_type="dataset", private=True, exist_ok=True, token=token)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="preflight/access_probe.json",
            repo_id=requested,
            repo_type="dataset",
            token=token,
            commit_message="Blind palaeography v1 access probe",
        )
        result.update({"write_ok": True, "repo": requested, "path": "preflight/access_probe.json"})
        return result
    except Exception as exc:
        result["requested_error"] = repr(exc)
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="blind_scribal_hands_v1/preflight/access_probe.json",
            repo_id=fallback,
            repo_type="dataset",
            token=token,
            commit_message="Blind palaeography v1 access probe fallback",
        )
        result.update({
            "write_ok": True,
            "repo": fallback,
            "path": "blind_scribal_hands_v1/preflight/access_probe.json",
            "used_fallback": True,
        })
    except Exception as exc:
        result.update({"write_ok": False, "fallback_repo": fallback, "fallback_error": repr(exc)})
    return result


def main() -> int:
    root = Path(os.environ.get("VDINO_ROOT", "/vdino3"))
    token = os.environ.get("HF_TOKEN")
    report: dict[str, Any] = {
        "status": "STARTED",
        "purpose": "non-scientific access/model/data smoke test",
        "davis_labels_loaded": False,
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
    }
    try:
        report["inventory"] = file_inventory(root)
        report["known_assets"] = inspect_known_assets(root)
        report["models"] = test_models(token)
        report["hub_write"] = test_hub_write(token, report)
        report["status"] = "PASS" if report["inventory"]["exists"] and report["hub_write"].get("write_ok") else "PARTIAL"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        import traceback

        report["traceback"] = traceback.format_exc()
    print("BLIND_PALAEOGRAPHY_PREFLIGHT " + json.dumps(report, sort_keys=True, default=str))
    return 0 if report["status"] in {"PASS", "PARTIAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
