#!/usr/bin/env python3
"""End-to-end Historical-WI smoke for preregistered calibration v1.3.

The script derives v1.3 from the immutable v1.2 smoke source, substitutes the
user-provided DINOv3 ViT-7B/16 bucket, verifies and localizes the checkpoint,
then runs the same external-control pipeline.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
import urllib.request

V12_SCRIPT_URL = (
    "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
    "edf418ff10a8f26db6e01fb5fcd1a5b68f0046e5/"
    "experiments/blind_stroke_palaeography_v1/code/uv_v1_2_smoke.py"
)
BUCKET_ID = "Digitalgoldfish79/dinov3-vit7b16-pretrain-lvd1689m-bucket"
BUCKET_MOUNT = pathlib.Path("/model")
LOCAL_MODEL = pathlib.Path("/tmp/dinov3-vit7b16-local")

MODEL_FILES = {
    "config.json": (746, "b317786dd342ad8f51ce8246f39754ba648c7d375ad75d7b415507fe58d74ce6"),
    "preprocessor_config.json": (585, "960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7"),
    "model.safetensors.index.json": (48723, "ae26856def93bcf537202109c60ef76ca22d1f373dc85d2b68aeb6fe940c85fd"),
    "model-00001-of-00006.safetensors": (4980241600, "7132627f25459ee8797cb2965d3427706a87119ccfcdfef1bd7977dd7580821f"),
    "model-00002-of-00006.safetensors": (4967510232, "a7b17660c408adf235c318328010ecded0ee24181b97b806c1e45b42efc5ff4b"),
    "model-00003-of-00006.safetensors": (4967510568, "b5937a7a7051239798a6984d07e2d68fb1f8f93d0947c63be9ffe1bbbbe8dab7"),
    "model-00004-of-00006.safetensors": (4967543448, "569c817cc4424410c1d49df48e053ba4206eb7a1bd2c53381b7f7dfb32c4d57e"),
    "model-00005-of-00006.safetensors": (4967543320, "51ab5686ebe67cb48b738caee366d6a0bd0fb19b3f0d37dc37b8e82bf34c0e66"),
    "model-00006-of-00006.safetensors": (2013860920, "d3f77e2cbd0f9a349eeaf2559213f2495e43e4229a02a33f083f48ab5539ecbb"),
}


def load_v12_script():
    destination = pathlib.Path("/tmp/uv_v1_2_smoke.py")
    with urllib.request.urlopen(V12_SCRIPT_URL) as response:
        raw = response.read()
    destination.write_bytes(raw)
    spec = importlib.util.spec_from_file_location("uv_v1_2_frozen", destination)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen v1.2 script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def derive_v1_3() -> str:
    v12 = load_v12_script()
    source = v12.derive(v12.reconstruct())
    old_dino = '''    if kind == "dinov3":
        mid = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        proc = AutoImageProcessor.from_pretrained(mid, token=token)
        model = AutoModel.from_pretrained(mid, token=token, torch_dtype=torch.float16)'''
    new_dino = '''    if kind == "dinov3":
        # v1.3: checksum-verified DINOv3 ViT-7B/16 bucket, localized before load.
        mid = os.environ["DINO_MODEL_PATH"]
        proc = AutoImageProcessor.from_pretrained(mid, local_files_only=True)
        model = AutoModel.from_pretrained(
            mid,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )'''
    substitutions = [
        (old_dino, new_dino),
        ('            if device == "cuda":\n                px = px.half()',
         '            if device == "cuda":\n                px = px.to(dtype=next(model.parameters()).dtype)'),
        ('"schema": "blind-pal-external-calibration-v1.2"',
         '"schema": "blind-pal-external-calibration-v1.3"'),
        ('"dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m"',
         f'"dinov3": "{BUCKET_ID}"'),
    ]
    for old, new in substitutions:
        if source.count(old) != 1:
            raise RuntimeError(f"unexpected v1.3 substitution count: {old[:100]!r}")
        source = source.replace(old, new, 1)
    return source


def localize_model() -> None:
    LOCAL_MODEL.mkdir(parents=True, exist_ok=True)
    total = sum(size for size, _ in MODEL_FILES.values())
    print("V13_LOCALIZE_BEGIN " + json.dumps({"bucket": BUCKET_ID, "bytes": total}, sort_keys=True), flush=True)
    for name, (expected_size, expected_sha) in MODEL_FILES.items():
        src, dst = BUCKET_MOUNT / name, LOCAL_MODEL / name
        started, copied = time.time(), 0
        digest = hashlib.sha256()
        print("V13_LOCALIZE_FILE_BEGIN " + json.dumps({"name": name, "bytes": expected_size}, sort_keys=True), flush=True)
        with src.open("rb", buffering=0) as r, dst.open("wb", buffering=0) as w:
            while True:
                chunk = r.read(16 * 1024 * 1024)
                if not chunk:
                    break
                w.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
                if copied == expected_size or copied % (1024 * 1024 * 1024) < len(chunk):
                    elapsed = max(time.time() - started, 1e-6)
                    print("V13_LOCALIZE_PROGRESS " + json.dumps({
                        "name": name, "copied": copied, "total": expected_size,
                        "pct": round(100 * copied / expected_size, 2),
                        "MiB_s": round(copied / 2**20 / elapsed, 2),
                    }, sort_keys=True), flush=True)
        actual_sha = digest.hexdigest()
        if copied != expected_size or actual_sha != expected_sha:
            raise RuntimeError(f"verification failed for {name}: {copied}, {actual_sha}")
        print("V13_LOCALIZE_FILE_END " + json.dumps({
            "name": name, "sha256": actual_sha,
            "seconds": round(time.time() - started, 2),
        }, sort_keys=True), flush=True)
    print("V13_LOCALIZE_COMPLETE", flush=True)


def load_calibration(source: str):
    raw = source.encode("utf-8")
    print("V13_DERIVED_SOURCE " + json.dumps({
        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
    }, sort_keys=True), flush=True)
    destination = pathlib.Path("/tmp/external_calibration_v1_3.py")
    destination.write_bytes(raw)
    spec = importlib.util.spec_from_file_location("external_calibration_v1_3", destination)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create v1.3 module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def main() -> int:
    source = derive_v1_3()
    if "--derive-only" in sys.argv:
        raw = source.encode("utf-8")
        print("V13_DERIVE_ONLY " + json.dumps({
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }, sort_keys=True), flush=True)
        return 0

    localize_model()
    os.environ["DINO_MODEL_PATH"] = str(LOCAL_MODEL)
    module = load_calibration(source)
    module.upload_directory = lambda path, repo, token, path_in_repo: {
        "transport": "job_log", "path": path_in_repo,
    }
    work = "/tmp/blindpal_smoke_v13"
    sys.argv = [
        "external_calibration_v1_3.py",
        "--corpus", "historical_wi",
        "--work", work,
        "--output-repo", "Digitalgoldfish79/blind-scribal-hands-v1",
        "--max-writers", "20",
        "--pages-per-writer", "3",
        "--fragments-per-page", "1",
        "--max-tiles", "2",
        "--workers", "32",
        "--batch-size", "4",
        "--permutations", "3",
        "--panel-seed", "20260717",
    ]
    rc = int(module.main())
    result = pathlib.Path(work) / "historical_wi" / "output" / "calibration_result.json"
    print("V13_END_TO_END_SMOKE_RESULT " + result.read_text(encoding="utf-8"), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
