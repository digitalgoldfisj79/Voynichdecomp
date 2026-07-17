#!/usr/bin/env python3
"""Deterministic v1.5.1 reproduction wrapper with durable Hub persistence.

This wrapper deliberately preserves the exact assembled v1.5.1 scientific source.
It adds only:

1. a fail-fast Hugging Face write-permission test before expensive computation;
2. durable upload of the complete output directory, including the checkpoint;
3. a separate, non-self-referential SHA-256 manifest and environment record.

Required environment variables:

- HF_TOKEN
- SAGHOG_RESULT_REPO, e.g. Digitalgoldfish79/voynich-saghog-v15-reproduction-20260717
- SAGHOG_RUN_ID, a stable path component for this reproduction
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import runpy
import sys
import time
import urllib.request
from typing import Any

from huggingface_hub import HfApi

PARTS_COMMIT = "7541f99629eb68c4e5663478b828054a07459039"
ROOT = (
    "https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/"
    + PARTS_COMMIT
    + "/experiments/blind_stroke_palaeography_v1/code/v1_5_parts/"
)
PARENT_BYTES = 23391
PARENT_SHA256 = "e064648d07e28eac56a2f46012012d5e472aacc4e44dfa81c7018235b220b934"
DERIVED_BYTES = 23391
DERIVED_SHA256 = "fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8"
OLD = "MAX_WRITERS = 48 if PREFLIGHT else None"
NEW = "MAX_WRITERS = 80 if PREFLIGHT else None"
OUT = pathlib.Path("/tmp/saghog_v15_full/output")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def assemble_source() -> bytes:
    parent = b"".join(
        urllib.request.urlopen(ROOT + f"part{i:02d}.pyfrag", timeout=120).read()
        for i in range(7)
    )
    parent_sha = hashlib.sha256(parent).hexdigest()
    if len(parent) != PARENT_BYTES or parent_sha != PARENT_SHA256:
        raise RuntimeError(
            f"parent v1.5 mismatch: bytes={len(parent)}, sha256={parent_sha}"
        )

    source = parent.decode("utf-8")
    if source.count(OLD) != 1:
        raise RuntimeError("expected exactly one v1.5.1 substitution target")

    derived = source.replace(OLD, NEW, 1).encode("utf-8")
    derived_sha = hashlib.sha256(derived).hexdigest()
    if len(derived) != DERIVED_BYTES or derived_sha != DERIVED_SHA256:
        raise RuntimeError(
            f"derived v1.5.1 mismatch: bytes={len(derived)}, sha256={derived_sha}"
        )
    return derived


def permission_preflight(api: HfApi, repo_id: str, run_id: str) -> dict[str, Any]:
    identity = api.whoami()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

    payload = {
        "schema": "blind-pal-saghog-persistence-preflight-v1",
        "run_id": run_id,
        "timestamp_unix": time.time(),
        "authenticated_as": identity.get("name"),
        "repo_id": repo_id,
    }
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"write_tests/{run_id}.json",
        path_or_fileobj=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        commit_message=f"Write-permission preflight for {run_id}",
    )
    print("SAGHOG_PERSIST_WRITE_TEST " + json.dumps(payload, sort_keys=True), flush=True)
    return payload


def package_versions() -> dict[str, str]:
    names = [
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "einops",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "pandas",
        "pytorch-metric-learning",
        "huggingface-hub",
    ]
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def build_manifest() -> dict[str, Any]:
    if not OUT.is_dir():
        raise RuntimeError(f"expected output directory does not exist: {OUT}")

    files: dict[str, Any] = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {
            "SHA256_MANIFEST.json",
            "REPRODUCTION_METADATA.json",
        }:
            files[str(path.relative_to(OUT))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return {
        "schema": "blind-pal-saghog-v1.5.1-persisted-files-v1",
        "files": files,
    }


def main() -> int:
    token = require_env("HF_TOKEN")
    repo_id = require_env("SAGHOG_RESULT_REPO")
    run_id = require_env("SAGHOG_RUN_ID")

    api = HfApi(token=token)

    # This must succeed before any model training or expensive data processing.
    permission_preflight(api, repo_id, run_id)

    derived = assemble_source()
    destination = pathlib.Path("/tmp/saghog_v1_5_1_full.py")
    destination.write_bytes(derived)
    print(
        "SAGHOG_PERSIST_ASSEMBLED "
        + json.dumps(
            {
                "bytes": len(derived),
                "sha256": hashlib.sha256(derived).hexdigest(),
                "parent_commit": PARTS_COMMIT,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    # Executes the byte-identical v1.5.1 scientific programme.
    runpy.run_path(str(destination), run_name="__main__")

    checkpoint = OUT / "saghog_v15_best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("scientific run completed without saghog_v15_best.pt")

    manifest = build_manifest()
    (OUT / "SHA256_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    metadata = {
        "schema": "blind-pal-saghog-v1.5.1-reproduction-metadata-v1",
        "run_id": run_id,
        "repo_id": repo_id,
        "scientific_source_bytes": len(derived),
        "scientific_source_sha256": hashlib.sha256(derived).hexdigest(),
        "parent_parts_commit": PARTS_COMMIT,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "environment_controls": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    (OUT / "REPRODUCTION_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=OUT,
        path_in_repo=f"runs/{run_id}",
        commit_message=f"Persist complete SAGHOG v1.5.1 reproduction {run_id}",
    )

    final = {
        "repo_id": repo_id,
        "run_id": run_id,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(OUT / "SHA256_MANIFEST.json"),
        "result_sha256": sha256_file(OUT / "result.json"),
        "feature_sha256": sha256_file(OUT / "exact_features.npz"),
        "writer_split_sha256": sha256_file(OUT / "writer_split.json"),
    }
    print("SAGHOG_PERSIST_RESULT " + json.dumps(final, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
