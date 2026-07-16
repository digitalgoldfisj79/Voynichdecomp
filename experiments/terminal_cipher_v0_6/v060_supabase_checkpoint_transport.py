#!/usr/bin/env python3
"""Execution-only lossless checkpoint sharding over Supabase Storage.

The transport preserves the exact bytes produced by ``torch.save``. It does not
alter tensors, model configuration, training data, optimisation, or selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import requests
from tusclient import client as tus_client

DEFAULT_CHUNK_BYTES = 40 * 1024 * 1024
TUS_CHUNK_BYTES = 6 * 1024 * 1024


def sha256_file(path: Path, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def split_exact_bytes(source: Path, working: Path, chunk_bytes: int) -> list[dict[str, Any]]:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    working.mkdir(parents=True, exist_ok=True)
    parts: list[dict[str, Any]] = []
    with source.open("rb") as src:
        index = 0
        while True:
            payload = src.read(chunk_bytes)
            if not payload:
                break
            part_path = working / f"{source.name}.part-{index:05d}"
            part_path.write_bytes(payload)
            parts.append(
                {
                    "index": index,
                    "filename": part_path.name,
                    "local_path": str(part_path),
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            index += 1
    if not parts:
        raise RuntimeError(f"checkpoint is empty: {source}")
    return parts


def _tus_headers(publishable_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {publishable_key}",
        "apikey": publishable_key,
    }


def upload_tus(
    local_path: Path,
    object_path: str,
    *,
    project_ref: str,
    publishable_key: str,
    bucket: str,
) -> None:
    endpoint = f"https://{project_ref}.storage.supabase.co/storage/v1/upload/resumable"
    client = tus_client.TusClient(endpoint, headers=_tus_headers(publishable_key))
    with local_path.open("rb") as stream:
        uploader = client.uploader(
            file_stream=stream,
            chunk_size=TUS_CHUNK_BYTES,
            metadata={
                "bucketName": bucket,
                "objectName": object_path,
                "contentType": "application/octet-stream",
                "cacheControl": "3600",
            },
        )
        uploader.upload()


def signed_download_url(signer_url: str, object_path: str) -> str:
    response = requests.post(
        signer_url,
        json={"action": "download", "path": object_path},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    url = payload.get("signedUrl")
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"signer did not return signedUrl for {object_path}")
    return url


def download_to(url: str, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for block in response.iter_content(4 * 1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                digest.update(block)
                total += len(block)
    return total, digest.hexdigest()


def persist_checkpoint(
    checkpoint_path: Path,
    *,
    object_prefix: str,
    project_ref: str,
    publishable_key: str,
    bucket: str,
    signer_url: str,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    verify_roundtrip: bool = True,
) -> dict[str, Any]:
    """Upload exact checkpoint bytes as bounded parts plus a manifest.

    When ``verify_roundtrip`` is true, every uploaded part is downloaded through
    a fresh signed URL, reassembled, and checked against the original byte-level
    SHA-256 before this function returns.
    """
    checkpoint_path = checkpoint_path.resolve()
    whole_size = checkpoint_path.stat().st_size
    whole_sha = sha256_file(checkpoint_path)

    with tempfile.TemporaryDirectory(prefix="v060-checkpoint-shards-") as tmp:
        working = Path(tmp)
        parts = split_exact_bytes(checkpoint_path, working / "parts", chunk_bytes)
        remote_parts: list[dict[str, Any]] = []
        for part in parts:
            local_part = Path(part["local_path"])
            object_path = f"{object_prefix}/{part['filename']}"
            upload_tus(
                local_part,
                object_path,
                project_ref=project_ref,
                publishable_key=publishable_key,
                bucket=bucket,
            )
            remote_parts.append(
                {
                    "index": part["index"],
                    "object_path": object_path,
                    "bytes": part["bytes"],
                    "sha256": part["sha256"],
                }
            )

        manifest: dict[str, Any] = {
            "format": "v060-exact-byte-shards-v1",
            "original_filename": checkpoint_path.name,
            "original_bytes": whole_size,
            "original_sha256": whole_sha,
            "chunk_bytes": chunk_bytes,
            "parts": remote_parts,
        }
        manifest_path = working / f"{checkpoint_path.name}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_object_path = f"{object_prefix}/{manifest_path.name}"
        upload_tus(
            manifest_path,
            manifest_object_path,
            project_ref=project_ref,
            publishable_key=publishable_key,
            bucket=bucket,
        )
        manifest["manifest_object_path"] = manifest_object_path
        manifest["manifest_sha256"] = sha256_file(manifest_path)

        if verify_roundtrip:
            rebuilt = working / f"rebuilt-{checkpoint_path.name}"
            with rebuilt.open("wb") as output:
                for part in remote_parts:
                    downloaded = working / "downloaded" / Path(part["object_path"]).name
                    size, digest = download_to(
                        signed_download_url(signer_url, part["object_path"]),
                        downloaded,
                    )
                    if size != part["bytes"] or digest != part["sha256"]:
                        raise RuntimeError(
                            f"part verification failed for {part['object_path']}: "
                            f"size={size}/{part['bytes']} sha={digest}/{part['sha256']}"
                        )
                    with downloaded.open("rb") as source:
                        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            rebuilt_size = rebuilt.stat().st_size
            rebuilt_sha = sha256_file(rebuilt)
            if rebuilt_size != whole_size or rebuilt_sha != whole_sha:
                raise RuntimeError(
                    "whole checkpoint verification failed: "
                    f"size={rebuilt_size}/{whole_size} sha={rebuilt_sha}/{whole_sha}"
                )
            manifest["roundtrip_verified"] = True
            manifest["roundtrip_bytes"] = rebuilt_size
            manifest["roundtrip_sha256"] = rebuilt_sha

    return manifest


def transport_from_environment() -> dict[str, str]:
    required = {
        "project_ref": "V060_SUPABASE_PROJECT_REF",
        "publishable_key": "V060_SUPABASE_PUBLISHABLE_KEY",
        "bucket": "V060_SUPABASE_BUCKET",
        "signer_url": "V060_SUPABASE_SIGNER_URL",
    }
    result: dict[str, str] = {}
    missing: list[str] = []
    for key, environment_name in required.items():
        value = os.environ.get(environment_name)
        if not value:
            missing.append(environment_name)
        else:
            result[key] = value
    if missing:
        raise RuntimeError("missing persistence environment: " + ", ".join(missing))
    return result
