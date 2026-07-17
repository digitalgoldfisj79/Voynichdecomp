#!/usr/bin/env python3
"""Run the frozen external calibration and emit a compact result bundle to logs.

This wrapper changes only result transport. It does not alter features, folds,
models, thresholds, seeds, or selection logic.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import external_calibration as calibration


def arg_value(flag: str, default: str | None = None) -> str | None:
    try:
        return sys.argv[sys.argv.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def no_remote_upload(path: Path, repo: str, token: str | None, path_in_repo: str):
    return {
        "transport": "job_log_bundle",
        "requested_repo": repo,
        "path": path_in_repo,
        "reason": "connected OAuth lacks Hugging Face repository commit scope",
    }


def emit_bundle(out_dir: Path) -> None:
    include = [
        "calibration_result.json",
        "data_audit.json",
        "panel_manifest.csv",
        "SHA256_MANIFEST.json",
    ]
    metadata = {"schema": "external-calibration-log-bundle-v1", "files": {}}
    for name in include + ["features_compact.npz"]:
        path = out_dir / name
        if path.is_file():
            metadata["files"][name] = {
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "included": name in include,
            }
    (out_dir / "LOG_BUNDLE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    include.append("LOG_BUNDLE_METADATA.json")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in include:
            path = out_dir / name
            if path.is_file():
                tar.add(path, arcname=name)
    raw = buf.getvalue()
    encoded = base64.b64encode(raw).decode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    print(
        "CAL_LOG_BUNDLE_BEGIN "
        + json.dumps(
            {
                "encoding": "base64(tar.gz)",
                "bytes": len(raw),
                "sha256": digest,
                "chunks": (len(encoded) + 2999) // 3000,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for index in range(0, len(encoded), 3000):
        print(
            f"CAL_LOG_BUNDLE_CHUNK {index // 3000:05d} {encoded[index:index + 3000]}",
            flush=True,
        )
    print("CAL_LOG_BUNDLE_END " + digest, flush=True)


def main() -> int:
    corpus = arg_value("--corpus")
    work = Path(arg_value("--work", "/tmp/blindpal") or "/tmp/blindpal")
    if not corpus:
        raise SystemExit("--corpus is required")
    calibration.upload_directory = no_remote_upload
    rc = calibration.main()
    emit_bundle(work / corpus / "output")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
