#!/usr/bin/env python3
"""Run preregistered external calibration v1.2 and emit compact results to logs.

This wrapper changes only result transport. Scientific source derivation and
its digest checks are delegated to external_calibration_v1_2_launcher.py.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import external_calibration_v1_2_launcher as frozen


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
    metadata = {"schema": "external-calibration-log-bundle-v1.2", "files": {}}
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


def load_frozen_module():
    source = frozen.derive_v1_2(frozen.reconstruct_parent())
    destination = Path("/tmp/external_calibration_v1_2.py")
    destination.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("external_calibration_v1_2", destination)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load v1.2 calibration module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    corpus = arg_value("--corpus")
    work = Path(arg_value("--work", "/tmp/blindpal") or "/tmp/blindpal")
    if not corpus:
        raise SystemExit("--corpus is required")
    calibration = load_frozen_module()
    calibration.upload_directory = no_remote_upload
    rc = calibration.main()
    emit_bundle(work / corpus / "output")
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
