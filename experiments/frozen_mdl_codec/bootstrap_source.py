#!/usr/bin/env python3
"""Reconstruct and safely extract the frozen MDL codec v0.1 source bundle."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "SOURCE_BUNDLE_MANIFEST.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_members(archive: tarfile.TarFile):
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive member: {member.name!r}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not permitted: {member.name!r}")
        yield member


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=HERE / "expanded")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    chunks = []
    for expected in manifest["parts"]:
        path = HERE / "source_bundle" / expected["name"]
        data = path.read_bytes()
        if len(data) != expected["bytes"]:
            raise SystemExit(f"size mismatch: {path}")
        if digest(data) != expected["sha256"]:
            raise SystemExit(f"SHA-256 mismatch: {path}")
        chunks.append(data)

    encoded = b"".join(chunks)
    if digest(encoded) != manifest["base64_sha256"]:
        raise SystemExit("combined base64 SHA-256 mismatch")
    if len(encoded) != manifest["base64_bytes"]:
        raise SystemExit("combined base64 size mismatch")

    archive_bytes = base64.b64decode(encoded, validate=True)
    if digest(archive_bytes) != manifest["archive_sha256"]:
        raise SystemExit("archive SHA-256 mismatch")
    if len(archive_bytes) != manifest["archive_bytes"]:
        raise SystemExit("archive size mismatch")

    archive_path = HERE / manifest["archive_name"]
    archive_path.write_bytes(archive_bytes)
    args.target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(args.target, members=safe_members(archive), filter="data")

    print(json.dumps({
        "status": "PASS",
        "archive": str(archive_path),
        "archive_sha256": digest(archive_bytes),
        "source_root": str(args.target / manifest["root"]),
    }, indent=2))


if __name__ == "__main__":
    main()
