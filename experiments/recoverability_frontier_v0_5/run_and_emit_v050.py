#!/usr/bin/env python3
"""Run a v0.5.0 decoder and emit its compressed result through immutable job logs.

The wrapper forwards all arguments to train_decoder_v050_optimized.py.  It then
prints a deterministic gzip+base64 representation of the result JSON in bounded
parts, together with raw and compressed SHA-256 hashes.  This is a fallback for
compute environments where Hub write credentials are unavailable.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PART_CHARS = 60000


def find_output(arguments: list[str]) -> Path:
    for index, value in enumerate(arguments):
        if value == "--output" and index + 1 < len(arguments):
            return Path(arguments[index + 1])
        if value.startswith("--output="):
            return Path(value.split("=", 1)[1])
    raise SystemExit("run_and_emit_v050.py requires --output PATH")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    forwarded = sys.argv[1:]
    output = find_output(forwarded)
    target = Path(__file__).with_name("train_decoder_v050_optimized.py")
    command = [sys.executable, "-u", str(target), *forwarded]
    subprocess.run(command, check=True)

    raw = output.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    parts = [encoded[i : i + PART_CHARS] for i in range(0, len(encoded), PART_CHARS)]
    metadata = {
        "format": "gzip+base64",
        "output_name": output.name,
        "raw_bytes": len(raw),
        "raw_sha256": sha256(raw),
        "compressed_bytes": len(compressed),
        "compressed_sha256": sha256(compressed),
        "encoded_chars": len(encoded),
        "parts": len(parts),
        "pid": os.getpid(),
    }
    print("V050_ARTIFACT_META " + json.dumps(metadata, sort_keys=True), flush=True)
    for index, part in enumerate(parts):
        print(f"V050_ARTIFACT_PART {index:04d}/{len(parts):04d} {part}", flush=True)
    print("V050_ARTIFACT_END " + metadata["raw_sha256"], flush=True)


if __name__ == "__main__":
    main()
