#!/usr/bin/env python3
"""Decode the frozen v0.2 training source payload.

The initial GitHub contents upload omitted three base64 characters at two
known boundaries. This decoder repairs only those literal omissions, validates
the frozen raw-source SHA-256, and writes the executable Python source.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
from pathlib import Path

EXPECTED_B64_LENGTH = 12944
EXPECTED_RAW_LENGTH = 32158
EXPECTED_RAW_SHA256 = "3ace252ecbf58f63b4144a2f61586e57763adb1a484fcc783c85cc1fc8baf7ec"
REPAIRS = (
    ("MDHJLilBzgevh", "MDHJLilBzgArevh"),
    ("CAUzPDFFtCNjGzKF", "CAUzPDFFtCN1jGzKF"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    encoded = args.payload.read_text(encoding="utf-8").strip()
    for old, new in REPAIRS:
        if encoded.count(old) != 1:
            raise RuntimeError(f"Expected exactly one repair site: {old!r}")
        encoded = encoded.replace(old, new, 1)
    if len(encoded) != EXPECTED_B64_LENGTH:
        raise RuntimeError(f"Unexpected repaired payload length: {len(encoded)}")

    source = gzip.decompress(base64.b64decode(encoded, validate=True))
    digest = hashlib.sha256(source).hexdigest()
    if len(source) != EXPECTED_RAW_LENGTH or digest != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"Frozen source verification failed: bytes={len(source)} sha256={digest}"
        )
    args.output.write_bytes(source)
    print(f"SOURCE_OK {len(source)} {digest}")


if __name__ == "__main__":
    main()
